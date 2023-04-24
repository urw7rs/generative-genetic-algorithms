from typing import Callable, Any, Optional

from flax import linen as nn
from flax import struct
from jax import lax
import jax
import jax.numpy as jnp
import numpy as np


@struct.dataclass
class TransformerConfig:
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

    input_size: int = 263
    output_size: int = 263
    latent_length: int = 7
    share_embeddings: bool = False
    recons_via_embedding: bool = False
    dtype: Any = jnp.float32
    emb_dim: int = 256
    num_heads: int = 4
    num_layers: int = 7
    qkv_dim: int = 256
    mlp_dim: int = 1024
    max_len: int = 1024
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    deterministic: bool = False
    decode: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    posemb_init: Optional[Callable] = None


def shift_right(x, axis=1):
    """Shift the input to the right by padding on axis 1."""
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (1, 0)
    padded = jnp.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(0))
    return padded[:, :-1]


def sinusoidal_init(max_len=2048, min_scale=1.0, max_scale=10000.0):
    """1D Sinusoidal Position Embedding Initializer.

    Args:
        max_len: maximum possible length for the input.
        min_scale: float: minimum frequency-scale in sine grating.
        max_scale: float: maximum frequency-scale in sine grating.

    Returns:
        output: init function returning `(1, max_len, d_feature)`
    """

    def init(key, shape, dtype=np.float32):
        """Sinusoidal init."""
        del key, dtype
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
        div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
        pe[:, : d_feature // 2] = np.sin(position * div_term)
        pe[:, d_feature // 2 : 2 * (d_feature // 2)] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
        return jnp.array(pe)

    return init


class AddPositionEmbs(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
      decode: whether to run in single-position autoregressive mode.
    """

    config: TransformerConfig
    decode: bool = False

    @nn.compact
    def __call__(self, inputs):
        """Applies AddPositionEmbs module.

        By default this layer uses a fixed sinusoidal embedding table. If a
        learned position embedding is desired, pass an initializer to
        posemb_init in the configuration.

        Args:
          inputs: input data.

        Returns:
          output: `(bs, timesteps, in_dim)`
        """
        config = self.config
        # inputs.shape is (batch_size, seq_len, emb_dim)
        assert inputs.ndim == 3, (
            "Number of dimensions should be 3," " but it is: %d" % inputs.ndim
        )
        length = inputs.shape[1]
        pos_emb_shape = (1, config.max_len, inputs.shape[-1])
        if config.posemb_init is None:
            # Use a fixed (non-learned) sinusoidal position embedding.
            pos_embedding = sinusoidal_init(max_len=config.max_len)(
                None, pos_emb_shape, None
            )
        else:
            pos_embedding = self.param(
                "pos_embedding", config.posemb_init, pos_emb_shape
            )
        pe = pos_embedding[:, :length, :]

        # We use a cache position index for tracking decoding position.
        if self.decode:
            is_initialized = self.has_variable("cache", "cache_index")
            cache_index = self.variable(
                "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.uint32)
            )
            if is_initialized:
                i = cache_index.value
                cache_index.value = i + 1
                _, _, df = pos_embedding.shape
                pe = lax.dynamic_slice(pos_embedding, jnp.array((0, i, 0)), (1, 1, df))
        return inputs + pe


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
      out_dim: optionally specify out dimension.
    """

    config: TransformerConfig
    out_dim: Optional[int] = None

    @nn.compact
    def __call__(self, inputs):
        """Applies Transformer MlpBlock module."""
        config = self.config
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(
            config.mlp_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(inputs)
        x = nn.relu(x)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=config.deterministic)
        output = nn.Dense(
            actual_out_dim,
            dtype=config.dtype,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
        )(x)
        output = nn.Dropout(rate=config.dropout_rate)(
            output, deterministic=config.deterministic
        )
        return output


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(self, inputs, encoder_mask=None):
        """Applies Encoder1DBlock module.

        Args:
          inputs: input data.
          encoder_mask: encoder self-attention mask.

        Returns:
          output after transformer encoder block.
        """
        config = self.config

        # Attention block.
        assert inputs.ndim == 3
        x = nn.LayerNorm(dtype=config.dtype)(inputs)
        x = nn.SelfAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            qkv_features=config.qkv_dim,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=config.attention_dropout_rate,
            deterministic=config.deterministic,
        )(x, encoder_mask)

        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=config.deterministic)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm(dtype=config.dtype)(x)
        y = MlpBlock(config=config)(y)

        return x + y


class EncoderDecoder1DBlock(nn.Module):
    """Transformer encoder-decoder layer.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(self, targets, encoded, decoder_mask=None, encoder_decoder_mask=None):
        """Applies EncoderDecoder1DBlock module.

        Args:
          targets: input data for decoder
          encoded: input data from encoder
          decoder_mask: decoder self-attention mask.
          encoder_decoder_mask: encoder-decoder attention mask.

        Returns:
          output after transformer encoder-decoder block.
        """
        config = self.config

        # Decoder block.
        assert targets.ndim == 3
        x = nn.LayerNorm(dtype=config.dtype)(targets)
        x = nn.SelfAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            qkv_features=config.qkv_dim,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=config.attention_dropout_rate,
            deterministic=config.deterministic,
            decode=config.decode,
        )(x, decoder_mask)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=config.deterministic)
        x = x + targets

        # Encoder-Decoder block.
        y = nn.LayerNorm(dtype=config.dtype)(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            qkv_features=config.qkv_dim,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=config.attention_dropout_rate,
            deterministic=config.deterministic,
        )(y, encoded, encoder_decoder_mask)

        y = nn.Dropout(rate=config.dropout_rate)(y, deterministic=config.deterministic)
        y = y + x

        # MLP block.
        z = nn.LayerNorm(dtype=config.dtype)(y)
        z = MlpBlock(config=config)(z)

        return y + z


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
      shared_embedding: a shared embedding layer to use.
    """

    config: TransformerConfig
    shared_embedding: Any = None

    @nn.compact
    def __call__(self, inputs, encoder_mask=None):
        """Applies Transformer model on the inputs.

        Args:
          inputs: input data
          encoder_mask: decoder self-attention mask.

        Returns:
          output of a transformer encoder.
        """
        config = self.config
        assert inputs.ndim == 3  # (batch, len, features)

        # Input Embedding
        if self.shared_embedding is None:
            input_embed = nn.Dense(
                features=config.emb_dim,
                kernel_init=config.kernel_init,
                bias_init=config.bias_init,
            )
        else:
            input_embed = self.shared_embedding
        x = input_embed(inputs)

        dist_tokens = self.param(
            "dist_tokens",
            nn.initializers.zeros,
            (1, config.latent_length * 2, config.emb_dim),
        )
        dist_tokens = jnp.tile(dist_tokens, [x.shape[0], 1, 1])
        x = jnp.concatenate([dist_tokens, x], axis=1)
        x = AddPositionEmbs(config=config, decode=False, name="posembed_input")(x)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=config.deterministic)

        x = x.astype(config.dtype)

        # Input Encoder
        for lyr in range(config.num_layers):
            x = Encoder1DBlock(config=config, name=f"encoderblock_{lyr}")(
                x, encoder_mask
            )

        encoded = nn.LayerNorm(dtype=config.dtype, name="encoder_norm")(x)
        dist_tokens = encoded[:, : dist_tokens.shape[1]]
        mu, logvar = jnp.split(dist_tokens, 2, axis=1)
        return mu, logvar


class Decoder(nn.Module):
    """Transformer Model Decoder for sequence to sequence translation.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
      shared_embedding: a shared embedding layer to use.
    """

    config: TransformerConfig
    shared_embedding: Any = None

    @nn.compact
    def __call__(
        self,
        encoded,
        targets,
        decoder_mask=None,
        encoder_decoder_mask=None,
    ):
        """Applies Transformer model on the inputs.

        Args:
          encoded: encoded input data from encoder.
          targets: target inputs.
          decoder_mask: decoder self-attention mask.
          encoder_decoder_mask: encoder-decoder attention mask.

        Returns:
          output of a transformer decoder.
        """
        config = self.config

        assert encoded.ndim == 3  # (batch, len, depth)
        assert targets.ndim == 3  # (batch, len, depth)

        # Target Embedding
        if self.shared_embedding is None:
            output_embed = nn.Dense(
                features=config.emb_dim,
                kernel_init=config.kernel_init,
                bias_init=config.bias_init,
            )
        else:
            output_embed = self.shared_embedding

        if not config.decode:
            y = shift_right(targets)
        else:
            y = jnp.zeros_like(targets)
        y = output_embed(y)
        y = AddPositionEmbs(
            config=config, decode=config.decode, name="posembed_output"
        )(y)
        y = nn.Dropout(rate=config.dropout_rate)(y, deterministic=config.deterministic)

        y = y.astype(config.dtype)

        # Target-Input Decoder
        for lyr in range(config.num_layers):
            y = EncoderDecoder1DBlock(config=config, name=f"encoderdecoderblock_{lyr}")(
                y,
                encoded,
                decoder_mask=decoder_mask,
                encoder_decoder_mask=encoder_decoder_mask,
            )
        y = nn.LayerNorm(dtype=config.dtype, name="encoderdecoder_norm")(y)

        # Decoded Logits
        if config.recons_via_embedding:
            # Use the transpose of embedding matrix for logit transform.
            logits = output_embed.attend(y.astype(jnp.float32))
            # Correctly normalize pre-softmax logits for this shared case.
            logits = logits / jnp.sqrt(y.shape[-1])
        else:
            recons = nn.Dense(
                config.output_size,
                dtype=config.dtype,
                kernel_init=config.kernel_init,
                bias_init=config.bias_init,
                name="logitdense",
            )(y)
        return recons


def get_mask(x):
    return jnp.sum(jnp.abs(x), axis=-1) != 0


class Transformer(nn.Module):
    """Transformer Model for sequence to sequence translation.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig
    rng_collection: str = "noise"

    def setup(self):
        config = self.config

        if config.share_embeddings:
            if config.output_size is not None:
                assert (
                    config.output_size == config.size
                ), "can't share embedding with different vocab sizes."
            self.shared_embedding = nn.Embed(
                num_embeddings=config.size,
                features=config.emb_dim,
                embedding_init=nn.initializers.normal(stddev=1.0),
            )
        else:
            self.shared_embedding = None

        self.encoder = Encoder(config=config, shared_embedding=self.shared_embedding)
        self.decoder = Decoder(config=config, shared_embedding=self.shared_embedding)

    def encode(self, inputs, input_mask):
        """Applies Transformer encoder-branch on the inputs.

        Args:
          inputs: input data.

        Returns:
          encoded feature array from the transformer encoder.
        """
        config = self.config
        # Make padding attention mask.
        input_mask = jnp.concatenate(
            [
                jnp.ones((inputs.shape[0], config.latent_length * 2)),
                input_mask,
            ],
            axis=1,
        )
        encoder_mask = nn.make_attention_mask(
            input_mask, input_mask, dtype=config.dtype
        )
        return self.encoder(inputs, encoder_mask=encoder_mask)

    def decode(
        self,
        encoded,
        target_mask,
    ):
        """Applies Transformer decoder-branch on encoded-input and target.

        Args:
          encoded: encoded input data from encoder.
          targets: target data.

        Returns:
          reconstructed array from transformer decoder.
          latent mean
          latent log variance
          noise used to sample latent
        """
        config = self.config

        # Make padding attention masks.
        if config.decode:
            # for fast autoregressive decoding only a special encoder-decoder mask is used
            decoder_mask = None
            encoded_mask = jnp.ones((encoded.shape[0], encoded.shape[1]))
            encoder_decoder_mask = nn.make_attention_mask(
                target_mask, encoded_mask, dtype=config.dtype
            )
        else:
            encoded_mask = jnp.ones((encoded.shape[0], encoded.shape[1]))
            decoder_mask = nn.make_attention_mask(
                target_mask,
                target_mask,
                dtype=config.dtype,
            )
            encoder_decoder_mask = nn.make_attention_mask(
                target_mask, encoded_mask, dtype=config.dtype
            )

        reconstructed = self.decoder(
            encoded,
            jnp.zeros(
                (encoded.shape[0], target_mask.shape[1], encoded.shape[2]),
                dtype=encoded.dtype,
            ),
            decoder_mask=decoder_mask,
            encoder_decoder_mask=encoder_decoder_mask,
        )

        reconstructed = jnp.where(
            jnp.tile(target_mask[:, :, None], [1, 1, reconstructed.shape[2]]),
            reconstructed,
            jnp.zeros_like(reconstructed),
        )

        return reconstructed.astype(self.config.dtype)

    def __call__(
        self,
        inputs,
        input_mask,
    ):
        """Applies Transformer model on the inputs.

        Args:
          inputs: input data.
          targets: target data.

        Returns:
          reconstructed array from full transformer.
        """
        mu, logvar = self.encode(inputs, input_mask)

        std = jnp.exp(0.5 * logvar)

        rng = self.make_rng(self.rng_collection)
        noise = jax.random.normal(rng, mu.shape, dtype=mu.dtype)
        sampled_latent = std * noise + mu

        recons = self.decode(sampled_latent, input_mask)
        return recons, mu, logvar, noise

    def generate(self, target_mask, dtype=jnp.float32):
        rng = self.make_rng(self.rng_collection)
        noise = jax.random.normal(
            rng, (*target_mask.shape, self.config.emb_dim), dtype=dtype
        )

        sample = self.decode(noise, target_mask)
        return sample
