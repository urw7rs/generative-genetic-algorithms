from typing import Any, Callable, Optional, Tuple
from flax.linen.dtypes import promote_dtype

import functools

import numpy as np

from jax import lax
import jax
import jax.numpy as jnp

from flax import linen as nn
from flax import struct
from flax.linen.initializers import Initializer

from flax.linen import initializers
from flax.linen.linear import default_kernel_init
from flax.linen.linear import DenseGeneral
from flax.linen.linear import DotGeneralT
from flax.linen.linear import PrecisionLike
from flax.linen.module import compact


PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


def linear_attention(
    query: Array,
    key: Array,
    value: Array,
    mask: Optional[Array] = None,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
):
    """Computes dot-product attention given query, key, and value.

    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762. It calculates the attention weights given
    query and key and combines the values using the attention weights.

    Note: query, key, value needn't have any batch dimensions.

    Args:
      query: queries for calculating attention with shape of
        `[batch..., q_length, num_heads, qk_depth_per_head]`.
      key: keys for calculating attention with shape of
        `[batch..., kv_length, num_heads, qk_depth_per_head]`.
      value: values to be used in attention with shape of
        `[batch..., kv_length, num_heads, v_depth_per_head]`.
      bias: bias for the attention weights. This should be broadcastable to the
        shape `[batch..., num_heads, q_length, kv_length]`.
        This can be used for incorporating causal masks, padding masks,
        proximity bias, etc.
      mask: mask for the attention weights. This should be broadcastable to the
        shape `[batch..., num_heads, q_length, kv_length]`.
        This can be used for incorporating causal masks.
        Attention weights are masked out if their corresponding mask value
        is `False`.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      dtype: the dtype of the computation (default: infer from inputs)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.

    Returns:
      Output of shape `[batch..., q_length, num_heads, v_depth_per_head]`.
    """
    query, key, value = promote_dtype(query, key, value, dtype=dtype)
    dtype = query.dtype
    assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), "q, k, v batch dims must match."
    assert (
        query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), "q, k, v num_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."

    # apply attention mask
    if mask is not None:
        mask = mask[..., None, None]
        big_neg = jnp.finfo(dtype).min
        key = jnp.where(mask, key, big_neg)
        value = jnp.where(mask, value, 0)

    dim = query.shape[-1]

    query = jax.nn.softmax(query, axis=-1).astype(dtype)
    key = jax.nn.softmax(key, axis=-3).astype(dtype)

    query = query * dim**-0.5
    context = jnp.einsum("...nhd,...nhe->...hde", key, value, precision=precision)
    return jnp.einsum("...hde,...nhd->...nhe", context, query)


class LinearMultiHeadAttention(nn.Module):
    """Multi-head linear attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation
        (default: infer from inputs and params)
      param_dtype: the dtype passed to parameter initializers (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rate: dropout rate
      deterministic: if false, the attention weight is masked randomly
        using dropout, whereas if true, the attention weights
        are deterministic.
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      use_bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts
        query, key, value, and returns output of shape
        `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``
      decode: whether to prepare and use an autoregressive cache.
    """

    num_heads: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    precision: PrecisionLike = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
    use_bias: bool = True
    decode: bool = False
    qkv_dot_general: DotGeneralT = lax.dot_general
    out_dot_general: DotGeneralT = lax.dot_general

    @compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Array,
        mask: Optional[Array] = None,
    ):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        Args:
          inputs_q: input queries of shape
            `[batch_sizes..., length, features]`.
          inputs_kv: key/values of shape
            `[batch_sizes..., length, features]`.
          mask: attention mask of shape
            `[batch_sizes..., num_heads, query_length, key/value_length]`.
            Attention weights are masked out if their corresponding mask value
            is `False`.
          deterministic: if false, the attention weight is masked randomly
            using dropout, whereas if true, the attention weights
            are deterministic.

        Returns:
          output of shape `[batch_sizes..., length, features]`.
        """
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert (
            qkv_features % self.num_heads == 0
        ), "Memory dimension must be divisible by number of heads."
        head_dim = qkv_features // self.num_heads

        dense = functools.partial(
            DenseGeneral,
            axis=-1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, head_dim),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
            dot_general=self.qkv_dot_general,
        )
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        query, key, value = (
            dense(name="query")(inputs_q),
            dense(name="key")(inputs_kv),
            dense(name="value")(inputs_kv),
        )

        # apply attention
        x = linear_attention(
            query,
            key,
            value,
            mask=mask,
            dtype=self.dtype,
            precision=self.precision,
        )  # pytype: disable=wrong-keyword-args
        # back to the original inputs dimensions
        out = DenseGeneral(
            features=features,
            axis=(-2, -1),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            dot_general=self.out_dot_general,
            name="out",  # type: ignore[call-arg]
        )(x)
        return out


class LinearSelfAttention(LinearMultiHeadAttention):
    """Self-attention special case of multi-head dot-product attention."""

    @compact
    def __call__(
        self,
        inputs_q: Array,
        mask: Optional[Array] = None,  # type: ignore
    ):
        """Applies multi-head dot product self-attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        Args:
          inputs_q: input queries of shape
            `[batch_sizes..., length, features]`.
          mask: attention mask of shape
            `[batch_sizes..., num_heads, query_length, key/value_length]`.
            Attention weights are masked out if their corresponding mask value
            is `False`.
          deterministic: if false, the attention weight is masked randomly
            using dropout, whereas if true, the attention weights
            are deterministic.

        Returns:
          output of shape `[batch_sizes..., length, features]`.
        """
        return super().__call__(inputs_q, inputs_q, mask)


CheckpointSelfAttention = nn.checkpoint(nn.SelfAttention)
CheckpointMultiHeadDotProductAttention = nn.checkpoint(nn.MultiHeadDotProductAttention)


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
    num_layers: int = 9
    qkv_dim: int = 256
    mlp_dim: int = 1024
    max_len: int = 1024
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    deterministic: bool = False
    decode: bool = False
    kernel_init: Callable = nn.initializers.xavier_uniform()
    bias_init: Callable = nn.initializers.normal(stddev=1e-6)
    posemb_init: Initializer = nn.initializers.xavier_uniform()


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
        x = nn.gelu(x)
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
        x = LinearSelfAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            qkv_features=config.qkv_dim,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            use_bias=False,
        )(inputs, encoder_mask)

        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=config.deterministic)
        x = x + inputs

        x = nn.LayerNorm(dtype=config.dtype)(x)

        # MLP block.
        y = MlpBlock(config=config)(x)
        y = y + x

        return nn.LayerNorm(dtype=config.dtype)(y)


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
        x = LinearSelfAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            qkv_features=config.qkv_dim,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            use_bias=False,
            decode=config.decode,
        )(targets, decoder_mask)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=config.deterministic)
        x = x + targets

        x = nn.LayerNorm(dtype=config.dtype)(x)

        # Encoder-Decoder block.
        y = LinearMultiHeadAttention(
            num_heads=config.num_heads,
            dtype=config.dtype,
            qkv_features=config.qkv_dim,
            kernel_init=config.kernel_init,
            bias_init=config.bias_init,
            use_bias=False,
        )(x, encoded, encoder_decoder_mask)

        y = nn.Dropout(rate=config.dropout_rate)(y, deterministic=config.deterministic)
        y = y + x

        y = nn.LayerNorm(dtype=config.dtype)(y)

        # MLP block.
        z = MlpBlock(config=config)(y)

        z = z + y

        return nn.LayerNorm(dtype=config.dtype)(y)


@struct.dataclass
class EncoderOutput:
    mu: jax.Array
    logvar: jax.Array


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
        assert config.num_layers % 2 == 1

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
            nn.initializers.xavier_uniform(),
            (1, config.latent_length * 2, config.emb_dim),
        )
        dist_tokens = jnp.tile(dist_tokens, [x.shape[0], 1, 1])
        x = jnp.concatenate([dist_tokens, x], axis=1)
        x = AddPositionEmbs(config=config, decode=False, name="posembed_input")(x)
        x = nn.Dropout(rate=config.dropout_rate)(x, deterministic=config.deterministic)

        x = x.astype(config.dtype)

        # Input Encoder
        outputs = []
        for lyr in range(config.num_layers):
            if lyr > config.num_layers // 2:
                x = jnp.concatenate([x, outputs.pop()], axis=-1)
                x = nn.Dense(config.emb_dim)(x)

            x = Encoder1DBlock(config=config, name=f"encoderblock_{lyr}")(
                x, encoder_mask
            )

            if lyr < config.num_layers // 2:
                outputs.append(x)

        dist_tokens = x[:, : dist_tokens.shape[1]]
        mu, logvar = jnp.split(dist_tokens, 2, axis=1)
        return EncoderOutput(mu, logvar)


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
        assert config.num_layers % 2 == 1

        if not config.decode:
            y = shift_right(targets)
        else:
            y = jnp.zeros_like(targets)
        y = AddPositionEmbs(
            config=config, decode=config.decode, name="posembed_output"
        )(y)
        y = nn.Dropout(rate=config.dropout_rate)(y, deterministic=config.deterministic)

        y = y.astype(config.dtype)

        # Target-Input Decoder
        outputs = []
        for lyr in range(config.num_layers):
            if lyr > config.num_layers // 2:
                y = jnp.concatenate([y, outputs.pop()], axis=-1)

            y = EncoderDecoder1DBlock(config=config, name=f"encoderdecoderblock_{lyr}")(
                y,
                encoded,
                decoder_mask=decoder_mask,
                encoder_decoder_mask=encoder_decoder_mask,
            )

            if lyr < config.num_layers // 2:
                outputs.append(y)

        # Decoded Logits
        if config.recons_via_embedding:
            # Use the transpose of embedding matrix for logit transform.
            recons = output_embed.attend(y.astype(jnp.float32))
            # Correctly normalize pre-softmax logits for this shared case.
            recons = recons / jnp.sqrt(y.shape[-1])
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


@struct.dataclass
class TransformerOutput:
    recons: jax.Array
    mu: jax.Array
    logvar: jax.Array
    noise: jax.Array


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
        if self.config.decode:
            pass
        else:
            encoder_mask = input_mask

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
            decoder_mask = target_mask
            encoder_decoder_mask = encoded_mask

        reconstructed = self.decoder(
            encoded,
            jnp.zeros(
                (encoded.shape[0], target_mask.shape[1], encoded.shape[2]),
                dtype=encoded.dtype,
            ),
            decoder_mask=decoder_mask,
            encoder_decoder_mask=encoder_decoder_mask,
        )

        reconstructed = reconstructed * jnp.expand_dims(target_mask, axis=-1)
        return reconstructed.astype(self.config.dtype)

    def __call__(
        self,
        inputs,
        input_mask,
    ) -> TransformerOutput:
        """Applies Transformer model on the inputs.

        Args:
          inputs: input data.
          targets: target data.

        Returns:
          reconstructed array from full transformer.
        """
        output = self.encode(inputs, input_mask)
        mu = output.mu
        logvar = output.logvar

        std = jnp.exp(0.5 * logvar)

        rng = self.make_rng(self.rng_collection)
        noise = jax.random.normal(rng, mu.shape, dtype=mu.dtype)
        sampled_latent = std * noise + mu

        recons = self.decode(sampled_latent, input_mask)
        return TransformerOutput(recons, mu, logvar, noise)

    def generate(self, target_mask, dtype=jnp.float32):
        rng = self.make_rng(self.rng_collection)
        noise = jax.random.normal(
            rng, (*target_mask.shape, self.config.emb_dim), dtype=dtype
        )

        sample = self.decode(noise, target_mask)
        return sample
