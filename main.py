from typing import Optional
from pathlib import Path
import functools

from jsonargparse import CLI

import jax

from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils

import numpy as onp

import tensorflow as tf

from gga import vae
from gga import humanml3d
from gga import plot_smpl

HumanML3DConfig = humanml3d.HumanML3DConfig
LoopConfig = vae.train.LoopConfig
TransformerConfig = vae.models.TransformerConfig


def train_vae(
    seed: int,
    use_mixed_precision: bool,
    data_config: HumanML3DConfig,
    loop_config: LoopConfig,
    checkpoint: Optional[Path] = None,
):
    print(data_config)
    print(loop_config)

    ds = humanml3d.load_motion(loop_config, data_config)
    train_ds, eval_ds = [ds[key] for key in ["train", "val"]]

    n_devices = jax.local_device_count()
    assert (
        loop_config.batch_size % n_devices == 0
    ), "batch size must be divisable by number of devices"

    model_config = TransformerConfig()
    state: vae.train.TrainState = vae.train.create_state(
        seed,
        use_mixed_precision,
        loop_config.batch_size // n_devices,
        model_config,
        loop_config,
    )

    if checkpoint is not None:
        state = checkpoints.restore_checkpoint(checkpoint, state)

    mean = jax_utils.replicate(ds["mean"])
    std = jax_utils.replicate(ds["std"])

    state, history = vae.train.train_loop(
        seed,
        state,
        train_ds,
        eval_ds,
        mean,
        std,
        loop_config,
        model_config,
    )


def plot_skeletons(
    seed: int,
    use_mixed_precision: bool,
    data_config: HumanML3DConfig,
    loop_config: LoopConfig,
    checkpoint: Optional[Path] = None,
):
    ds = humanml3d.load_motion_text(loop_config, data_config)
    train_ds, eval_ds = [ds[key] for key in ["train", "val"]]

    n_devices = jax.local_device_count()
    assert (
        loop_config.batch_size % n_devices == 0
    ), "batch size must be divisable by number of devices"

    model_config = TransformerConfig()
    state: vae.train.TrainState = vae.train.create_state(
        seed,
        use_mixed_precision,
        loop_config.batch_size // n_devices,
        model_config,
        loop_config,
    )

    if checkpoint is not None:
        state = checkpoints.restore_checkpoint(checkpoint, state)

    state = jax_utils.replicate(state)

    rng = jax.random.PRNGKey(seed)
    rng, dropout_rng, noise_rng = jax.random.split(rng, 3)
    dropout_rngs = jax.random.split(dropout_rng, jax.local_device_count())
    noise_rngs = jax.random.split(noise_rng, jax.local_device_count())

    p_generate_step = jax.pmap(
        functools.partial(vae.train.generate_step, config=model_config),
        axis_name="device",
    )

    mean = jax_utils.replicate(ds["mean"])
    std = jax_utils.replicate(ds["std"])

    for batch in train_ds.as_numpy_iterator():
        batched_text: onp.ndarray = batch.pop("text")

        batch = common_utils.shard(batch)

        positions = p_generate_step(
            state,
            batch,
            mean=mean,
            std=std,
            dropout_rng=dropout_rngs,
            noise_rng=noise_rngs,
        )[0]
        mask = batch["mask"][0]

        positions = onp.array(positions)
        batched_mask = onp.array(mask)

        for i, (pos, mask, text) in enumerate(
            zip(positions, batched_mask, batched_text)
        ):
            text = text[0].decode()
            plot_smpl.plot_skeleton(f"{i}.gif", pos[mask], text, fps=20)

        break


def plot_skeletons_recons(
    seed: int,
    use_mixed_precision: bool,
    data_config: HumanML3DConfig,
    loop_config: LoopConfig,
    checkpoint: Optional[Path] = None,
):
    ds = humanml3d.load_motion_text(loop_config, data_config)
    train_ds, eval_ds = [ds[key] for key in ["train", "val"]]

    n_devices = jax.local_device_count()
    assert (
        loop_config.batch_size % n_devices == 0
    ), "batch size must be divisable by number of devices"

    model_config = TransformerConfig()
    state: vae.train.TrainState = vae.train.create_state(
        seed,
        use_mixed_precision,
        loop_config.batch_size // n_devices,
        model_config,
        loop_config,
    )

    if checkpoint is not None:
        state = checkpoints.restore_checkpoint(checkpoint, state)

    state = jax_utils.replicate(state)

    rng = jax.random.PRNGKey(seed)
    rng, dropout_rng, noise_rng = jax.random.split(rng, 3)
    dropout_rngs = jax.random.split(dropout_rng, jax.local_device_count())
    noise_rngs = jax.random.split(noise_rng, jax.local_device_count())

    p_reconstruct = jax.pmap(
        functools.partial(vae.train.reconstruct, config=model_config),
        axis_name="device",
    )

    mean = jax_utils.replicate(ds["mean"])
    std = jax_utils.replicate(ds["std"])

    for batch in train_ds.as_numpy_iterator():
        batched_text: onp.ndarray = batch.pop("text")

        batch = common_utils.shard(batch)

        positions = p_reconstruct(
            state,
            batch,
            mean=mean,
            std=std,
            dropout_rng=dropout_rngs,
            noise_rng=noise_rngs,
        )[0]
        mask = batch["mask"][0]

        positions = onp.array(positions)
        batched_mask = onp.array(mask)

        for i, (pos, mask, text) in enumerate(
            zip(positions, batched_mask, batched_text)
        ):
            text = text[0].decode()
            plot_smpl.plot_skeleton(f"{i}.gif", pos[mask], text, fps=20)

        break


if __name__ == "__main__":
    tf.random.set_seed(0)
    tf.config.set_visible_devices([], "GPU")

    CLI()
