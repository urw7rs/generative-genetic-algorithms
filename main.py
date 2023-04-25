from typing import Optional
from pathlib import Path
import functools

from jsonargparse import CLI

import numpy as onp

import tensorflow as tf

import jax

from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils

from gga import vae
from gga import humanml3d
from gga import plot_smpl


def train_vae(
    seed: int = 0,
    use_mixed_precision: bool = True,
    restore_checkpoints: bool = False,
    workdir: Optional[Path] = None,
    data_config: Optional[humanml3d.HumanML3DConfig] = None,
    loop_config: Optional[vae.train.LoopConfig] = None,
    model_config: Optional[vae.models.TransformerConfig] = None,
):
    if data_config is None:
        data_config = humanml3d.HumanML3DConfig()

    if loop_config is None:
        loop_config = vae.train.LoopConfig()

    if model_config is None:
        model_config = vae.models.TransformerConfig()

    ds = humanml3d.load_motion(loop_config, data_config)
    train_ds, eval_ds = [ds[key] for key in ["train", "val"]]

    n_devices = jax.local_device_count()
    assert (
        loop_config.batch_size % n_devices == 0
    ), "batch size must be divisable by number of devices"

    state: vae.train.TrainState = vae.train.create_state(
        seed,
        use_mixed_precision,
        loop_config.batch_size // n_devices,
        model_config,
        loop_config,
    )

    if restore_checkpoints:
        assert workdir is not None, "Speicfy checkpoint dir in workdir"
        state = checkpoints.restore_checkpoint(workdir, state)

    state, history = vae.train.train_loop(
        seed,
        state,
        train_ds,
        eval_ds,
        loop_config,
        model_config,
    )


def plot_skeletons(
    seed: int = 0,
    use_mixed_precision: bool = True,
    restore_checkpoints: bool = False,
    workdir: Optional[Path] = None,
    data_config: Optional[humanml3d.HumanML3DConfig] = None,
    loop_config: Optional[vae.train.LoopConfig] = None,
    model_config: Optional[vae.models.TransformerConfig] = None,
):
    if data_config is None:
        data_config = humanml3d.HumanML3DConfig()

    if loop_config is None:
        loop_config = vae.train.LoopConfig()

    if model_config is None:
        model_config = vae.models.TransformerConfig()

    ds = humanml3d.load_motion_text(loop_config, data_config)
    train_ds, eval_ds = [ds[key] for key in ["train", "val"]]

    n_devices = jax.local_device_count()
    assert (
        loop_config.batch_size % n_devices == 0
    ), "batch size must be divisable by number of devices"

    state: vae.train.TrainState = vae.train.create_state(
        seed,
        use_mixed_precision,
        loop_config.batch_size // n_devices,
        model_config,
        loop_config,
    )

    if restore_checkpoints:
        assert workdir is not None, "Speicfy checkpoint dir in workdir"
        state = checkpoints.restore_checkpoint(workdir, state)

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
    seed: int = 0,
    use_mixed_precision: bool = True,
    restore_checkpoints: bool = False,
    workdir: Optional[Path] = None,
    data_config: Optional[humanml3d.HumanML3DConfig] = None,
    loop_config: Optional[vae.train.LoopConfig] = None,
    model_config: Optional[vae.models.TransformerConfig] = None,
):
    if data_config is None:
        data_config = humanml3d.HumanML3DConfig()

    if loop_config is None:
        loop_config = vae.train.LoopConfig()

    if model_config is None:
        model_config = vae.models.TransformerConfig()

    ds = humanml3d.load_motion_text(loop_config, data_config)
    train_ds, eval_ds = [ds[key] for key in ["train", "val"]]

    n_devices = jax.local_device_count()
    assert (
        loop_config.batch_size % n_devices == 0
    ), "batch size must be divisable by number of devices"

    state: vae.train.TrainState = vae.train.create_state(
        seed,
        use_mixed_precision,
        loop_config.batch_size // n_devices,
        model_config,
        loop_config,
    )

    if restore_checkpoints:
        assert workdir is not None, "Speicfy checkpoint dir in workdir"
        state = checkpoints.restore_checkpoint(workdir, state)

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
