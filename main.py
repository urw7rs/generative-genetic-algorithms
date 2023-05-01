from typing import Any, Optional
from pathlib import Path
import functools

import jax

from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils

import numpy as onp

import tensorflow as tf

from jsonargparse import CLI

from gga import vae
from gga import humanml3d
from gga import plot_smpl
from gga.console import console

LoopConfig = vae.train.LoopConfig
TransformerConfig = vae.models.TransformerConfig


def prepare_dataset(dataset: Any, batch_size, total_steps):
    train_ds, train_info = dataset.prepare("train")
    train_ds, train_info = dataset.batch(
        train_ds,
        train_info,
        batch_size,
        total_steps,
        drop_remainder=True,
    )

    eval_ds, eval_info = dataset.prepare("train")
    eval_ds, eval_info = dataset.batch(eval_ds, eval_info, batch_size, shuffle=False)

    return train_ds, train_info, eval_ds, eval_info


def train_vae(
    seed: int,
    use_mixed_precision: bool,
    dataset: Any,
    loop_config: LoopConfig,
    checkpoint: Optional[Path] = None,
):
    with console.status("preparing..."):
        train_ds, train_info, eval_ds, eval_info = prepare_dataset(
            dataset, loop_config.batch_size, loop_config.total_steps
        )

        console.log("dataset ready")

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
        console.log("model ready")

        if checkpoint is not None:
            state = checkpoints.restore_checkpoint(checkpoint, state)
            console.log("checkpoint loaded")

    state, history = vae.train.train_loop(
        seed,
        state,
        train_ds,
        train_info,
        eval_ds,
        eval_info,
        loop_config,
        model_config,
    )


def plot_skeletons(
    seed: int,
    use_mixed_precision: bool,
    dataset: Any,
    loop_config: LoopConfig,
    checkpoint: Optional[Path] = None,
):
    with console.status("preparing..."):
        train_ds, train_info, eval_ds, eval_info = prepare_dataset(
            dataset, loop_config.batch_size, loop_config.total_steps
        )

        console.log("dataset ready")

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

        console.log("model ready")

        if checkpoint is not None:
            state = checkpoints.restore_checkpoint(checkpoint, state)
            console.log("checkpoint loaded")

        state = jax_utils.replicate(state)

        rng = jax.random.PRNGKey(seed)
        rng, noise_rng = jax.random.split(rng)
        noise_rngs = jax.random.split(noise_rng, jax.local_device_count())

        eval_config = model_config.replace(deterministic=True)
        p_generate_step = jax.pmap(
            functools.partial(
                vae.train.generate_step, ds_info=eval_info, config=eval_config
            ),
            axis_name="device",
        )

    with console.status("plotting..."):
        for batch in eval_ds.as_numpy_iterator():
            batched_text: onp.ndarray = batch.pop("text")

            batch = common_utils.shard(batch)

            positions = p_generate_step(state, batch, noise_rng=noise_rngs)[0]
            mask = batch["mask"][0]

            positions = onp.array(positions)
            batched_mask = onp.array(mask)

            for pos, mask, text in zip(positions, batched_mask, batched_text):
                text = text[0].decode()
                file_name = f"{text}.gif"
                plot_smpl.plot_skeleton(file_name, pos[mask], text, fps=20)
                console.log(f"saved {file_name}")

            break


def plot_skeletons_recons(
    seed: int,
    use_mixed_precision: bool,
    dataset: Any,
    loop_config: LoopConfig,
    checkpoint: Optional[Path] = None,
):
    with console.status("preparing..."):
        train_ds, train_info, eval_ds, eval_info = prepare_dataset(
            dataset, loop_config.batch_size, loop_config.total_steps
        )

        console.log("dataset ready")

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

        console.log("model ready")

        if checkpoint is not None:
            state = checkpoints.restore_checkpoint(checkpoint, state)
            console.log("loaded checkpoint")

        state = jax_utils.replicate(state)

        rng = jax.random.PRNGKey(seed)
        rng, noise_rng = jax.random.split(rng)
        noise_rngs = jax.random.split(noise_rng, jax.local_device_count())

        eval_config = model_config.replace(deterministic=True)
        p_reconstruct = jax.pmap(
            functools.partial(
                vae.train.reconstruct, ds_info=eval_info, config=eval_config
            ),
            axis_name="device",
        )

    with console.status("plotting..."):
        for batch in eval_ds.as_numpy_iterator():
            batched_text: onp.ndarray = batch.pop("text")

            batch = common_utils.shard(batch)

            positions = p_reconstruct(
                state,
                batch,
                noise_rng=noise_rngs,
            )[0]
            mask = batch["mask"][0]

            positions = onp.array(positions)
            batched_mask = onp.array(mask)

            for i, (pos, mask, text) in enumerate(
                zip(positions, batched_mask, batched_text)
            ):
                text = text[0].decode()
                file_name = f"recons_{text}.gif"
                plot_smpl.plot_skeleton(file_name, pos[mask], text, fps=20)
                console.log(f"saved {file_name}")

            break


if __name__ == "__main__":
    tf.random.set_seed(0)
    tf.config.set_visible_devices([], "GPU")

    CLI([train_vae, plot_skeletons, plot_skeletons_recons])
