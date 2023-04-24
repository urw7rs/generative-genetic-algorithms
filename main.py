from typing import Optional
from pathlib import Path

from jsonargparse import CLI

import numpy as onp

import jax
from flax.training import checkpoints

import tensorflow as tf

from gga import vae
from gga import humanml3d
from gga import plot_smpl
from gga import smpl


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

    batch = next(iter(eval_ds.as_numpy_iterator()))

    motion: onp.ndarray = batch["motion"]
    motion = motion * ds["std"] + ds["mean"]
    batched_text: onp.ndarray = batch["text"]

    batched_positions = onp.array(jax.jit(smpl.recover_from_ric)(motion))
    for i, (positions, text) in enumerate(zip(batched_positions, batched_text)):
        text = text[0].decode()
        plot_smpl.plot_skeleton(f"{i}.gif", positions, text, fps=20)


if __name__ == "__main__":
    tf.random.set_seed(0)
    tf.config.set_visible_devices([], "GPU")

    CLI()
