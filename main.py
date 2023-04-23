from typing import Optional
from pathlib import Path

from jsonargparse import CLI

import jax
from flax.training import checkpoints

import tensorflow as tf

from gga import vae
from gga import humanml3d


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

    ds = humanml3d.load_vae(loop_config, data_config)
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


if __name__ == "__main__":
    tf.random.set_seed(0)
    tf.config.set_visible_devices([], "GPU")

    CLI()
