from typing import Any, Dict, Mapping

import functools
from pathlib import Path
from dataclasses import asdict

from tqdm.rich import tqdm

import dataclasses

import numpy as np

import tensorflow as tf

import jax
import jax.numpy as jnp
from jax import tree_util

import optax
from orbax.checkpoint import (
    CheckpointManagerOptions,
    CheckpointManager,
    Checkpointer,
    PyTreeCheckpointHandler,
)


from flax import jax_utils
from flax import struct
from flax.training import common_utils
from flax.training import train_state
from flax.training import dynamic_scale as dynamic_scale_lib

from torch.utils.tensorboard import SummaryWriter

from gga import smpl
from gga.metrics import ReconstructionMetrics, GenerationMetrics
from gga.metrics import compute_fid, compute_diversity

from . import models


TFDS = tf.data.Dataset


def normalize(x, mean, std):
    return (x - mean) / std


def denormalize(x, mean, std):
    return x * std + mean


def to_pos(x, ds_info):
    return smpl.recover_from_ric(denormalize(x, ds_info["mean"], ds_info["std"]))


def kl_divergence(mean, logvar):
    return -0.5 * (1 + logvar - jnp.square(mean) - jnp.exp(logvar))


def mse_loss(preds, target):
    return ((preds - target) ** 2).mean()


def smooth_l1_loss(preds, target, beta: float = 1.0):
    l1 = jnp.abs(preds - target)
    loss = jnp.where(l1 < beta, 0.5 / beta * l1**2, l1 - 0.5 / beta)
    return loss


@struct.dataclass
class Losses:
    total_loss: jax.Array
    recons_loss: jax.Array
    pos_recons_loss: jax.Array
    kl_loss: jax.Array


def train_step(state, batch, ds_info, config, dropout_rng=None, noise_rng=None):
    """Perform a single training step."""
    train_keys = ["motion", "mask"]
    (inputs, input_mask) = (batch.get(k, None) for k in train_keys)

    dropout_rng = jax.random.fold_in(dropout_rng, state.step)
    noise_rng = jax.random.fold_in(noise_rng, state.step)

    def loss_fn(params):
        """loss function used for training."""
        output = models.Transformer(config).apply(
            {"params": params},
            inputs,
            input_mask,
            rngs={"dropout": dropout_rng, "noise": noise_rng},
        )

        gt_pos = to_pos(inputs, ds_info)
        pos = to_pos(output.recons, ds_info)

        pos_recons_loss = smooth_l1_loss(pos, gt_pos).mean()
        recons_loss = smooth_l1_loss(output.recons, inputs).mean()
        kl_loss = kl_divergence(output.mu, output.logvar).mean() * 1e-4

        loss = recons_loss + pos_recons_loss + kl_loss

        return loss, Losses(loss, recons_loss, pos_recons_loss, kl_loss)

    if state.dynamic_scale:
        # dynamic scale takes care of averaging gradients across replicas
        grad_fn = state.dynamic_scale.value_and_grad(
            loss_fn, has_aux=True, axis_name="device"
        )
        (
            dynamic_scale,
            is_fin,
            (loss, losses),
            grads,
        ) = grad_fn(state.params)
        state = state.replace(dynamic_scale=dynamic_scale)
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, losses), grads = grad_fn(state.params)
        grads = jax.lax.pmean(grads, axis_name="device")

    new_state = state.apply_gradients(grads=grads)

    losses = jax.lax.pmean(losses, axis_name="device")

    if state.dynamic_scale:
        # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
        # params should be restored (= skip this step).
        select_fn = functools.partial(jnp.where, is_fin)
        new_state = new_state.replace(
            opt_state=jax.tree_util.tree_map(
                select_fn, new_state.opt_state, state.opt_state
            ),
            params=jax.tree_util.tree_map(select_fn, new_state.params, state.params),
        )

    logs = {}
    logs["losses"] = asdict(losses)
    return logs, new_state


def eval_step(state, batch, ds_info, config, noise_rng=None):
    """Calculate evaluation metrics on a batch."""
    eval_keys = ["motion", "mask"]
    (inputs, input_mask) = (batch.get(k, None) for k in eval_keys)

    noise_rng = jax.random.fold_in(noise_rng, state.step)

    output = models.Transformer(config).apply(
        {"params": state.params},
        inputs,
        input_mask,
        rngs={"noise": noise_rng},
    )

    encoded_recons = models.Transformer(config).apply(
        {"params": state.params},
        output.recons,
        input_mask,
        method="encode",
    )

    gt_pos = to_pos(inputs, ds_info)
    pos = to_pos(output.recons, ds_info)

    pos_recons_loss = smooth_l1_loss(pos, gt_pos).mean()
    recons_loss = smooth_l1_loss(output.recons, inputs).mean()
    kl_loss = kl_divergence(output.mu, output.logvar).mean() * 1e-4

    loss = recons_loss + pos_recons_loss + kl_loss

    losses = Losses(loss, recons_loss, pos_recons_loss, kl_loss)
    losses = jax.lax.pmean(losses, axis_name="device")

    logs = {}
    logs["losses"] = asdict(losses)

    logs["reconstruction_metrics"] = {
        "predictions": pos,
        "targets": gt_pos,
        "mask": input_mask,
    }

    logs["generation_metrics"] = {
        "predictions": output.mu,
        "targets": encoded_recons.mu,
    }

    return logs, state


def collect(list_of_dicts, key):
    output = []
    for entry in list_of_dicts:
        output.append(entry[key])
    return output


def compute_metrics(history):
    metrics = {}

    # add epoch level losses
    losses = collect(history, "losses")
    for name in losses[0].keys():
        metric = jnp.stack(collect(losses, name), axis=0).mean()
        metric = jax.lax.pmean(metric, axis_name="device")
        metrics[name] = metric

    # add reconstruction metrics
    reconstruction_metrics = collect(history, "reconstruction_metrics")
    predictions = jnp.concatenate(
        collect(reconstruction_metrics, "predictions"), axis=0
    )
    targets = jnp.concatenate(collect(reconstruction_metrics, "targets"), axis=0)
    mask = jnp.concatenate(collect(reconstruction_metrics, "mask"), axis=0)

    reconstruction_metrics = ReconstructionMetrics.create(predictions, targets, mask)
    reconstruction_metrics = jax.lax.psum(reconstruction_metrics, axis_name="device")
    metrics.update(reconstruction_metrics.compute())

    return metrics


def compute_np_metrics(history, key):
    metrics = {}
    # add generation metrics
    generation_metrics = collect(history, "generation_metrics")
    predictions = jnp.concatenate(collect(generation_metrics, "predictions"), axis=0)
    targets = jnp.concatenate(collect(generation_metrics, "targets"), axis=0)
    generation_metrics = GenerationMetrics.create(predictions, targets)
    metrics.update(generation_metrics.compute(key))
    return metrics


def generate_samples(state, batch, config, noise_rng=None):
    eval_keys = ["mask"]
    (input_mask,) = (batch.get(k, None) for k in eval_keys)

    motion = models.Transformer(config).apply(
        {"params": state.params},
        input_mask,
        config.dtype,
        rngs={"noise": noise_rng},
        method="generate",
    )
    return motion


def generate_step(state, batch, ds_info, config, noise_rng=None):
    motion = generate_samples(state, batch, config, noise_rng)
    joints = to_pos(motion, ds_info)
    joints *= batch["mask"][:, :, None, None]
    return joints


def reconstruct(state, batch, ds_info, config, noise_rng=None):
    inputs = batch["motion"]
    input_mask = batch["mask"]

    output = models.Transformer(config).apply(
        {"params": state.params},
        inputs,
        input_mask,
        rngs={"noise": noise_rng},
    )
    joints = to_pos(output.recons, ds_info)
    joints *= batch["mask"][:, :, None, None]
    return joints


def preferred_dtype(use_mixed_precision):
    platform = jax.local_devices()[0].platform
    if use_mixed_precision:
        if platform == "tpu":
            return jnp.bfloat16
        elif platform == "gpu":
            return jnp.float16
        return jnp.float32


class TrainState(train_state.TrainState):
    dynamic_scale: dynamic_scale_lib.DynamicScale


@dataclasses.dataclass
class LoopConfig:
    batch_size: int = 64
    eval_steps: int = 100
    epochs: int = 6000
    learning_rate: float = 1e-4
    grad_accum_steps: int = 2


def create_state(
    seed: int,
    use_mixed_precision: bool,
    per_device_batch_size: int,
    train_config: models.TransformerConfig,
    loop_config: LoopConfig,
):
    dtype = preferred_dtype(use_mixed_precision)
    train_config = train_config.replace(dtype=dtype)

    eval_config = train_config.replace(deterministic=True)
    m = models.Transformer(eval_config)

    rng = jax.random.PRNGKey(seed)
    rng, init_rng, noise_rng = jax.random.split(rng, 3)
    input_shape = (
        per_device_batch_size,
        train_config.max_len - train_config.latent_length * 2,
        train_config.input_size,
    )
    input_mask_shape = (
        per_device_batch_size,
        train_config.max_len - train_config.latent_length * 2,
    )

    initial_variables = jax.jit(m.init)(
        {"params": init_rng, "noise": noise_rng},
        jnp.ones(input_shape, jnp.float32),
        jnp.ones(input_mask_shape, jnp.bool_),
    )

    dynamic_scale = None
    if dtype == jnp.float16:
        dynamic_scale = dynamic_scale_lib.DynamicScale()

    optimizer = optax.adamw(learning_rate=loop_config.learning_rate)

    if loop_config.grad_accum_steps > 1:
        optimizer = optax.MultiSteps(
            optimizer, every_k_schedule=loop_config.grad_accum_steps
        )

    state = TrainState.create(
        apply_fn=m.apply,
        params=initial_variables["params"],
        tx=optimizer,
        dynamic_scale=dynamic_scale,
    )

    return state


def train_loop(
    checkpoint: Path,
    seed: int,
    state: TrainState,
    train_ds: TFDS,
    train_info: Dict[str, Any],
    eval_ds: TFDS,
    eval_info: Dict[str, Any],
    loop_config: LoopConfig,
    config: models.TransformerConfig,
):
    def best_fn(metrics: Mapping[str, float]) -> float:
        return metrics["mpjpe"]

    options = CheckpointManagerOptions(max_to_keep=5, best_fn=best_fn, best_mode="min")

    checkpoint_manager = CheckpointManager(
        checkpoint, Checkpointer(PyTreeCheckpointHandler()), options
    )

    if checkpoint_manager.latest_step() is not None:
        state = checkpoint_manager.restore(
            checkpoint_manager.latest_step(), items=state
        )

    state = jax_utils.replicate(state)

    rng = jax.random.PRNGKey(seed)
    rng, dropout_rng, noise_rng, eval_rng = jax.random.split(rng, 4)
    dropout_rngs = jax.random.split(dropout_rng, jax.local_device_count())
    noise_rngs = jax.random.split(noise_rng, jax.local_device_count())

    p_train_step = jax.pmap(
        functools.partial(train_step, ds_info=train_info, config=config),
        axis_name="device",
        donate_argnums=(0,),
    )

    eval_config = config.replace(deterministic=True)
    p_eval_step = jax.pmap(
        functools.partial(eval_step, ds_info=eval_info, config=eval_config),
        axis_name="device",
        donate_argnums=(0,),
    )

    p_compute_metrics = jax.pmap(compute_metrics, axis_name="device")

    writer = SummaryWriter()

    if checkpoint_manager.latest_step() is None:
        step = 0
    else:
        step = checkpoint_manager.latest_step() + 1

    steps_per_epoch = train_info["num_samples"] // loop_config.batch_size
    total_steps = loop_config.epochs * steps_per_epoch - step

    for batch in tqdm(train_ds.as_numpy_iterator(), total=total_steps):
        batch.pop("text")
        batch = common_utils.shard(batch)

        logs, state = p_train_step(
            state, batch, dropout_rng=dropout_rngs, noise_rng=noise_rngs
        )

        if step % 50 == 0:
            # state is replicated for each device
            logs = jax_utils.unreplicate(logs)

            losses = logs["losses"]
            for name, metric in losses.items():
                writer.add_scalar(f"train/{name}", metric.item(), step)

        if step % 10_000 == 0:
            history = []
            for batch in eval_ds.as_numpy_iterator():
                batch.pop("text")
                batch = common_utils.shard(batch)

                log, state = p_eval_step(
                    state,
                    batch,
                    noise_rng=noise_rngs,
                )

                history.append(log)

            metrics = p_compute_metrics(history)
            metrics = jax_utils.unreplicate(metrics)

            history = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), history)
            metrics.update(compute_np_metrics(history, eval_rng))

            for name, metric in metrics.items():
                writer.add_scalar(f"val/{name}", metric.item(), step)

            checkpoint_manager.save(
                step,
                jax_utils.unreplicate(state),
                metrics=tree_util.tree_map(lambda x: x.item(), metrics),
            )

        step += 1

        if step > total_steps:
            break

    writer.close()

    return state
