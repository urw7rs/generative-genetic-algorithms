from time import time
import functools
from pathlib import Path

import dataclasses

import numpy as onp

import tensorflow as tf

import ciclo

import jax
import jax.numpy as jnp

import optax

from flax import jax_utils
from flax import struct
from flax.training import common_utils
from flax.training import train_state
from flax.training import dynamic_scale as dynamic_scale_lib

from torch.utils.tensorboard import SummaryWriter

from gga import smpl
from gga.metrics import ReconstructionMetrics

from . import models


TFDS = tf.data.Dataset


def rsqrt_schedule(
    init_value: float,
    shift: int = 0,
):
    """Applies a reverse square-root schedule.

    The reverse square root schedule is simply `lr = init_value / sqrt(step)`.

    Args:
      init_value: Base learning rate (before applying the rsqrt schedule).
      shift: How many steps the rsqrt should be shifted. Shifting the rsqrt
        schedule makes it less steep in the beginning (close to 0).

    Returns:
      A schedule `count -> learning_rate`.
    """

    def schedule(count):
        return init_value * (count + shift) ** -0.5 * shift**0.5

    return schedule


def create_learning_rate_schedule(learning_rate: float, warmup_steps: int):
    """Creates a rsqrt schedule with linear warmup."""
    return optax.join_schedules(
        [
            optax.linear_schedule(
                init_value=0, end_value=learning_rate, transition_steps=warmup_steps
            ),
            rsqrt_schedule(init_value=learning_rate, shift=warmup_steps),
        ],
        boundaries=[warmup_steps],
    )


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
    recons_loss: jax.Array
    pos_recons_loss: jax.Array
    kl_loss: jax.Array


def train_step(state, batch, config, mean, std, dropout_rng=None, noise_rng=None):
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

        gt_pos = smpl.recover_from_ric(inputs * std + mean)
        pos = smpl.recover_from_ric(output.recons * std + mean)

        pos_recons_loss = smooth_l1_loss(pos, gt_pos).mean()
        recons_loss = smooth_l1_loss(output.recons, inputs).mean()
        kl_loss = kl_divergence(output.mu, output.logvar).mean() * 1e-4

        loss = recons_loss + pos_recons_loss + kl_loss

        return loss, Losses(recons_loss, pos_recons_loss, kl_loss)

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

    loss = jax.lax.pmean(loss, axis_name="device")
    recons_loss = jax.lax.pmean(losses.recons_loss, axis_name="device")
    pos_recons_loss = jax.lax.pmean(losses.pos_recons_loss, axis_name="device")
    kl_loss = jax.lax.pmean(losses.kl_loss, axis_name="device")

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

    logs = ciclo.logs()
    logs.add_metric("train/recons_loss", recons_loss)
    logs.add_metric("train/pos_recons_loss", pos_recons_loss)
    logs.add_metric("train/kl_loss", kl_loss)
    logs.add_metric("train/loss", loss)
    return logs, new_state


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


def generate_step(state, batch, mean, std, config, noise_rng=None):
    motion = generate_samples(state, batch, config, noise_rng)
    motion = motion * std + mean
    joints = smpl.recover_from_ric(motion)
    joints *= batch["mask"][:, :, None, None]
    return joints


def reconstruct(state, batch, mean, std, config, noise_rng=None):
    inputs = batch["motion"]
    input_mask = batch["mask"]

    output = models.Transformer(config).apply(
        {"params": state.params},
        inputs,
        input_mask,
        rngs={"noise": noise_rng},
    )
    motion = output.recons * std + mean
    joints = smpl.recover_from_ric(motion)
    joints *= batch["mask"][:, :, None, None]
    return joints


def eval_step(state, batch, config, mean, std, noise_rng=None):
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

    gt_pos = smpl.recover_from_ric(inputs * std + mean)
    pos = smpl.recover_from_ric(output.recons * std + mean)

    pos_recons_loss = smooth_l1_loss(pos, gt_pos).mean()
    recons_loss = smooth_l1_loss(output.recons, inputs).mean()
    kl_loss = kl_divergence(output.mu, output.logvar).mean() * 1e-4

    loss = recons_loss + pos_recons_loss + kl_loss

    losses = Losses(recons_loss, pos_recons_loss, kl_loss)

    loss = jax.lax.pmean(loss, axis_name="device")
    recons_loss = jax.lax.pmean(losses.recons_loss, axis_name="device")
    pos_recons_loss = jax.lax.pmean(losses.pos_recons_loss, axis_name="device")
    kl_loss = jax.lax.pmean(losses.kl_loss, axis_name="device")

    logs = ciclo.logs()
    logs.add_metric("recons_loss", recons_loss)
    logs.add_metric("pos_recons_loss", pos_recons_loss)
    logs.add_metric("kl_loss", kl_loss)
    logs.add_metric("loss", loss)
    return logs, state


def reset_step(state):
    return state.replace(metrics=state.metrics.reset())


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
    total_steps: int = 600_000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-7
    warmup_steps: int = 5000
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

    learning_rate_fn = create_learning_rate_schedule(
        learning_rate=loop_config.learning_rate, warmup_steps=loop_config.warmup_steps
    )

    dynamic_scale = None
    if dtype == jnp.float16:
        dynamic_scale = dynamic_scale_lib.DynamicScale()

    optimizer = optax.adamw(
        learning_rate=learning_rate_fn,
        b1=0.9,
        b2=0.98,
        eps=1e-9,
        weight_decay=loop_config.weight_decay,
    )

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
    seed: int,
    state: TrainState,
    train_ds: TFDS,
    eval_ds: TFDS,
    mean,
    std,
    loop_config: LoopConfig,
    config: models.TransformerConfig,
):
    # Replicate state.
    state = jax_utils.replicate(state)

    rng = jax.random.PRNGKey(seed)
    rng, dropout_rng, noise_rng = jax.random.split(rng, 3)
    dropout_rngs = jax.random.split(dropout_rng, jax.local_device_count())
    noise_rngs = jax.random.split(noise_rng, jax.local_device_count())

    # compile multidevice versions of train/eval/predict step and cache init fn.
    p_train_step = jax.pmap(
        functools.partial(train_step, config=config),
        axis_name="device",
        donate_argnums=(0,),
    )

    eval_config = config.replace(deterministic=True)
    p_eval_step = jax.pmap(
        functools.partial(eval_step, config=eval_config),
        axis_name="device",
        donate_argnums=(0,),
    )

    call_checkpoint = ciclo.every(steps=1)
    checkpoint = ciclo.checkpoint(f"logdir/{Path(__file__).stem}/{int(time())}")
    checkpoint_best = ciclo.checkpoint(
        f"logdir/{Path(__file__).stem}/{int(time())}/best",
        monitor="val/recons_loss",
        mode="min",
        keep=3,
    )

    call_eval = ciclo.every(10_00)
    call_writer = ciclo.every(50)

    history = ciclo.history()

    writer = SummaryWriter()

    logs = ciclo.logs()
    # keras averages metrics by default set always_stateful to True to disable this behavior
    keras_bar = ciclo.keras_bar(total=loop_config.total_steps, always_stateful=True)

    for elapsed, batch in ciclo.elapse(
        train_ds.as_numpy_iterator(), stop=loop_config.total_steps
    ):
        batch = common_utils.shard(batch)
        new_logs, state = p_train_step(
            state,
            batch,
            mean=mean,
            std=std,
            dropout_rng=dropout_rngs,
            noise_rng=noise_rngs,
        )
        logs.updates = jax.tree_map(lambda x: x[0], new_logs)

        if call_writer(elapsed):
            # state is replicated for each device
            step = int(state.step[0])
            for key, value in logs["metrics"].items():
                if "val" not in key:
                    writer.add_scalar(key, onp.asarray(value), step)

        keras_bar(elapsed, logs)
        history.commit(elapsed, logs)

        if call_eval(elapsed):
            eval_history = ciclo.history()

            eval_logs = ciclo.logs()

            for eval_elapsed, batch in ciclo.elapse(
                eval_ds.as_numpy_iterator(), stop=loop_config.eval_steps
            ):
                batch = common_utils.shard(batch)
                eval_logs.updates, state = p_eval_step(
                    state,
                    batch,
                    mean=mean,
                    std=std,
                    noise_rng=noise_rngs,
                )
                eval_history.commit(eval_elapsed, eval_logs)

            step = int(state.step[0])
            for key in eval_logs["metrics"].keys():
                metric = onp.stack(eval_history.collect(key)).mean(axis=0)[0]
                logs.add_metric(f"val/{key}", metric)
                writer.add_scalar(f"val/{key}", onp.asarray(metric), step)

            single_state = jax.tree_util.tree_map(lambda x: x[0], state)
            checkpoint_best(elapsed, single_state, logs)

            if call_checkpoint(elapsed):
                single_state = jax.tree_util.tree_map(lambda x: x[0], state)
                checkpoint(elapsed, single_state, logs)

    writer.close()

    return state, history
