import functools

import dataclasses

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from . import train

AUTOTUNE = tf.data.AUTOTUNE


def pad_motions(batch):
    batch["motion"]


def calc_mean(motion_ds) -> np.ndarray:
    n = 0
    total_sum = 0
    for motion in motion_ds.as_numpy_iterator():
        n += motion.shape[0]
        total_sum += motion.sum(0)

    mean = total_sum / n
    return mean


def calc_std(motion_ds, mean: np.ndarray, num_joints: int = 22):
    variance = 0
    n = 0
    for motion in motion_ds.as_numpy_iterator():
        n += motion.shape[0]
        variance += np.sum((motion - mean) ** 2, axis=0)

    std = np.sqrt(variance / n)

    std[0:1] = std[0:1].mean() / 1.0
    std[1:3] = std[1:3].mean() / 1.0
    std[3:4] = std[3:4].mean() / 1.0
    std[4 : 4 + (num_joints - 1) * 3] = std[4 : 4 + (num_joints - 1) * 3].mean() / 1.0
    std[4 + (num_joints - 1) * 3 : 4 + (num_joints - 1) * 9] = (
        std[4 + (num_joints - 1) * 3 : 4 + (num_joints - 1) * 9].mean() / 1.0
    )
    std[4 + (num_joints - 1) * 9 : 4 + (num_joints - 1) * 9 + num_joints * 3] = (
        std[4 + (num_joints - 1) * 9 : 4 + (num_joints - 1) * 9 + num_joints * 3].mean()
        / 1.0
    )
    std[4 + (num_joints - 1) * 9 + num_joints * 3 :] = (
        std[4 + (num_joints - 1) * 9 + num_joints * 3 :].mean() / 1.0
    )

    return std


def normalize(batch, mean, std):
    batch["motion"] = (batch["motion"] - mean) / std
    return batch


@dataclasses.dataclass
class HumanML3DConfig:
    min_length: int = 40
    max_length: int = 196


def humanml3d(
    loop_config: train.LoopConfig, config: HumanML3DConfig
) -> tf.data.Dataset:
    train_ds, eval_ds = tfds.load("humanml3d", split=["train", "val"])

    def load_dataset(ds):
        ds = ds.filter(
            lambda batch: tf.math.reduce_any(
                tf.logical_and(
                    tf.math.greater_equal(batch["length"], config.min_length),
                    tf.math.less_equal(batch["length"], config.max_length),
                )
            )
        )
        return ds

    train_ds, eval_ds = [load_dataset(ds) for ds in [train_ds, eval_ds]]

    motion_ds = train_ds.map(lambda x: x["motion"], num_parallel_calls=AUTOTUNE)

    mean = calc_mean(motion_ds)
    std = calc_std(motion_ds, mean)

    def norm_and_batch(ds, shuffle: bool):
        ds = ds.map(
            functools.partial(normalize, mean=mean, std=std),
            num_parallel_calls=AUTOTUNE,
        ).map(
            lambda x: {"motion": x["motion"]},
            num_parallel_calls=AUTOTUNE,
        )

        num_samples = 0
        for batch in ds:
            num_samples += 1

        count = loop_config.total_steps // num_samples
        if loop_config.total_steps % num_samples > 0:
            count += 1

        if shuffle:
            ds = ds.shuffle(num_samples, reshuffle_each_iteration=True)

        ds = (
            ds.repeat(count * loop_config.batch_size)
            .padded_batch(
                loop_config.batch_size,
                padded_shapes={
                    "motion": (config.max_length, None),
                },
            )
            .prefetch(AUTOTUNE)
        )
        return ds

    train_ds = norm_and_batch(train_ds, shuffle=True)
    eval_ds = norm_and_batch(eval_ds, shuffle=False)

    return train_ds, eval_ds
