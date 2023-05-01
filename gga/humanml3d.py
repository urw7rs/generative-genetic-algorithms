from typing import Callable, Optional

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE


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


def normalize(x, mean, std):
    return (x - mean) / std


def length_mask(length, max_length):
    idx = tf.range(max_length, dtype=length.dtype)
    mask = idx < length
    return mask


class HumanML3D:
    def __init__(
        self,
        min_length: int = 40,
        max_length: int = 196,
        tokenizer: Optional[Callable] = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.tokenizer = tokenizer

    def prepare(self, split, shuffle: bool = True) -> tf.data.Dataset:
        ds: tf.data.Dataset = tfds.load("humanml3d", split=split, shuffle_files=True)

        ds = ds.filter(
            lambda batch: tf.math.reduce_any(
                tf.logical_and(
                    tf.math.greater_equal(batch["length"], self.min_length),
                    tf.math.less_equal(batch["length"], self.max_length),
                )
            )
        )

        motion_ds = ds.map(lambda x: x["motion"], num_parallel_calls=AUTOTUNE)

        mean = calc_mean(motion_ds)
        std = calc_std(motion_ds, mean)

        def collate(x):
            batch = {
                "motion": normalize(x["motion"], mean=mean, std=std),
                "text": tf.expand_dims(x["caption"], axis=0),
                "mask": length_mask(x["length"], max_length=self.max_length),
            }
            if self.tokenizer is not None:
                batch["text"] = self.tokenizer(batch["text"])

            return batch

        ds = ds.map(collate, num_parallel_calls=AUTOTUNE)

        num_samples = 0
        for _ in ds:
            num_samples += 1

        if shuffle:
            ds = ds.shuffle(num_samples, reshuffle_each_iteration=True)
        info = {}
        info["mean"] = mean
        info["std"] = std
        info["num_samples"] = num_samples

        return ds, info

    def batch(
        self,
        ds,
        batch_size: int,
        drop_remainder=False,
        shuffle=True,
    ):
        ds = ds.padded_batch(
            batch_size,
            padded_shapes={
                "motion": (self.max_length, None),
                "text": (None,),
                "mask": (self.max_length,),
            },
            drop_remainder=drop_remainder,
        ).prefetch(AUTOTUNE)

        return ds
