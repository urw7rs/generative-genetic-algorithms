from gga import humanml3d
from gga import vae


def test_load_motion():
    loop_config = vae.train.LoopConfig(
        batch_size=4, eval_steps=2, total_steps=10, warmup_steps=4
    )
    data_config = humanml3d.HumanML3DConfig()

    ds = humanml3d.load_vae(loop_config, data_config)

    assert "train" in ds.keys()
    assert "val" in ds.keys()

    batch = next(iter(ds["train"].as_numpy_iterator()))

    assert "motion" in batch.keys()
    assert batch["motion"].shape[-1] == 263
    assert batch["motion"].shape[0] == 4


def test_load_motion_text():
    loop_config = vae.train.LoopConfig(
        batch_size=4, eval_steps=2, total_steps=10, warmup_steps=4
    )
    data_config = humanml3d.HumanML3DConfig()

    ds = humanml3d.load_with_text(loop_config, data_config)

    assert "train" in ds.keys()
    assert "val" in ds.keys()

    batch = next(iter(ds["train"].as_numpy_iterator()))

    assert "motion" in batch.keys()
    assert batch["motion"].shape[-1] == 263
    assert batch["motion"].shape[0] == 4

    assert "text" in batch.keys()
    assert batch["text"].shape[0] == 4
