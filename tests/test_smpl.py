import jax
import jax.numpy as jnp

from gga import smpl


def test_recover_from_ric():
    rng = jax.random.PRNGKey(0)
    data = jax.random.normal(rng, (2, 16, 263))

    data = smpl.recover_from_ric(data, 22)

    breakpoint()
