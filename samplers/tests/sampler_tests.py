"""File containing the unittests for samplers."""

# built-in libs
import unittest

# external libs
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

# deps
from interfaces.continuous import SiTInterface
from samplers import EulerSampler

class DummyMlp(nn.Module):
    in_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        W = self.param('W', lambda rng, shape: jnp.eye(shape[-1]), x.shape)
        return jnp.matmul(x, W, precision='highest')
    
class TestEulerSampler(unittest.TestCase):
    pass