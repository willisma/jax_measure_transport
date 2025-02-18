"""File containing the unittests for interfaces."""

# built-in libs
import unittest

# external libs
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

# deps
from continuous import SiTInterface, EDMInterface

class DummyMlp(nn.Module):
    in_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        W = self.param('W', lambda rng, shape: jnp.eye(shape[-1]), x.shape)
        return jnp.matmul(x, W, precision='highest')

class TestSiTInterface(unittest.TestCase):
    
    def setUp(self):
        self.mlp = DummyMlp(in_dim=3)
        self.interface = SiTInterface(
            network=self.mlp,
        )
        self.params = self.interface.init(jax.random.PRNGKey(0), jnp.ones((4, 64, 64, 3)))
        self.rng = jax.random.PRNGKey(0)
        self.shape = (16, 64, 64, 3)
    
    def test_sample_t(self):
        t = self.interface.apply(
            self.params,
            shape=(self.shape[0],),
            rngs={'time': self.rng},
            method='sample_t'
        )
        self.assertEqual(t.shape, (self.shape[0],))
    
    def test_sample_n(self):
        n = self.interface.apply(
            self.params,
            shape=self.shape,
            rngs={'noise': self.rng},
            method='sample_n'
        )
        self.assertEqual(n.shape, self.shape)
    
    def test_pred(self):

        x = jax.random.normal(self.rng, self.shape)
        t = self.interface.apply(
            self.params,
            shape=(self.shape[0],),
            rngs={'time': self.rng},
            method='sample_t'
        )
        n = self.interface.apply(
            self.params,
            shape=self.shape,
            rngs={'noise': self.rng},
            method='sample_n'
        )

        target = self.interface.apply(
            self.params,
            x=x,
            n=n,
            t=t,
            method='target'
        )

        pred = self.interface.apply(
            self.params,
            x_t=x-n,
            t=t,
            method='pred'
        )

        self.assertTrue(jnp.allclose(target, x - n))
        self.assertTrue(jnp.allclose(target, pred))


class TestEDMInterface(unittest.TestCase):
    # TODO
    pass

if __name__ == "__main__":
    unittest.main()