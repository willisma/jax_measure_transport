"""File containing the unittests for interfaces."""

# built-in libs
import unittest

# external libs
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

# deps
from interfaces.continuous import SiTInterface, EDMInterface

class DummyMlp1(nn.Module):
    in_dim: int

    @nn.compact
    def __call__(self, x_t: jnp.ndarray, t: jnp.ndarray, x: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
        W = self.param('W', lambda rng, shape: jnp.eye(shape[-1]), x.shape)
        return jnp.matmul(x - n, W, precision='highest')

class TestSiTInterface(unittest.TestCase):
    
    def setUp(self):
        self.mlp = DummyMlp1(in_dim=3)
        self.interface = SiTInterface(
            network=self.mlp,
        )
        self.params = self.interface.init(
            jax.random.PRNGKey(0), jnp.ones((4, 64, 64, 3)), jnp.ones((4, 64, 64, 3)), jnp.ones((4, 64, 64, 3))
        )
        self.rng = jax.random.PRNGKey(0)
        self.shape = (16, 64, 64, 3)

        self.sim_iters = 1e5
    
    def test_sample_t(self):
        t = self.interface.apply(
            self.params,
            shape=(self.shape[0],),
            rngs={'time': self.rng},
            method='sample_t'
        )
        self.assertEqual(t.shape, (self.shape[0],))

        sim_ts = []
        for i in range(int(self.sim_iters)):
            t = self.interface.apply(
                self.params,
                shape=(self.shape[0],),
                rngs={'time': jax.random.PRNGKey(i)},
                method='sample_t'
            )
            sim_ts.append(t)
        sim_ts = jnp.concatenate(sim_ts, axis=0)
        mu = jnp.mean(sim_ts, axis=0)
        sigma = jnp.var(sim_ts, axis=0)

        self.assertTrue(jnp.allclose(mu, 0.5, atol=1e-3))
        self.assertTrue(jnp.allclose(sigma, 1 / 12, atol=1e-3))
    
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

        x_t = self.interface.apply(
            self.params,
            x=x,
            n=n,
            t=t,
            method='sample_x_t'
        )

        pred = self.interface.apply(
            self.params,
            x_t=x_t,
            t=t,
            x=x,
            n=n,
            method='pred'
        )

        self.assertTrue(jnp.allclose(target, x - n))
        self.assertTrue(jnp.allclose(target, pred))


class DummyMlp2(nn.Module):
    in_dim: int

    @nn.compact
    def __call__(self, x_t: jnp.ndarray, t: jnp.ndarray, x: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
        W = self.param('W', lambda rng, shape: jnp.eye(shape[-1]), x.shape)
        return jnp.matmul(x, W, precision='highest')


class TestEDMInterface(unittest.TestCase):

    def setUp(self):
        self.mlp = DummyMlp2(in_dim=3)
        self.interface = EDMInterface(
            network=self.mlp,
        )
        self.params = self.interface.init(
            jax.random.PRNGKey(0), jnp.ones((4, 64, 64, 3)), jnp.ones((4, 64, 64, 3)), jnp.ones((4, 64, 64, 3))
        )
        self.rng = jax.random.PRNGKey(0)
        self.shape = (16, 64, 64, 3)

        self.sim_iters = 1e5
    
    def test_sample_t(self):
        t = self.interface.apply(
            self.params,
            shape=(self.shape[0],),
            rngs={'time': self.rng},
            method='sample_t'
        )
        self.assertEqual(t.shape, (self.shape[0],))

        sim_ts = []
        for i in range(int(self.sim_iters)):
            t = self.interface.apply(
                self.params,
                shape=(self.shape[0],),
                rngs={'time': jax.random.PRNGKey(i)},
                method='sample_t'
            )
            sim_ts.append(t)
        sim_ts = jnp.concatenate(sim_ts, axis=0)
        mu = jnp.mean(sim_ts, axis=0)
        sigma = jnp.var(sim_ts, axis=0)

        self.assertTrue(jnp.allclose(
            mu, jnp.exp(self.interface.t_mu + 0.5 * self.interface.t_sigma ** 2), atol=1e-3
        ))
        self.assertTrue(jnp.allclose(
            sigma,
            (jnp.exp(self.interface.t_sigma ** 2) - 1) * jnp.exp(2 * self.interface.t_mu + self.interface.t_sigma ** 2),
            atol=1e-1
        ))
    
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

        x_t = self.interface.apply(
            self.params,
            x=x,
            n=n,
            t=t,
            method='sample_x_t'
        )

        pred = self.interface.apply(
            self.params,
            x_t=x_t,
            t=t,
            x=x,
            n=n,
            method='pred'
        )

        self.assertTrue(jnp.allclose(target, x))
        self.assertTrue(jnp.allclose(pred, n, atol=1e-5))

if __name__ == "__main__":
    unittest.main()