"""File containing the unittests for interfaces."""

# built-in libs
import unittest

# external libs
import flax
import flax.linen as nn
from flax import nnx
import jax
import jax.numpy as jnp

# deps
from interfaces.continuous import SiTInterface, EDMInterface, TrainingTimeDistType

class DummyMlp1(nn.Module):
    in_dim: int

    @nn.compact
    def __call__(self, x_t: jnp.ndarray, t: jnp.ndarray, x: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
        W = self.param('W', lambda rng, shape: jnp.eye(shape[-1]), x.shape)
        return jnp.matmul(x - n, W, precision='highest')

class TestSiTInterface(unittest.TestCase):
    
    def setUp(self):
        self.mlp = DummyMlp1(in_dim=3)
        # self.params = self.interface.init(
        #     jax.random.PRNGKey(0), jnp.ones((4, 64, 64, 3)), jnp.ones((4, 64, 64, 3)), jnp.ones((4, 64, 64, 3))
        # )
        self.rngs = nnx.Rngs(
            params=0, dropout=0, label_dropout=0, time=0, noise=0
        )
        self.mlp = nnx.bridge.ToNNX(
            self.mlp, rngs=self.rngs
        )
        nnx.bridge.lazy_init(
            self.mlp, jnp.ones((4, 64, 64, 3)), jnp.ones((4,)), jnp.ones((4, 64, 64, 3)), jnp.ones((4, 64, 64, 3))
        )
        self.interface = SiTInterface(
            network=self.mlp,
            train_time_dist_type=TrainingTimeDistType.UNIFORM
        )
        self.shape = (16, 64, 64, 3)

        self.sim_iters = 1e5
    
    def test_sample_t(self):
        t = self.interface.sample_t((self.shape[0],))
        self.assertEqual(t.shape, (self.shape[0],))

        sim_ts = []
        for i in range(int(self.sim_iters)):
            t = self.interface.sample_t((self.shape[0],))
            sim_ts.append(t)
        sim_ts = jnp.concatenate(sim_ts, axis=0)
        mu = jnp.mean(sim_ts, axis=0)
        sigma = jnp.var(sim_ts, axis=0)

        self.assertTrue(jnp.allclose(mu, 0.5, atol=1e-3))
        self.assertTrue(jnp.allclose(sigma, 1 / 12, atol=1e-3))
    
    def test_sample_n(self):
        n = self.interface.sample_n(self.shape)
        self.assertEqual(n.shape, self.shape)
    
    def test_pred(self):

        x = jax.random.normal(self.rngs.noise(), self.shape)
        t = self.interface.sample_t((self.shape[0],))
        n = self.interface.sample_n(self.shape)

        target = self.interface.target(x, n, t)

        x_t = self.interface.sample_x_t(x, n, t)

        pred = self.interface.pred(x_t, t, x, n)

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

        self.rngs = nnx.Rngs(
            params=0, dropout=0, label_dropout=0, time=0, noise=0
        )
        self.mlp = nnx.bridge.ToNNX(
            self.mlp, rngs=self.rngs
        )
        nnx.bridge.lazy_init(
            self.mlp, jnp.ones((4, 64, 64, 3)), jnp.ones((4,)), jnp.ones((4, 64, 64, 3)), jnp.ones((4, 64, 64, 3))
        )
        self.interface = EDMInterface(
            network=self.mlp,
            train_time_dist_type=TrainingTimeDistType.LOGNORMAL
        )
        self.shape = (16, 64, 64, 3)

        self.sim_iters = 1e5
    
    def test_sample_t(self):
        t = self.interface.sample_t((self.shape[0],))
        self.assertEqual(t.shape, (self.shape[0],))

        sim_ts = []
        for i in range(int(self.sim_iters)):
            t = self.interface.sample_t((self.shape[0],))
            sim_ts.append(t)
        sim_ts = jnp.concatenate(sim_ts, axis=0)
        mu = jnp.mean(sim_ts, axis=0)
        sigma = jnp.var(sim_ts, axis=0)

        self.assertTrue(jnp.allclose(
            mu, jnp.exp(self.interface.t_mu + 0.5 * self.interface.t_sigma ** 2), atol=1e-2
        ))
        self.assertTrue(jnp.allclose(
            sigma,
            (jnp.exp(self.interface.t_sigma ** 2) - 1) * jnp.exp(2 * self.interface.t_mu + self.interface.t_sigma ** 2),
            atol=1e-1
        ))
    
    def test_sample_n(self):
        n = self.interface.sample_n(self.shape)
        self.assertEqual(n.shape, self.shape)
    
    def test_pred(self):

        x = jax.random.normal(self.rngs.noise(), self.shape)
        t = self.interface.sample_t((self.shape[0],))
        n = self.interface.sample_n(self.shape)

        target = self.interface.target(x, n, t)

        x_t = self.interface.sample_x_t(x, n, t)

        pred = self.interface.pred(x_t, t, x, n)

        self.assertTrue(jnp.allclose(target, x))
        self.assertTrue(jnp.allclose(pred, n, atol=1e-5))

if __name__ == "__main__":
    unittest.main()