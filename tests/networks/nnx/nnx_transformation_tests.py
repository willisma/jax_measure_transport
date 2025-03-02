"""File containing unittests for nnx transformations."""

# built-in libs
import unittest
import time

# external libs
from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import ml_collections


class TestGrad(unittest.TestCase):
    """Test Linear module in nnx"""

    def setUp(self):
        self.in_dim = 4
        self.out_dim = 1152
        self.dtype = jnp.float32
        self.dshape = (4, 32, 32, 4)

        self.nn_rng = jax.random.PRNGKey(0)
        self.nn_linear = nn.Dense(
            features=self.out_dim, dtype=self.dtype
        )
        self.nn_param = self.nn_linear.init(
            self.nn_rng,
            jnp.ones(self.dshape, dtype=self.dtype)
        )

        self.nnx_rng = nnx.Rngs(params=42)
        self.nnx_linear = nnx.Linear(
            self.in_dim, self.out_dim, dtype=self.dtype, rngs=self.nnx_rng,
            kernel_init=nn.initializers.lecun_uniform(),
            bias_init=nn.initializers.zeros
        )

        self.data_rng = jax.random.PRNGKey(42)
        self.sim_iter = 1_000


    def test_grad(self):
        """Test Linear module."""
        
        # check equivalence in random state

        nn_param = self.nn_param['params']
        nn_param['kernel'] = nn.initializers.lecun_uniform()(
            jax.random.fold_in(jax.random.PRNGKey(42), 0), (self.in_dim, self.out_dim)
        )
        nn_param['bias'] = nn.initializers.zeros(
            jax.random.fold_in(jax.random.PRNGKey(42), 1), (self.out_dim,)
        )

        self.assertTrue(
            jnp.allclose(nn_param['kernel'], self.nnx_linear.kernel.value)
        )
        self.assertTrue(
            jnp.allclose(nn_param['bias'], self.nnx_linear.bias.value)
        )

        # benchmark grad results
        x = jax.random.normal(self.data_rng, self.dshape, dtype=self.dtype)

        def nn_forward(params, x):
            x = nn.activation.silu(x)
            return jnp.sum(self.nn_linear.apply({'params': params}, x))

        def nnx_forward(model, x):
            x = nnx.silu(x)
            return jnp.sum(model(x))
    
        nn_forward = jax.value_and_grad(nn_forward)
        nn_y, nn_grad = nn_forward(nn_param, x)

        nnx_forward = nnx.value_and_grad(nnx_forward)
        nnx_y, nnx_grad = nnx_forward(self.nnx_linear, x)
        
        self.assertTrue(
            jnp.allclose(nn_y, nnx_y)
        )
        self.assertTrue(
            jnp.allclose(nn_grad['kernel'], nnx_grad['kernel'].value)
        )
        self.assertTrue(
            jnp.allclose(nn_grad['bias'], nnx_grad['bias'].value)
        )


if __name__ == "__main__":

    # check rng equivalence
    # flax rng defaults to two separate streamlines:
    # .params() & .dropout()

    # calling rngs.[attr]() will increment the counter of the rng by 1
    # equivalently, it's equal to jax.random.fold_in(jax.random.PRNGKey(seed), counter)

    rng1 = nnx.Rngs(params=0)
    rng2 = jax.random.PRNGKey(0)

    key = rng1.params()

    print(jnp.asarray(key), jax.random.fold_in(rng2, 0))  # should be equal

    unittest.main()
