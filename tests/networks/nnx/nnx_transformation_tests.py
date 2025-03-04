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
import numpy as np
import optax


class TestGrad(unittest.TestCase):
    """Test .grad transformation in nnx"""

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


class TestRngSplit(unittest.TestCase):
    """Test .split_rngs in nnx."""

    def setUp(self):
        self.rngs = nnx.Rngs(to_split_1=1, to_split_2=2, to_broadcast=0)
    
    def test_rng_split(self):

        @nnx.split_rngs(
            splits=jax.local_device_count(), only=['to_split_1', 'to_split_2']
        )
        def forward(rngs):
            # to_split will be splitted into jax.local_device_count() **different** streams
            self.assertEqual(rngs.to_split_1.key.value.shape, (8,))
            self.assertEqual(rngs.to_split_2.key.value.shape, (8,))

            # to_broadcast will be broadcasted to all devices
            self.assertEqual(rngs.to_broadcast.key.value.shape, ())
        
        forward(self.rngs)
        self.assertEqual(self.rngs.to_split_1.key.value.shape, ())
        self.assertEqual(self.rngs.to_split_2.key.value.shape, ())
        self.assertEqual(self.rngs.to_broadcast.key.value.shape, ())


class DummyModule(nnx.Module):

    def __init__(self, in_dim, out_dim, rngs, dtype=jnp.float32):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dtype = dtype
        self.rngs = rngs
        self.linear1 = nnx.Linear(
            self.in_dim, self.out_dim, dtype=self.dtype, rngs=rngs
        )
        self.linear2 = nnx.Linear(
            self.out_dim, self.out_dim, dtype=self.dtype, rngs=rngs
        )
    
    def __call__(self, x):
        x = self.linear1(x)
        x = nnx.silu(x)
        x = self.linear2(x)
        return x


class TestPmap(unittest.TestCase):
    """Test pmap transformation in nnx."""

    def setUp(self):
        self.in_dim = 128
        self.out_dim = 128
        self.dtype = jnp.float32
        self.shape = (4, 128, 128)

        self.nnx_rng = nnx.Rngs(params=42, noise=0)
        self.nnx_module = DummyModule(
            self.in_dim, self.out_dim, dtype=self.dtype, rngs=self.nnx_rng
        )
    
    def test_pmap(self):

        x = jax.random.normal(
            jax.random.key(42), (jax.local_device_count(),) + self.shape, dtype=self.dtype
        )

        # this means we are broadcasting both rngs and params to all devices
        state_axes = nnx.StateAxes({(nnx.RngState, nnx.Param): None})

        @nnx.pmap(in_axes=(state_axes, 0), out_axes=0)
        def forward_with_fix_randomness(model, x):
            return model(x), jax.random.normal(model.rngs.noise(), x.shape, dtype=model.dtype)
        
        y, n = forward_with_fix_randomness(self.nnx_module, x)
        self.assertEqual(y.shape, (8, 4, 128, 128))

        # noises are the same since rngs are broadcasted
        self.assertTrue(jnp.allclose(n[0], n[1]))

        # now, we will split the noise stream of the rngs
        state_axes = nnx.StateAxes({'noise': 0, ...: None})  # ... means broadcast the rest of the states

        @nnx.split_rngs(splits=jax.local_device_count(), only=['noise'])
        @nnx.pmap(in_axes=(state_axes, 0), out_axes=0)
        def forward_with_randomness(model, x):
            return model(x), jax.random.normal(model.rngs.noise(), x.shape, dtype=model.dtype)
        
        y, n = forward_with_randomness(self.nnx_module, x)
        self.assertEqual(y.shape, (8, 4, 128, 128))

        # noises are the same since rngs are broadcasted
        self.assertTrue(not jnp.allclose(n[0], n[1]))
    
    def test_pmean(self):

        x = jax.random.normal(
            jax.random.key(42), (jax.local_device_count(),) + self.shape, dtype=self.dtype
        )
    
        state_axes = nnx.StateAxes({'noise': 0, ...: None})  # ... means broadcast the rest of the states

        def loss_fn(model, x):
            return jnp.mean(
                jnp.square(model(x)) - x # + jax.random.normal(model.rngs.noise(), x.shape, dtype=model.dtype)))
            )

        tx = optax.adam(1e-3)
        state = nnx.Optimizer(self.nnx_module, tx)
        init_state = nnx.state(self.nnx_module)

        @nnx.split_rngs(splits=jax.local_device_count(), only=['noise'])
        @nnx.pmap(in_axes=(state_axes, 0), out_axes=0, axis_name='data')
        def forward_with_randomness(state, x):
            model = state.model
            loss, grads = nnx.value_and_grad(loss_fn)(model, x)
            grads =jax.lax.pmean(grads, axis_name='data')  # <-- remember to gather the grads
            state.update(grads=grads)
            return grads

        # gradient is not gathered
        grads = forward_with_randomness(state, x)

        @nnx.jit
        def forward(state, x):
            model = state.model
            loss, grads = nnx.value_and_grad(loss_fn)(model, x)
            state.update(grads=grads)
            return grads

        test_tx = optax.adam(1e-3)
        graphdef, _ = nnx.split(self.nnx_module)
        test_state = nnx.Optimizer(nnx.merge(graphdef, init_state), test_tx)
        test_grads = forward(test_state, x.reshape(-1, *x.shape[2:]))

        jax.tree.map(
            lambda x, y: self.assertTrue(np.allclose(x, jnp.mean(y, axis=0))), test_grads, grads
        )

        # state is unified across devices; implicit gathering happens in state.update
        test_state.model.rngs.noise()  # <-- for some reason, broadcasting will call this rng once
        jax.tree.map(
            lambda x, y: self.assertTrue(np.allclose(x, y)),
            nnx.state(test_state.model.linear1),
            nnx.state(state.model.linear1)
        )
        jax.tree.map(
            lambda x, y: self.assertTrue(np.allclose(x, y)),
            nnx.state(test_state.model.linear2),
            nnx.state(state.model.linear2)
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
