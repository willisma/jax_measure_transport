"""File containing the unit tests for NNX checkpointing with Orbax."""

# built-in libs
import unittest

# external libs
import jax
import jax.numpy as jnp
import flax
from flax import nnx
import optax
import orbax.checkpoint as ocp

# deps



ckpt_dir = ocp.test_utils.erase_and_create_empty('/tmp/my-checkpoints/')


class TwoLayerMLP(nnx.Module):
    def __init__(self, dim, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(dim, dim, rngs=rngs, use_bias=False)
        self.linear2 = nnx.Linear(dim, dim, rngs=rngs, use_bias=False)

    def __call__(self, x):
        x = self.linear1(x)
        return self.linear2(nnx.silu(x))

def loss_fn(model: nnx.Module, batch):
    pred = model(batch)
    return jnp.mean(jnp.square(pred - batch))

@nnx.jit
def train_step(
    model: nnx.Module, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch
):
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch)
    metrics.update(loss=loss)
    optimizer.update(grads)

if __name__ == "__main__":

    model = TwoLayerMLP(128, nnx.Rngs(params=42))

    learning_rate = 0.005
    momentum = 0.9

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average('loss'),
    )

    train_steps = 1024

    for i in range(train_steps):
        batch = jax.random.normal(jax.random.PRNGKey(i), (32, 128))
        train_step(model, optimizer, metrics, batch)

        if i % 128 == 0:
            for metric, value in metrics.compute().items():
                print(f"{metric}: {value}")