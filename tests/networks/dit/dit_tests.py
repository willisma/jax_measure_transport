"""File containing unittests for dit modules."""

# built-in libs
import unittest

# external libs
import flax
from flax import nnx
import jax
import jax.numpy as jnp

# deps
from networks.transformers import dit_nnx

if __name__ == "__main__":

    rngs = nnx.Rngs(params=0, dropout=0, label_dropout=0)
    model = dit_nnx.DiT(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        rngs=rngs,
    )

    nnx.display(model)
