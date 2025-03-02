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
from tests.networks.dit import dit_torch


def convert_flax_to_torch(flax_module, torch_module):
    """Convert flax.nnx module to torch module."""



class TestDiT(unittest.TestCase):
    """Test consistency of DiT modules in dit_nnx with torch."""

    def setUp(self):
        rngs = nnx.Rngs(params=0, dropout=0, label_dropout=0)
        self.nnx_model = dit_nnx.DiT(
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            rngs=rngs,
        )

        self.th_model = dit_torch.DiT_B_2()
    
    def test_embedders(self):
        """Test consistency of embedders."""
        pass

    def test_blocks(self):
        """Test consistency of blocks."""
        pass


    

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

    th_model = dit_torch.DiT_B_2()

    print(th_model)
