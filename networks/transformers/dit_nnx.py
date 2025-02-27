"""File containing the model definition for DiT."""

# built-in libs

# external libs
import flax
import flax.linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

# deps
from networks.transformers import utils

PRECISION = None

class DiscreteTimeEmbedder(nnx.Module):
    """Embedding **Discrete Time** into vector representations. This embedder admits time range in {0, 1000}."""

    def __init__(
        self, hidden_size: int, freq_embed_size: int, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32
    ):
        self.mlp = nnx.Sequential(
            nnx.Linear(
                freq_embed_size, hidden_size,
                kernel_init=nn.initializers.normal(stddev=0.02), bias_init=nn.initializers.zeros,
                dtype=dtype, precision=PRECISION, rngs=rngs
            ),
            nnx.silu(),
            nnx.Linear(
                hidden_size, hidden_size,
                kernel_init=nn.initializers.normal(stddev=0.02), bias_init=nn.initializers.zeros,
                dtype=dtype, precision=PRECISION, rngs=rngs
            ),
        )
        self.dtype = dtype
        self.freq_embed_size = freq_embed_size
    
    @staticmethod
    def timestep_embedding(
        t: jnp.ndarray, *, dim: int, max_period: int = 10000, dtype: jnp.dtype = jnp.float32
    ) -> jnp.ndarray:
        """Compute the sinusoidal timestep embedding."""
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = jnp.exp(
            -jnp.log(max_period) * jnp.arange(0, half, dtype=dtype) / half
        )
        args = jnp.expand_dims(t, axis=1) * jnp.expand_dims(freqs, axis=0)
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Forward pass for the DiscreteTimeEmbedder."""
        return self.mlp(self.timestep_embedding(t, dim=self.freq_embed_size, dtype=self.dtype))


class ContinuousTimeEmbedder(nnx.Module):
    """Embedding **Continuous Time** into vector representations. This embedder admits time range in [0, 1]."""
    
    def __init__(
        self, hidden_size: int, freq_embed_size: int, *, rngs: nnx.Rngs, scale: float = 1.0, dtype: jnp.dtype = jnp.float32
    ):
        key = rngs.params()
        self.gaussian_basis = nnx.Param(
            jax.random.normal(key, (freq_embed_size // 2,)) * scale
        )
        self.proj = nnx.Sequential(
            nnx.Linear(
                freq_embed_size, hidden_size,
                kernel_init=nn.initializers.normal(stddev=0.02), bias_init=nn.initializers.zeros,
                dtype=dtype, precision=PRECISION, rngs=rngs
            ),
            nnx.silu(),
            nnx.Linear(
                hidden_size, hidden_size,
                kernel_init=nn.initializers.normal(stddev=0.02), bias_init=nn.initializers.zeros,
                dtype=dtype, precision=PRECISION, rngs=rngs
            ),
        )
    
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        # gaussian_basis should be registered as buffer
        t = t[..., None] * jax.lax.stop_gradient(self.gaussian_basis[None, :]) * 2 * np.pi
        t = jnp.concatenate(
            [jnp.sin(t), jnp.cos(t)], axis=-1
        )
        t = self.proj(t)


class ClassEmbedder(nnx.Module):
    """Lookup Table for class embeddings."""

    def __init__(
        self, num_classes: int, hidden_size: int, dropout_prob: float, *, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32
    ):
        take_null_class = dropout_prob > 0.0
        self.embedding_table = nnx.Embed(
            num_classes + take_null_class, hidden_size, 
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=dtype, rngs=rngs
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(
        self, labels: jnp.ndarray, *, train: bool, rngs: nnx.Rngs
    ) -> jnp.ndarray:
        """Drop tokens with probability `dropout_prob`."""
        if not train or self.dropout_prob == 0.0:
            return labels
        key = rngs.label_dropout()
        drop_ids = jax.random.uniform(key, (labels.shape[0],)) < self.dropout_prob
        labels = jnp.where(drop_ids, self.num_classes, labels)
        return labels

    def __call__(self, labels: jnp.ndarray, *, train: bool, rngs: nnx.Rngs) -> jnp.ndarray:
        labels = self.token_drop(labels, train=train, rngs=rngs)
        return self.embedding_table(labels)


class PositionEmbedder(nnx.Module):
    """Adds positional embeddings to the input sequence."""
    
    def __init__(
        self, input_shape: tuple[int, ...], *, sincos: bool, rngs: nnx.Rngs, dtype: jnp.dtype = jnp.float32
    ):
        key = rngs.params()
        h, w, c = input_shape

        if not self.sincos:
            self.pe = nnx.Param(
                jax.random.normal(key, (1, h * w, c), dtype=dtype) * 0.02
            )
        else:
            pe_array = utils.get_2d_sincos_pos_embed(c, (h, w), dtype=dtype)
            self.pe = nnx.Param(
                jnp.full((1, h * w, c), pe_array, dtype=dtype)
            )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        pe = jax.lax.stop_gradient(self.pe) if self.sincos else self.pe
        return x + pe

class DiT(nnx.Module):
    pass