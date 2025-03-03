"""File containing the utility functions for DiT."""

# built-in libs
import functools

# external libs
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def patch_kernel(dtype: jnp.dtype = jnp.float32):
    """
    ViT patch embedding initializer:
    As patch_embed is implemented as Conv, we view its 4D params as 2D
    """
    def init(key, shape, dtype=dtype):
        h, w, c, n = shape
        fan_in = h * w * c
        fan_out = n
        denominator = (fan_in + fan_out) / 2
        variance = jnp.array(1. / denominator, dtype=dtype)
        return jax.random.uniform(key, shape, dtype, -1) * jnp.sqrt(3 * variance)

    return init


INIT_TABLE = {
    'patch': {
        'kernel': patch_kernel(),
        'bias': nn.initializers.zeros
    },
    'time_embed': {
        'kernel': nn.initializers.normal(stddev=0.02),
        'bias': nn.initializers.zeros
    },
    'class_embed': nn.initializers.normal(stddev=0.02),
    'mod': {
        'kernel': nn.initializers.zeros,
        'bias': nn.initializers.zeros
    },
    'mlp': {
        'kernel': nn.initializers.xavier_uniform(),
        'bias': nn.initializers.zeros
    },
    'attn': {
        'qkv_kernel': functools.partial(
            nn.initializers.variance_scaling, 0.5, "fan_avg", "uniform"
        )(),
        'out_kernel': nn.initializers.xavier_uniform(),
    },
}


def modulation(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Modulation for AdaLN.
    
    Args:
    - x: input sequence (N, L, D)
    - shift: (N, D)
    - scale: (N, D)
    """
    return x * (1 + scale[:, None, ...]) + shift[:, None, ...]  # expand to make shape broadcastable


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_w, emb_h], axis=1) # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int tuple of the grid, (height, width)
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    h, w = grid_size

    grid_h = np.arange(h, dtype=np.float32)
    grid_w = np.arange(w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def unpatchify(
    x: jnp.ndarray, *, patch_sizes: tuple[int, int], channels: int = 3
) -> jnp.ndarray:
    p, q = patch_sizes
    h = w = int(x.shape[1]**.5)

    x = jnp.reshape(x, (x.shape[0], h, w, p, q, channels))
    x = jnp.einsum('nhwpqc->nhpwqc', x)
    imgs = jnp.reshape(x, (x.shape[0], h * p, w * q, channels))
    return imgs