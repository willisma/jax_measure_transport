"""File containing evaluation code for calculating the FID score."""

# built-in libs
import functools
from typing import Callable, Iterable

# external libs
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

# deps
from eval import inception

model = inception.InceptionV3(pretrained=True)

def inception_forward(
    renormalize_data: bool = False,
    run_all_gather: bool = True
):
    """Forward pass of the inception model to extract features."""
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 299, 299, 3)))
    params = flax.jax_utils.replicate(params)

    def forward(params, x):
        if renormalize_data:
            x = x.astype(jnp.float32) / 127.5 - 1
        
        # TODO: ablate following resize choices
        x = jax.image.resize(x, (299, 299), method='bilinear')
        features = model.apply(params, x, train=False).squeeze()
        if run_all_gather:
            features = jax.lax.all_gather(features, axis_name='data', tiled=True)
        
        return features

    return params, jax.pmap(forward, axis_name='data')


def calculate_stats_for_iterable(
    image_iter: Iterable[jnp.ndarray] | jnp.ndarray,
    detector: Callable[[dict, jnp.ndarray], jnp.ndarray],
    detector_params: dict,
    batch_size: int = 64,
    total_num_images: int = 50_000
) -> dict[str, jnp.ndarray]:
    """Calculate the statistics for an iterable of images.
    
    Args:
    - image_iter: Iterable / Array of images to calculate statistics for.
    - detector: Function to extract features. **Note: detector is assumed to be pmap / pjit'd.**
    - detector_params: Parameters for the detector. **Note: detector_params is assumed to be processed to match detector.**
    - batch_size: Batch size for processing images.
    - total_num_images: Total number of images to evaluate

    Returns:
    - stats: Statistics for the images.
    """
    if isinstance(image_iter, jnp.ndarray):
        assert len(image_iter.shape) == 4, 'Image array should have shape (N, H, W, C)'
    
    stats = {}
    # TODO: remove the hardcoding here
    running_mu = np.zeros(2048, dtype=np.float64)
    running_cov = np.zeros(2048, dtype=np.float64)

    for i in range(0, len(image_iter), batch_size):
        batch = image_iter[i:i + batch_size]
        batch = batch.reshape(jax.local_device_count(), -1, *batch.shape[1:])
        batch_features = detector(detector_params, batch)

        batch_features = np.asarray(jax.device_get(batch_features), dtype=np.float64)

        running_mu = running_mu + np.sum(batch_features, axis=0)
        running_cov = running_cov + np.matmul(batch_features.T, batch_features)
    
    running_mu = running_mu[:total_num_images]
    running_cov = running_cov[:total_num_images, :total_num_images]

    mu = running_mu / total_num_images
    cov = (running_cov - np.outer(mu, mu) * total_num_images) / (total_num_images - 1)
    
    return {'mu': mu, 'sigma': cov}
