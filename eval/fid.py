"""File containing evaluation code for calculating the FID score."""

# built-in libs
import functools
import math
from typing import Any, Callable, Iterable

# external libs
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import torch

# deps
from eval import utils
from samplers import samplers


def calculate_stats_for_iterable(
    image_iter: Iterable[jnp.ndarray] | jnp.ndarray,
    detector: Callable[[dict, jnp.ndarray], jnp.ndarray],
    detector_params: dict,
    batch_size: int = 64,
    num_eval_images: int | None = None
) -> dict[str, np.ndarray]:
    """Calculate the statistics for an iterable of images. This function is ddp-agnostic.
    
    Args:
    - image_iter: Iterable / Array of images to calculate statistics for.
    - detector: Function to extract features. **Note: detector is assumed to be pmap / pjit'd.**
    - detector_params: Parameters for the detector. **Note: detector_params is assumed to be processed to match detector.**
    - batch_size: Batch size for processing images.
    - num_eval_images: Total number of images to evaluate

    Returns:
    - stats: Inception statistics for the images.
    """
    if isinstance(image_iter, jnp.ndarray):
        assert len(image_iter.shape) == 4, 'Image array should have shape (N, H, W, C)'
        image_iter = image_iter.reshape(-1, batch_size, *image_iter.shape[1:])
    
    total_num_images = 0

    # TODO: remove the hardcoding here
    running_mu = np.zeros(2048, dtype=np.float64)
    running_cov = np.zeros((2048, 2048), dtype=np.float64)

    for i, batch in enumerate(image_iter):
        batch = batch.reshape(jax.local_device_count(), -1, *batch.shape[1:])
        batch_features = detector(detector_params, batch)
        total_num_images += batch_features.shape[0]

        # TODO: check if this is necessary
        utils.lock()

        batch_features = np.asarray(jax.device_get(batch_features), dtype=np.float64)

        if num_eval_images is not None and total_num_images > num_eval_images:
            batch_features = batch_features[:(total_num_images - num_eval_images)]

        running_mu = running_mu + np.sum(batch_features, axis=0)
        running_cov = running_cov + np.matmul(batch_features.T, batch_features)

        if num_eval_images is not None and total_num_images >= num_eval_images:
            total_num_images = num_eval_images
            break

    mu = running_mu / total_num_images
    cov = (running_cov - np.outer(mu, mu) * total_num_images) / (total_num_images - 1)
    
    return {'mu': mu, 'sigma': cov}


def calculate_real_stats(
    config: ml_collections.ConfigDict,
    dataset: torch.utils.data.Dataset,
    detector: Callable[[dict, jnp.ndarray], jnp.ndarray],
    detector_params: dict,
) -> dict[str, np.ndarray]:
    """Calculate the statistics for real images.
    
    Args:
    - config: Overall config for experiment.
    - dataset: Image Dataset to calculate statistics for.
    - detector: Function to extract features. **Note: detector is assumed to be pmap / pjit'd.**
    - detector_params: Parameters for the detector. **Note: detector_params is assumed to be processed to match detector.**

    Returns:
    - stats: Inception statistics for the images.
    """
    
    # build distributed loader
    loader = utils.build_eval_loader(dataset, config.eval.inception_batch_size, config.data.num_workers)
    return calculate_stats_for_iterable(loader, detector, detector_params)


def calculate_cls_fake_stats(
    config: ml_collections.ConfigDict,
    rng: jnp.ndarray,

    # generator
    sampler: samplers.Samplers,
    generator: nn.Module,
    generator_params: dict,

    # detector
    detector: Callable[[dict, jnp.ndarray], jnp.ndarray],
    detector_params: dict,

    # guidance
    guide_generator: nn.Module | None = None,
    guide_generator_params: dict | None = None,
    guidance_scale: float = 1.0,
    all_eval_sample_nums: list[int] = [50000],

    # utils
    save_samples_path: str | None = None
)-> dict[str, np.ndarray]:
    """Extract and calculate the statistics for class-conditioned synthesized images.
    
    Args:
    - config: Overall config for experiment.
    - rng: PRNG key for sampling random noises and conditions.

    - generator: Generator.
    - generator_params: Parameters for the generator.

    - detector: Function to extract features. **Note: detector is assumed to be pmap / pjit'd.**
    - detector_params: Parameters for the detector. **Note: detector_params is assumed to be processed to match detector.**

    - guide_generator: Guiding generator.
    - guide_generator_params: Parameters for the guiding generator.
    - guidance_scale: scale for generation guidance.
    - all_eval_sample_nums: a list of number of total samples to generate.

    Returns:
    - stats: Inception statistics for the images.
    """

    batch_size = config.eval.batch_size
    sample_size = config.data.image_size // config.vae.downsample_factor if config.is_ldm else config.data.image_size
    sample_channels = config.model.sample_channels

    if guide_generator is not None:
        assert guide_generator_params is not None, 'Guide generator params should be provided when guide generator is provided.'
    else:
        guide_generator = generator
        guide_generator_params = generator_params

    def sample_step(rng, params, g_params):
        rng = jax.random.fold_in(rng, jax.lax.axis_index('data'))
        x_rng, y_rng = jax.random.split(rng)
        x = jax.random.normal(
            x_rng, (batch_size, sample_size, sample_size, sample_channels)
        )
        y = jax.random.randint(
            y_rng, (batch_size,), 0, config.data.num_classes
        )
        
        def generator_step(x, t, y):
            return generator.apply(params, x, t,y=y, method='pred')
        
        def guide_generator_step(x, t, y):
            return guide_generator.apply(
                g_params, x, t, y=jnp.zeros_like(y, dtype=jnp.int32) if config.eval.g_net.uncond else y, method='pred'
            )
        
        samples = sampler.sample(
            generator_step, x, g_net=guide_generator_step, guidance_scale=guidance_scale, y=y
        )
    
        return samples

    p_sample_step = jax.pmap(sample_step, axis_name='data')

    max_eval_samples = max(all_eval_sample_nums)
    eval_iters = math.ceil(
        max_eval_samples / (batch_size * jax.process_count() * jax.local_device_count())
    )
    assert (
        (eval_iters * jax.local_device_count() * batch_size) % jax.local_device_count() == 0,
        f"n_samples must be divisible by local_device_count: {max_eval_samples} % {jax.local_device_count()} != 0"
    )
    rngs = jax.random.split(rng, eval_iters)

    total_num_samples = 0
    per_process_samples = []

    for i in range(eval_iters):
        samples = p_sample_step(rngs[i], generator_params, guide_generator_params)

        per_process_samples.append(samples)
        total_num_samples = total_num_samples + samples.shape[0] * jax.process_count()

    per_process_samples = jnp.concatenate(per_process_samples, axis=0)

    all_stats = {}
    for num_eval_sampels in all_eval_sample_nums:
        all_stats[num_eval_sampels] = calculate_stats_for_iterable(
            per_process_samples, detector, detector_params, config.eval.inception_batch_size, num_eval_sampels
        )
    
    if save_samples_path is not None:
        # directly save to gcs
        pass

    # TODO: check if this is necessary
    utils.lock()
    
    return all_stats

    
def calculate_fid(
    config: ml_collections.ConfigDict,
    rng: jnp.ndarray,

    # real dataset
    dataset: torch.utils.data.Dataset,

    # generator
    sampler: samplers.Samplers,
    generator: nn.Module,
    generator_params: dict,

    # sampler
    guidance_scale: float = 1.0,

    # guidance
    guide_generator: nn.Module | None = None,
    guide_generator_params: dict | None = None,
) -> dict[str, float]:
    """Calculate the FID score betwee the synthesized images and real dataset."""
    detector_params, detector = utils.get_detector(config)

    real_stats = calculate_real_stats(config, dataset, detector, detector_params)
    fake_stats = calculate_cls_fake_stats(
        config, rng, sampler, generator, generator_params, detector, detector_params,
        guide_generator, guide_generator_params, guidance_scale, config.eval.all_eval_samples_nums, config.eval.save_samples_path
    )

    all_fid_scores = {}
    for num_samples, stats in fake_stats.items():
        fid = utils.calculate_fid(stats, ref_stats=real_stats)
        all_fid_scores[num_samples] = fid

    return all_fid_scores
