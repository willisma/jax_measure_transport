"""File containing the training loop for DiT on ImageNet."""

# built-in libs
from collections import defaultdict
import time
import warnings

# external libs
from absl import logging
from clu import metric_writers
from flax import nnx
import jax
import jax.numpy as jnp
import ml_collections

# deps
from data import local_imagenet_dataset, utils as data_util
from interfaces import continuous
from utils import wandb as wandb_util, initialize as init_util, checkpoint as ckpt_util


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

Batch = dict[str, jnp.ndarray]
Interfaces = continuous.Interfaces


@nnx.split_rngs(splits=jax.local_device_count(), only='dropout, time, noise, label_dropout')
def train_step(
    model: Interfaces,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch: Batch,
):
    """Training step for DiT on ImageNet. **All updates happened in-place.**
    
    Args:
    - model: DiT model wrapped by Interfaces.
    - optimizer: optimizer for training.
    - metrics: metrics for training.
    - batch: batch of samples and labels.
    """
    samples, labels = batch["samples"], batch["labels"]

    def loss_fn(model):
        loss = model(samples, labels)
        return loss.mean()

    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model)
    metrics.update(loss=loss)
    optimizer.update(grads)


def train_and_evaluate(
    config: ml_collections.ConfigDict,
    workdir: str
):
    """Train and evaluate DiT on ImageNet.
    
    Args:
    - config: configuration for training and evaluation.
    - workdir: working directory for saving checkpoints and logs.
    """

    image_size = config.data.image_size
    image_channels = config.network.in_channels

    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0
    )

    wandb_util.initialize(
        config, exp_name=config.exp_name, project_name=config.project_name
    )

    if config.data.batch_size % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')

    dataset = local_imagenet_dataset.build_imagenet_dataset(
        is_train=True,
        data_dir=config.data.data_dir,
        image_size=image_size,
        latent_dataset=config.data.latent_dataset
    )

    model, optimizer, metrics, ema = init_util.build_model(config)
    ckpt_mngr = ckpt_util.build_checkpoint_manager(
        workdir, **config.checkpoint.options
    )
    step = ckpt_mngr.latest_step()

    restored_state = ckpt_util.restore_checkpoints(
        workdir, step, nnx.split(optimizer)[-1], nnx.split(ema)[-1], mngr=ckpt_mngr
    )
    optimizer = nnx.merge(optimizer, restored_state.state)
    ema = nnx.merge(ema, restored_state.ema_state)
    model = optimizer.model

    loader = local_imagenet_dataset.build_imagenet_loader(
        config, dataset, offset_seed=step
    )

    state_axes = nnx.StateAxes(
        {'dropout': 0, 'time': 0, 'noise': 0, 'label_dropout': 0, ...: None}
    )
    p_train_step = nnx.pmap(train_step, in_axes=(state_axes, 0), out_axes=0)

    metrics_history = defaultdict(list)
    train_metrics_last_t = time.time()

    for _ in range(config.total_steps):

        batch = data_util.parse_batch(next(loader))
        p_train_step(model, optimizer, metrics, batch)

        if config.get('log_every_steps'):
            if step % config.log_every_steps == 0:
                for metric, value in metrics.compute().items():
                    metrics_history[metric].append(value)
                metrics.reset()

                summary = {
                    f'train_{k}': float(v)
                    for k, v in jax.tree_map(lambda x: x.mean(), metrics_history).items()
                }
                summary['steps_per_second'] = (
                    config.log_every_steps / (time.time() - train_metrics_last_t)
                )

                wandb_util.log_copy(summary)
                writer.write_scalars(step + 1, summary)
                metrics_history = defaultdict(list)
                train_metrics_last_t = time.time()
        
        if (step + 1) % config.save_every_steps == 0 or step + 1 == config.total_steps:
            ckpt_util.save_checkpoints(
                workdir, step + 1, nnx.split(optimizer)[-1], nnx.split(ema)[-1], mngr=ckpt_mngr
            )

            assert step + 1 == int(nnx.split(optimizer)[-1].step[0])
            
        step += 1
    
    return model, optimizer, ema