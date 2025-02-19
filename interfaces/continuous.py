"""File containing the interface for measure transport."""

# built-in libs
from abc import ABC, abstractmethod
from enum import Enum

# external libs
import flax
import flax.linen as nn

import jax
import jax.numpy as jnp


class TrainingTimeDistType(Enum):
    """Class for Training Time Distribution Types."""
    UNIFORM = 1
    LOGNORMAL = 2
    LOGITNORMAL = 3

    # TODO: Add more training time distribution types


class Interfaces(nn.Module, ABC):
    r"""Base class for all measure transport interfaces.
    
    All interfaces be a wrapper around network backbone and should support:
    - Calculate losses for training
        - Define transport path (\alpha_t & \sigma_t)
        - Sample t
        - Sample X_t
    - Give tangent for sampling

    Required RNG Key:
    - 
    """

    network: nn.Module
    train_time_dist_type: TrainingTimeDistType = TrainingTimeDistType.UNIFORM

    @abstractmethod
    def sample_t(self, shape: tuple[int, ...]) -> jnp.ndarray:
        r"""Sample t from the training time distribution.
        
        Args:
        - shape: shape of timestep t.

        Return:
        - t: sampled timestep t.
        """
    
    @abstractmethod
    def sample_n(self, shape: tuple[int, ...]) -> jnp.ndarray:
        r"""Sample noises.
        
        Args:
        - shape: shape of noise.

        Return:
        - n: sampled noise.
        """
        # Exposing this function to the interface allows for more flexibility in noise sampling

    @abstractmethod
    def sample_x_t(self, x: jnp.ndarray, n: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        r"""Sample X_t according to the defined interface.
        
        Args:
        - x: input clean sample.
        - n: noise.
        - t: current timestep.

        Return:
        - x_t: Sampled X_t according to tranport path.
        """

    @abstractmethod
    def target(self, x: jnp.ndarray, n: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        r"""Get training target.
        
        Args:
        - x: input clean sample.
        - n: noise.
        - t: current timestep.

        Return:
        - target: training target.
        """

    @abstractmethod
    def pred(self, x_t: jnp.ndarray, t: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        r"""Predict ODE tangent according to the defined interface.
        
        Args:
        - x_t: input noisy sample.
        - t: current timestep.
        """
    
    @abstractmethod
    def loss(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        r"""Calculate loss for training.
        
        Args:
        - x: input clean sample.
        - args: additional arguments for network forward.
        - kwargs: additional keyword arguments for network forward.

        Return:
        - loss: calculated loss.
        """
    
    ########## Helper Functions ##########
    @staticmethod
    def mean_flat(x: jnp.ndarray) -> jnp.ndarray:
        r"""Take mean w.r.t. all dimensions of x except the first."""
        return jnp.mean(x, axis=list(range(1, x.ndim)))
    
    @staticmethod
    def bcast_right(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        r"""Broadcast x to the right to match the shape of y."""
        assert len(y.shape) >= x.ndim
        return x.reshape(x.shape + (1,) * (len(y.shape) - x.ndim))


class SiTInterface(Interfaces):
    r"""Interface for SiT.
    
    Transport path:
    - x_t = (1 - t) * x + t * n

    Losses:
    - L = |D - (x - n)|^2

    Predictions:
    - x = xt + t * D
    """

    network: nn.Module
    train_time_dist_type: TrainingTimeDistType = TrainingTimeDistType.UNIFORM

    # hyperparams
    t_mu: float = 0.
    t_sigma: float = 1.0

    n_mu: float = 0.
    n_sigma: float = 1.0

    x_sigma: float = 0.5

    def sample_t(self, shape: tuple[int, ...]) -> jnp.ndarray:
        rng = self.make_rng('time')

        if self.train_time_dist_type == TrainingTimeDistType.UNIFORM:
            return jax.random.uniform(rng, shape=shape)
        elif self.train_time_dist_type == TrainingTimeDistType.LOGNORMAL:
            return jax.nn.sigmoid(jax.random.normal(rng, shape=shape) * self.t_sigma + self.t_mu)
        else:
            raise ValueError(f"Training Time Distribution Type {self.train_time_dist_type} not supported.")
    
    def sample_n(self, shape: tuple[int, ...]) -> jnp.ndarray:
        rng = self.make_rng('noise')

        return jax.random.normal(rng, shape=shape) * self.n_sigma + self.n_mu
    
    def sample_x_t(self, x: jnp.ndarray, n: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        t = self.bcast_right(t, x)
        return (1 - t) * x + t * n
    
    def target(self, x: jnp.ndarray, n: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return x - n
    
    def pred(self, x_t: jnp.ndarray, t: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return self.network(x_t, t, *args, **kwargs)
    
    def loss(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        t = self.sample_t((x.shape[0],))
        n = self.sample_n(x.shape)

        x_t = self.sample_x_t(x, n, t)
        target = self.target(x, n, t)

        net_out = self.network(x_t, t, *args, **kwargs)

        return self.mean_flat((net_out - target) ** 2)
    
    def __call__(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return self.loss(x, *args, **kwargs)


class EDMInterface(Interfaces):
    r"""Interface for EDM.
    
    Transport Path:
    - x_t = x + \sigma * n
    
    Losses:
    - L - |D - x| ^ 2

    Predictions:
    - x = D
    """

    network: nn.Module
    train_time_dist_type: TrainingTimeDistType = TrainingTimeDistType.LOGNORMAL

    # hyperparams
    t_mu: float = -1.2
    t_sigma: float = 1.2

    n_mu: float = 0.
    n_sigma: float = 1.0

    x_sigma: float = 0.5

    def sample_t(self, shape: tuple[int, ...]) -> jnp.ndarray:
        rng = self.make_rng('time')

        if self.train_time_dist_type == TrainingTimeDistType.UNIFORM:
            return jax.random.uniform(rng, shape=shape)
        elif self.train_time_dist_type == TrainingTimeDistType.LOGNORMAL:
            return jnp.exp(jax.random.normal(rng, shape=shape) * self.t_sigma + self.t_mu)
        elif self.train_time_dist_type == TrainingTimeDistType.LOGITNORMAL:
            return jax.nn.sigmoid(jax.random.normal(rng, shape=shape) * self.t_sigma + self.t_mu)
        else:
            raise ValueError(f"Training Time Distribution Type {self.train_time_dist_type} not supported.")
    
    def sample_n(self, shape: tuple[int, ...]) -> jnp.ndarray:
        rng = self.make_rng('noise')

        return jax.random.normal(rng, shape=shape) * self.n_sigma + self.n_mu
    
    def sample_x_t(self, x: jnp.ndarray, n: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return x + self.bcast_right(t, n) * n
    
    def target(self, x: jnp.ndarray, n: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        return x
    
    def pred(self, x_t: jnp.ndarray, t: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return (x_t - self.network(x_t, t, *args, **kwargs)) / self.bcast_right(t, x_t)
    
    def loss(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        sigma = self.sample_t((x.shape[0],))
        n = self.sample_n(x.shape)

        x_t = self.sample_x_t(x, n, sigma)
        target = self.target(x, n, sigma)

        # preconditionings
        c_skip = self.x_sigma ** 2 / (sigma ** 2 + self.x_sigma ** 2)
        c_out = sigma * self.x_sigma / jnp.sqrt(sigma ** 2 + self.x_sigma ** 2)
        c_in = 1 / jnp.sqrt(self.x_sigma ** 2 + sigma ** 2)
        c_noise = jnp.log(sigma) / 4

        F_x = self.network((self.bcast_right(c_in, x_t) * x_t), c_noise, *args, **kwargs)  # F_x
        D_x = self.bcast_right(c_skip, x_t) * x_t + self.bcast_right(c_out, F_x) * F_x  # D_x

        return self.mean_flat((D_x - target) ** 2)
    
    def __call__(self, x: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return self.loss(x, *args, **kwargs)




    
