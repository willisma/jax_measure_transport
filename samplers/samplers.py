"""File containing samplers. Samplers are made model / interface agnostic."""

# built-in libs
from abc import ABC, abstractmethod
from enum import Enum

# external libs
import flax.linen as nn
import jax
import jax.numpy as jnp


class SamplingTimeDistType(Enum):
    """Class for Sampling Time Distribution Types."""
    UNIFORM = 1

    # TODO: Add more sampling time distribution types


class Samplers(ABC):
    r"""Base class for all samplers.

    All samplers should support:
    - Sample discretized timegrid t
    - A single forward step in integration
    """

    @abstractmethod
    def sample_t(self, steps: int) -> jnp.ndarray:
        r"""Sample discretized timegrid t.

        Args:
        - steps: number of steps in the timegrid.

        Return:
        - t: discretized timegrid.
        """
    
    @abstractmethod
    def forward(
        self, net: nn.Module, x: jnp.ndarray, t: jnp.ndarray, dt: jnp.ndarray,
        g_net: nn.Module = None, guidance_scale: float = 1.0
    ):
        r"""A single forward step in integration.

        Args:
        - net: network to integrate vector field with.
        - x: current state.
        - t: current time.
        - g_net: guidance network.
        - guidance_scale: scale of guidance.

        Return:
        - x_next: next state.
        """

    ########## Sampling ##########
    def sample(
        self, net: nn.Module, x: jnp.ndarray,
        g_net: nn.Module = None, guidance_scale: float = 1.0
    ) -> jnp.ndarray:
        r"""Main sample loop

        Args:
        - net: network to integrate vector field with.
        - x: current state.
        - t: current time.
        - g_net: guidance network.
        - guidance_scale: scale of guidance.

        Return:
        - x_next: next state.
        """

        timegrid = self.sample_t()

        def _fn(carry, t_curr):
            x_curr, t_prev = carry
            dt = t_curr - t_prev
            x_next = self.forward(net, x_curr, t_curr, dt, g_net, guidance_scale)
            return (x_next, t_curr), x_next

        return jax.lax.scan(_fn, x, timegrid)[0][0]
    

    ########## Helper Functions ##########
    def expand_right(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
            Expand x to match the batch dimension
            and broadcast x to the right to match the shape of y.
        """
        assert len(y.shape) >= x.ndim
        return jnp.ones((y.shape[0],)) * x


    def bcast_right(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        r"""Broadcast x to the right to match the shape of y."""
        assert len(y.shape) >= x.ndim
        return x.reshape(x.shape + (1,) * (len(y.shape) - x.ndim))
    

class EulerSampler(Samplers):
    r"""Euler Sampler.

    First Order Deterministic Sampler.
    """

    def sample_t(self, steps: int) -> jnp.ndarray:
        return jnp.linspace(0, 1, steps)
    
    def forward(
        self, net: nn.Module, x: jnp.ndarray, t: jnp.ndarray, dt: jnp.ndarray,
        g_net: nn.Module = None, guidance_scale: float = 1.0
    ) -> jnp.ndarray:
        
        t = self.expand_right(t, x)

        net_out = net(x, t)

        if g_net is None:
            g_net = net
        
        def guided_fn(x, t):
            g_net_out = g_net(x, t)
            return g_net_out + guidance_scale * (net_out - g_net_out)

        def unguided_fn(x, t):
            return net_out
        
        out = jax.lax.cond(
            guidance_scale == 0., unguided_fn, guided_fn, x, t
        )

        d_cur = (x - out) / t

        return x + d_cur * dt