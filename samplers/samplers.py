"""File containing samplers. Samplers are made model / interface agnostic."""

# built-in libs
from abc import ABC, abstractmethod
import copy
from enum import Enum

# external libs
import flax.linen as nn
import jax
import jax.numpy as jnp


class SamplingTimeDistType(Enum):
    """Class for Sampling Time Distribution Types."""
    UNIFORM = 1
    EXP = 2

    # TODO: Add more sampling time distribution types


DEFAULT_SAMPLING_TIME_KWARGS = {
    SamplingTimeDistType.UNIFORM: {
        't_start': 1.0,
        't_end': 0.0
    },
    SamplingTimeDistType.EXP: {
        'sigma_min': 0.002,
        'sigma_max': 80.0,
        'rho': 7.0
    }
}


class Samplers(ABC):
    r"""Base class for all samplers.

    All samplers should support:
    - Sample discretized timegrid t
    - A single forward step in integration
    """

    def __init__(
        self,
        num_sampling_steps: int,
        sampling_time_dist: SamplingTimeDistType,
        sampling_time_kwargs: dict = {},

    ):
        self.num_sampling_steps = num_sampling_steps
        self.sampling_time_dist = sampling_time_dist
        self.sampling_time_kwargs = self.get_default_sampling_kwargs(
            sampling_time_kwargs, self.sampling_time_dist
        )
    
    @abstractmethod
    def forward(
        self, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_next: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ):
        r"""A single forward step in integration.

        Args:
        - net: network to integrate vector field with.
        - x: current state.
        - t_curr: current time step.
        - t_next: next time step.
        - g_net: guidance network.
        - guidance_scale: scale of guidance.
        - net_kwargs: extra net args.

        Return:
        - x_next: next state.
        """
    
    @abstractmethod
    def last_step(
        self, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_last: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ):
        r"""Last step in integration. 
        This interface is exposed since lots of samplers have special treatment for the last step:
        - Heun: last step is one first order Euler step.
        - Stochastic: last step returns the expected marginal value.
        
        Args:
        - net: network to integrate vector field with.
        - x: current state.
        - t_curr: current time step.
        - t_last: last time step. Note: model is never evaluated at this step.
        - g_net: guidance network.
        - guidance_scale: scale of guidance.
        - net_kwargs: extra net args.

        Return:
        - x_last: final state.
        """

    ########## Sampling ##########
    def sample_t(self, steps: int) -> jnp.ndarray:
        if self.sampling_time_dist == SamplingTimeDistType.UNIFORM:
            t_start = self.sampling_time_kwargs['t_start']
            t_end = self.sampling_time_kwargs['t_end']
            return jnp.linspace(t_start, t_end, steps)
        elif self.sampling_time_dist == SamplingTimeDistType.EXP:
            # following aligns with EDM implementation
            step_indices = jnp.arange(steps)
            sigma_min = self.sampling_time_kwargs['sigma_min']
            sigma_max = self.sampling_time_kwargs['sigma_max']
            rho = self.sampling_time_kwargs['rho']

            t_steps = (
                sigma_max ** (1 / rho)
                +
                step_indices / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
            ) ** rho

            # ensure last step is 0
            return jnp.concatenate([t_steps, jnp.array([0.])])
        else:
            raise ValueError(f"Sampling Time Distribution {self.sampling_time_dist} not supported.")

    def sample(
        self, net: nn.Module, x: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        num_sampling_steps: int | None = None,
        **net_kwargs
    ) -> jnp.ndarray:
        r"""Main sample loop

        Args:
        - net: network to integrate vector field with.
        - x: current state.
        - t: current time.
        - g_net: guidance network.
        - guidance_scale: scale of guidance.
        - net_kwargs: extra net args.

        Return:
        - x_final: final clean state.
        """
        if num_sampling_steps is not None:
            # exposing this pathway for flexibility in sampling
            timegrid = self.sample_t(num_sampling_steps)
        else:
            # if not provided, use the default number of sampling steps
            timegrid = self.sample_t(self.num_sampling_steps)

        def _fn(carry, t_next):
            x_curr, t_curr = carry
            x_next = self.forward(net, x_curr, t_curr, t_next, g_net, guidance_scale, **net_kwargs)
            return (x_next, t_next), x_next

        (x_curr, _), _ = jax.lax.scan(_fn, (x, timegrid[0]), timegrid[1:-1])
        x_final = self.last_step(net, x_curr, timegrid[-2], timegrid[-1], g_net, guidance_scale, **net_kwargs)

        return x_final
    

    ########## Helper Functions ##########
    def get_default_sampling_kwargs(self, kwargs: dict, sampling_time_dist: SamplingTimeDistType) -> dict:
        r"""Get default kwargs for sampling time distribution."""
        default_kwargs = copy.deepcopy(DEFAULT_SAMPLING_TIME_KWARGS[sampling_time_dist])
        for key, value in default_kwargs.items():
            if key in kwargs:
                # overwrite default value
                default_kwargs[key] = kwargs[key]
        
        return default_kwargs
                

    def expand_right(self, x: jnp.ndarray | float, y: jnp.ndarray) -> jnp.ndarray:
        """
            Expand x to match the batch dimension
            and broadcast x to the right to match the shape of y.
        """
        if isinstance(x, jnp.ndarray):
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

    def forward(
        self, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_next: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ) -> jnp.ndarray:
        
        t_curr = self.expand_right(t_curr, x)

        net_out = net(x, t_curr, **net_kwargs)

        if g_net is None:
            g_net = net
        
        def guided_fn(x, t):
            # TODO: consider using different set of args for g_net
            g_net_out = g_net(x, t, **net_kwargs)
            return g_net_out + guidance_scale * (net_out - g_net_out)

        def unguided_fn(x, t):
            return net_out
        
        d_curr = jax.lax.cond(
            guidance_scale == 1., unguided_fn, guided_fn, x, t_curr
        )

        dt = t_next - t_curr
        return x + d_curr * self.bcast_right(dt, d_curr)
    
    def last_step(
        self, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_next: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ) -> jnp.ndarray:
        return self.forward(net, x, t_curr, t_next, g_net, guidance_scale, **net_kwargs)
    

class HeunSampler(Samplers):
    r"""Heun Sampler.

    Second Order Deterministic Sampler.
    """
    
    def forward(
        self, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_next: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ) -> jnp.ndarray:
        
        t_curr = self.expand_right(t_curr, x)

        net_out = net(x, t_curr, **net_kwargs)

        if g_net is None:
            g_net = net
        
        def guided_fn(x, t):
            # TODO: consider using different set of args for g_net
            g_net_out = g_net(x, t, **net_kwargs)
            return g_net_out + guidance_scale * (net_out - g_net_out)

        def unguided_fn(x, t):
            return net_out
        
        d_curr = jax.lax.cond(
            guidance_scale == 1., unguided_fn, guided_fn, x, t_curr
        )

        dt = t_next - t_curr
        x_next = x + d_curr * self.bcast_right(dt, d_curr)

        # Heun's Method
        d_next = jax.lax.cond(
            guidance_scale == 1., unguided_fn, guided_fn, x_next, t_next
        )

        return x + 0.5 * self.bcast_right(dt, d_curr) * (d_curr + d_next)
    
    def last_step(
        self, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_next: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ) -> jnp.ndarray:
        # Heun's last step is one first order Euler step

        t_curr = self.expand_right(t_curr, x)

        net_out = net(x, t_curr, **net_kwargs)

        if g_net is None:
            g_net = net
        
        def guided_fn(x, t):
            # TODO: consider using different set of args for g_net
            g_net_out = g_net(x, t, **net_kwargs)
            return g_net_out + guidance_scale * (net_out - g_net_out)

        def unguided_fn(x, t):
            return net_out
        
        d_curr = jax.lax.cond(
            guidance_scale == 0., unguided_fn, guided_fn, x, t_curr
        )

        dt = t_next - t_curr
        return x + d_curr * self.bcast_right(dt, d_curr)