"""File containing samplers. Samplers are made model / interface agnostic."""

# built-in libs
from abc import ABC, abstractmethod
import copy
from enum import Enum
from typing import Callable

# external libs
import flax.linen as nn
import jax
import jax.numpy as jnp


class SamplingTimeDistType(Enum):
    """Class for Sampling Time Distribution Types."""
    UNIFORM = 1
    EXP     = 2

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
        self, rng, net: nn.Module, x: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        num_sampling_steps: int | None = None,
        **net_kwargs
    ) -> jnp.ndarray:
        r"""Main sample loop

        Args:
        - rng: random key for potentially stochastic samplers
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
            x_curr, t_curr, rng = carry
            rng, cur_rng = jax.random.split(rng)
            x_next = self.forward(
                cur_rng, net, x_curr, t_curr, t_next, g_net, guidance_scale, **net_kwargs
            )
            return (x_next, t_next, rng), x_next

        (x_curr, _, rng), _ = jax.lax.scan(_fn, (x, timegrid[0], rng), timegrid[1:-1])
        x_final = self.last_step(rng, net, x_curr, timegrid[-2], timegrid[-1], g_net, guidance_scale, **net_kwargs)

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
        self, rng, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_next: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ) -> jnp.ndarray:
        del rng
        t_curr = self.expand_right(t_curr, x)

        net_out = net.pred(x, t_curr, **net_kwargs)

        if g_net is None:
            g_net = net
        
        def guided_fn(x, t):
            # TODO: consider using different set of args for g_net
            g_net_out = g_net.pred(x, t, **net_kwargs)
            return g_net_out + guidance_scale * (net_out - g_net_out)

        def unguided_fn(x, t):
            return net_out
        
        d_curr = jax.lax.cond(
            guidance_scale == 1., unguided_fn, guided_fn, x, t_curr
        )

        dt = t_next - t_curr
        return x + d_curr * self.bcast_right(dt, d_curr)
    
    def last_step(
        self, rng, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_next: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ) -> jnp.ndarray:
        return self.forward(rng, net, x, t_curr, t_next, g_net, guidance_scale, **net_kwargs)
    

class HeunSampler(Samplers):
    r"""Heun Sampler.

    Second Order Deterministic Sampler.
    """
    
    def forward(
        self, rng, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_next: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ) -> jnp.ndarray:
        del rng
        t_curr = self.expand_right(t_curr, x)

        net_out = net.pred(x, t_curr, **net_kwargs)

        if g_net is None:
            g_net = net
        
        def guided_fn(x, t):
            # TODO: consider using different set of args for g_net
            g_net_out = g_net.pred(x, t, **net_kwargs)
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
        self, rng, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_next: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ) -> jnp.ndarray:
        del rng
        # Heun's last step is one first order Euler step
        t_curr = self.expand_right(t_curr, x)

        net_out = net.pred(x, t_curr, **net_kwargs)

        if g_net is None:
            g_net = net
        
        def guided_fn(x, t):
            # TODO: consider using different set of args for g_net
            g_net_out = g_net.pred(x, t, **net_kwargs)
            return g_net_out + guidance_scale * (net_out - g_net_out)

        def unguided_fn(x, t):
            return net_out
        
        d_curr = jax.lax.cond(
            guidance_scale == 0., unguided_fn, guided_fn, x, t_curr
        )

        dt = t_next - t_curr
        return x + d_curr * self.bcast_right(dt, d_curr)
    

class DiffusionCoeffType(Enum):
    """Class for Sampling Time Distribution Types."""
    CONSTANT = 1
    KL       = 2
    SIGMA    = 3
    LINEAR   = 4
    COS      = 5
    SIN      = 6
    CONCAVE  = 7
    CONVEX   = 8


class EulerMaruyamaSampler(Samplers):
    r"""EulerMaruyama Sampler.
    
    First Order Stochastic Sampler.
    """

    def __init__(
        self,
        num_sampling_steps: int,
        sampling_time_dist: SamplingTimeDistType,
        sampling_time_kwargs: dict = {},

        # below are args for stochastic samplers
        diffusion_coeff: DiffusionCoeffType | Callable[[jnp.ndarray], jnp.ndarray] = DiffusionCoeffType.KL,
        diffusion_coeff_norm: float = 1.0
    ):
        super().__init__(
            num_sampling_steps,
            sampling_time_dist,
            sampling_time_kwargs
        )

        self.diffusion_coeff_fn = self.instantiate_diffusion_coeff(
            diffusion_coeff, diffusion_coeff_norm
        )

    def instantiate_diffusion_coeff(
        self, coeff: DiffusionCoeffType | Callable[[jnp.ndarray], jnp.ndarray], norm: float
    ):
        """Instantiate the diffusion coefficient for SDE sampling.
        
        Args:
        - diffusion_coeff: the desired diffusion coefficient. If a Callable is passed in, directly returned;
            otherwise instantiate the coefficient function based on our default settings.
        Returns:
        - diffusion_coeff_fn w(t)
        """

        if type(coeff) == Callable:
            return coeff
        
        choices = {
            DiffusionCoeffType.CONSTANT: lambda t: norm,
            DiffusionCoeffType.KL:       lambda t: norm * self.compute_drift(x, t)[1],
            DiffusionCoeffType.SIGMA:    lambda t: norm * self.compute_sigma_t(t)[0],
            DiffusionCoeffType.LINEAR:   lambda t: norm * (1 - t),
            DiffusionCoeffType.COS:      lambda t: 0.25 * (norm * jnp.cos(jnp.pi * (1 - t)) + 1) ** 2,
            DiffusionCoeffType.SIN:      lambda t: 0.25 * (norm * jnp.cos(jnp.pi * t) + 1) ** 2,
            DiffusionCoeffType.CONCAVE:  lambda t: norm * jnp.sin(jnp.pi * t) ** 2,
            DiffusionCoeffType.CONVEX:   lambda t: norm * jnp.cos(jnp.pi * t) ** 2,
        }

        try:
            fn = choices[coeff]
        except KeyError:
            raise ValueError(f"Diffusion coefficient function {coeff} not supported. Consider using custom functions.")
        
        return fn

    def drift(
        self, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, **net_kwargs
    ):
        tangent = net.pred(x, t_curr, **net_kwargs)
        score = net.score(x, t_curr, **net_kwargs)

        return tangent - 0.5 * self.diffusion_coeff_fn(t_curr) * score

    def forward(
        self, rng, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_next: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ) -> jnp.ndarray:
        
        t_curr = self.expand_right(t_curr, x)
        
        net_out = self.drift(net, x, t_curr, **net_kwargs)

        if g_net is None:
            g_net = net
        
        def guided_fn(x, t):
            # TODO: consider using different set of args for g_net
            g_net_out = self.drift(g_net, x, t, **net_kwargs)
            return g_net_out + guidance_scale * (net_out - g_net_out)

        def unguided_fn(x, t):
            return net_out
        
        d_curr = jax.lax.cond(
            guidance_scale == 1., unguided_fn, guided_fn, x, t_curr
        )

        dt = t_next - t_curr

        x_mean = x + d_curr * self.bcast_right(dt, d_curr)
        wiener = jax.random.normal(rng, x_mean) * jnp.sqrt(jnp.abs(dt))
        x = x_mean + self.bcast_right(
            jnp.sqrt(self.diffusion_coeff_fn(t_curr))
        ) * wiener

        return x

    def last_step(
        self, rng, net: nn.Module, x: jnp.ndarray, t_curr: jnp.ndarray, t_next: jnp.ndarray,
        g_net: nn.Module | None = None, guidance_scale: float = 1.0,
        **net_kwargs
    ) -> jnp.ndarray:
        del rng
        t_curr = self.expand_right(t_curr, x)

        net_out = self.drift(net, x, t_curr, **net_kwargs)

        if g_net is None:
            g_net = net
        
        def guided_fn(x, t):
            # TODO: consider using different set of args for g_net
            g_net_out = self.drift(g_net, x, t, **net_kwargs)
            return g_net_out + guidance_scale * (net_out - g_net_out)

        def unguided_fn(x, t):
            return net_out
        
        d_curr = jax.lax.cond(
            guidance_scale == 1., unguided_fn, guided_fn, x, t_curr
        )

        dt = t_next - t_curr

        return x + d_curr * self.bcast_right(dt, d_curr)


class EDMSampler(Samplers):
    r"""EDM Stochastic Sampler.
    
    Second Order Stochastic Sampler proposed in https://arxiv.org/abs/2206.00364
    """
    pass
