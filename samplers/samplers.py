"""File containing samplers. Samplers are made model / interface agnostic."""

# built-in libs
from abc import ABC, abstractmethod
from enum import Enum


class SamplingTimeDistType(Enum):
    """Class for Sampling Time Distribution Types."""
    UNIFORM = 1

    # TODO: Add more sampling time distribution types