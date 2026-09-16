"""Nested sampling for feasibility analysis.

A standalone port of the feasibility-analysis nested sampler (mc::NSFEAS) of
Benoit Chachuat's MAGNUS library. Pass a constraint function and a box domain,
get back the live, dead and discarded points.

    >>> from nested_sampler import NestedSampler
    >>> def g(x):
    ...     y = x[0] ** 2 + x[1]
    ...     return [0.2 - y, y - 0.75]      # feasible iff all values <= 0
    >>> result = NestedSampler(g, lb=[-1, -1], ub=[1, 1], num_live=200).sample()
    >>> result.live.x.shape
    (200, 2)
"""

from ._core import Status
from .result import History, PointSet, Result, Stats
from .sampler import NestedSampler

__all__ = ["NestedSampler", "Result", "PointSet", "Stats", "History", "Status"]
__version__ = "0.1.0"
