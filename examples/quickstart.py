"""Quick start: the illustrative example of Kusumo et al. (2020).

The response y = p * d0^2 + d1 must lie in [0.2, 0.75] on the box [-1, 1]^2.
"""

import numpy as np

from nested_sampler import NestedSampler


def constraints(d, p=None):
    p0 = 1.0 if p is None else p[0]
    y = p0 * d[0] ** 2 + d[1]
    return [y - 0.75, 0.20 - y]  # feasible iff all values <= 0


# Nominal parameter p = 1.
result = NestedSampler(constraints, lb=[-1, -1], ub=[1, 1], num_live=500, num_prop=16).sample()
print(result.summary())
print("first live points:\n", result.live.x[:5])

# Uncertain parameter: 100 scenarios, at most 5 percent of them may violate the constraints.
scenarios = np.random.default_rng(0).normal(1.0, np.sqrt(0.3), size=(100, 1))
result = NestedSampler(constraints, lb=[-1, -1], ub=[1, 1], scenarios=scenarios, criterion="var",
                       threshold=0.05, num_live=500, num_prop=16).sample()
print(result.summary())
result.save("quadratic_probabilistic.npz")
