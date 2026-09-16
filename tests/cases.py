"""Test problems shared by the port and the MAGNUS reference runs.

Each case gives the constraints twice: as a plain Python function for the
port, and as a builder of MC++ DAG expressions for MAGNUS.
"""

import numpy as np


def _quadratic_py(x, p=None):
    p0 = 1.0 if p is None else p[0]
    y = p0 * x[0] ** 2 + x[1]
    return [y - 0.75, 0.20 - y]


def _quadratic_dag(d, p):
    p0 = 1.0 if not p else p[0]
    y = p0 * d[0] ** 2 + d[1]
    return [y - 0.75, 0.20 - y]


# Example 2 of Paulen et al. (2020): exponential fit with error bounds.
_T = 0.1 * np.arange(10)
_Y = 1.0 * np.exp(1.0 * _T)


def _expfit_py(x):
    g = []
    for t, y in zip(_T, _Y):
        m = x[0] * np.exp(x[1] * t)
        g += [m - y - 1.0, -m + y - 1.0]
    return g


def _expfit_dag(d, p):
    import pymc

    g = []
    for t, y in zip(_T, _Y):
        m = d[0] * pymc.exp(d[1] * float(t))
        g += [m - float(y) - 1.0, -m + float(y) - 1.0]
    return g


def _cubic3d_py(x, p=None):
    p0, p1 = (1.0, 0.5) if p is None else (p[0], p[1])
    y = p0 * x[0] ** 2 + x[1] - p1 * x[2] ** 3
    return [y - 0.75, 0.20 - y, x[0] + x[1] + x[2] - 1.0]


def _cubic3d_dag(d, p):
    p0, p1 = (1.0, 0.5) if not p else (p[0], p[1])
    y = p0 * d[0] ** 2 + d[1] - p1 * d[2] ** 3
    return [y - 0.75, 0.20 - y, d[0] + d[1] + d[2] - 1.0]


_rng = np.random.RandomState(0)
_PSAM1 = _rng.normal(1.0, np.sqrt(0.3), size=(50, 1))
_PSAM2 = np.column_stack([_rng.normal(1.0, 0.3, 40), _rng.normal(0.5, 0.2, 40)])

CASES = {
    "quadratic_nominal": dict(
        py=_quadratic_py, dag_constraints=_quadratic_dag, lb=[-1, -1], ub=[1, 1],
        criterion="var", threshold=0.05, num_live=100, num_prop=16),
    "quadratic_var": dict(
        py=_quadratic_py, dag_constraints=_quadratic_dag, lb=[-1, -1], ub=[1, 1],
        scenarios=_PSAM1, criterion="var", threshold=0.1, num_live=200, num_prop=16),
    "quadratic_cvar": dict(
        py=_quadratic_py, dag_constraints=_quadratic_dag, lb=[-1, -1], ub=[1, 1],
        scenarios=_PSAM1, criterion="cvar", threshold=0.1, num_live=200, num_prop=8),
    "expfit": dict(
        py=_expfit_py, dag_constraints=_expfit_dag, lb=[-10, -10], ub=[10, 10],
        criterion="var", threshold=0.0, num_live=300, num_prop=16),
    "cubic3d_nominal": dict(
        py=_cubic3d_py, dag_constraints=_cubic3d_dag, lb=[-1, -1, -1], ub=[1, 1, 1],
        criterion="var", threshold=0.1, num_live=300, num_prop=16),
    "cubic3d_var": dict(
        py=_cubic3d_py, dag_constraints=_cubic3d_dag, lb=[-1, -1, -1], ub=[1, 1, 1],
        scenarios=_PSAM2, criterion="var", threshold=0.1, num_live=200, num_prop=16),
    "quadratic_maxiter": dict(
        py=_quadratic_py, dag_constraints=_quadratic_dag, lb=[-1, -1], ub=[1, 1],
        criterion="var", threshold=0.05, num_live=100, num_prop=16, max_iter=5),
}
