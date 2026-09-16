"""User-facing sampler: wraps the C++ core and the constraint function."""

import numpy as np

from . import _core
from .result import Result

_CRITERIA = {"var": _core.Options.Criterion.VAR, "cvar": _core.Options.Criterion.CVAR}


class NestedSampler:
    """Nested sampling of the feasible region of a set of constraints.

    Parameters
    ----------
    constraints : callable
        Constraint function. The design point x is feasible iff every value
        returned is <= 0.

        * without scenarios: ``constraints(x)`` returns the ng constraint values
          for one design point x (1-D array of length nx);
        * with scenarios: ``constraints(x, p)`` is called for every scenario
          row p of ``scenarios``;
        * ``vectorized=True``: ``constraints(X)`` or ``constraints(X, P)`` gets
          all points X (m, nx) and all scenarios P (ns, np) at once and returns
          an array of shape (m, ng) or (m, ns, ng).

        An exception raised for a point (non-vectorized mode) or a NaN among
        its constraint values marks the evaluation as failed; the point is
        dropped and counted in ``Result.stats.failures``.
    lb, ub : array-like
        Lower and upper bounds of the box domain.
    scenarios : array-like, optional
        Parameter scenarios, shape (ns, np) (a 1-D array is one parameter).
    weights : array-like, optional
        Scenario probabilities (normalised to sum one); equal by default.
    vectorized : bool
        Whether ``constraints`` accepts batches of points.
    num_live : int
        Number of live points.
    num_prop : int
        Proposals generated per iteration.
    criterion : {"var", "cvar"}
        Risk measure of the maximal constraint value across scenarios.
    threshold : float
        VaR/CVaR percentile (the accepted probability of violation).
    ell_mag, ell_red : float
        Initial magnification of the enclosing ellipsoid and its reduction
        exponent.
    max_iter, max_err, max_time : int, int, float
        Limits on iterations, failed evaluations and wall time in seconds
        (0 = no limit).
    seed : int
        Seed of the pseudo-random stream used for ellipsoid sampling. The
        default (5489) is the Armadillo default used by MAGNUS.
    display : int
        0 = silent, 1 = progress table.
    display_iter : int
        Print every display_iter iterations (0 = every iteration).
    rng_skip : int
        Advanced: pseudo-random draws to discard after seeding, to align with a
        MAGNUS process whose random stream has already been used.
    """

    def __init__(self, constraints, lb, ub, *, scenarios=None, weights=None, vectorized=False,
                 num_live=256, num_prop=16, criterion="var", threshold=0.1, ell_mag=0.30,
                 ell_red=0.20, max_iter=0, max_err=0, max_time=0.0, seed=5489, display=1,
                 display_iter=25, rng_skip=0):
        self.constraints = constraints
        self.vectorized = bool(vectorized)

        self.lb = np.array(lb, dtype=float).ravel()
        self.ub = np.array(ub, dtype=float).ravel()
        if self.lb.size == 0 or self.lb.shape != self.ub.shape:
            raise ValueError("lb and ub must be non-empty and of equal length")
        if not np.all(self.lb < self.ub):
            raise ValueError("lb must be strictly below ub in every dimension")

        if scenarios is None:
            self.scenarios = None
            self.weights = np.empty(0)
            if weights is not None:
                raise ValueError("weights given without scenarios")
        else:
            self.scenarios = np.array(scenarios, dtype=float)
            if self.scenarios.ndim == 1:
                self.scenarios = self.scenarios.reshape(-1, 1)
            if self.scenarios.ndim != 2 or self.scenarios.shape[0] == 0:
                raise ValueError("scenarios must have shape (ns, np)")
            ns = self.scenarios.shape[0]
            if weights is None:
                self.weights = np.full(ns, 1.0 / ns)
            else:
                self.weights = np.array(weights, dtype=float).ravel()
                if self.weights.shape != (ns,):
                    raise ValueError("weights must have one entry per scenario")
                if not np.all(self.weights > 0):
                    raise ValueError("weights must be positive")
                self.weights = self.weights / self.weights.sum()  # as MAGNUS: w / sum(w)

        opt = _core.Options()
        key = criterion.lower() if isinstance(criterion, str) else criterion
        if key not in _CRITERIA and key not in _CRITERIA.values():
            raise ValueError("criterion must be 'var' or 'cvar'")
        opt.criterion = _CRITERIA.get(key, key)
        opt.threshold = float(threshold)
        opt.num_live = int(num_live)
        opt.num_prop = int(num_prop)
        opt.ell_mag = float(ell_mag)
        opt.ell_red = float(ell_red)
        opt.max_iter = int(max_iter)
        opt.max_err = int(max_err)
        opt.max_time = float(max_time)
        opt.seed = int(seed)
        opt.rng_skip = int(rng_skip)
        opt.display = int(display)
        opt.display_iter = int(display_iter)
        if opt.num_live <= 0 or opt.num_prop <= 0:
            raise ValueError("num_live and num_prop must be positive")
        if opt.criterion == _core.Options.Criterion.CVAR and opt.threshold <= 0:
            raise ValueError("threshold must be positive for the CVaR criterion")
        self.options = opt

        self._num_constraints = None
        self._last_error = None

    @property
    def dimension(self):
        return self.lb.size

    @property
    def num_scenarios(self):
        return 1 if self.scenarios is None else self.scenarios.shape[0]

    def sample(self):
        """Run the sampler and return a :class:`Result`."""
        raw = _core.run(self.lb.tolist(), self.ub.tolist(), self.weights.tolist(), self.options, self._evaluate)
        return Result.from_raw(raw, self.lb, self.ub)

    # Evaluation of a batch of points: returns an array (m, ns, ng), NaN rows
    # for failed evaluations.
    def _evaluate(self, X):
        X = np.asarray(X, dtype=float)
        m = X.shape[0]
        ns = self.num_scenarios

        if self.vectorized:
            if self.scenarios is None:
                G = np.asarray(self.constraints(X), dtype=float)
                if G.ndim == 2 and G.shape[0] == m:
                    G = G.reshape(m, 1, -1)
            else:
                G = np.asarray(self.constraints(X, self.scenarios), dtype=float)
            if G.ndim != 3 or G.shape[0] != m or G.shape[1] != ns:
                raise ValueError("vectorized constraints must return an array of shape "
                                 "(m, ng) without scenarios or (m, ns, ng) with scenarios")
            self._check_num_constraints(G.shape[2])
            return G

        rows = []
        for x in X:
            try:
                if self.scenarios is None:
                    g = np.asarray(self.constraints(x), dtype=float).reshape(1, -1)
                else:
                    g = np.array([np.asarray(self.constraints(x, p), dtype=float).ravel()
                                  for p in self.scenarios])
            except Exception as err:  # failed evaluation: recorded by the core
                self._last_error = err
                rows.append(None)
                continue
            self._check_num_constraints(g.shape[-1])
            rows.append(g)

        if self._num_constraints is None:
            # Every evaluation so far failed: most likely a broken function.
            raise RuntimeError("all constraint evaluations failed") from self._last_error

        ng = self._num_constraints
        G = np.full((m, ns, ng), np.nan)
        for i, g in enumerate(rows):
            if g is not None:
                G[i] = g
        return G

    def _check_num_constraints(self, ng):
        if ng == 0:
            raise ValueError("constraints returned no values")
        if self._num_constraints is None:
            self._num_constraints = ng
        elif ng != self._num_constraints:
            raise ValueError("constraints returned %d values, expected %d" % (ng, self._num_constraints))
