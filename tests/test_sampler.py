"""Behavioural tests of the Python interface (no MAGNUS needed)."""

import numpy as np
import pytest

from nested_sampler import NestedSampler, Result, Status


def quadratic(x, p=None):
    p0 = 1.0 if p is None else p[0]
    y = p0 * x[0] ** 2 + x[1]
    return [y - 0.75, 0.20 - y]


def quadratic_vec(X, P=None):
    if P is None:
        y = X[:, 0] ** 2 + X[:, 1]
        return np.stack([y - 0.75, 0.20 - y], axis=1)
    y = P[None, :, 0] * X[:, None, 0] ** 2 + X[:, None, 1]
    return np.stack([y - 0.75, 0.20 - y], axis=2)


def make(**kw):
    kw.setdefault("display", 0)
    return NestedSampler(quadratic, [-1, -1], [1, 1], num_live=100, num_prop=16, **kw)


def test_converges_to_feasible_live_set():
    r = make().sample()
    assert r.status == Status.NORMAL and r.converged
    assert len(r.live) == 100
    assert np.all(r.live.feasible)
    assert np.all(np.diff(r.live.value) >= 0)
    y = r.live.x[:, 0] ** 2 + r.live.x[:, 1]
    assert np.all((y >= 0.2) & (y <= 0.75))
    assert np.all((r.live.x >= -1) & (r.live.x <= 1))
    # every dead point was the worst live point at its time of death
    assert np.all(np.diff(r.dead.value) <= 0)
    assert len(r.discarded) > 0
    assert r.stats.evaluations == 100 + 16 * r.stats.iterations
    assert len(r.history.contour) == r.stats.iterations


def test_deterministic():
    a, b = make().sample(), make().sample()
    np.testing.assert_array_equal(a.live.x, b.live.x)
    np.testing.assert_array_equal(a.dead.x, b.dead.x)
    c = make(seed=1).sample()
    assert not np.array_equal(a.dead.x, c.dead.x)


def test_vectorized_matches_pointwise():
    a = make().sample()
    b = NestedSampler(quadratic_vec, [-1, -1], [1, 1], num_live=100, num_prop=16, vectorized=True,
                      display=0).sample()
    np.testing.assert_array_equal(a.live.x, b.live.x)
    np.testing.assert_array_equal(a.discarded.value, b.discarded.value)


def test_scenarios_var_and_cvar():
    psam = np.random.RandomState(0).normal(1.0, 0.5, size=(30, 1))
    for crit in ("var", "cvar"):
        a = make(scenarios=psam, criterion=crit, threshold=0.1).sample()
        b = NestedSampler(quadratic_vec, [-1, -1], [1, 1], num_live=100, num_prop=16, scenarios=psam,
                          criterion=crit, threshold=0.1, vectorized=True, display=0).sample()
        assert a.converged
        assert np.all(a.live.prob <= 1.0) and np.all(a.live.prob >= 0.9 - 1e-12)
        np.testing.assert_array_equal(a.live.x, b.live.x)
    # explicit weights are normalised internally (w / sum, as in MAGNUS)
    w = make(scenarios=psam, weights=np.ones(30)).sample()
    e = make(scenarios=psam).sample()
    np.testing.assert_array_equal(w.live.x, e.live.x)


def test_max_iter_interrupts():
    r = make(max_iter=3).sample()
    assert r.status == Status.INTERRUPT
    assert r.stats.iterations == 4  # limit checked at the first replacement of iteration max_iter
    assert not r.converged


def test_failed_evaluations_are_dropped():
    calls = {"n": 0}

    def flaky(x):
        calls["n"] += 1
        if calls["n"] % 7 == 0:
            raise ValueError("boom")
        return quadratic(x)

    r = NestedSampler(flaky, [-1, -1], [1, 1], num_live=50, num_prop=8, display=0).sample()
    assert r.converged and len(r.live) == 50
    assert r.stats.failures > 0

    def nan_sometimes(x):
        g = quadratic(x)
        return [np.nan, g[1]] if x[0] > 0.9 else g

    r = NestedSampler(nan_sometimes, [-1, -1], [1, 1], num_live=50, num_prop=8, display=0).sample()
    assert r.converged and r.stats.failures > 0
    assert np.all(r.all_points().x[:, 0] <= 0.9)


def test_broken_function_raises():
    def broken(x):
        raise RuntimeError("always")

    with pytest.raises(RuntimeError, match="all constraint evaluations failed"):
        NestedSampler(broken, [-1, -1], [1, 1], num_live=10, display=0).sample()

    def inconsistent(x):
        return [1.0] if x[0] > 0 else [1.0, 2.0]

    with pytest.raises(ValueError):
        NestedSampler(inconsistent, [-1, -1], [1, 1], num_live=10, display=0).sample()


def test_argument_validation():
    with pytest.raises(ValueError):
        NestedSampler(quadratic, [0, 0], [1, 0], display=0)
    with pytest.raises(ValueError):
        NestedSampler(quadratic, [0, 0], [1, 1], criterion="mean", display=0)
    with pytest.raises(ValueError):
        NestedSampler(quadratic, [0, 0], [1, 1], criterion="cvar", threshold=0.0, display=0)
    with pytest.raises(ValueError):
        NestedSampler(quadratic, [0, 0], [1, 1], weights=[1.0], display=0)


def test_save_load_and_dataframe(tmp_path):
    r = make().sample()
    path = tmp_path / "run.npz"
    r.save(path)
    s = Result.load(path)
    assert s.status == r.status and s.stats == r.stats
    np.testing.assert_array_equal(s.live.x, r.live.x)
    np.testing.assert_array_equal(s.dead.value, r.dead.value)
    np.testing.assert_array_equal(s.history.contour, r.history.contour)
    assert "NORMAL" in r.summary()
    pd = pytest.importorskip("pandas")
    df = r.to_dataframe()
    assert isinstance(df, pd.DataFrame) and len(df) == len(r.all_points())
    assert r.feasible_points().shape[1] == 2
