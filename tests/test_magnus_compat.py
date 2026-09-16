"""Point-by-point comparison with the reference MAGNUS sampler.

Skipped unless the MAGNUS Python modules are importable from NS_MAGNUS_LIB
(default /opt/magnus/lib).
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from cases import CASES
from nested_sampler import NestedSampler

HERE = os.path.dirname(os.path.abspath(__file__))
MAGNUS_LIB = os.environ.get("NS_MAGNUS_LIB", "/opt/magnus/lib")


def magnus_available():
    return os.path.exists(os.path.join(MAGNUS_LIB, "magnus.so"))


def run_reference(name, tmp_path):
    out = str(tmp_path / (name + ".npz"))
    env = dict(os.environ, NS_MAGNUS_LIB=MAGNUS_LIB)
    subprocess.run([sys.executable, os.path.join(HERE, "magnus_reference.py"), name, out],
                   check=True, env=env, cwd=HERE, stdout=subprocess.DEVNULL)
    return np.load(out)


def run_port(case):
    ns = NestedSampler(case["py"], case["lb"], case["ub"], scenarios=case.get("scenarios"),
                       criterion=case["criterion"], threshold=case["threshold"],
                       num_live=case["num_live"], num_prop=case["num_prop"],
                       max_iter=case.get("max_iter", 0), display=0)
    return ns.sample()


def sorted_set(x, value, prob):
    order = np.lexsort([x[:, i] for i in range(x.shape[1] - 1, -1, -1)] + [value])
    return x[order], value[order], prob[order]


@pytest.mark.skipif(not magnus_available(), reason="MAGNUS Python modules not found")
@pytest.mark.parametrize("name", sorted(CASES))
def test_same_points_as_magnus(name, tmp_path):
    case = CASES[name]
    ref = run_reference(name, tmp_path)
    res = run_port(case)

    assert int(res.status) == int(ref["status"])
    assert res.stats.failures == int(ref["failures"]) == 0
    assert res.stats.evaluations == int(ref["evaluations"])
    assert res.stats.feasible == int(ref["feasible"])

    for set_name, pts in (("live", res.live), ("dead", res.dead), ("discarded", res.discarded)):
        rx, rv, rp = sorted_set(ref[set_name + "_x"], ref[set_name + "_value"], ref[set_name + "_prob"])
        px, pv, pp = sorted_set(pts.x, pts.value, pts.prob)
        assert px.shape == rx.shape, "%s: %d points in port, %d in MAGNUS" % (set_name, len(px), len(rx))
        np.testing.assert_allclose(px, rx, rtol=0, atol=1e-12, err_msg=set_name + " coordinates")
        np.testing.assert_allclose(pv, rv, rtol=1e-12, atol=1e-12, err_msg=set_name + " values")
        np.testing.assert_allclose(pp, rp, rtol=0, atol=1e-12, err_msg=set_name + " probabilities")
