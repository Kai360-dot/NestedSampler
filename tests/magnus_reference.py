"""Run the reference MAGNUS sampler (mc::NSFEAS) on a named test case.

Usage: python magnus_reference.py CASE OUTPUT.npz

Runs in its own process so that Armadillo's global random stream starts from
its default seed, exactly as in a fresh MAGNUS session. Needs the MAGNUS Python
modules (pymc, magnus) on the path; set NS_MAGNUS_LIB to their directory.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("NS_MAGNUS_LIB", "/opt/magnus/lib"))
import pymc  # noqa: E402
from magnus import NSFeas  # noqa: E402

from cases import CASES  # noqa: E402


def run(case):
    DAG = pymc.FFGraph()
    DAG.options.MAXTHREAD = 1
    NS = NSFeas()
    NS.options.DISPLEVEL = 0
    NS.options.FEASCRIT = NS.options.CVAR if case["criterion"] == "cvar" else NS.options.VAR
    NS.options.FEASTHRES = case["threshold"]
    NS.options.NUMLIVE = case["num_live"]
    NS.options.NUMPROP = case["num_prop"]
    NS.options.MAXITER = case.get("max_iter", 0)
    NS.set_dag(DAG)

    nx = len(case["lb"])
    d = DAG.add_vars(nx, "d")
    scenarios = case.get("scenarios")
    if scenarios is None:
        p = []
    else:
        p = DAG.add_vars(np.shape(scenarios)[1], "p")
    NS.set_constraint(case["dag_constraints"](d, p))
    NS.set_control(d, list(case["lb"]), list(case["ub"]))
    if scenarios is not None:
        NS.set_parameter(p, [list(row) for row in np.atleast_2d(scenarios)])
    NS.setup()
    status = NS.sample()

    def unpack(t):
        return np.asarray(t[0]), np.asarray(t[1]).ravel(), np.asarray(t[2]).ravel()

    out = {"status": status, "evaluations": NS.stats.numfct, "failures": NS.stats.numerr,
           "feasible": NS.stats.numfeas, "iterations": NS.stats.iter}
    for name, t in (("live", NS.live_points), ("dead", NS.dead_points), ("discarded", NS.discard_points)):
        x, value, prob = unpack(t)
        out[name + "_x"], out[name + "_value"], out[name + "_prob"] = x, value, prob
    return out


if __name__ == "__main__":
    np.savez(sys.argv[2], **run(CASES[sys.argv[1]]))
