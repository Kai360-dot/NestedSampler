"""Result containers: plain numpy arrays with a few conveniences."""

from dataclasses import dataclass, field

import numpy as np

from ._core import Status


@dataclass
class PointSet:
    """A set of design points with their feasibility criterion.

    x : (N, nx) coordinates
    value : (N,) criterion (max constraint value, or its VaR/CVaR across scenarios); feasible iff <= 0
    prob : (N,) probability of feasibility across the scenarios
    """

    x: np.ndarray
    value: np.ndarray
    prob: np.ndarray

    @property
    def feasible(self):
        """Boolean mask of the feasible points."""
        return self.value <= 0.0

    def __len__(self):
        return self.value.shape[0]

    @staticmethod
    def concatenate(sets):
        sets = list(sets)
        return PointSet(np.concatenate([s.x for s in sets]), np.concatenate([s.value for s in sets]),
                        np.concatenate([s.prob for s in sets]))


@dataclass
class Stats:
    iterations: int
    evaluations: int
    failures: int
    feasible: int
    walltime: float


@dataclass
class History:
    """Per-iteration record: contour level, feasible and dead counts, magnification factor."""

    iteration: np.ndarray
    contour: np.ndarray
    feasible: np.ndarray
    dead: np.ndarray
    factor: np.ndarray
    box_sampling: np.ndarray


@dataclass
class Result:
    """Outcome of a nested sampling run."""

    status: Status
    stats: Stats
    live: PointSet        # final live set, ascending criterion
    dead: PointSet        # in order of death
    discarded: PointSet   # rejected proposals, in order of rejection
    history: History
    lb: np.ndarray = field(repr=False)
    ub: np.ndarray = field(repr=False)

    @classmethod
    def from_raw(cls, raw, lb, ub):
        def points(d):
            return PointSet(np.asarray(d["x"]), np.asarray(d["value"]), np.asarray(d["prob"]))

        return cls(status=Status(raw["status"]), stats=Stats(**raw["stats"]), live=points(raw["live"]),
                   dead=points(raw["dead"]), discarded=points(raw["discarded"]),
                   history=History(**{k: np.asarray(v) for k, v in raw["history"].items()}),
                   lb=np.asarray(lb), ub=np.asarray(ub))

    @property
    def converged(self):
        """True when every live point satisfies the constraints."""
        return self.status == Status.NORMAL

    def all_points(self):
        """Live, dead and discarded points as one set."""
        return PointSet.concatenate([self.live, self.dead, self.discarded])

    def feasible_points(self):
        """Coordinates of every evaluated point that satisfies the constraints."""
        pts = self.all_points()
        return pts.x[pts.feasible]

    def summary(self):
        s = self.stats
        lines = [
            "status:      %s" % self.status.name,
            "iterations:  %d" % s.iterations,
            "evaluations: %d (%d failed)" % (s.evaluations, s.failures),
            "wall time:   %.2f s" % s.walltime,
            "live:        %d points, %d feasible, contour %.4e" % (
                len(self.live), int(self.live.feasible.sum()), self.live.value.max() if len(self.live) else np.nan),
            "dead:        %d points" % len(self.dead),
            "discarded:   %d points" % len(self.discarded),
        ]
        return "\n".join(lines)

    def save(self, path):
        """Save to a .npz file (see :meth:`load`)."""
        np.savez(path, status=int(self.status), lb=self.lb, ub=self.ub,
                 stats=np.array([self.stats.iterations, self.stats.evaluations, self.stats.failures,
                                 self.stats.feasible, self.stats.walltime]),
                 live_x=self.live.x, live_value=self.live.value, live_prob=self.live.prob,
                 dead_x=self.dead.x, dead_value=self.dead.value, dead_prob=self.dead.prob,
                 discarded_x=self.discarded.x, discarded_value=self.discarded.value,
                 discarded_prob=self.discarded.prob,
                 **{"history_" + k: v for k, v in vars(self.history).items()})

    @classmethod
    def load(cls, path):
        with np.load(path) as f:
            st = f["stats"]
            stats = Stats(int(st[0]), int(st[1]), int(st[2]), int(st[3]), float(st[4]))
            history = History(**{k: f["history_" + k] for k in History.__dataclass_fields__})

            def points(name):
                return PointSet(f[name + "_x"], f[name + "_value"], f[name + "_prob"])

            return cls(status=Status(int(f["status"])), stats=stats, live=points("live"),
                       dead=points("dead"), discarded=points("discarded"), history=history,
                       lb=f["lb"], ub=f["ub"])

    def to_dataframe(self):
        """All points as a pandas DataFrame with a 'set' column (needs pandas)."""
        import pandas as pd

        frames = []
        for name, pts in (("live", self.live), ("dead", self.dead), ("discarded", self.discarded)):
            df = pd.DataFrame(pts.x, columns=["x%d" % i for i in range(pts.x.shape[1])])
            df["value"] = pts.value
            df["prob"] = pts.prob
            df["feasible"] = pts.feasible
            df["set"] = name
            frames.append(df)
        return pd.concat(frames, ignore_index=True)
