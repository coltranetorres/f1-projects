"""Pure replay computation. No FastF1 imports: everything here works on plain arrays."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HZ = 4.0
MAX_WINDOW_S = 120.0
PIT_START_DIST = 50.0   # raw units (1/10 m): a t=0 position this far from the outline is the pit lane
RUNNING, FINISHED, DNF = 0, 1, 2
PIT_START = 3   # pit-lane starter that has not moved yet (DNF wins over it)


@dataclass
class DriverInput:
    code: str
    number: str
    team: str
    color: str                    # hex, no '#'
    t: np.ndarray                 # seconds since race start, increasing
    x: np.ndarray
    y: np.ndarray
    classified_pos: int | None    # official finishing position; None if not classified
    finish_t: float | None        # seconds when the driver's last lap ended
    n_laps: int


@dataclass
class RaceInput:
    event: str
    drivers: list[DriverInput]
    outline: np.ndarray           # (N, 2): one lap, first point on the start/finish line
    total_laps: int
    best_lap: tuple[str, float]   # (driver code, seconds)


@dataclass
class Replay:
    meta: dict
    clock: np.ndarray             # (T,)
    x: np.ndarray                 # (D, T) int32
    y: np.ndarray
    pos: np.ndarray               # (D, T) int8, 1-based
    gap: np.ndarray               # (D, T) float32 seconds behind the leader
    down: np.ndarray              # (D, T) int8 laps behind the leader
    status: np.ndarray            # (D, T) int8: RUNNING / FINISHED / DNF / PIT_START
    lap: np.ndarray               # (T,) int16 leader lap
    lap_time: np.ndarray          # (T,) float32 leader's current lap time

    def window(self, t0: float, t1: float) -> dict:
        hz = self.meta["hz"]
        n = len(self.clock)
        i0 = min(n, max(0, math.ceil(t0 * hz - 1e-9)))
        i1 = min(n, max(i0, math.ceil(t1 * hz - 1e-9)))
        sl = slice(i0, i1)
        drivers = {
            d["code"]: {
                "x": self.x[i, sl].tolist(),
                "y": self.y[i, sl].tolist(),
                "pos": self.pos[i, sl].tolist(),
                "gap": np.round(self.gap[i, sl], 3).tolist(),
                "down": self.down[i, sl].tolist(),
                "status": self.status[i, sl].tolist(),
            }
            for i, d in enumerate(self.meta["drivers"])
        }
        return {
            "t0": i0 / hz,
            "hz": hz,
            "frames": i1 - i0,
            "lap": self.lap[sl].tolist(),
            "lap_time": np.round(self.lap_time[sl], 3).tolist(),
            "drivers": drivers,
        }

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, meta=np.array(json.dumps(self.meta)), clock=self.clock, x=self.x, y=self.y,
            pos=self.pos, gap=self.gap, down=self.down, status=self.status,
            lap=self.lap, lap_time=self.lap_time,
        )

    @classmethod
    def load(cls, path: Path) -> "Replay":
        with np.load(path, allow_pickle=False) as z:
            return cls(json.loads(str(z["meta"])), z["clock"], z["x"], z["y"], z["pos"],
                       z["gap"], z["down"], z["status"], z["lap"], z["lap_time"])


def arc_lengths(outline: np.ndarray) -> tuple[np.ndarray, float]:
    seg = np.hypot(*np.diff(outline, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    closing = float(np.hypot(*(outline[0] - outline[-1])))
    return cum, float(cum[-1] + closing)


def project_to_outline(outline: np.ndarray, cum: np.ndarray, L: float,
                       pts: np.ndarray, tree: cKDTree | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Project points onto the closed outline polyline. Returns (arc length s, distance).
    Continuous between vertices, unlike nearest-vertex snapping."""
    n = len(outline)
    tree = tree or cKDTree(outline)
    _, j = tree.query(pts)
    best_s = np.empty(len(pts))
    best_d = np.full(len(pts), np.inf)
    for k in ((j - 1) % n, j):                       # the two segments touching the nearest vertex
        a, b = outline[k], outline[(k + 1) % n]
        end = np.where(k < n - 1, cum[np.minimum(k + 1, n - 1)], L)
        seg_len = end - cum[k]
        ab = b - a
        t = np.clip(((pts - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
        dist = np.hypot(*(a + t[:, None] * ab - pts).T)
        better = dist < best_d
        best_d = np.where(better, dist, best_d)
        best_s = np.where(better, cum[k] + t * seg_len, best_s)
    return best_s, best_d


def unwrap_progress(s: np.ndarray, L: float) -> np.ndarray:
    """Along-track distance -> continuous laps. Grid slots sit just behind the line, so a
    first sample in the back half of the lap is treated as negative progress."""
    s = np.asarray(s, float).copy()
    if s[0] > L / 2:
        s[0] -= L
    d = np.diff(s)
    d = d - L * np.round(d / L)
    dist = s[0] + np.concatenate([[0.0], np.cumsum(d)])
    return np.maximum.accumulate(dist) / L


def last_moving_time(t: np.ndarray, x: np.ndarray, y: np.ndarray, min_step: float = 5.0) -> float:
    step = np.hypot(np.diff(x), np.diff(y))
    moving = np.nonzero(step > min_step)[0]
    return float(t[moving[-1] + 1]) if len(moving) else float(t[0])


def first_moving_time(t: np.ndarray, x: np.ndarray, y: np.ndarray, min_step: float = 5.0,
                      after: float = 0.0) -> float:
    """First time at/after `after` that the car moves; inf if it never does."""
    step = np.hypot(np.diff(x), np.diff(y))
    moving = np.nonzero((step > min_step) & (t[1:] >= after))[0]
    return float(t[moving[0] + 1]) if len(moving) else float("inf")


def compute_replay(race: RaceInput, hz: float = HZ) -> Replay:
    drivers = [d for d in race.drivers if len(d.t) >= 2]
    if not drivers:
        raise ValueError("no drivers with position data")
    drivers = [replace(d, finish_t=None) if d.finish_t is not None and not math.isfinite(d.finish_t) else d
               for d in drivers]
    finish_ts = [d.finish_t for d in drivers if d.finish_t is not None]
    duration = (max(finish_ts) if finish_ts else max(d.t[-1] for d in drivers)) + 5.0
    clock = np.arange(0.0, duration, 1.0 / hz)
    D, T = len(drivers), len(clock)

    cum, L = arc_lengths(race.outline)
    tree = cKDTree(race.outline)
    X, Y, prog = np.empty((D, T)), np.empty((D, T)), np.empty((D, T))
    status = np.full((D, T), RUNNING, dtype=np.int8)
    score_key = np.empty((D, T))
    finish_idx = {}
    pit_start = np.zeros(D, dtype=bool)

    for i, d in enumerate(drivers):
        X[i] = np.interp(clock, d.t, d.x)
        Y[i] = np.interp(clock, d.t, d.y)
        s_arc, dist = project_to_outline(race.outline, cum, L, np.column_stack([X[i], Y[i]]), tree)
        prog[i] = np.minimum(unwrap_progress(s_arc, L), race.total_laps)
        pit_start[i] = dist[0] > PIT_START_DIST
        if d.classified_pos is not None and d.finish_t is not None:
            k = int(np.searchsorted(clock, d.finish_t))
            finish_idx[i] = k
            status[i, k:] = FINISHED
            prog[i] = np.minimum(prog[i], d.n_laps)   # cool-down lap must not move the order or gaps
        else:
            status[i, clock >= last_moving_time(d.t, d.x, d.y)] = DNF
        if pit_start[i]:
            status[i, (clock < first_moving_time(d.t, d.x, d.y)) & (status[i] == RUNNING)] = PIT_START

    env = np.maximum.accumulate(prog.max(axis=0))   # leader's progress over time
    gap = np.maximum(clock[None, :] - np.interp(prog, env, clock), 0.0)
    down = np.maximum(np.floor(env[None, :] - prog), 0).astype(np.int8)

    winner = min((d for d in drivers if d.classified_pos is not None and d.finish_t is not None),
                 key=lambda d: d.classified_pos, default=None)
    for i, d in enumerate(drivers):
        if i in finish_idx and winner is not None:
            k = finish_idx[i]
            gap[i, k:] = max(d.finish_t - winner.finish_t, 0.0)
            down[i, k:] = race.total_laps - d.n_laps
        key = 100 - prog[i]
        if i in finish_idx:   # official position only breaks ties (also while still RUNNING at the cap,
            key += 1e-6 * d.classified_pos   # so a car reaching its final lap count never jumps a finished one)
        if pit_start[i]:   # starts from the pit lane: behind every running car until it moves
            key = np.where(clock < first_moving_time(d.t, d.x, d.y), key + 500, key)
        key = np.where(status[i] == DNF, 2000 + key, key)
        score_key[i] = key

    pos = (np.argsort(np.argsort(score_key, axis=0, kind="stable"), axis=0) + 1).astype(np.int8)

    lead_laps = np.floor(env)
    lap = np.clip(lead_laps + 1, 1, race.total_laps).astype(np.int16)
    lap_start = np.maximum(np.interp(lead_laps, env, clock), 0.0)
    lap_time = np.maximum(clock - lap_start, 0.0)
    if winner is not None:
        kf = max(min(int(np.searchsorted(clock, winner.finish_t)), T) - 1, 0)
        lap[kf:] = lap[kf]
        lap_time[kf:] = lap_time[kf]

    meta = {
        "event": race.event,
        "total_laps": race.total_laps,
        "duration": T / hz,
        "hz": hz,
        "drivers": [{"code": d.code, "number": d.number, "team": d.team, "color": d.color} for d in drivers],
        "outline": np.rint(race.outline).astype(int).tolist(),
        "best_lap": {"driver": race.best_lap[0], "time_s": race.best_lap[1]},
    }
    return Replay(meta, clock, np.rint(X).astype(np.int32), np.rint(Y).astype(np.int32), pos,
                  gap.astype(np.float32), down, status, lap, lap_time.astype(np.float32))
