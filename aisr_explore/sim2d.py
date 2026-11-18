"""Light-weight 2-D simulator used for batch experiments.

It mirrors the Webots set-up: a unicycle (differential-drive) robot with a 2-D
360 deg lidar in one of the maps from maps.py. The *same* FrontierExplorer
object as in Webots is driven here, so results transfer directly. It is much
faster than real time, which makes repeated runs over maps x strategies x
start poses practical.
"""

import math

import numpy as np

from .maps import MAPS, rasterise, reachable_free
from .motion import unicycle_to_wheels, wrap_angle
from .occupancy_grid import FREE, OCC, UNKNOWN
from .explorer import FrontierExplorer
from .planner import disk_offsets


class Lidar2D:
    def __init__(self, truth, size, res, n_beams=180, max_range=5.0, noise_std=0.01, rng=None):
        self.truth, self.res, self.n = truth, res, truth.shape[0]
        self.origin = -size / 2
        self.angles = -math.pi + np.arange(n_beams) * (2 * math.pi / n_beams)
        self.max_range, self.noise_std = max_range, noise_std
        self.rng = rng or np.random.default_rng(0)
        self._t = np.arange(0, max_range, res * 0.5)

    def scan(self, x, y, th):
        phi = th + self.angles
        px = x + self._t[None, :] * np.cos(phi)[:, None]
        py = y + self._t[None, :] * np.sin(phi)[:, None]
        ix = np.floor((px - self.origin) / self.res).astype(int)
        iy = np.floor((py - self.origin) / self.res).astype(int)
        out = (ix < 0) | (ix >= self.n) | (iy < 0) | (iy >= self.n)
        hit = np.ones_like(out)
        inside = ~out
        hit[inside] = self.truth[iy[inside], ix[inside]]
        first = np.argmax(hit, axis=1)
        any_hit = hit.any(axis=1)
        r = np.where(any_hit, self._t[first], np.inf)
        r = r + self.rng.normal(0, self.noise_std, r.shape)
        return r, self.angles


class Robot2D:
    def __init__(self, truth, size, res, pose, radius=0.24):
        self.truth, self.res, self.origin = truth, res, -size / 2
        self.x, self.y, self.th = pose
        self.oy, self.ox = disk_offsets(int(math.ceil(radius / res)))
        self.collisions = 0

    def _collides(self, x, y):
        iy = int((y - self.origin) / self.res) + self.oy
        ix = int((x - self.origin) / self.res) + self.ox
        ok = (iy >= 0) & (iy < self.truth.shape[0]) & (ix >= 0) & (ix < self.truth.shape[1])
        return bool(self.truth[iy[ok], ix[ok]].any()) or not ok.all()

    def step(self, v, w, dt):
        th = self.th + w * dt
        x = self.x + v * dt * math.cos(self.th + 0.5 * w * dt)
        y = self.y + v * dt * math.sin(self.th + 0.5 * w * dt)
        self.th = wrap_angle(th)
        if self._collides(x, y):
            self.collisions += 1
            return 0.0
        d = math.hypot(x - self.x, y - self.y)
        self.x, self.y = x, y
        return d


def coverage_and_accuracy(state, truth, area):
    known = state != UNKNOWN
    coverage = float((known & area).sum()) / max(int(area.sum()), 1)
    # accuracy only where the map makes a claim and the cell is inside the arena
    roi = known & (area | truth)
    est_occ = state == OCC
    acc = float((est_occ[roi] == truth[roi]).mean()) if roi.any() else 1.0
    return coverage, acc


def run_episode(map_name, cfg, start_index=0, dt=0.1, max_time=900.0,
                n_beams=180, noise_std=0.01, seed=0, record_every=1.0, verbose=False):
    m = MAPS[map_name]
    truth = rasterise(map_name, cfg.map_size, cfg.resolution)
    pose = m["starts"][start_index]
    robot = Robot2D(truth, cfg.map_size, cfg.resolution, pose)
    lidar = Lidar2D(truth, cfg.map_size, cfg.resolution, n_beams, cfg.max_range,
                    noise_std, np.random.default_rng(seed))
    ex = FrontierExplorer(cfg, log=print if verbose else None)
    n = truth.shape[0]
    s_iy = int((pose[1] + cfg.map_size / 2) / cfg.resolution)
    s_ix = int((pose[0] + cfg.map_size / 2) / cfg.resolution)
    area = reachable_free(truth, (s_iy, s_ix))

    t, travelled, next_rec = 0.0, 0.0, 0.0
    trace = []
    traj = []
    while t <= max_time:
        r, a = lidar.scan(robot.x, robot.y, robot.th)
        v, w = ex.step(t, (robot.x, robot.y, robot.th), r, a)
        # same wheel-speed saturation as on the Pioneer
        wl, wr = unicycle_to_wheels(v, w, cfg.wheel_radius, cfg.axle_length, cfg.max_wheel_speed)
        v = cfg.wheel_radius * (wl + wr) / 2
        w = cfg.wheel_radius * (wr - wl) / cfg.axle_length
        if t >= next_rec or ex.done:
            cov, acc = coverage_and_accuracy(ex.grid.classify(), truth, area)
            trace.append((t, travelled, cov, acc))
            traj.append((robot.x, robot.y))
            next_rec += record_every
        if ex.done:
            break
        travelled += robot.step(v, w, dt)
        t += dt

    trace = np.array(trace)
    cov_final = trace[-1, 2]

    def first(th):
        idx = np.nonzero(trace[:, 2] >= th)[0]
        return (float(trace[idx[0], 0]), float(trace[idx[0], 1])) if idx.size else (math.nan, math.nan)

    t90, d90 = first(0.90)
    t95, d95 = first(0.95)
    summary = dict(map=map_name, start=start_index, seed=seed,
                   finished=ex.done, time=float(trace[-1, 0]), distance=float(trace[-1, 1]),
                   coverage=float(cov_final), accuracy=float(trace[-1, 3]),
                   t90=t90, d90=d90, t95=t95, d95=d95,
                   collisions=robot.collisions, **ex.stats())
    return summary, trace, np.array(traj), ex.grid.classify(), truth
