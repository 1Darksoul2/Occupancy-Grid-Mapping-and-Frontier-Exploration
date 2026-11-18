"""Log-odds occupancy-grid mapping (Thrun, Burgard & Fox, Probabilistic Robotics, ch. 9).

Each cell stores l = log(p / (1 - p)). For every lidar beam the cells it passes
through receive l_free and the cell at the end point (if the beam hit something)
receives l_occ. Updates are fully vectorised in NumPy: all beams are sampled at
half-cell spacing in one array operation, and every cell is updated at most once
per scan (np.unique), which keeps a 360-beam scan at a few milliseconds.
"""

import numpy as np

UNKNOWN, FREE, OCC = -1, 0, 1


class OccupancyGrid:
    def __init__(self, size=12.0, resolution=0.05, origin=None,
                 l_occ=0.85, l_free=-0.4, l_min=-5.0, l_max=5.0,
                 occ_threshold=0.2, free_threshold=-0.2):
        self.res = float(resolution)
        self.n = int(round(size / resolution))
        self.shape = (self.n, self.n)                 # (rows = y, cols = x)
        if origin is None:
            origin = (-size / 2.0, -size / 2.0)       # world coords of the grid corner
        self.origin = np.asarray(origin, dtype=float)
        self.l_occ, self.l_free = l_occ, l_free
        self.l_min, self.l_max = l_min, l_max
        self.occ_threshold, self.free_threshold = occ_threshold, free_threshold
        self.log_odds = np.zeros(self.shape, dtype=np.float32)

    @classmethod
    def from_config(cls, cfg):
        return cls(cfg.map_size, cfg.resolution, None, cfg.l_occ, cfg.l_free,
                   cfg.l_min, cfg.l_max, cfg.occ_threshold, cfg.free_threshold)

    # ------------------------------------------------------------------ coords
    def world_to_grid(self, x, y):
        ix = np.floor((np.asarray(x) - self.origin[0]) / self.res).astype(np.int64)
        iy = np.floor((np.asarray(y) - self.origin[1]) / self.res).astype(np.int64)
        return iy, ix

    def grid_to_world(self, iy, ix):
        x = self.origin[0] + (np.asarray(ix) + 0.5) * self.res
        y = self.origin[1] + (np.asarray(iy) + 0.5) * self.res
        return x, y

    def in_bounds(self, iy, ix):
        return (iy >= 0) & (iy < self.n) & (ix >= 0) & (ix < self.n)

    def _flat(self, x, y):
        iy, ix = self.world_to_grid(x, y)
        ok = self.in_bounds(iy, ix)
        return iy[ok] * self.n + ix[ok]

    # ------------------------------------------------------------------ update
    def update(self, x, y, theta, ranges, angles, max_range, min_range=0.05):
        """Integrate one scan taken at pose (x, y, theta).

        ranges: beam lengths [m]; inf/NaN or >= max_range means "no return".
        angles: beam angles in the robot frame [rad].
        """
        r = np.asarray(ranges, dtype=float)
        a = np.asarray(angles, dtype=float)
        r = np.where(np.isfinite(r), r, max_range)
        keep = r > min_range
        r, a = r[keep], a[keep]
        if r.size == 0:
            return
        hit = r < max_range - 1e-3
        r = np.minimum(r, max_range)
        phi = theta + a
        c, s = np.cos(phi), np.sin(phi)

        # Free space: sample every half cell from the robot up to one cell short of the end.
        step = 0.5 * self.res
        t = np.arange(int(np.ceil(r.max() / step))) * step
        mask = t[None, :] < (r[:, None] - self.res)
        px = x + t[None, :] * c[:, None]
        py = y + t[None, :] * s[:, None]
        free_idx = np.unique(self._flat(px[mask], py[mask]))

        # Occupied: beam end points that are real returns.
        hit_idx = np.unique(self._flat(x + r[hit] * c[hit], y + r[hit] * s[hit]))
        free_idx = np.setdiff1d(free_idx, hit_idx, assume_unique=True)

        flat = self.log_odds.reshape(-1)
        flat[free_idx] += self.l_free
        flat[hit_idx] += self.l_occ
        np.clip(self.log_odds, self.l_min, self.l_max, out=self.log_odds)

    def mark_free_disc(self, x, y, radius):
        """The cells under the robot are free by definition (used at start-up)."""
        iy, ix = self.world_to_grid(x, y)
        rc = int(np.ceil(radius / self.res))
        yy, xx = np.mgrid[-rc:rc + 1, -rc:rc + 1]
        disc = yy ** 2 + xx ** 2 <= rc ** 2
        Y, X = iy + yy[disc], ix + xx[disc]
        ok = self.in_bounds(Y, X)
        self.log_odds[Y[ok], X[ok]] = np.minimum(self.log_odds[Y[ok], X[ok]], 2 * self.free_threshold)

    # ------------------------------------------------------------------ queries
    def probabilities(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))

    def classify(self):
        """int8 grid with UNKNOWN (-1), FREE (0), OCC (1)."""
        state = np.full(self.shape, UNKNOWN, dtype=np.int8)
        state[self.log_odds < self.free_threshold] = FREE
        state[self.log_odds > self.occ_threshold] = OCC
        return state
