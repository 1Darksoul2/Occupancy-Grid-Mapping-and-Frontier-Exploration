"""Test environments, defined once as rectangles and used twice:
  * rasterised into a ground-truth grid for the NumPy simulator and for the map
    accuracy metric, and
  * written into Webots .wbt worlds by tools/make_worlds.py,
so the offline experiments and the Webots demo run in identical geometry.

Rectangle = (centre_x, centre_y, size_x, size_y) in metres. Arena: 10 m x 10 m.
"""

import math

import numpy as np

T = 0.15  # wall thickness

_OUTER = [(0, 5, 10 + T, T), (0, -5, 10 + T, T), (5, 0, T, 10 + T), (-5, 0, T, 10 + T)]

MAPS = {
    # Four rooms joined by doors, with furniture.
    "office": {
        "walls": _OUTER + [
            (0, -4.0, T, 2.0), (0, 0.0, T, 4.0), (0, 4.0, T, 2.0),        # x = 0, doors at y=+-2.5
            (-4.0, 0, 2.0, T), (-1.0, 0, 2.0, T),                          # y = 0 left, door x=-2.5
            (1.25, 1.0, 2.5, T), (4.25, 1.0, 1.5, T),                      # y = 1 right, door x=3
        ],
        "boxes": [(-3.0, 3.0, 0.8, 0.8), (3.0, 3.5, 1.0, 0.6), (-2.5, -3.0, 0.6, 1.2),
                  (3.0, -2.5, 1.2, 1.2), (1.5, -4.2, 0.6, 0.6), (-1.2, 2.2, 0.5, 0.5)],
        "starts": [(-4.0, -4.0, 0.0), (4.0, 4.2, math.pi), (-1.0, 1.5, 0.0)],
    },
    # Open hall with scattered obstacles (warehouse-like).
    "clutter": {
        "walls": _OUTER,
        "boxes": [(-3.0, -3.0, 1.0, 0.6), (-1.0, -3.5, 0.5, 0.5), (1.5, -3.0, 1.5, 0.5),
                  (3.5, -3.8, 0.6, 0.6), (-3.5, -0.5, 0.5, 1.5), (-1.5, 0.5, 0.8, 0.8),
                  (1.0, 0.8, 0.5, 0.5), (3.0, 0.0, 0.8, 1.6), (-3.0, 2.5, 1.2, 0.5),
                  (-0.5, 3.0, 0.6, 0.6), (1.8, 3.2, 1.0, 1.0), (4.0, 2.5, 0.5, 0.5),
                  (0.5, -1.5, 0.6, 0.6)],
        "starts": [(-4.0, -4.0, 0.0), (4.0, 4.0, math.pi), (0.0, -0.5, math.pi / 2)],
    },
    # Zig-zag corridors (long, narrow, many dead ends).
    "maze": {
        "walls": _OUTER + [
            (-1.0, -2.5, 8.0, T), (1.0, 0.0, 8.0, T), (-1.0, 2.5, 8.0, T),
            (-1.0, -4.45, T, 1.1), (2.0, -1.95, T, 1.1),     # stubs leave ~1.3 m gaps
            (-1.0, 0.55, T, 1.1), (1.0, 4.45, T, 1.1),
        ],
        "boxes": [(3.8, -4.2, 0.5, 0.5), (-4.2, 1.0, 0.5, 0.5)],
        "starts": [(-4.0, -4.0, 0.0), (-4.0, 4.0, 0.0), (4.0, 1.25, math.pi)],
    },
}


def rectangles(name):
    m = MAPS[name]
    return list(m["walls"]) + list(m["boxes"])


def rasterise(name, size=12.0, res=0.05):
    """Boolean ground-truth occupancy grid with the same geometry as OccupancyGrid."""
    n = int(round(size / res))
    c = -size / 2 + (np.arange(n) + 0.5) * res
    X, Y = np.meshgrid(c, c)                       # rows = y, cols = x
    occ = np.zeros((n, n), dtype=bool)
    for cx, cy, sx, sy in rectangles(name):
        occ |= (np.abs(X - cx) <= sx / 2) & (np.abs(Y - cy) <= sy / 2)
    return occ


def reachable_free(occ, start_cell):
    """Free cells 4-connected to the start: the area that *should* be explored."""
    from collections import deque
    ny, nx = occ.shape
    seen = np.zeros_like(occ)
    q = deque([start_cell])
    seen[start_cell] = True
    while q:
        y, x = q.popleft()
        for v, u in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= v < ny and 0 <= u < nx and not occ[v, u] and not seen[v, u]:
                seen[v, u] = True
                q.append((v, u))
    return seen
