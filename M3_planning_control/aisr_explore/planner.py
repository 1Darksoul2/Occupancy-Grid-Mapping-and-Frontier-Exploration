"""Cost map construction and grid path planning.

* Obstacles are inflated by the robot radius plus a safety margin (configuration
  space), so the robot can be planned as a point.
* A soft cost band around the inflated region makes paths keep away from walls.
* One Dijkstra wavefront from the robot gives the path cost to EVERY reachable
  cell, so all frontier candidates are costed with a single search instead of
  one A* per frontier.
"""

import heapq
import math

import numpy as np

from .occupancy_grid import FREE, OCC

SQRT2 = math.sqrt(2.0)


def disk_offsets(r_cells):
    r = int(r_cells)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    keep = yy ** 2 + xx ** 2 <= r * r
    return yy[keep], xx[keep]


def dilate(mask, r_cells):
    """Binary dilation with a disc: stamp the disc on every set cell."""
    out = mask.copy()
    if r_cells <= 0 or not mask.any():
        return out
    ys, xs = np.nonzero(mask)
    oy, ox = disk_offsets(r_cells)
    Y = (ys[:, None] + oy[None, :]).ravel()
    X = (xs[:, None] + ox[None, :]).ravel()
    ok = (Y >= 0) & (Y < mask.shape[0]) & (X >= 0) & (X < mask.shape[1])
    out[Y[ok], X[ok]] = True
    return out


def build_costmap(state, res, robot_radius, margin, band, band_penalty):
    """Per-cell traversal cost; np.inf = not traversable (occupied, inflated or unknown)."""
    occ = state == OCC
    r_in = int(math.ceil((robot_radius + margin) / res))
    r_band = int(math.ceil((robot_radius + margin + band) / res))
    inflated = dilate(occ, r_in)
    near = dilate(occ, r_band)
    cost = np.full(state.shape, np.inf)
    trav = (state == FREE) & ~inflated
    cost[trav] = 1.0
    cost[trav & near] = 1.0 + band_penalty
    return cost


def clear_start(cost, state, start, r_cells, value):
    """If the robot sits inside the inflated zone, let it plan its way out."""
    sy, sx = start
    oy, ox = disk_offsets(r_cells)
    Y, X = sy + oy, sx + ox
    ok = (Y >= 0) & (Y < cost.shape[0]) & (X >= 0) & (X < cost.shape[1])
    Y, X = Y[ok], X[ok]
    sel = (state[Y, X] != OCC) & ~np.isfinite(cost[Y, X])
    cost[Y[sel], X[sel]] = value


class DijkstraResult:
    def __init__(self, dist, parent, width):
        self.dist = dist          # (ny, nx) cost-to-come in cells (inf = unreachable)
        self._parent = parent     # flat parent list over the padded grid
        self._w = width

    def path_to(self, goal):
        gy, gx = goal
        if not np.isfinite(self.dist[gy, gx]):
            return []
        w = self._w
        u = (gy + 1) * w + (gx + 1)
        path = []
        while u != -1:
            y, x = divmod(u, w)
            path.append((y - 1, x - 1))
            u = self._parent[u]
        path.reverse()
        return path


def dijkstra(cost, start):
    """8-connected Dijkstra on a cost grid. Edge cost = step length * mean cell cost."""
    ny, nx = cost.shape
    w = nx + 2
    padded = np.full((ny + 2, nx + 2), np.inf)
    padded[1:-1, 1:-1] = cost
    c = padded.ravel().tolist()                    # Python lists are much faster to index
    inf = math.inf
    dist = [inf] * len(c)
    parent = [-1] * len(c)
    s = (start[0] + 1) * w + (start[1] + 1)
    if c[s] != inf:
        dist[s] = 0.0
        nbrs = ((1, 1.0), (-1, 1.0), (w, 1.0), (-w, 1.0),
                (w + 1, SQRT2), (w - 1, SQRT2), (-w + 1, SQRT2), (-w - 1, SQRT2))
        heap = [(0.0, s)]
        pop, push = heapq.heappop, heapq.heappush
        while heap:
            d, u = pop(heap)
            if d > dist[u]:
                continue
            cu = c[u]
            for off, step in nbrs:
                v = u + off
                cv = c[v]
                if cv == inf:
                    continue
                nd = d + step * 0.5 * (cu + cv)
                if nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    push(heap, (nd, v))
    d = np.array(dist).reshape(ny + 2, nx + 2)[1:-1, 1:-1]
    return DijkstraResult(d, parent, w)


def path_length(points):
    p = np.asarray(points, dtype=float)
    if len(p) < 2:
        return 0.0
    return float(np.sum(np.hypot(*np.diff(p, axis=0).T)))
