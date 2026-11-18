"""Frontier detection (Yamauchi, 1997) and information-gain estimation.

A frontier cell is a FREE cell with at least one UNKNOWN 4-neighbour. Frontier
cells are grouped into 8-connected clusters; clusters below a minimum size are
discarded as sensor noise. Information gain of a candidate goal is the number of
UNKNOWN cells in a square window around it, computed in O(1) per query from an
integral image (summed-area table).
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .occupancy_grid import UNKNOWN, FREE


def frontier_mask(state):
    free = state == FREE
    unk = np.pad(state == UNKNOWN, 1, constant_values=False)
    nb_unknown = unk[:-2, 1:-1] | unk[2:, 1:-1] | unk[1:-1, :-2] | unk[1:-1, 2:]
    return free & nb_unknown


@dataclass
class FrontierCluster:
    cells: np.ndarray        # (k, 2) array of (iy, ix)
    centroid: tuple          # (iy, ix) float

    @property
    def size(self):
        return len(self.cells)


_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def cluster_frontiers(mask, min_size=1):
    """8-connected component labelling by BFS. Returns a list of FrontierCluster."""
    ny, nx = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    clusters = []
    for sy, sx in np.argwhere(mask):
        if seen[sy, sx]:
            continue
        seen[sy, sx] = True
        q, cells = deque([(sy, sx)]), []
        while q:
            y, x = q.popleft()
            cells.append((y, x))
            for dy, dx in _N8:
                v, u = y + dy, x + dx
                if 0 <= v < ny and 0 <= u < nx and mask[v, u] and not seen[v, u]:
                    seen[v, u] = True
                    q.append((v, u))
        if len(cells) >= min_size:
            arr = np.array(cells, dtype=np.int64)
            clusters.append(FrontierCluster(arr, tuple(arr.mean(axis=0))))
    return clusters


def integral_image(a):
    ii = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.int64)
    ii[1:, 1:] = np.asarray(a, dtype=np.int64).cumsum(0).cumsum(1)
    return ii


def box_sum(ii, cy, cx, half):
    """Sum of the source array over the square [cy-half, cy+half] x [cx-half, cx+half]."""
    ny, nx = ii.shape[0] - 1, ii.shape[1] - 1
    y0, y1 = max(cy - half, 0), min(cy + half + 1, ny)
    x0, x1 = max(cx - half, 0), min(cx + half + 1, nx)
    return int(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])
