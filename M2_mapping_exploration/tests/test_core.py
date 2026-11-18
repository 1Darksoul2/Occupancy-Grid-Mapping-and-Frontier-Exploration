"""Unit tests for the core algorithms.  Run:  python -m unittest discover tests -v"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aisr_explore import FREE, OCC, UNKNOWN, OccupancyGrid, make_config  # noqa: E402
from aisr_explore.frontier import box_sum, cluster_frontiers, frontier_mask, integral_image  # noqa: E402
from aisr_explore.motion import PathFollower, unicycle_to_wheels  # noqa: E402
from aisr_explore.planner import build_costmap, dijkstra, dilate  # noqa: E402
from aisr_explore.sim2d import run_episode  # noqa: E402


class TestOccupancyGrid(unittest.TestCase):
    def test_single_beam(self):
        g = OccupancyGrid(size=4.0, resolution=0.1)
        for _ in range(3):
            g.update(0.0, 0.0, 0.0, [1.0], [0.0], max_range=3.0)
        st = g.classify()
        iy, ix = g.world_to_grid(0.5, 0.0)
        self.assertEqual(st[iy, ix], FREE)
        iy, ix = g.world_to_grid(1.0, 0.0)
        self.assertEqual(st[iy, ix], OCC)
        iy, ix = g.world_to_grid(1.5, 0.0)
        self.assertEqual(st[iy, ix], UNKNOWN)

    def test_no_return_clears_but_marks_nothing(self):
        g = OccupancyGrid(size=4.0, resolution=0.1)
        g.update(0.0, 0.0, 0.0, [np.inf], [0.0], max_range=1.5)
        st = g.classify()
        self.assertFalse((st == OCC).any())
        self.assertEqual(st[g.world_to_grid(1.0, 0.0)], FREE)

    def test_log_odds_clamped(self):
        g = OccupancyGrid(size=2.0, resolution=0.1, l_max=2.0)
        for _ in range(50):
            g.update(0.0, 0.0, 0.0, [0.5], [0.0], max_range=1.0)
        self.assertLessEqual(g.log_odds.max(), 2.0)


class TestFrontier(unittest.TestCase):
    def test_frontier_and_clusters(self):
        st = np.full((20, 20), UNKNOWN, np.int8)
        st[5:15, 5:10] = FREE
        f = frontier_mask(st)
        self.assertTrue(f[5:15, 9].all())       # border with unknown
        self.assertFalse(f[7:13, 7].any())      # interior
        cl = cluster_frontiers(f, min_size=1)
        self.assertEqual(len(cl), 1)

    def test_integral_image(self):
        a = np.random.default_rng(0).integers(0, 2, (30, 30))
        ii = integral_image(a)
        self.assertEqual(box_sum(ii, 10, 12, 3), int(a[7:14, 9:16].sum()))
        self.assertEqual(box_sum(ii, 0, 0, 3), int(a[0:4, 0:4].sum()))


class TestPlanner(unittest.TestCase):
    def test_dilate(self):
        m = np.zeros((11, 11), bool)
        m[5, 5] = True
        d = dilate(m, 2)
        self.assertTrue(d[5, 7] and d[3, 5])
        self.assertFalse(d[7, 7])

    def test_dijkstra_around_wall(self):
        st = np.full((40, 40), FREE, np.int8)
        st[0:30, 20] = OCC
        cost = build_costmap(st, 0.1, 0.1, 0.0, 0.0, 0.0)
        dj = dijkstra(cost, (5, 5))
        path = dj.path_to((5, 35))
        self.assertTrue(path)
        self.assertEqual(path[0], (5, 5))
        self.assertEqual(path[-1], (5, 35))
        self.assertTrue(all(np.isfinite(cost[p]) for p in path))
        self.assertGreater(max(p[0] for p in path), 29)   # had to go around the wall


class TestMotion(unittest.TestCase):
    def test_wheels(self):
        wl, wr = unicycle_to_wheels(0.3, 0.0, 0.1, 0.3, 10.0)
        self.assertAlmostEqual(wl, 3.0)
        self.assertAlmostEqual(wr, 3.0)
        wl, wr = unicycle_to_wheels(0.0, 1.0, 0.1, 0.3, 10.0)
        self.assertAlmostEqual(wr, -wl)
        wl, wr = unicycle_to_wheels(5.0, 0.0, 0.1, 0.3, 10.0)   # saturation
        self.assertAlmostEqual(max(abs(wl), abs(wr)), 10.0)

    def test_pure_pursuit_turns_toward_path(self):
        f = PathFollower(lookahead=0.4)
        f.set_path([(0, 0), (0.5, 0.5), (1, 1), (1.5, 1.5)])
        v, w, done = f.compute(0, 0, 0.0)     # path is 45 deg to the left
        self.assertGreater(w, 0)
        self.assertFalse(done)


class TestEndToEnd(unittest.TestCase):
    def test_explores_clutter(self):
        s, *_ = run_episode("clutter", make_config("utility_commit"), 0, max_time=600)
        self.assertTrue(s["finished"])
        self.assertGreater(s["coverage"], 0.95)
        self.assertEqual(s["collisions"], 0)


if __name__ == "__main__":
    unittest.main()
