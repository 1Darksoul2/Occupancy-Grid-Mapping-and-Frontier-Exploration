"""Frontier exploration state machine (simulator independent).

Every control step:   scan -> map update -> (re)plan if needed -> pure pursuit.
Every re-plan:        classify map -> frontiers -> cost map -> one Dijkstra ->
                      one candidate goal per frontier cluster -> score -> path.

Scoring:
    nearest : U = 1 / (d + 0.1)                      (Yamauchi 1997)
    utility : U = IG * exp(-lambda * d)              (Gonzalez-Banos & Latombe 2002)
with d the planned path length [m] and IG the unknown area [m^2] around the goal.
With commit_margin m > 0 the current goal is kept unless a new goal has
U_new > (1 + m) * U_current (hysteresis against goal oscillation).
Goals that are reached but still produce a frontier, or that time out, are
blacklisted so the robot cannot loop forever on an unobservable frontier.
"""

import math
import time

import numpy as np

from .frontier import box_sum, cluster_frontiers, frontier_mask, integral_image
from .motion import PathFollower
from .occupancy_grid import FREE, OCC, UNKNOWN, OccupancyGrid
from .planner import build_costmap, clear_start, dijkstra, dilate


class Candidate:
    __slots__ = ("cell", "xy", "dist_m", "ig_m2", "utility", "cluster_size")

    def __init__(self, cell, xy, dist_m, ig_m2, utility, cluster_size):
        self.cell, self.xy, self.dist_m = cell, xy, dist_m
        self.ig_m2, self.utility, self.cluster_size = ig_m2, utility, cluster_size


class FrontierExplorer:
    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.grid = OccupancyGrid.from_config(cfg)
        self.follower = PathFollower(cfg.lookahead, cfg.v_max, cfg.w_max,
                                     cfg.rotate_threshold, cfg.goal_tolerance)
        self.log = log or (lambda *a, **k: None)
        self.goal = None              # Candidate currently pursued
        self.goal_since = 0.0
        self.path_xy = None
        self.last_plan_t = -math.inf
        self.blacklist = []
        self.empty_plans = 0
        self.escape_until = -1.0
        self.escape_w = 0.0
        self.done = False
        self.done_reason = ""
        self.initialised = False
        # statistics
        self.n_plans = 0
        self.n_goals = 0
        self.n_blacklisted = 0
        self.n_escapes = 0
        self.plan_time_total = 0.0
        self.last_frontiers = []      # for visualisation

    # ------------------------------------------------------------------ main step
    def step(self, t, pose, ranges, angles):
        """Return (v, w). Sets self.done when no reachable frontier is left."""
        cfg = self.cfg
        x, y, th = pose
        if not self.initialised:
            self.grid.mark_free_disc(x, y, cfg.robot_radius)
            self.initialised = True
        ranges = np.asarray(ranges, dtype=float)
        angles = np.asarray(angles, dtype=float)
        self.grid.update(x, y, th, ranges, angles, cfg.max_range, cfg.min_range)
        if self.done:
            return 0.0, 0.0

        # reactive safety layer: something unexpected right in front -> turn away
        if t < self.escape_until:
            return 0.0, self.escape_w
        # danger zone = rectangle in front of the robot, as wide as the robot
        r = np.where(np.isfinite(ranges), ranges, cfg.max_range)
        valid = r > cfg.min_range
        bx, by = r * np.cos(angles), r * np.sin(angles)
        danger = valid & (bx > 0) & (bx < cfg.stop_distance) & (np.abs(by) < cfg.robot_radius + 0.02)
        moving_forward = self.follower.path is not None
        if moving_forward and danger.any():
            left = r[(angles > 0.3) & (angles < 2.0)]
            right = r[(angles < -0.3) & (angles > -2.0)]
            turn_left = (left.mean() if left.size else 0) >= (right.mean() if right.size else 0)
            self.escape_w = cfg.w_max * (0.8 if turn_left else -0.8)
            self.escape_until = t + 0.8
            self.last_plan_t = -math.inf          # force a re-plan after escaping
            self.n_escapes += 1
            return 0.0, self.escape_w

        need_plan = (self.follower.path is None
                     or t - self.last_plan_t >= cfg.replan_period)
        v, w, reached = self.follower.compute(x, y, th)
        if reached:
            self._blacklist_goal("reached")
            need_plan = True
        if self.goal is not None and t - self.goal_since > cfg.goal_timeout:
            self._blacklist_goal("timeout")
            need_plan = True

        if need_plan:
            self.plan(t, (x, y, th))
            if self.done:
                return 0.0, 0.0
            if self.follower.path is None:          # nothing reachable yet: look around
                return 0.0, 0.6
            v, w, _ = self.follower.compute(x, y, th)
        return v, w

    def _blacklist_goal(self, why):
        if self.goal is not None:
            self.blacklist.append(self.goal.xy)
            self.n_blacklisted += why == "timeout"
        self.goal = None
        self.follower.clear()

    # ------------------------------------------------------------------ planning
    def plan(self, t, pose):
        t0 = time.perf_counter()
        cfg, grid = self.cfg, self.grid
        x, y, _ = pose
        res = grid.res
        self.last_plan_t = t
        self.n_plans += 1

        state = grid.classify()
        clusters = cluster_frontiers(frontier_mask(state), cfg.min_frontier_size)
        self.last_frontiers = clusters

        start = tuple(int(v) for v in grid.world_to_grid(x, y))
        cost = build_costmap(state, res, cfg.robot_radius, cfg.safety_margin,
                             cfg.cost_band, cfg.band_penalty)
        clear_start(cost, state, start,
                    int(math.ceil((cfg.robot_radius + cfg.safety_margin) / res)),
                    1.0 + cfg.band_penalty)
        dj = dijkstra(cost, start)

        candidates = self._score(clusters, dj.dist, state, res)
        self.plan_time_total += time.perf_counter() - t0

        if not candidates:
            self.empty_plans += 1
            self.goal = None
            self.follower.clear()
            if self.empty_plans >= cfg.empty_plans_to_finish:
                self.done = True
                self.done_reason = ("no frontiers left" if not clusters
                                    else "remaining frontiers unreachable")
                self.log(f"[explorer] t={t:.1f}s finished: {self.done_reason}")
            return
        self.empty_plans = 0

        best = max(candidates, key=lambda c: c.utility)
        if self.goal is not None and cfg.commit_margin > 0:
            same = [c for c in candidates
                    if math.dist(c.xy, self.goal.xy) < cfg.blacklist_radius + 0.1]
            if same:
                cur = max(same, key=lambda c: c.utility)
                if best.utility <= (1.0 + cfg.commit_margin) * cur.utility:
                    best = cur

        if self.goal is None or math.dist(best.xy, self.goal.xy) > cfg.blacklist_radius:
            self.goal_since = t
            self.n_goals += 1
        self.goal = best

        cells = dj.path_to(best.cell)
        iy = np.array([c[0] for c in cells])
        ix = np.array([c[1] for c in cells])
        px, py = grid.grid_to_world(iy, ix)
        pts = np.column_stack([px, py])[::2]           # thin to every second cell
        if len(cells) and (len(cells) - 1) % 2:
            pts = np.vstack([pts, [px[-1], py[-1]]])
        self.path_xy = pts
        self.follower.set_path(pts)

    def _score(self, clusters, dist, state, res):
        cfg = self.cfg
        R = int(math.ceil(cfg.approach_radius / res))
        ig_half = int(math.ceil(cfg.ig_radius / res))
        ii = integral_image(state == UNKNOWN)
        ny, nx = dist.shape
        out = []
        for cl in clusters:
            cy, cx = cl.cells[:, 0], cl.cells[:, 1]
            y0, y1 = max(cy.min() - R, 0), min(cy.max() + R + 1, ny)
            x0, x1 = max(cx.min() - R, 0), min(cx.max() + R + 1, nx)
            win = np.zeros((y1 - y0, x1 - x0), dtype=bool)
            win[cy - y0, cx - x0] = True
            dwin = dist[y0:y1, x0:x1]
            # approach cells: reachable, within approach_radius of the cluster,
            # and not inside a blacklisted disc
            cand = dilate(win, R) & np.isfinite(dwin)
            wy_idx, wx_idx = np.mgrid[y0:y1, x0:x1]
            wxs, wys = self.grid.grid_to_world(wy_idx, wx_idx)
            for bx, by in self.blacklist:
                cand &= np.hypot(wxs - bx, wys - by) >= cfg.blacklist_radius
            if not cand.any():
                continue
            # the approach cell closest to the cluster centroid (middle of long frontiers)
            d2c = (wy_idx - cl.centroid[0]) ** 2 + (wx_idx - cl.centroid[1]) ** 2
            k = int(np.argmin(np.where(cand, d2c, np.inf)))
            gy, gx = divmod(k, win.shape[1])
            gy, gx = gy + y0, gx + x0
            xy = (float(wxs[gy - y0, gx - x0]), float(wys[gy - y0, gx - x0]))
            d_m = float(dist[gy, gx]) * res
            ig = box_sum(ii, gy, gx, ig_half) * res * res
            if cfg.selection == "nearest":
                u = 1.0 / (d_m + 0.1)
            else:
                u = ig * math.exp(-cfg.lambda_cost * d_m)
            out.append(Candidate((gy, gx), xy, d_m, ig, u, cl.size))
        return out

    # ------------------------------------------------------------------ info
    def stats(self):
        return dict(n_plans=self.n_plans, n_goals=self.n_goals,
                    n_blacklisted=self.n_blacklisted, n_escapes=self.n_escapes,
                    mean_plan_ms=1000.0 * self.plan_time_total / max(self.n_plans, 1),
                    done=self.done, done_reason=self.done_reason)
