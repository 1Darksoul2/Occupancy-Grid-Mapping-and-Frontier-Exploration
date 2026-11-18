"""Path following for a differential-drive robot.

Pure pursuit (Coulter, 1992): pick the path point one look-ahead distance ahead,
compute the arc through it, curvature k = 2 sin(alpha) / L_d, and set w = v * k.
Large heading errors are handled by turning on the spot first (a differential
drive can do that; a car could not). Unicycle commands (v, w) are converted to
wheel speeds with the differential-drive inverse kinematics.
"""

import math

import numpy as np


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class PathFollower:
    def __init__(self, lookahead=0.45, v_max=0.35, w_max=1.2,
                 rotate_threshold=1.0, goal_tolerance=0.25):
        self.lookahead = lookahead
        self.v_max, self.w_max = v_max, w_max
        self.rotate_threshold = rotate_threshold
        self.goal_tolerance = goal_tolerance
        self.path = None
        self._idx = 0

    def set_path(self, points):
        self.path = np.asarray(points, dtype=float) if len(points) else None
        self._idx = 0

    def clear(self):
        self.path = None

    def compute(self, x, y, theta):
        """Return (v, w, reached_goal)."""
        if self.path is None:
            return 0.0, 0.0, False
        p = self.path
        gx, gy = p[-1]
        if math.hypot(gx - x, gy - y) < self.goal_tolerance:
            return 0.0, 0.0, True

        # progress index: closest point at or after the previous one (no going back)
        d = np.hypot(p[self._idx:, 0] - x, p[self._idx:, 1] - y)
        self._idx += int(np.argmin(d))
        # look-ahead point: first point at least L_d away
        rest = p[self._idx:]
        dist = np.hypot(rest[:, 0] - x, rest[:, 1] - y)
        far = np.nonzero(dist >= self.lookahead)[0]
        tx, ty = rest[far[0]] if far.size else rest[-1]

        alpha = wrap_angle(math.atan2(ty - y, tx - x) - theta)
        if abs(alpha) > self.rotate_threshold:
            return 0.0, math.copysign(self.w_max * 0.8, alpha), False

        ld = max(math.hypot(tx - x, ty - y), 1e-3)
        dist_goal = math.hypot(gx - x, gy - y)
        v = self.v_max * max(0.25, math.cos(alpha)) * min(1.0, 0.5 + dist_goal)
        w = v * 2.0 * math.sin(alpha) / ld
        if abs(w) > self.w_max:                    # respect turn-rate limit, keep curvature
            v *= self.w_max / abs(w)
            w = math.copysign(self.w_max, w)
        return v, w, False


def unicycle_to_wheels(v, w, wheel_radius, axle_length, max_wheel_speed):
    """Differential-drive inverse kinematics, saturating while keeping the v/w ratio."""
    wl = (v - 0.5 * w * axle_length) / wheel_radius
    wr = (v + 0.5 * w * axle_length) / wheel_radius
    m = max(abs(wl), abs(wr))
    if m > max_wheel_speed:
        wl, wr = wl * max_wheel_speed / m, wr * max_wheel_speed / m
    return wl, wr
