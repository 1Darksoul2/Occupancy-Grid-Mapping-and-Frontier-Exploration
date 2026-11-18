"""Webots controller: Pioneer 3-DX occupancy-grid mapping + frontier exploration.

Devices (added in the robot's extensionSlot by tools/make_worlds.py):
  lidar        360-deg 2-D lidar, 360 beams, 5 m
  gps, imu     pose (x, y, yaw)
  map_display  live map: grey unknown, white free, black occupied,
               blue frontiers, green planned path, red robot

Controller arguments (controllerArgs in the .wbt):
  --map=<office|clutter|maze>   only used to score the map against ground truth
  --strategy=<nearest|utility|utility_commit>
  --max-time=<s>                stop after this simulated time (default 900)
  --flip-lidar                  reverse beam order if the map comes out mirrored

Results are written to results/webots/<map>_<strategy>/ when exploration ends.
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

from controller import Display, Robot  # noqa: E402  (Webots API)

from aisr_explore import FREE, OCC, UNKNOWN, make_config  # noqa: E402
from aisr_explore.explorer import FrontierExplorer  # noqa: E402
from aisr_explore.maps import MAPS, rasterise, reachable_free  # noqa: E402
from aisr_explore.motion import unicycle_to_wheels  # noqa: E402
from aisr_explore.sim2d import coverage_and_accuracy  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="office")
    ap.add_argument("--strategy", default="utility_commit")
    ap.add_argument("--max-time", type=float, default=900.0)
    ap.add_argument("--flip-lidar", action="store_true")
    return ap.parse_args(sys.argv[1:])


def render(display, ex, pose, traj):
    """Paint the map into the Display (row 0 at the top, so flip y)."""
    state = ex.grid.classify()
    img = np.empty(state.shape + (4,), dtype=np.uint8)          # BGRA
    img[..., 3] = 255
    img[state == UNKNOWN, :3] = 150
    img[state == FREE, :3] = 255
    img[state == OCC, :3] = 0
    for cl in ex.last_frontiers:
        img[cl.cells[:, 0], cl.cells[:, 1], :3] = (230, 120, 40)       # blue (BGR)
    g = ex.grid
    if ex.path_xy is not None:
        iy, ix = g.world_to_grid(ex.path_xy[:, 0], ex.path_xy[:, 1])
        ok = g.in_bounds(iy, ix)
        img[iy[ok], ix[ok], :3] = (60, 170, 30)                         # green
    if traj:
        t = np.asarray(traj)
        iy, ix = g.world_to_grid(t[:, 0], t[:, 1])
        ok = g.in_bounds(iy, ix)
        img[iy[ok], ix[ok], :3] = (40, 100, 235)                        # orange trail
    ry, rx = g.world_to_grid(pose[0], pose[1])
    img[max(ry - 2, 0):ry + 3, max(rx - 2, 0):rx + 3, :3] = (30, 30, 220)  # red robot
    img = np.ascontiguousarray(img[::-1])
    w, h = display.getWidth(), display.getHeight()
    if img.shape[0] != h or img.shape[1] != w:                          # nearest-neighbour resize
        ys = (np.arange(h) * img.shape[0] / h).astype(int)
        xs = (np.arange(w) * img.shape[1] / w).astype(int)
        img = np.ascontiguousarray(img[ys][:, xs])
    ref = display.imageNew(img.tobytes(), Display.BGRA, w, h)
    display.imagePaste(ref, 0, 0, False)
    display.imageDelete(ref)


def save_results(out_dir, ex, traj, trace, summary):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "log_odds.npy"), ex.grid.log_odds)
    with open(os.path.join(out_dir, "trace.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "distance_m", "coverage", "accuracy", "x", "y"])
        w.writerows(trace)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        w.writeheader()
        w.writerow(summary)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(ex.grid.classify() + 1, cmap=ListedColormap(["#b9b8b2", "#ffffff", "#1d1d1b"]),
                  vmin=0, vmax=2, origin="lower", extent=[-6, 6, -6, 6], interpolation="nearest")
        t = np.asarray(traj)
        ax.plot(t[:, 0], t[:, 1], color="#2a78d6", linewidth=1.5)
        ax.set_xlim(-5.4, 5.4); ax.set_ylim(-5.4, 5.4)
        ax.set_title(f"Webots: {summary['map']} / {summary['strategy']}  "
                     f"({summary['distance']:.0f} m, {summary['time']:.0f} s)", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "final_map.png"), dpi=200)
        plt.close(fig)
    except Exception as e:                                  # plotting is optional
        print("could not save figure:", e)


def main():
    args = parse_args()
    cfg = make_config(args.strategy)
    robot = Robot()
    dt_ms = int(robot.getBasicTimeStep())

    lidar = robot.getDevice("lidar")
    lidar.enable(dt_ms)
    gps = robot.getDevice("gps")
    gps.enable(dt_ms)
    imu = robot.getDevice("imu")
    imu.enable(dt_ms)
    display = robot.getDevice("map_display")
    left = robot.getDevice("left wheel")
    right = robot.getDevice("right wheel")
    for m in (left, right):
        m.setPosition(float("inf"))
        m.setVelocity(0.0)

    n = lidar.getHorizontalResolution()
    fov = lidar.getFov()
    # Webots orders lidar beams from left (+fov/2) to right (-fov/2).
    angles = fov / 2.0 - np.arange(n) * (fov / n)
    if args.flip_lidar:
        angles = -angles
    cfg.max_range = min(cfg.max_range, lidar.getMaxRange())

    ex = FrontierExplorer(cfg)
    truth = area = None
    if args.map in MAPS:
        truth = rasterise(args.map, cfg.map_size, cfg.resolution)

    traj, trace = [], []
    travelled, last_xy, next_rec, next_draw = 0.0, None, 0.0, 0.0
    finished = False
    while robot.step(dt_ms) != -1:
        t = robot.getTime()
        p = gps.getValues()
        x, y = p[0], p[1]
        if not all(map(math.isfinite, (x, y))):            # GPS not ready on the first step
            continue
        yaw = imu.getRollPitchYaw()[2]
        if truth is not None and area is None:
            s = (int((y + cfg.map_size / 2) / cfg.resolution), int((x + cfg.map_size / 2) / cfg.resolution))
            area = reachable_free(truth, s)
        if last_xy is not None:
            travelled += math.hypot(x - last_xy[0], y - last_xy[1])
        last_xy = (x, y)

        ranges = np.asarray(lidar.getRangeImage(), dtype=float)
        v, w = (0.0, 0.0) if finished else ex.step(t, (x, y, yaw), ranges, angles)
        wl, wr = unicycle_to_wheels(v, w, cfg.wheel_radius, cfg.axle_length, cfg.max_wheel_speed)
        left.setVelocity(wl)
        right.setVelocity(wr)

        if finished:
            continue
        if t >= next_rec:
            traj.append((x, y))
            cov, acc = (coverage_and_accuracy(ex.grid.classify(), truth, area)
                        if truth is not None else (math.nan, math.nan))
            trace.append((round(t, 2), round(travelled, 3), cov, acc, x, y))
            next_rec += 1.0
        if t >= next_draw:
            render(display, ex, (x, y), traj)
            next_draw += 0.5

        if ex.done or t > args.max_time:
            finished = True
            render(display, ex, (x, y), traj)
            cov, acc = trace[-1][2], trace[-1][3]
            summary = dict(map=args.map, strategy=args.strategy, finished=ex.done,
                           time=t, distance=travelled, coverage=cov, accuracy=acc, **ex.stats())
            out = os.path.join(ROOT, "results", "webots", f"{args.map}_{args.strategy}")
            save_results(out, ex, traj, trace, summary)
            print(f"[controller] done at t={t:.1f}s, distance={travelled:.1f} m, "
                  f"coverage={100 * cov:.1f} %, accuracy={100 * acc:.2f} %  -> {out}")


if __name__ == "__main__":
    main()
