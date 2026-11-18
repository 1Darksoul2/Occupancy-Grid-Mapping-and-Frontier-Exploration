# Occupancy-Grid Mapping and Frontier Exploration

CSAI4014 AISR project (Batch B3–B4). A Pioneer 3-DX with a 2-D lidar explores an
unknown indoor environment in **Webots**. It builds a probabilistic occupancy grid
as it goes and keeps driving to the most useful frontier (the boundary between known
free space and unknown space) until nothing reachable is left.
All algorithms are written in **Python + NumPy**.

```
lidar ──► log-odds occupancy grid ──► frontier detection + clustering
                                              │
              cost map (inflation) ──► Dijkstra wavefront (one search costs every frontier)
                                              │
                    frontier scoring (nearest | utility | utility + commitment)
                                              │
      pure-pursuit path follower ──► differential-drive kinematics ──► wheel speeds
                 ▲
     reactive safety layer (front danger zone → turn away, re-plan)
```

## Repository layout

| Path | What it is |
|---|---|
| `aisr_explore/occupancy_grid.py` | Log-odds occupancy grid with a vectorised inverse sensor model |
| `aisr_explore/frontier.py` | Frontier cells, 8-connected clustering, integral-image information gain |
| `aisr_explore/planner.py` | Obstacle inflation, soft cost band, 8-connected Dijkstra, path extraction |
| `aisr_explore/motion.py` | Pure pursuit and differential-drive inverse kinematics |
| `aisr_explore/explorer.py` | Exploration state machine: re-planning, goal scoring, commitment, blacklist, safety layer |
| `aisr_explore/config.py` | Every parameter, plus the three named strategies |
| `aisr_explore/maps.py` | The three test environments (office, clutter, maze), each defined once as rectangles |
| `aisr_explore/sim2d.py` | Fast 2-D simulator (unicycle + ray-cast lidar) for batch experiments |
| `controllers/frontier_explorer/` | Webots controller (same `FrontierExplorer` as offline) with a live map display |
| `worlds/*.wbt` | Webots worlds, generated from `maps.py` by `tools/make_worlds.py` |
| `experiments/run_experiments.py` | Batch comparison, tables and plots → `results/offline/` |
| `tests/test_core.py` | Unit tests and one end-to-end test |

## Setup

```bash
pip install -r requirements.txt
```

Webots R2025a or newer (any version with ENU coordinates, i.e. R2022a or newer, works if you change
`WEBOTS_VERSION` in `tools/make_worlds.py`). In Webots, go to **Tools → Preferences → Python command** and
point it at the Python that has NumPy installed.

## Run in Webots

1. **File → Open World** → `worlds/office.wbt` (or `clutter.wbt`, `maze.wbt`).
2. Press play. The robot's **map_display** window shows the map as it is built:
   grey = unknown, white = free, black = occupied, blue = frontiers, green = planned path,
   orange = trail, red = robot. If the window is hidden, open it from the robot's context menu
   (**Overlays → Display Devices**).
3. When exploration ends, the controller prints a summary and writes
   `results/webots/<map>_<strategy>/` with `final_map.png`, `trace.csv`, `summary.csv` and `log_odds.npy`.

To change the strategy or start pose, regenerate the worlds:

```bash
python tools/make_worlds.py --strategy nearest --start 1
```

or edit `controllerArgs` on the robot in the scene tree (`--strategy=nearest|utility|utility_commit`).
If the map ever comes out mirrored, add `--flip-lidar` to `controllerArgs`.

## Run the offline experiments

```bash
python experiments/run_experiments.py --lambda-sweep 0.05 0.15 0.3 0.6 1.0
```

This runs 3 maps × 3 strategies × 3 start poses × 3 sensor-noise seeds = 81 episodes, plus the λ sweep.
It takes a few minutes on a multi-core PC. Use `--quick` for a fast smoke run.
Outputs in `results/offline/`: `summary.md` (table for the report), `summary.csv`,
`coverage_vs_distance.png`, `coverage_vs_time.png`, `bars_distance95.png`, `bars_time95.png`,
`final_maps.png`, `lambda_sweep.png`.

## Tests

```bash
python -m unittest discover tests -v
```

## Methods compared

| Strategy | Goal score | Source |
|---|---|---|
| `nearest` | 1 / (path length + 0.1) | Yamauchi, 1997 |
| `utility` | IG · exp(−λ · path length) | González-Baños & Latombe, 2002 |
| `utility_commit` | utility, but keep the current goal unless a new one is ≥ 50 % better | this project |

IG is the unknown area (m²) in a 4 m × 4 m window around the candidate goal. It is computed in O(1) per
candidate with an integral image. Path lengths for all frontiers come from one Dijkstra search from the
robot instead of one A* search per frontier.

Metrics: coverage (the share of reachable free space that is now known), map accuracy (the share of known
cells whose free/occupied label matches ground truth), time and distance to 90 % and 95 % coverage,
number of goals, safety-layer escapes, planning time, and collisions (offline simulator only).


* The pose comes from GPS and the inertial unit, so this is mapping with known poses, not SLAM.
  Odometry drift is not modelled.
* The map is 2-D. Obstacles above or below the lidar plane are invisible.
* The 12 m × 12 m map size is fixed; larger worlds need a larger `map_size`.
* The Dijkstra search runs in pure Python. That takes about 60 ms per re-plan at 240 × 240 cells, which is
  fine here but would need a compiled implementation for large maps.
