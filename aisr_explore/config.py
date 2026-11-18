"""All tunable parameters in one place (units: metres, seconds, radians)."""

from dataclasses import dataclass, replace


@dataclass
class ExplorerConfig:
    # ---- map -------------------------------------------------------------
    map_size: float = 12.0          # square map side [m], centred on the origin
    resolution: float = 0.05        # cell size [m]
    l_occ: float = 0.85             # log-odds increment for a hit   (p ~ 0.70)
    l_free: float = -0.40           # log-odds increment for a pass  (p ~ 0.40)
    l_min: float = -5.0             # clamping keeps the map able to change
    l_max: float = 5.0
    occ_threshold: float = 0.20     # log-odds above  -> occupied
    free_threshold: float = -0.20   # log-odds below  -> free

    # ---- sensor ----------------------------------------------------------
    max_range: float = 5.0          # lidar max range [m]
    min_range: float = 0.10

    # ---- robot (Pioneer 3-DX) -------------------------------------------
    robot_radius: float = 0.26
    safety_margin: float = 0.08     # extra inflation beyond the robot radius
    wheel_radius: float = 0.0975
    axle_length: float = 0.33
    max_wheel_speed: float = 12.3   # rad/s

    # ---- planning --------------------------------------------------------
    cost_band: float = 0.30         # soft band around inflated obstacles [m]
    band_penalty: float = 4.0       # extra per-cell cost inside the band (free cell = 1)
    min_frontier_size: int = 6      # cells; smaller clusters are noise
    approach_radius: float = 0.8    # goal may be this far from the frontier
    replan_period: float = 1.5      # s
    goal_timeout: float = 40.0      # s on one goal before it is blacklisted
    blacklist_radius: float = 0.5   # m
    empty_plans_to_finish: int = 3  # consecutive plans with no goal -> done

    # ---- frontier selection ---------------------------------------------
    selection: str = "utility"      # "nearest" | "utility"
    ig_radius: float = 2.0          # half-width of the information-gain window [m]
    lambda_cost: float = 0.30       # utility = IG * exp(-lambda * path_length)
    commit_margin: float = 0.0      # keep current goal unless best is (1+m) better

    # ---- motion control (pure pursuit) -----------------------------------
    v_max: float = 0.35
    w_max: float = 1.2
    lookahead: float = 0.35
    rotate_threshold: float = 1.0   # |heading error| above which robot turns on the spot
    goal_tolerance: float = 0.25
    stop_distance: float = 0.31    # obstacle closer than this ahead (within robot width) -> escape


# Named strategies compared in the experiments.
STRATEGIES = {
    # Yamauchi (1997): go to the closest reachable frontier.
    "nearest": dict(selection="nearest", commit_margin=0.0),
    # Gonzalez-Banos & Latombe (2002) style: information gain discounted by cost.
    "utility": dict(selection="utility", commit_margin=0.0),
    # Ours: utility + goal commitment (hysteresis) to stop goal oscillation.
    "utility_commit": dict(selection="utility", commit_margin=0.5),
}


def make_config(strategy: str = "utility_commit", **overrides) -> ExplorerConfig:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {list(STRATEGIES)}")
    cfg = replace(ExplorerConfig(), **STRATEGIES[strategy])
    return replace(cfg, **overrides) if overrides else cfg
