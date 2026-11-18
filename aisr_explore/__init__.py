"""AISR project: occupancy-grid mapping and frontier-based exploration.

The package is simulator-agnostic: the same FrontierExplorer is driven by the
Webots controller (controllers/frontier_explorer) and by the lightweight NumPy
simulator (aisr_explore.sim2d) used for batch experiments.
"""

from .config import ExplorerConfig, STRATEGIES, make_config
from .occupancy_grid import OccupancyGrid, UNKNOWN, FREE, OCC
from .explorer import FrontierExplorer

__all__ = [
    "ExplorerConfig", "STRATEGIES", "make_config",
    "OccupancyGrid", "UNKNOWN", "FREE", "OCC",
    "FrontierExplorer",
]
