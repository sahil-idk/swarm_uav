"""Search area partitioning and sweep pattern generation for multi-drone coverage."""

from typing import Dict, List

import numpy as np

from .utils import setup_logger


class AreaCoveragePlanner:
    """Partitions a search area among drones and generates sweep patterns.

    Divides the search region into vertical strips (one per drone) and
    generates lawnmower sweep waypoints within each strip.

    Attributes:
        config: Configuration dictionary.
        sweep_spacing: Distance between sweep lines.
        altitude: Flight altitude (NED).
        overlap: Overlap ratio between adjacent strips.
    """

    def __init__(self, config: Dict):
        """Initialize the area coverage planner.

        Args:
            config: Configuration dict with area coverage parameters.
        """
        self.config = config
        self.sweep_spacing = config.get("sweep_line_spacing", 8.0)
        self.altitude = config.get("flight_altitude", -8)
        self.overlap = config.get("overlap_ratio", 0.2)
        self.logger = setup_logger("AreaCoveragePlanner")

    def partition_and_assign(self, drone_names: List[str],
                             search_area: Dict) -> Dict[str, List[np.ndarray]]:
        """Partition the search area and generate waypoints for each drone.

        Splits the area into vertical strips along the Y axis,
        then generates a lawnmower sweep pattern for each strip.

        Args:
            drone_names: List of drone identifiers.
            search_area: Dict with x_min, x_max, y_min, y_max.

        Returns:
            Dict mapping drone name to list of waypoint arrays.
        """
        n = len(drone_names)
        x_min = search_area["x_min"]
        x_max = search_area["x_max"]
        y_min = search_area["y_min"]
        y_max = search_area["y_max"]

        # Divide along Y axis into strips
        strip_width = (y_max - y_min) / n
        overlap_width = strip_width * self.overlap

        assignments = {}
        for i, name in enumerate(drone_names):
            strip_y_min = y_min + i * strip_width - (overlap_width if i > 0 else 0)
            strip_y_max = y_min + (i + 1) * strip_width + (
                overlap_width if i < n - 1 else 0
            )

            waypoints = self._generate_sweep_pattern(
                x_min, x_max, strip_y_min, strip_y_max
            )
            assignments[name] = waypoints
            self.logger.info(
                f"{name}: Assigned strip Y=[{strip_y_min:.1f}, {strip_y_max:.1f}] "
                f"with {len(waypoints)} waypoints"
            )

        return assignments

    def _generate_sweep_pattern(self, x_min: float, x_max: float,
                                y_min: float, y_max: float
                                ) -> List[np.ndarray]:
        """Generate a lawnmower sweep pattern within a rectangular region.

        Creates back-and-forth sweep lines along the X axis,
        spaced by sweep_spacing along the Y axis.

        Args:
            x_min: Minimum X boundary.
            x_max: Maximum X boundary.
            y_min: Minimum Y boundary.
            y_max: Maximum Y boundary.

        Returns:
            Ordered list of waypoints for the sweep.
        """
        waypoints = []
        y_positions = np.arange(y_min, y_max + self.sweep_spacing, self.sweep_spacing)

        for i, y in enumerate(y_positions):
            if y > y_max:
                y = y_max

            if i % 2 == 0:
                # Forward sweep
                waypoints.append(np.array([x_min, y, self.altitude]))
                waypoints.append(np.array([x_max, y, self.altitude]))
            else:
                # Reverse sweep
                waypoints.append(np.array([x_max, y, self.altitude]))
                waypoints.append(np.array([x_min, y, self.altitude]))

        return waypoints

    def compute_coverage_percentage(self, trajectories: Dict[str, List[np.ndarray]],
                                    search_area: Dict,
                                    resolution: float = 2.0) -> float:
        """Estimate the percentage of the search area covered.

        Discretizes the search area into a grid and marks cells
        visited by any drone trajectory.

        Args:
            trajectories: Dict mapping drone name to list of positions.
            search_area: Search area boundaries.
            resolution: Grid cell size in meters.

        Returns:
            Coverage percentage (0.0 to 100.0).
        """
        x_min = search_area["x_min"]
        x_max = search_area["x_max"]
        y_min = search_area["y_min"]
        y_max = search_area["y_max"]

        nx = int((x_max - x_min) / resolution)
        ny = int((y_max - y_min) / resolution)
        grid = np.zeros((nx, ny), dtype=bool)

        sensor_range = self.config.get("sensor_range", 15.0)

        for name, positions in trajectories.items():
            for pos in positions:
                cx = int((pos[0] - x_min) / resolution)
                cy = int((pos[1] - y_min) / resolution)
                r = int(sensor_range / resolution)

                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        gx, gy = cx + dx, cy + dy
                        if 0 <= gx < nx and 0 <= gy < ny:
                            if dx * dx + dy * dy <= r * r:
                                grid[gx, gy] = True

        total_cells = nx * ny
        covered_cells = np.sum(grid)
        return float(covered_cells / total_cells * 100) if total_cells > 0 else 0.0
