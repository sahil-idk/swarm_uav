"""Processes lidar point clouds and depth images for obstacle detection."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import setup_logger


class SensorProcessor:
    """Processes raw sensor data into obstacle information.

    Converts lidar point clouds and depth images into obstacle
    positions usable by the collision avoidance and path planning modules.

    Attributes:
        config: Configuration dictionary.
        sensor_range: Maximum lidar range.
        safe_distance: Minimum distance to consider as obstacle.
    """

    def __init__(self, config: Dict):
        """Initialize the sensor processor.

        Args:
            config: Configuration dict with sensor parameters.
        """
        self.config = config
        self.sensor_range = config.get("sensor_range", 15.0)
        self.safe_distance = config.get("safe_distance_obstacle", 3.0)
        self.logger = setup_logger("SensorProcessor")

    def get_obstacle_positions(self, point_cloud: Optional[np.ndarray]
                               ) -> np.ndarray:
        """Extract obstacle positions from a lidar point cloud.

        Filters out ground points and distant points, then clusters
        remaining points into obstacle positions.

        Args:
            point_cloud: Raw lidar points (N, 3) in drone body frame, or None.

        Returns:
            Obstacle positions as (M, 3) array. Empty array if no obstacles.
        """
        if point_cloud is None or len(point_cloud) == 0:
            return np.empty((0, 3))

        # Filter by range
        distances = np.linalg.norm(point_cloud[:, :2], axis=1)
        in_range = distances < self.sensor_range
        filtered = point_cloud[in_range]

        if len(filtered) == 0:
            return np.empty((0, 3))

        # Filter out ground points (points far below the drone)
        non_ground = filtered[:, 2] < 2.0  # Keep points not too far below
        filtered = filtered[non_ground]

        if len(filtered) == 0:
            return np.empty((0, 3))

        # Simple grid-based clustering
        obstacles = self._cluster_points(filtered)
        return obstacles

    def _cluster_points(self, points: np.ndarray,
                        grid_size: float = 2.0) -> np.ndarray:
        """Cluster point cloud into obstacle centroids using a voxel grid.

        Args:
            points: Filtered point cloud (N, 3).
            grid_size: Voxel grid cell size in meters.

        Returns:
            Cluster centroids as (M, 3) array.
        """
        if len(points) == 0:
            return np.empty((0, 3))

        # Quantize to grid
        grid_indices = np.floor(points / grid_size).astype(int)
        unique_cells = {}

        for i, idx in enumerate(grid_indices):
            key = tuple(idx)
            if key not in unique_cells:
                unique_cells[key] = []
            unique_cells[key].append(points[i])

        # Compute centroids
        centroids = []
        for cell_points in unique_cells.values():
            centroid = np.mean(cell_points, axis=0)
            centroids.append(centroid)

        return np.array(centroids)

    def process_depth_image(self, depth_image: np.ndarray,
                            fov_deg: float = 90.0
                            ) -> np.ndarray:
        """Convert a depth image to 3D obstacle points in camera frame.

        Args:
            depth_image: Depth image (H, W) with distances in meters.
            fov_deg: Horizontal field of view in degrees.

        Returns:
            3D points (N, 3) in camera frame.
        """
        h, w = depth_image.shape
        fov_rad = np.radians(fov_deg)
        fx = w / (2.0 * np.tan(fov_rad / 2.0))
        fy = fx  # Assume square pixels

        cx, cy = w / 2.0, h / 2.0

        # Create pixel coordinate grids
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Filter valid depths
        valid = (depth_image > 0.1) & (depth_image < self.sensor_range)

        z = depth_image[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy

        points = np.stack([x, y, z], axis=1)
        return points

    def get_nearest_obstacle_distance(self, point_cloud: Optional[np.ndarray]
                                      ) -> float:
        """Get the distance to the nearest obstacle from lidar data.

        Args:
            point_cloud: Lidar point cloud (N, 3), or None.

        Returns:
            Distance to nearest obstacle, or inf if none detected.
        """
        if point_cloud is None or len(point_cloud) == 0:
            return float("inf")

        distances = np.linalg.norm(point_cloud, axis=1)
        return float(np.min(distances))
