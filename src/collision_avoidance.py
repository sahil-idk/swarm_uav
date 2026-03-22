"""Artificial Potential Field (APF) based collision avoidance for drones."""

from typing import Dict, List, Optional

import numpy as np

from .utils import clamp_velocity, euclidean_distance, setup_logger


class APFCollisionAvoidance:
    """Collision avoidance using Artificial Potential Fields.

    Computes velocity commands that attract the drone toward its goal
    while repelling it from obstacles and other drones.

    Attributes:
        config: Configuration dictionary with APF parameters.
        attractive_gain: Scaling factor for attractive force.
        repulsive_gain: Scaling factor for repulsive force.
        safe_distance_obstacle: Minimum distance to obstacles.
        safe_distance_drone: Minimum inter-drone distance.
        influence_distance: Range of obstacle influence.
    """

    def __init__(self, config: Dict):
        """Initialize APF collision avoidance.

        Args:
            config: Configuration dict with APF parameters.
        """
        self.config = config
        self.attractive_gain = config.get("apf_attractive_gain", 1.0)
        self.repulsive_gain = config.get("apf_repulsive_gain", 5.0)
        self.safe_distance_obstacle = config.get("safe_distance_obstacle", 3.0)
        self.safe_distance_drone = config.get("safe_distance_drone", 4.0)
        self.influence_distance = config.get("apf_influence_distance", 10.0)
        self.max_speed = config.get("cruise_speed", 3.0)
        self.logger = setup_logger("APFCollisionAvoidance")

    def compute_velocity(self, current_pos: np.ndarray, target_pos: np.ndarray,
                         obstacles: Optional[np.ndarray] = None,
                         other_drones: Optional[Dict[str, np.ndarray]] = None
                         ) -> np.ndarray:
        """Compute a collision-free velocity command using APF.

        Args:
            current_pos: Current drone position [x, y, z].
            target_pos: Target/goal position [x, y, z].
            obstacles: Obstacle positions as (N, 3) array, or None.
            other_drones: Dict of other drone positions {name: [x,y,z]}.

        Returns:
            Velocity command [vx, vy, vz] in NED frame.
        """
        # Attractive force toward goal
        f_attractive = self._attractive_force(current_pos, target_pos)

        # Repulsive force from obstacles
        f_repulsive_obs = np.zeros(3)
        if obstacles is not None and len(obstacles) > 0:
            f_repulsive_obs = self._repulsive_force_obstacles(
                current_pos, obstacles
            )

        # Repulsive force from other drones
        f_repulsive_drones = np.zeros(3)
        if other_drones:
            f_repulsive_drones = self._repulsive_force_drones(
                current_pos, other_drones
            )

        # Total force
        total_force = f_attractive + f_repulsive_obs + f_repulsive_drones

        # Convert force to velocity and clamp
        velocity = clamp_velocity(total_force, self.max_speed)
        return velocity

    def _attractive_force(self, current_pos: np.ndarray,
                          target_pos: np.ndarray) -> np.ndarray:
        """Compute attractive force toward the target.

        Uses a linear attractive potential that pulls the drone
        toward the goal position.

        Args:
            current_pos: Current position.
            target_pos: Goal position.

        Returns:
            Attractive force vector.
        """
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)

        if distance < 0.1:
            return np.zeros(3)

        return self.attractive_gain * direction / distance

    def _repulsive_force_obstacles(self, current_pos: np.ndarray,
                                   obstacles: np.ndarray) -> np.ndarray:
        """Compute repulsive force from obstacles.

        Only obstacles within the influence distance generate a force.
        Force magnitude increases as the drone approaches an obstacle.

        Args:
            current_pos: Current drone position.
            obstacles: Obstacle positions (N, 3).

        Returns:
            Total repulsive force vector from obstacles.
        """
        total_force = np.zeros(3)

        for obs in obstacles:
            diff = current_pos - obs
            distance = np.linalg.norm(diff)

            if distance < 0.01:
                distance = 0.01

            if distance < self.influence_distance:
                magnitude = self.repulsive_gain * (
                    1.0 / distance - 1.0 / self.influence_distance
                ) * (1.0 / distance ** 2)
                force = magnitude * diff / distance
                total_force += force

        return total_force

    def _repulsive_force_drones(self, current_pos: np.ndarray,
                                other_drones: Dict[str, np.ndarray]
                                ) -> np.ndarray:
        """Compute repulsive force from other drones.

        Args:
            current_pos: Current drone position.
            other_drones: Dict mapping drone name to position.

        Returns:
            Total repulsive force vector from other drones.
        """
        total_force = np.zeros(3)

        for name, pos in other_drones.items():
            diff = current_pos - pos
            distance = np.linalg.norm(diff)

            if distance < 0.01:
                distance = 0.01

            if distance < self.safe_distance_drone * 2:
                magnitude = self.repulsive_gain * 2.0 * (
                    1.0 / distance - 1.0 / (self.safe_distance_drone * 2)
                ) * (1.0 / distance ** 2)
                force = magnitude * diff / distance
                total_force += force

        return total_force

    def check_collision_risk(self, current_pos: np.ndarray,
                             obstacles: Optional[np.ndarray] = None,
                             other_drones: Optional[Dict[str, np.ndarray]] = None
                             ) -> bool:
        """Check if any collision risk exists.

        Args:
            current_pos: Current drone position.
            obstacles: Obstacle positions.
            other_drones: Other drone positions.

        Returns:
            True if there is a collision risk.
        """
        if obstacles is not None and len(obstacles) > 0:
            distances = np.linalg.norm(obstacles - current_pos, axis=1)
            if np.any(distances < self.safe_distance_obstacle):
                return True

        if other_drones:
            for name, pos in other_drones.items():
                if euclidean_distance(current_pos, pos) < self.safe_distance_drone:
                    return True

        return False
