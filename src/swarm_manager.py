"""Swarm manager that coordinates N drones, tracks states, and issues commands."""

import time
from typing import Dict, List, Optional, Tuple

import airsim
import numpy as np
import yaml

from .drone_client import DroneClient
from .path_planner import RRTStarPlanner
from .collision_avoidance import APFCollisionAvoidance
from .area_coverage import AreaCoveragePlanner
from .target_detector import TargetDetector
from .sensor_processor import SensorProcessor
from .utils import euclidean_distance, setup_logger


class DroneState:
    """Tracks the state of a single drone in the swarm.

    Attributes:
        name: Drone identifier.
        position: Current NED position.
        velocity: Current NED velocity.
        waypoints: List of waypoints to follow.
        current_wp_idx: Index of current waypoint.
        status: Current status string (idle, moving, searching, returning).
        targets_found: List of detected target positions.
    """

    def __init__(self, name: str):
        self.name = name
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.waypoints: List[np.ndarray] = []
        self.current_wp_idx = 0
        self.status = "idle"
        self.targets_found: List[np.ndarray] = []
        self.trajectory: List[np.ndarray] = []


class SwarmManager:
    """Coordinates multiple drones for a forest search mission.

    Manages drone lifecycle, assigns search areas, handles path planning
    with collision avoidance, and aggregates detection results.

    Attributes:
        config: Loaded swarm configuration dictionary.
        client: Shared AirSim client.
        drones: Dict mapping drone name to DroneClient.
        states: Dict mapping drone name to DroneState.
    """

    def __init__(self, config_path: str = "config/swarm_config.yaml"):
        """Initialize the swarm manager.

        Args:
            config_path: Path to the swarm configuration YAML file.
        """
        self.logger = setup_logger("SwarmManager")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.num_drones = self.config["num_drones"]
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.drones: Dict[str, DroneClient] = {}
        self.states: Dict[str, DroneState] = {}
        self.collision_avoidance = APFCollisionAvoidance(self.config)
        self.area_planner = AreaCoveragePlanner(self.config)
        self.target_detector = TargetDetector(self.config)
        self.sensor_processor = SensorProcessor(self.config)
        self.path_planner = RRTStarPlanner(self.config)

        self._initialize_drones()

    def _initialize_drones(self):
        """Create DroneClient and DroneState for each drone."""
        for i in range(self.num_drones):
            name = f"Drone{i}"
            self.drones[name] = DroneClient(self.client, name)
            self.states[name] = DroneState(name)
        self.logger.info(f"Initialized {self.num_drones} drones.")

    def setup(self):
        """Enable control and take off all drones."""
        for name, drone in self.drones.items():
            drone.enable_control()
            drone.takeoff()
            drone.move_to_position(
                *drone.get_position()[:2],
                self.config["flight_altitude"],
                self.config["cruise_speed"]
            )
            self.states[name].status = "ready"
        self.logger.info("All drones airborne and ready.")

    def assign_search_areas(self):
        """Partition the search area and assign waypoints to each drone."""
        drone_names = list(self.drones.keys())
        assignments = self.area_planner.partition_and_assign(
            drone_names, self.config["search_area"]
        )

        for name, waypoints in assignments.items():
            self.states[name].waypoints = waypoints
            self.states[name].current_wp_idx = 0
            self.states[name].status = "searching"
        self.logger.info("Search areas assigned to all drones.")

    def update(self):
        """Run one update cycle: sense, plan, act for all drones."""
        all_positions = self._get_all_positions()

        for name, drone in self.drones.items():
            state = self.states[name]
            state.position = drone.get_position()
            state.velocity = drone.get_velocity()
            state.trajectory.append(state.position.copy())

            # Check for target detection
            self._check_for_targets(name)

            # Navigate toward next waypoint with collision avoidance
            if state.status == "searching" and state.current_wp_idx < len(state.waypoints):
                target_wp = state.waypoints[state.current_wp_idx]

                # Get obstacle data from lidar
                obstacles = self.sensor_processor.get_obstacle_positions(
                    drone.get_lidar_data()
                )

                # Compute velocity with collision avoidance
                other_positions = {
                    n: p for n, p in all_positions.items() if n != name
                }
                velocity = self.collision_avoidance.compute_velocity(
                    current_pos=state.position,
                    target_pos=target_wp,
                    obstacles=obstacles,
                    other_drones=other_positions
                )

                drone.move_by_velocity(velocity[0], velocity[1], velocity[2])

                # Check if waypoint reached
                if euclidean_distance(state.position, target_wp) < 2.0:
                    state.current_wp_idx += 1
                    self.logger.debug(
                        f"{name}: Reached waypoint {state.current_wp_idx}"
                    )

            elif state.current_wp_idx >= len(state.waypoints):
                state.status = "complete"

    def _get_all_positions(self) -> Dict[str, np.ndarray]:
        """Get current positions of all drones.

        Returns:
            Dict mapping drone name to position array.
        """
        return {
            name: drone.get_position()
            for name, drone in self.drones.items()
        }

    def _check_for_targets(self, drone_name: str):
        """Run target detection on the drone's camera image.

        Args:
            drone_name: Name of the drone to check.
        """
        drone = self.drones[drone_name]
        state = self.states[drone_name]

        image = drone.get_rgb_image()
        detections = self.target_detector.detect(image)

        if detections:
            for det in detections:
                self.logger.info(
                    f"{drone_name}: Target detected at pixel {det}"
                )
            state.targets_found.append(state.position.copy())

    def is_mission_complete(self) -> bool:
        """Check if all drones have completed their search patterns.

        Returns:
            True if all drones are done.
        """
        return all(
            s.status == "complete" for s in self.states.values()
        )

    def shutdown(self):
        """Land all drones and disable control."""
        for name, drone in self.drones.items():
            drone.land()
            drone.disable_control()
            self.states[name].status = "idle"
        self.logger.info("All drones landed. Mission complete.")

    def get_results(self) -> Dict:
        """Collect mission results from all drones.

        Returns:
            Dict with trajectories, targets found, and timing info.
        """
        results = {}
        for name, state in self.states.items():
            results[name] = {
                "trajectory": [pos.tolist() for pos in state.trajectory],
                "targets_found": [pos.tolist() for pos in state.targets_found],
                "waypoints_completed": state.current_wp_idx,
                "total_waypoints": len(state.waypoints),
            }
        return results
