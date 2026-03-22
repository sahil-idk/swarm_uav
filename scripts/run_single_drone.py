"""Test script: one drone navigates through a forest environment."""

import sys
import time

import airsim
import numpy as np
import yaml

sys.path.insert(0, "..")

from src.drone_client import DroneClient
from src.collision_avoidance import APFCollisionAvoidance
from src.sensor_processor import SensorProcessor
from src.target_detector import TargetDetector
from src.utils import euclidean_distance, setup_logger


def main():
    """Run a single drone through a forest navigation test."""
    logger = setup_logger("SingleDroneTest")

    # Load config
    with open("config/swarm_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()

    drone = DroneClient(client, "Drone0")
    collision_avoidance = APFCollisionAvoidance(config)
    sensor_processor = SensorProcessor(config)
    target_detector = TargetDetector(config)

    # Setup
    drone.enable_control()
    drone.takeoff()

    altitude = config["flight_altitude"]
    speed = config["cruise_speed"]

    # Move to flight altitude
    pos = drone.get_position()
    drone.move_to_position(pos[0], pos[1], altitude, speed)
    logger.info(f"At flight altitude: {altitude}m")

    # Define waypoints for a simple forest traversal
    waypoints = [
        np.array([0, 0, altitude]),
        np.array([20, 0, altitude]),
        np.array([20, 20, altitude]),
        np.array([0, 20, altitude]),
        np.array([0, 0, altitude]),
    ]

    trajectory = []

    try:
        for i, wp in enumerate(waypoints):
            logger.info(f"Navigating to waypoint {i}: {wp}")

            while euclidean_distance(drone.get_position(), wp) > 2.0:
                current_pos = drone.get_position()
                trajectory.append(current_pos.copy())

                # Get obstacle data
                lidar_data = drone.get_lidar_data()
                obstacles = sensor_processor.get_obstacle_positions(lidar_data)

                # Compute velocity with collision avoidance
                velocity = collision_avoidance.compute_velocity(
                    current_pos=current_pos,
                    target_pos=wp,
                    obstacles=obstacles
                )

                drone.move_by_velocity(velocity[0], velocity[1], velocity[2])

                # Check for targets
                image = drone.get_rgb_image()
                detections = target_detector.detect(image)
                if detections:
                    logger.info(f"Target detected at {current_pos}!")

                time.sleep(0.1)

            logger.info(f"Reached waypoint {i}")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        drone.land()
        drone.disable_control()
        logger.info("Single drone test complete.")

        # Save trajectory
        np.save("results/single_drone_trajectory.npy",
                np.array([t.tolist() for t in trajectory]))
        logger.info(f"Saved trajectory ({len(trajectory)} points)")


if __name__ == "__main__":
    main()
