"""Wrapper around AirSim MultirotorClient for controlling a single drone."""

import logging
from typing import Dict, List, Optional, Tuple

import airsim
import numpy as np

from .utils import setup_logger


class DroneClient:
    """Interface for controlling a single drone in AirSim.

    Wraps the AirSim MultirotorClient to provide higher-level
    drone control methods for takeoff, movement, and sensor access.

    Attributes:
        name: Unique drone identifier (e.g., 'Drone0').
        client: Shared AirSim MultirotorClient instance.
        logger: Logger for this drone.
    """

    def __init__(self, client: airsim.MultirotorClient, name: str):
        """Initialize the drone client.

        Args:
            client: Shared AirSim MultirotorClient.
            name: Vehicle name as defined in AirSim settings.
        """
        self.client = client
        self.name = name
        self.logger = setup_logger(f"DroneClient-{name}")

    def enable_control(self):
        """Enable API control and arm the drone."""
        self.client.enableApiControl(True, vehicle_name=self.name)
        self.client.armDisarm(True, vehicle_name=self.name)
        self.logger.info(f"{self.name}: API control enabled, armed.")

    def disable_control(self):
        """Disarm and disable API control."""
        self.client.armDisarm(False, vehicle_name=self.name)
        self.client.enableApiControl(False, vehicle_name=self.name)
        self.logger.info(f"{self.name}: Disarmed, API control disabled.")

    def takeoff(self, timeout: float = 10.0):
        """Command the drone to take off.

        Args:
            timeout: Maximum time to wait for takeoff completion.
        """
        self.logger.info(f"{self.name}: Taking off...")
        self.client.takeoffAsync(timeout_sec=timeout, vehicle_name=self.name).join()

    def land(self, timeout: float = 30.0):
        """Command the drone to land.

        Args:
            timeout: Maximum time to wait for landing.
        """
        self.logger.info(f"{self.name}: Landing...")
        self.client.landAsync(timeout_sec=timeout, vehicle_name=self.name).join()

    def move_to_position(self, x: float, y: float, z: float, speed: float,
                         timeout: float = 30.0):
        """Move drone to a target position in NED coordinates.

        Args:
            x: North position (meters).
            y: East position (meters).
            z: Down position (meters, negative = up).
            speed: Flight speed (m/s).
            timeout: Maximum time for the movement.
        """
        self.logger.debug(f"{self.name}: Moving to ({x:.1f}, {y:.1f}, {z:.1f})")
        self.client.moveToPositionAsync(
            x, y, z, speed,
            timeout_sec=timeout,
            vehicle_name=self.name
        ).join()

    def move_by_velocity(self, vx: float, vy: float, vz: float,
                         duration: float = 0.1):
        """Move drone with a velocity command in NED frame.

        Args:
            vx: North velocity (m/s).
            vy: East velocity (m/s).
            vz: Down velocity (m/s).
            duration: Duration to apply the velocity command.
        """
        self.client.moveByVelocityAsync(
            vx, vy, vz, duration,
            vehicle_name=self.name
        )

    def hover(self):
        """Command the drone to hover in place."""
        self.client.hoverAsync(vehicle_name=self.name).join()

    def get_position(self) -> np.ndarray:
        """Get the current drone position in NED coordinates.

        Returns:
            Position as numpy array [x, y, z].
        """
        state = self.client.getMultirotorState(vehicle_name=self.name)
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val])

    def get_velocity(self) -> np.ndarray:
        """Get the current drone velocity in NED frame.

        Returns:
            Velocity as numpy array [vx, vy, vz].
        """
        state = self.client.getMultirotorState(vehicle_name=self.name)
        vel = state.kinematics_estimated.linear_velocity
        return np.array([vel.x_val, vel.y_val, vel.z_val])

    def get_orientation(self) -> Tuple[float, float, float, float]:
        """Get the current drone orientation as a quaternion.

        Returns:
            Quaternion (w, x, y, z).
        """
        state = self.client.getMultirotorState(vehicle_name=self.name)
        q = state.kinematics_estimated.orientation
        return q.w_val, q.x_val, q.y_val, q.z_val

    def get_rgb_image(self, camera_name: str = "front_rgb") -> np.ndarray:
        """Capture an RGB image from the specified camera.

        Args:
            camera_name: Camera name as defined in AirSim settings.

        Returns:
            RGB image as numpy array (H, W, 3).
        """
        responses = self.client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
        ], vehicle_name=self.name)

        response = responses[0]
        img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        img = img.reshape(response.height, response.width, 3)
        return img

    def get_depth_image(self, camera_name: str = "front_depth") -> np.ndarray:
        """Capture a depth image from the specified camera.

        Args:
            camera_name: Camera name as defined in AirSim settings.

        Returns:
            Depth image as numpy float array (H, W).
        """
        responses = self.client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.DepthPlanar, True, False)
        ], vehicle_name=self.name)

        response = responses[0]
        depth = airsim.list_to_2d_float_array(response.image_data_float,
                                               response.width, response.height)
        return depth

    def get_lidar_data(self, lidar_name: str = "LidarSensor") -> Optional[np.ndarray]:
        """Get lidar point cloud data.

        Args:
            lidar_name: Lidar sensor name from AirSim settings.

        Returns:
            Point cloud as numpy array (N, 3) or None if no data.
        """
        lidar_data = self.client.getLidarData(
            lidar_name=lidar_name, vehicle_name=self.name
        )

        if len(lidar_data.point_cloud) < 3:
            return None

        points = np.array(lidar_data.point_cloud, dtype=np.float32)
        points = points.reshape(-1, 3)
        return points
