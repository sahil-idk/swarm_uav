"""Utility functions for coordinate transforms, distance calculations, and logging."""

import logging
import math
import time
from typing import List, Tuple

import numpy as np


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger with console and file handlers.

    Args:
        name: Logger name.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute Euclidean distance between two points.

    Args:
        p1: First point as numpy array.
        p2: Second point as numpy array.

    Returns:
        Euclidean distance.
    """
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def ned_to_enu(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert NED (North-East-Down) to ENU (East-North-Up) coordinates.

    Args:
        x: North component.
        y: East component.
        z: Down component.

    Returns:
        Tuple of (east, north, up).
    """
    return y, x, -z


def enu_to_ned(e: float, n: float, u: float) -> Tuple[float, float, float]:
    """Convert ENU (East-North-Up) to NED (North-East-Down) coordinates.

    Args:
        e: East component.
        n: North component.
        u: Up component.

    Returns:
        Tuple of (north, east, down).
    """
    return n, e, -u


def quaternion_to_euler(w: float, x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert quaternion to Euler angles (roll, pitch, yaw).

    Args:
        w, x, y, z: Quaternion components.

    Returns:
        Tuple of (roll, pitch, yaw) in radians.
    """
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def clamp_velocity(velocity: np.ndarray, max_speed: float) -> np.ndarray:
    """Clamp a velocity vector to a maximum magnitude.

    Args:
        velocity: Velocity vector.
        max_speed: Maximum allowed speed.

    Returns:
        Clamped velocity vector.
    """
    speed = np.linalg.norm(velocity)
    if speed > max_speed:
        return velocity * (max_speed / speed)
    return velocity


def normalize_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi].

    Args:
        angle: Angle in radians.

    Returns:
        Normalized angle.
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


class Timer:
    """Simple timer for profiling code sections."""

    def __init__(self):
        self._start_time = None

    def start(self):
        """Start the timer."""
        self._start_time = time.perf_counter()

    def elapsed(self) -> float:
        """Return elapsed time in seconds since start."""
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time
