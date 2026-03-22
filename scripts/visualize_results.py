"""Visualize mission results: 3D trajectories, coverage, and metrics."""

import json
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, "..")

from src.utils import setup_logger


def plot_3d_trajectories(results: Dict, title: str = "Drone Trajectories",
                         save_path: Optional[str] = None):
    """Plot 3D trajectories of all drones.

    Args:
        results: Mission results dict with trajectory data per drone.
        title: Plot title.
        save_path: Path to save the figure, or None to display.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))

    for (name, data), color in zip(results.items(), colors):
        if name.startswith("Drone") and "trajectory" in data:
            traj = np.array(data["trajectory"])
            if len(traj) == 0:
                continue

            # Convert NED to display (x=North, y=East, z=Up)
            ax.plot(traj[:, 0], traj[:, 1], -traj[:, 2],
                    label=name, color=color, linewidth=1.5)

            # Mark start and end
            ax.scatter(*traj[0, :2], -traj[0, 2], color=color,
                       marker="o", s=100, edgecolors="black")
            ax.scatter(*traj[-1, :2], -traj[-1, 2], color=color,
                       marker="s", s=100, edgecolors="black")

            # Mark detected targets
            if "targets_found" in data and data["targets_found"]:
                targets = np.array(data["targets_found"])
                ax.scatter(targets[:, 0], targets[:, 1], -targets[:, 2],
                           color="red", marker="*", s=200,
                           label=f"{name} targets")

    ax.set_xlabel("North (m)")
    ax.set_ylabel("East (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title(title)
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_2d_coverage(results: Dict, search_area: Dict,
                     save_path: Optional[str] = None):
    """Plot 2D top-down view of drone coverage.

    Args:
        results: Mission results dict.
        search_area: Search area boundaries.
        save_path: Path to save the figure, or None to display.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = plt.cm.Set1(np.linspace(0, 1, 10))

    for i, (name, data) in enumerate(results.items()):
        if name.startswith("Drone") and "trajectory" in data:
            traj = np.array(data["trajectory"])
            if len(traj) == 0:
                continue
            ax.plot(traj[:, 1], traj[:, 0], label=name,
                    color=colors[i], linewidth=1.0, alpha=0.7)

    # Draw search area boundary
    sa = search_area
    rect = plt.Rectangle(
        (sa["y_min"], sa["x_min"]),
        sa["y_max"] - sa["y_min"],
        sa["x_max"] - sa["x_min"],
        fill=False, edgecolor="black", linewidth=2, linestyle="--"
    )
    ax.add_patch(rect)

    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("Coverage Map (Top-Down)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_inter_drone_distances(comparison_data: Dict,
                               save_path: Optional[str] = None):
    """Plot inter-drone distance over time for comparison runs.

    Args:
        comparison_data: Full comparison results with distance data.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for label, data in comparison_data.items():
        if "min_inter_drone_distances" in data:
            distances = data["min_inter_drone_distances"]
            ax.plot(distances, label=label, alpha=0.8)

    ax.axhline(y=4.0, color="red", linestyle="--", label="Safe distance (4m)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Min Inter-Drone Distance (m)")
    ax.set_title("Inter-Drone Distance Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


def main():
    """Load and visualize mission results."""
    logger = setup_logger("Visualizer")

    # Try to load swarm mission results
    try:
        with open("results/swarm_mission_results.json", "r") as f:
            results = json.load(f)

        plot_3d_trajectories(
            results, "Swarm Mission Trajectories",
            "results/trajectories_3d.png"
        )
        plot_2d_coverage(
            results,
            {"x_min": -50, "x_max": 50, "y_min": -50, "y_max": 50},
            "results/coverage_2d.png"
        )
        logger.info("Swarm mission plots generated.")
    except FileNotFoundError:
        logger.warning("No swarm mission results found. Run run_swarm.py first.")

    # Try to load comparison results
    try:
        with open("results/comparison_results.json", "r") as f:
            comparison = json.load(f)

        logger.info("Comparison summary:")
        for mode, data in comparison.items():
            logger.info(f"  {mode}: {data}")
    except FileNotFoundError:
        logger.warning("No comparison results found. Run run_comparison.py first.")


if __name__ == "__main__":
    main()
