"""Comparison script: runs missions with and without collision avoidance."""

import json
import sys
import time
from copy import deepcopy

import numpy as np
import yaml

sys.path.insert(0, "..")

from src.swarm_manager import SwarmManager
from src.area_coverage import AreaCoveragePlanner
from src.utils import setup_logger, Timer


def run_mission(config_path: str, enable_collision_avoidance: bool,
                label: str) -> dict:
    """Run a single swarm mission and return results.

    Args:
        config_path: Path to swarm config YAML.
        enable_collision_avoidance: Whether to enable APF collision avoidance.
        label: Label for this run (e.g., 'with_ca' or 'without_ca').

    Returns:
        Dict of mission results.
    """
    logger = setup_logger(f"Comparison-{label}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    mission_timeout = config.get("mission_timeout", 300)
    update_rate = config.get("update_rate", 10)
    dt = 1.0 / update_rate

    swarm = SwarmManager(config_path)

    # Disable collision avoidance if requested
    if not enable_collision_avoidance:
        swarm.collision_avoidance.repulsive_gain = 0.0
        logger.info("Collision avoidance DISABLED")
    else:
        logger.info("Collision avoidance ENABLED")

    swarm.setup()
    swarm.assign_search_areas()

    timer = Timer()
    timer.start()
    iteration = 0
    min_inter_drone_distances = []

    try:
        while not swarm.is_mission_complete():
            if timer.elapsed() > mission_timeout:
                break

            swarm.update()
            iteration += 1

            # Track minimum inter-drone distance
            positions = list(swarm._get_all_positions().values())
            if len(positions) >= 2:
                min_dist = float("inf")
                for i in range(len(positions)):
                    for j in range(i + 1, len(positions)):
                        d = np.linalg.norm(positions[i] - positions[j])
                        min_dist = min(min_dist, d)
                min_inter_drone_distances.append(min_dist)

            time.sleep(dt)

    except KeyboardInterrupt:
        logger.info("Interrupted")

    finally:
        elapsed = timer.elapsed()
        swarm.shutdown()

    # Compute coverage
    trajectories = {
        name: state.trajectory for name, state in swarm.states.items()
    }
    coverage_planner = AreaCoveragePlanner(config)
    coverage = coverage_planner.compute_coverage_percentage(
        trajectories, config["search_area"]
    )

    results = swarm.get_results()
    results["label"] = label
    results["collision_avoidance"] = enable_collision_avoidance
    results["mission_duration"] = elapsed
    results["coverage_percentage"] = coverage
    results["min_inter_drone_distances"] = min_inter_drone_distances
    results["avg_min_distance"] = (
        float(np.mean(min_inter_drone_distances))
        if min_inter_drone_distances else 0.0
    )

    return results


def main():
    """Run comparison between collision avoidance enabled/disabled."""
    logger = setup_logger("ComparisonTest")
    config_path = "config/swarm_config.yaml"

    logger.info("=== Running WITH collision avoidance ===")
    results_with = run_mission(config_path, True, "with_ca")

    # Allow AirSim to reset
    logger.info("Waiting for reset...")
    time.sleep(5)

    logger.info("=== Running WITHOUT collision avoidance ===")
    results_without = run_mission(config_path, False, "without_ca")

    # Summary
    comparison = {
        "with_collision_avoidance": {
            "duration": results_with["mission_duration"],
            "coverage": results_with["coverage_percentage"],
            "avg_min_inter_drone_dist": results_with["avg_min_distance"],
        },
        "without_collision_avoidance": {
            "duration": results_without["mission_duration"],
            "coverage": results_without["coverage_percentage"],
            "avg_min_inter_drone_dist": results_without["avg_min_distance"],
        },
    }

    logger.info("=== COMPARISON RESULTS ===")
    for mode, data in comparison.items():
        logger.info(f"{mode}:")
        for key, val in data.items():
            logger.info(f"  {key}: {val:.2f}")

    output_path = "results/comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Comparison saved to {output_path}")


if __name__ == "__main__":
    main()
