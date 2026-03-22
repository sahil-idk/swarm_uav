"""Main script: full swarm search mission in a forest environment."""

import json
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, "..")

from src.swarm_manager import SwarmManager
from src.utils import setup_logger, Timer


def main():
    """Run the full multi-drone swarm search mission."""
    logger = setup_logger("SwarmMission")

    # Load config
    with open("config/swarm_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    mission_timeout = config.get("mission_timeout", 300)
    update_rate = config.get("update_rate", 10)
    dt = 1.0 / update_rate

    # Initialize swarm
    logger.info("Initializing swarm...")
    swarm = SwarmManager("config/swarm_config.yaml")

    # Setup: arm and takeoff
    logger.info("Setting up drones...")
    swarm.setup()

    # Assign search areas
    logger.info("Assigning search areas...")
    swarm.assign_search_areas()

    # Main mission loop
    timer = Timer()
    timer.start()
    iteration = 0

    logger.info("Starting search mission...")

    try:
        while not swarm.is_mission_complete():
            if timer.elapsed() > mission_timeout:
                logger.warning("Mission timeout reached!")
                break

            swarm.update()
            iteration += 1

            # Periodic status logging
            if iteration % (update_rate * config.get("log_interval", 1)) == 0:
                elapsed = timer.elapsed()
                for name, state in swarm.states.items():
                    logger.info(
                        f"[{elapsed:.1f}s] {name}: status={state.status}, "
                        f"pos=({state.position[0]:.1f}, {state.position[1]:.1f}), "
                        f"wp={state.current_wp_idx}/{len(state.waypoints)}, "
                        f"targets={len(state.targets_found)}"
                    )

            time.sleep(dt)

    except KeyboardInterrupt:
        logger.info("Mission interrupted by user")

    finally:
        # Shutdown
        elapsed = timer.elapsed()
        logger.info(f"Mission duration: {elapsed:.1f}s")
        swarm.shutdown()

        # Save results
        results = swarm.get_results()
        results["mission_duration"] = elapsed
        results["total_iterations"] = iteration

        output_path = "results/swarm_mission_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
