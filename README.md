# Forest Swarm

Multi-drone swarm system for autonomous forest search and rescue using AirSim.

## Overview

This project implements a coordinated multi-drone system that autonomously searches a forest environment for targets (e.g., lost persons) using:

- **RRT\* path planning** for collision-free 3D navigation
- **Artificial Potential Field (APF)** collision avoidance for inter-drone and obstacle safety
- **Lawnmower sweep patterns** for systematic area coverage
- **Color-based target detection** from onboard cameras
- **Lidar point cloud processing** for real-time obstacle detection

## Project Structure

```
forest-swarm/
├── config/
│   ├── airsim_settings.json      # AirSim multi-drone configuration
│   └── swarm_config.yaml         # Tunable parameters
├── src/
│   ├── drone_client.py           # Single drone AirSim wrapper
│   ├── swarm_manager.py          # Multi-drone coordination
│   ├── path_planner.py           # RRT* 3D path planning
│   ├── collision_avoidance.py    # APF-based avoidance
│   ├── area_coverage.py          # Area partitioning & sweep generation
│   ├── target_detector.py        # Color-based target detection
│   ├── sensor_processor.py       # Lidar & depth image processing
│   └── utils.py                  # Coordinate transforms & utilities
├── scripts/
│   ├── run_single_drone.py       # Single drone test
│   ├── run_swarm.py              # Full swarm mission
│   ├── run_comparison.py         # With/without collision avoidance
│   └── visualize_results.py      # Plot trajectories & metrics
├── results/                      # Output data & plots
└── requirements.txt
```

## Setup

### Prerequisites

- [AirSim](https://microsoft.github.io/AirSim/) with an Unreal Engine forest environment
- Python 3.8+

### Installation

```bash
pip install -r requirements.txt
```

### AirSim Configuration

Copy `config/airsim_settings.json` to your AirSim settings directory:

- **Windows**: `%USERPROFILE%\Documents\AirSim\settings.json`
- **Linux**: `~/Documents/AirSim/settings.json`

## Usage

### Single Drone Test

```bash
python scripts/run_single_drone.py
```

### Full Swarm Mission

```bash
python scripts/run_swarm.py
```

### Comparison (With/Without Collision Avoidance)

```bash
python scripts/run_comparison.py
```

### Visualize Results

```bash
python scripts/visualize_results.py
```

## Configuration

Edit `config/swarm_config.yaml` to tune:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_drones` | 3 | Number of drones in the swarm |
| `flight_altitude` | -8 | Flight altitude in NED (negative = above ground) |
| `cruise_speed` | 3.0 | Cruise speed in m/s |
| `safe_distance_obstacle` | 3.0 | Minimum obstacle distance (m) |
| `safe_distance_drone` | 4.0 | Minimum inter-drone distance (m) |
| `apf_repulsive_gain` | 5.0 | APF repulsive force scaling |
| `apf_attractive_gain` | 1.0 | APF attractive force scaling |
| `sensor_range` | 15.0 | Lidar range (m) |
