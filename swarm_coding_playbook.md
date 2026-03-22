# UAV Swarm Navigation — Claude Code Prompt Playbook

## How to use this document

Feed these prompts **sequentially** to Claude Code or any agentic AI coding assistant.
Each prompt builds on the output of the previous one. Wait for each to complete and verify
it works before moving to the next. Prompts are grouped by phase matching your project timeline.

---

## PHASE 1: Project scaffolding + AirSim connectivity

### Prompt 1 — Project structure

```
Create a Python project called "forest-swarm" with the following structure:

forest-swarm/
├── config/
│   ├── airsim_settings.json      # AirSim multi-drone configuration
│   └── swarm_config.yaml         # Tunable parameters (speeds, distances, sensor ranges)
├── src/
│   ├── __init__.py
│   ├── drone_client.py           # Wrapper around AirSim MultirotorClient for single drone
│   ├── swarm_manager.py          # Coordinates N drones, tracks states, issues commands
│   ├── path_planner.py           # RRT* path planning in 3D with obstacle awareness
│   ├── collision_avoidance.py    # APF-based inter-drone + obstacle avoidance
│   ├── area_coverage.py          # Search area partitioning and sweep pattern generation
│   ├── target_detector.py        # Simple color-based target detection from camera images
│   ├── sensor_processor.py       # Processes lidar point clouds and depth images
│   └── utils.py                  # Coordinate transforms, distance calculations, logging
├── scripts/
│   ├── run_single_drone.py       # Test script: one drone navigates forest
│   ├── run_swarm.py              # Main script: full swarm search mission
│   ├── run_comparison.py         # Runs with/without collision avoidance for comparison
│   └── visualize_results.py      # Plots 3D trajectories and metrics
├── results/                      # Saved trajectory data, metrics, screenshots
├── requirements.txt
└── README.md

Create all files with proper docstrings, imports, and placeholder class/function signatures.
The requirements.txt should include: airsim, numpy, scipy, matplotlib, pyyaml, opencv-python.

For config/swarm_config.yaml, include these parameters with sensible defaults:
- num_drones: 3
- search_area: {x_min: -50, x_max: 50, y_min: -50, y_max: 50}
- flight_altitude: -8 (NED, so negative = above ground)
- cruise_speed: 3.0  # m/s
- sensor_range: 15.0  # lidar range in meters
- safe_distance_obstacle: 3.0
- safe_distance_drone: 4.0
- apf_repulsive_gain: 5.0
- apf_attractive_gain: 1.0
- target_color_hsv_lower: [0, 120, 70]  # red detection
- target_color_hsv_upper: [10, 255, 255]

For config/airsim_settings.json, create a valid AirSim settings file that spawns 3 drones
named Drone0, Drone1, Drone2 at positions (0,0,-2), (5,0,-2), (10,0,-2) respectively.
Each drone should have:
- VehicleType: SimpleFlight
- A front-facing camera (ImageType 0 for RGB, 1 for depth)
- A lidar sensor with Range 15, NumberOfChannels 16, PointsPerSecond 10000,
  VerticalFOVUpper 15, VerticalFOVLower -15
```

### Prompt 2 — Drone client wrapper

```
In src/drone_client.py, implement the DroneClient class that wraps AirSim's MultirotorClient
for controlling a single drone. This is the low-level interface each drone uses.

Requirements:
- __init__(self, client: airsim.MultirotorClient, drone_name: str, config: dict)
  Stores reference to the shared AirSim client, the drone's name, and config params.

- connect_and_arm(self) -> bool
  Enables API control and arms the drone. Returns True if successful.

- takeoff(self, altitude: float = None) -> None
  Takes off to the configured flight altitude (from config). Async with join.

- move_to_position(self, x, y, z, speed=None) -> None
  Moves to absolute NED position at cruise speed. Uses moveToPositionAsync.

- move_on_path(self, waypoints: list[tuple], speed=None) -> None
  Follows a list of (x,y,z) waypoints. Uses moveOnPathAsync with airsim.Vector3r.

- get_position(self) -> np.ndarray
  Returns current [x, y, z] as numpy array from getMultirotorState().

- get_velocity(self) -> np.ndarray
  Returns current [vx, vy, vz] as numpy array.

- get_lidar_data(self) -> np.ndarray
  Returns lidar point cloud as Nx3 numpy array. Handle the case where no points
  are returned (return empty array). Uses getLidarData(lidar_name, drone_name).

- get_depth_image(self) -> np.ndarray
  Returns depth image as 2D numpy array. Uses simGetImages with ImageType.DepthPerspective.

- get_rgb_image(self) -> np.ndarray
  Returns RGB image as HxWx3 numpy array (uint8). Uses simGetImages with ImageType.Scene.

- check_collision(self) -> bool
  Returns True if collision detected via simGetCollisionInfo.

- hover(self) -> None
  Commands drone to hold current position.

- land(self) -> None
  Lands the drone. Uses landAsync with join.

- disarm(self) -> None
  Disarms and releases API control.

Include proper error handling and logging. All async calls should use .join() for blocking
behavior by default, with an optional non_blocking parameter.
```

### Prompt 3 — Sensor processor

```
In src/sensor_processor.py, implement the SensorProcessor class that converts raw sensor
data into useful obstacle information.

Requirements:

class SensorProcessor:

- __init__(self, config: dict)
  Store sensor_range, safe_distance_obstacle from config.

- process_lidar(self, point_cloud: np.ndarray, drone_position: np.ndarray) -> dict
  Takes the raw Nx3 lidar point cloud (in drone body frame) and drone's world position.
  Returns a dict with:
    - "obstacles": list of obstacle positions in world frame (Nx3 array)
    - "nearest_obstacle_distance": float
    - "nearest_obstacle_direction": unit vector pointing from drone to nearest obstacle
    - "obstacle_density": float (points per cubic meter in sensor range, rough measure
      of how cluttered the area is — useful for speed adaptation)
    - "clear_directions": list of unit vectors where no obstacles were detected within
      sensor_range (useful for escape routes)

  To find clear directions: divide the horizontal plane into 12 sectors (30 degrees each).
  A sector is "clear" if it has fewer than 3 lidar points within sensor_range.

- process_depth_image(self, depth_image: np.ndarray) -> dict
  Takes the depth image (HxW float array, values in meters).
  Returns a dict with:
    - "min_depth": float (closest obstacle in image)
    - "min_depth_direction": "left", "center", or "right" (divide image into thirds)
    - "depth_histogram": np.ndarray of shape (3,) with average depth for left/center/right

- estimate_local_obstacle_map(self, point_cloud: np.ndarray, grid_resolution: float = 1.0) -> np.ndarray
  Creates a 2D occupancy grid around the drone from lidar data.
  Grid size: (2*sensor_range / grid_resolution) squared.
  Each cell is 1 if any lidar point falls in it, 0 otherwise.
  Returns the 2D binary grid. This is used by the path planner.

Include numpy vectorized operations wherever possible for performance.
```

---

## PHASE 2: Path planning + obstacle avoidance

### Prompt 4 — RRT* 3D path planner

```
In src/path_planner.py, implement a 3D RRT* path planner for navigating through the forest.

Requirements:

class Node:
    - position: np.ndarray (3D)
    - parent: Node or None
    - cost: float (distance from start)

class RRTStarPlanner:

- __init__(self, config: dict)
  Params from config: step_size=2.0, max_iterations=1000, goal_tolerance=2.0,
  search_radius=5.0 (for rewiring), bounds from search_area config.

- plan(self, start: np.ndarray, goal: np.ndarray, obstacle_map: np.ndarray,
       obstacle_positions: np.ndarray = None) -> list[np.ndarray]
  Main planning function. Takes start/goal as 3D positions, a 2D occupancy grid
  from sensor processor, and optionally raw obstacle positions for finer checking.
  Returns a list of 3D waypoints from start to goal, or empty list if no path found.

  Algorithm:
  1. Initialize tree with start node
  2. For each iteration:
     a. Sample random point in bounds (with 10% bias toward goal)
     b. Find nearest node in tree
     c. Steer from nearest toward random point by step_size
     d. Check if the path segment is collision-free (no obstacles in the way)
     e. Find all nodes within search_radius of new point
     f. Choose the parent that minimizes cost-to-come
     g. Add new node
     h. Rewire nearby nodes through new node if it reduces their cost
     i. If new node is within goal_tolerance of goal, record solution
  3. Return the lowest-cost path to goal, smoothed

- _is_collision_free(self, point_a: np.ndarray, point_b: np.ndarray,
                      obstacle_positions: np.ndarray, clearance: float = 3.0) -> bool
  Check if the line segment from A to B is at least `clearance` meters from all obstacles.
  Sample points along the segment at 0.5m intervals and check distance to all obstacles.

- _smooth_path(self, waypoints: list[np.ndarray], obstacle_positions: np.ndarray) -> list
  Remove unnecessary waypoints. For each triplet (A, B, C), if A-to-C is collision-free,
  remove B. Repeat until no more removable.

- _sample_random_point(self, goal: np.ndarray) -> np.ndarray
  Returns a random 3D point within bounds. 10% of the time, return the goal instead
  (goal biasing for faster convergence).

Also implement a simpler fallback planner:

class SimpleWaypointPlanner:
- plan(self, start, goal, obstacle_positions) -> list[np.ndarray]
  If RRT* is too slow for real-time, this generates a straight-line path from start to goal
  with intermediate waypoints every 5 meters. The collision avoidance module handles
  the actual obstacle dodging reactively. This is the "plan loosely, avoid reactively" approach.

Include timing: print how long planning takes. If RRT* takes >2 seconds, log a warning.
```

### Prompt 5 — Collision avoidance (APF)

```
In src/collision_avoidance.py, implement Artificial Potential Field based collision avoidance
for both static obstacles (trees) and dynamic obstacles (other drones).

This is the REACTIVE layer — it runs every control loop iteration and modifies the drone's
velocity command to avoid collisions in real-time.

Requirements:

class APFCollisionAvoidance:

- __init__(self, config: dict)
  Load: safe_distance_obstacle, safe_distance_drone, apf_repulsive_gain,
  apf_attractive_gain, max_avoidance_speed from config.

- compute_avoidance_velocity(self, drone_position: np.ndarray,
                              drone_velocity: np.ndarray,
                              goal_position: np.ndarray,
                              obstacle_positions: np.ndarray,
                              other_drone_positions: list[np.ndarray]) -> np.ndarray
  Main function. Returns a 3D velocity vector that the drone should follow.

  The velocity is the sum of:
  1. Attractive force toward goal:
     F_att = attractive_gain * (goal - position) / ||goal - position||
     Capped at cruise_speed magnitude.

  2. Repulsive force from obstacles:
     For each obstacle within sensor_range:
       distance = ||obstacle - position||
       if distance < safe_distance_obstacle:
         direction = (position - obstacle) / distance  # pointing AWAY from obstacle
         magnitude = repulsive_gain * (1/distance - 1/safe_distance_obstacle) * (1/distance^2)
         F_rep += direction * magnitude

  3. Repulsive force from other drones:
     Same formula as obstacles but using safe_distance_drone (larger buffer).
     Use 1.5x the repulsive_gain for drones (they're moving toward you too).

  4. Sum all forces, clip to max_avoidance_speed, return as velocity vector.

- compute_emergency_stop(self, drone_position: np.ndarray,
                          obstacle_positions: np.ndarray,
                          other_drone_positions: list) -> bool
  Returns True if ANY obstacle or drone is within 1.0 meter (hard safety limit).
  If True, the drone should immediately hover/stop.

- get_escape_velocity(self, drone_position: np.ndarray,
                       obstacle_positions: np.ndarray) -> np.ndarray
  When emergency stop is triggered and drone needs to back away.
  Returns velocity pointing directly away from the nearest obstacle at 1 m/s.

Include these Reynolds-inspired additions for smoother swarm behavior:

- compute_separation(self, drone_position, other_positions, desired_separation=4.0) -> np.ndarray
  Steer away from nearby drones (weighted by inverse distance).

- compute_alignment(self, drone_velocity, other_velocities) -> np.ndarray
  Steer toward average heading of nearby drones (for cohesive group movement).

The final compute_avoidance_velocity should blend all of these:
  result = 0.4 * attractive + 0.3 * obstacle_repulsive + 0.2 * drone_repulsive + 0.1 * separation

Normalize and scale to cruise_speed.
```

---

## PHASE 3: Swarm coordination

### Prompt 6 — Area coverage

```
In src/area_coverage.py, implement the search area partitioning and sweep pattern generation.

This module divides the total search area among N drones and generates a lawnmower/sweep
path for each drone within its assigned sector.

Requirements:

class AreaCoverage:

- __init__(self, config: dict)
  Load search_area bounds, num_drones, flight_altitude, sensor_range.
  The effective sweep width per pass = sensor_range * 0.6 (60% overlap for safety
  in dense forest where visibility is limited).

- partition_area(self, num_drones: int) -> list[dict]
  Divide the rectangular search area into N vertical strips (simplest approach).
  Each strip is a dict with keys: x_min, x_max, y_min, y_max, drone_id.

  For 3 drones over area x=[-50,50], y=[-50,50]:
    Drone0: x=[-50, -16.7], y=[-50, 50]
    Drone1: x=[-16.7, 16.7], y=[-50, 50]
    Drone2: x=[16.7, 50], y=[-50, 50]

  Return list of sector dicts.

- generate_sweep_path(self, sector: dict, start_position: np.ndarray,
                       altitude: float) -> list[np.ndarray]
  Generate a lawnmower pattern within the sector.
  The drone flies back and forth along the Y axis, stepping along X by sweep_width
  after each pass.

  Example for a sector x=[0,30], y=[0,50], sweep_width=9:
    Waypoints: (0,0,alt) -> (0,50,alt) -> (9,50,alt) -> (9,0,alt) -> (18,0,alt) ->
               (18,50,alt) -> (27,50,alt) -> (27,0,alt)

  Start from the corner nearest to start_position to minimize transit distance.
  Return list of 3D waypoints.

- estimate_coverage(self, trajectory: list[np.ndarray], sector: dict,
                     sensor_range: float) -> float
  Given a drone's actual flown trajectory (list of positions), estimate what percentage
  of the sector area was covered. Discretize the sector into 1m x 1m cells,
  mark a cell as covered if any trajectory point is within sensor_range of it.
  Return coverage as percentage (0-100).

- get_start_positions(self, num_drones: int) -> list[np.ndarray]
  Return starting positions for each drone. Place them at the edge of their
  assigned sector, spaced apart, all at flight_altitude.
```

### Prompt 7 — Swarm manager (the brain)

```
In src/swarm_manager.py, implement the SwarmManager that coordinates all drones.
This is the central orchestrator.

Requirements:

class DroneState(Enum):
    IDLE = "idle"
    TAKING_OFF = "taking_off"
    NAVIGATING = "navigating"
    SEARCHING = "searching"
    AVOIDING = "avoiding"
    TARGET_FOUND = "target_found"
    RETURNING = "returning"
    LANDED = "landed"

class SwarmManager:

- __init__(self, config_path: str = "config/swarm_config.yaml")
  Load config. Initialize AirSim client (one shared client for all drones).
  Create DroneClient instances for each drone. Initialize SensorProcessor,
  APFCollisionAvoidance, AreaCoverage, RRTStarPlanner, TargetDetector.
  Create data structures:
    - drone_states: dict mapping drone_name -> DroneState
    - drone_positions: dict mapping drone_name -> np.ndarray (updated each tick)
    - drone_paths: dict mapping drone_name -> list of waypoints to follow
    - drone_trajectories: dict mapping drone_name -> list of all positions visited
    - found_targets: list of (position, drone_name, timestamp)
    - mission_start_time: float
    - collision_count: int

- initialize(self) -> None
  Connect all drones, arm them, confirm API control is active.

- run_mission(self) -> dict
  Main mission loop. Returns a metrics dict when complete.
  1. Takeoff all drones (simultaneously using async, then join all)
  2. Partition search area, assign sectors, generate sweep paths
  3. Enter main control loop (runs at ~10 Hz):
     a. Update all drone positions
     b. For each drone:
        - Get lidar data, process obstacles
        - Get camera image, check for targets
        - Get positions of all OTHER drones
        - Compute next waypoint from assigned sweep path
        - Compute avoidance velocity (APF) given obstacles + other drones
        - If emergency stop needed: hover
        - Else: command drone toward modified waypoint
        - Log position to trajectory
        - Check for collisions, increment counter if any
     c. Check termination conditions:
        - All sweep paths completed, OR
        - Mission time exceeded max_mission_time (from config), OR
        - All targets found (if target count is known)
  4. Return all drones to start positions
  5. Land all drones
  6. Return metrics dict

- _update_drone_positions(self) -> None
  Query each drone's position and update drone_positions dict.

- _get_other_drone_positions(self, exclude_name: str) -> list[np.ndarray]
  Return positions of all drones except the named one.

- _command_drone_step(self, drone_name: str, goal_waypoint: np.ndarray,
                       obstacle_positions: np.ndarray,
                       other_positions: list[np.ndarray]) -> None
  Single control step for one drone:
  1. Compute avoidance velocity from APF
  2. If emergency: hover
  3. Else: moveToPositionAsync with the avoidance-modified velocity direction
     Use a short lookahead (1-2 seconds at cruise speed) so the drone doesn't
     commit to a long trajectory that might hit something.

- _check_waypoint_reached(self, drone_name: str, waypoint: np.ndarray,
                           tolerance: float = 2.0) -> bool
  Returns True if drone is within tolerance of the waypoint.

- _handle_target_found(self, drone_name: str, target_position: np.ndarray) -> None
  Log the find. Optionally: broadcast to other drones (for now, just log it).

- get_metrics(self) -> dict
  Return: total_time, per_drone_distance, per_drone_coverage, total_coverage,
  collision_count, targets_found, num_emergency_stops.

Use threading or asyncio if needed, but the simplest approach is a synchronous loop
at 10 Hz using time.sleep(0.1) between ticks. All AirSim commands within a tick
use non-blocking async calls, and we don't join until the next tick.

CRITICAL: Include extensive logging (Python logging module, INFO level) so we can
debug what each drone is doing at each step.
```

---

## PHASE 4: Target detection + scripts

### Prompt 8 — Target detector

```
In src/target_detector.py, implement simple color-based target detection.

In the forest environment, we place bright red objects as "survivors".
The drone's camera captures RGB images, and we detect red blobs.

Requirements:

class TargetDetector:

- __init__(self, config: dict)
  Load HSV thresholds for target color from config.
  min_blob_area = 500 pixels (to filter noise).

- detect(self, rgb_image: np.ndarray) -> list[dict]
  Takes an HxWx3 uint8 RGB image.
  1. Convert RGB to HSV (cv2.cvtColor)
  2. Create mask for target color range (cv2.inRange)
  3. Find contours in mask (cv2.findContours)
  4. Filter contours by area (> min_blob_area)
  5. For each valid contour:
     - Get bounding box (cv2.boundingRect)
     - Get centroid (cv2.moments)
     - Estimate relative direction: "left", "center", "right" based on centroid x
  6. Return list of dicts: [{bbox, centroid, area, direction}]

- estimate_target_distance(self, depth_image: np.ndarray, centroid: tuple) -> float
  Given the depth image and the pixel centroid of a detected target,
  return the estimated distance in meters. Sample a 5x5 patch around
  the centroid in the depth image and take the median value.

- annotate_image(self, rgb_image: np.ndarray, detections: list[dict]) -> np.ndarray
  Draw bounding boxes and labels on the image for visualization/recording.
  Returns the annotated image. Use cv2.rectangle and cv2.putText.
```

### Prompt 9 — Run scripts

```
Create three executable scripts:

1. scripts/run_single_drone.py
   - Connects to AirSim
   - Takes off one drone (Drone0)
   - Gets lidar data, prints obstacle count
   - Plans a path using RRT* from current position to a goal 30m away
   - Flies the path using move_on_path
   - Plots the 3D trajectory with matplotlib when done
   - Handles Ctrl+C gracefully (lands drone, disconnects)

2. scripts/run_swarm.py
   - Creates SwarmManager with default config
   - Calls run_mission()
   - Prints metrics summary
   - Saves trajectory data to results/trajectories.npz
   - Calls visualize_results automatically

3. scripts/run_comparison.py
   - Runs the mission TWICE:
     a. First with collision avoidance enabled (normal mode)
     b. Then with collision avoidance disabled (drones ignore each other)
   - For the no-avoidance run, the APF module returns zero avoidance velocity
   - Saves both sets of metrics
   - Prints a comparison table:
     | Metric               | With Avoidance | Without Avoidance |
     |----------------------|----------------|-------------------|
     | Collisions           | 0              | 7                 |
     | Coverage %           | 85%            | 62%               |
     | Total time           | 120s           | 95s               |
     | Targets found        | 3/3            | 1/3               |
   - Generates side-by-side trajectory plots

All scripts should:
- Use argparse for any configurable parameters
- Have proper __main__ guards
- Include try/finally blocks to ensure drones land on error
- Print progress to console with timestamps
```

### Prompt 10 — Visualization

```
In scripts/visualize_results.py, create publication-quality visualizations
for the demo and report.

Requirements:

def plot_3d_trajectories(trajectories: dict, obstacles: np.ndarray = None,
                         targets: list = None, save_path: str = None):
    """
    3D plot showing all drone trajectories in the forest.
    - Each drone's path in a different color (use a colorblind-friendly palette)
    - Obstacles as gray scatter points (if provided)
    - Targets as large red stars
    - Start positions as green circles, end positions as blue squares
    - Labels for each drone
    - Title, axis labels (X [m], Y [m], Z [m])
    - Use matplotlib's 3D projection
    - Save to file if save_path provided, also plt.show()
    """

def plot_2d_coverage_map(trajectories: dict, search_area: dict,
                          sensor_range: float, save_path: str = None):
    """
    Top-down 2D heatmap showing coverage.
    - Discretize search area into 1m grid
    - For each cell, compute how many times a drone was within sensor_range
    - Plot as heatmap (white=uncovered, blue=covered once, green=covered multiple times)
    - Overlay drone paths as colored lines
    - Overlay sector boundaries as dashed lines
    - Show coverage percentage as text annotation
    """

def plot_metrics_dashboard(metrics: dict, save_path: str = None):
    """
    A 2x2 subplot figure:
    - Top-left: Per-drone distance traveled (bar chart)
    - Top-right: Per-drone coverage percentage (bar chart)
    - Bottom-left: Inter-drone minimum distance over time (line plot, with
      safe_distance threshold as red dashed line)
    - Bottom-right: Summary text box with total time, collisions, targets found
    """

def plot_comparison(metrics_with: dict, metrics_without: dict, save_path: str = None):
    """
    Side-by-side comparison of with/without collision avoidance.
    Two columns of bar charts comparing key metrics.
    Make it clear that the avoidance system works.
    """

All plots should use:
- plt.style.use('seaborn-v0_8-whitegrid') or similar clean style
- Figure size (12, 8) for single plots, (16, 10) for dashboards
- Font size 12 for labels, 14 for titles
- Tight layout with proper spacing
- DPI 150 for saved files
```

---

## PHASE 5: Testing without AirSim (mock mode)

### Prompt 11 — Mock AirSim client (CRITICAL for development)

```
Create src/mock_airsim.py — a mock AirSim client that simulates drone physics
WITHOUT needing AirSim/Unreal Engine installed. This is CRITICAL for development
because AirSim setup takes days and your team can code and test algorithms immediately.

Requirements:

class MockMultirotorClient:
    """
    Simulates multiple drones in a 3D space with simple physics.
    Same API surface as airsim.MultirotorClient so code works with both.
    """

- __init__(self)
  Create an internal state dict for each drone: position, velocity, armed, api_enabled.
  Generate a random forest: 200 cylinder obstacles (trees) with random positions
  in the search area, radius 0.3-1.0m, stored as list of (x, y, radius).

- confirmConnection(self) -> True

- enableApiControl(self, enabled, vehicle_name="") -> None

- armDisarm(self, arm, vehicle_name="") -> None

- takeoffAsync(self, vehicle_name="") -> MockFuture
  Set drone altitude to flight_altitude over 2 seconds (simulated).

- moveToPositionAsync(self, x, y, z, speed, vehicle_name="") -> MockFuture
  Update drone position linearly toward target at given speed.
  The mock tracks time and interpolates position.

- moveOnPathAsync(self, waypoints, speed, vehicle_name="") -> MockFuture
  Same but follows waypoint list.

- getMultirotorState(self, vehicle_name="") -> MockState
  Returns mock state with current position, velocity.

- getLidarData(self, lidar_name="", vehicle_name="") -> MockLidarData
  Generate synthetic lidar data based on drone position and forest obstacles.
  For each tree within sensor_range, generate a cluster of 3D points
  approximating what a lidar would see on a cylinder.
  Add gaussian noise (sigma=0.1m) to make it realistic.

- simGetImages(self, requests, vehicle_name="") -> list
  Return synthetic images:
  - For depth: generate a depth image where pixels corresponding to nearby
    obstacles have small depth values
  - For RGB: generate a simple image, with red blobs if a target is nearby

- simGetCollisionInfo(self, vehicle_name="") -> MockCollisionInfo
  Check if drone position is within any tree cylinder. Return has_collided=True if so.

- listVehicles(self) -> list of vehicle names

class MockFuture:
    def join(self): pass  # instant in mock mode

class MockState, MockLidarData, MockCollisionInfo:
    Mirror the airsim data structures with the same attribute names.

Also add a factory function:

def get_client(use_mock: bool = False) -> client:
    if use_mock:
        return MockMultirotorClient()
    else:
        import airsim
        return airsim.MultirotorClient()

Update drone_client.py to accept either real or mock client via this factory.
Add --mock flag to all run scripts so the team can test without AirSim.

Place 3 red target objects at random positions in the mock forest.
```

---

## PHASE 6: Standalone demo (for presentation)

### Prompt 12 — Standalone algorithm demo (no AirSim needed)

```
Create scripts/demo_pathplanning.py — a standalone visual demo that does NOT need
AirSim. This is for the March 24 presentation and for quick algorithm testing.

Requirements:
- Generate a random 2D forest: 150 trees as circles with random positions and radii
  in a 100m x 100m area
- Place a start point and goal point on opposite sides
- Run the RRT* planner to find a path
- Visualize with matplotlib:
  - Trees as gray filled circles
  - RRT* tree branches as light blue lines
  - Final path as thick red line
  - Start as green dot, goal as blue dot
  - Title: "RRT* Path Planning in Dense Forest"
- Also show the path smoothing (before and after)
- Print: planning time, path length, number of tree nodes explored

Create a second demo: scripts/demo_apf.py
- Place 5 drones and 50 obstacles in a 2D area
- Each drone has a goal on the other side
- Simulate 200 timesteps of APF-based movement
- Animate with matplotlib.animation:
  - Drones as colored moving dots with velocity arrows
  - Obstacles as gray circles
  - Goal positions as colored X marks
  - Trails showing paths taken
  - Real-time inter-drone distance printout
- Save animation as .gif or .mp4

These demos should work on any machine with just numpy, matplotlib, scipy installed.
No AirSim dependency.
```

---

## Notes for using these prompts

### Order matters
Prompts 1-3 can be done in parallel by different team members.
Prompt 11 (mock client) should be done RIGHT AFTER Prompt 1 —
it unblocks ALL other development without waiting for AirSim setup.
Prompt 12 (demos) can be done immediately for the presentation.

### After each prompt, verify:
1. The code runs without import errors
2. Basic unit test: instantiate each class, call each method with dummy data
3. The mock mode works: `python scripts/run_single_drone.py --mock`

### Adapting prompts:
- If Claude Code asks clarifying questions, answer with specifics from the paper
- If a prompt generates code that doesn't work, paste the error back and say
  "Fix this error. The traceback is: [paste error]"
- If you want to change an algorithm, say "Replace RRT* with A* on a 3D grid
  because [reason]" — be specific about what to change and why

### Integration testing prompt (use after all modules are built):
```
Run scripts/run_swarm.py --mock and fix any errors.
The full pipeline should:
1. Initialize 3 mock drones
2. Take off
3. Partition the area
4. Each drone follows its sweep path with APF collision avoidance
5. Detect targets when nearby
6. Print metrics at the end
7. Generate trajectory plot saved to results/

Fix all errors until this runs end-to-end without crashing.
Then run it 3 times and print averaged metrics.
```
