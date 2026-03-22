"""RRT* path planning in 3D with obstacle awareness."""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import euclidean_distance, setup_logger


class Node:
    """A node in the RRT* tree.

    Attributes:
        position: 3D position [x, y, z].
        parent: Parent node or None for root.
        cost: Cost from root to this node.
    """

    def __init__(self, position: np.ndarray, parent: Optional["Node"] = None,
                 cost: float = 0.0):
        self.position = position
        self.parent = parent
        self.cost = cost


class RRTStarPlanner:
    """RRT* path planner for 3D environments with obstacles.

    Generates collision-free paths from start to goal positions
    while optimizing path length through rewiring.

    Attributes:
        config: Configuration dictionary with planning parameters.
    """

    def __init__(self, config: Dict):
        """Initialize the RRT* planner.

        Args:
            config: Configuration dict with keys:
                - rrt_max_iterations
                - rrt_step_size
                - rrt_goal_sample_rate
                - rrt_search_radius
                - search_area
                - flight_altitude
                - safe_distance_obstacle
        """
        self.config = config
        self.max_iterations = config.get("rrt_max_iterations", 1000)
        self.step_size = config.get("rrt_step_size", 2.0)
        self.goal_sample_rate = config.get("rrt_goal_sample_rate", 0.1)
        self.search_radius = config.get("rrt_search_radius", 5.0)
        self.safe_distance = config.get("safe_distance_obstacle", 3.0)
        self.logger = setup_logger("RRTStarPlanner")

    def plan(self, start: np.ndarray, goal: np.ndarray,
             obstacles: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """Plan a path from start to goal avoiding obstacles.

        Args:
            start: Start position [x, y, z].
            goal: Goal position [x, y, z].
            obstacles: Obstacle positions as (N, 3) array, or None.

        Returns:
            List of waypoints from start to goal. Empty if no path found.
        """
        if obstacles is None:
            obstacles = np.empty((0, 3))

        root = Node(start, cost=0.0)
        tree = [root]

        for i in range(self.max_iterations):
            # Sample a random point or the goal
            sample = self._sample(goal)

            # Find nearest node in tree
            nearest = self._nearest(tree, sample)

            # Steer toward the sample
            new_pos = self._steer(nearest.position, sample)

            # Check collision
            if self._is_collision(nearest.position, new_pos, obstacles):
                continue

            # Find nearby nodes for rewiring
            near_nodes = self._near(tree, new_pos)

            # Choose best parent
            best_parent = nearest
            best_cost = nearest.cost + euclidean_distance(nearest.position, new_pos)

            for node in near_nodes:
                cost = node.cost + euclidean_distance(node.position, new_pos)
                if cost < best_cost and not self._is_collision(
                    node.position, new_pos, obstacles
                ):
                    best_parent = node
                    best_cost = cost

            new_node = Node(new_pos, parent=best_parent, cost=best_cost)
            tree.append(new_node)

            # Rewire nearby nodes
            self._rewire(tree, new_node, near_nodes, obstacles)

            # Check if goal reached
            if euclidean_distance(new_pos, goal) < self.step_size:
                if not self._is_collision(new_pos, goal, obstacles):
                    goal_node = Node(
                        goal, parent=new_node,
                        cost=new_node.cost + euclidean_distance(new_pos, goal)
                    )
                    tree.append(goal_node)
                    path = self._extract_path(goal_node)
                    self.logger.info(
                        f"Path found with {len(path)} waypoints "
                        f"(cost: {goal_node.cost:.2f})"
                    )
                    return path

        self.logger.warning("RRT*: No path found within iteration limit.")
        return []

    def _sample(self, goal: np.ndarray) -> np.ndarray:
        """Sample a random point in the search space.

        Args:
            goal: Goal position (sampled with probability goal_sample_rate).

        Returns:
            Sampled 3D point.
        """
        if random.random() < self.goal_sample_rate:
            return goal.copy()

        area = self.config["search_area"]
        x = random.uniform(area["x_min"], area["x_max"])
        y = random.uniform(area["y_min"], area["y_max"])
        z = self.config["flight_altitude"]
        return np.array([x, y, z])

    def _nearest(self, tree: List[Node], point: np.ndarray) -> Node:
        """Find the nearest node in the tree to a given point.

        Args:
            tree: List of nodes.
            point: Query point.

        Returns:
            Nearest node.
        """
        distances = [euclidean_distance(n.position, point) for n in tree]
        return tree[int(np.argmin(distances))]

    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray) -> np.ndarray:
        """Steer from one position toward another, limited by step size.

        Args:
            from_pos: Starting position.
            to_pos: Target position.

        Returns:
            New position within step_size of from_pos toward to_pos.
        """
        direction = to_pos - from_pos
        dist = np.linalg.norm(direction)
        if dist <= self.step_size:
            return to_pos.copy()
        return from_pos + direction / dist * self.step_size

    def _near(self, tree: List[Node], point: np.ndarray) -> List[Node]:
        """Find all nodes within search_radius of a point.

        Args:
            tree: List of nodes.
            point: Query point.

        Returns:
            List of nearby nodes.
        """
        return [
            n for n in tree
            if euclidean_distance(n.position, point) < self.search_radius
        ]

    def _is_collision(self, from_pos: np.ndarray, to_pos: np.ndarray,
                      obstacles: np.ndarray) -> bool:
        """Check if a path segment collides with any obstacle.

        Args:
            from_pos: Segment start.
            to_pos: Segment end.
            obstacles: Obstacle positions (N, 3).

        Returns:
            True if collision detected.
        """
        if len(obstacles) == 0:
            return False

        # Check points along the segment
        direction = to_pos - from_pos
        dist = np.linalg.norm(direction)
        num_checks = max(int(dist / (self.safe_distance / 2)), 2)

        for t in np.linspace(0, 1, num_checks):
            point = from_pos + t * direction
            distances = np.linalg.norm(obstacles - point, axis=1)
            if np.any(distances < self.safe_distance):
                return True

        return False

    def _rewire(self, tree: List[Node], new_node: Node,
                near_nodes: List[Node], obstacles: np.ndarray):
        """Rewire nearby nodes through new_node if it reduces cost.

        Args:
            tree: The full tree.
            new_node: Newly added node.
            near_nodes: Nodes near the new node.
            obstacles: Obstacle positions.
        """
        for node in near_nodes:
            new_cost = new_node.cost + euclidean_distance(
                new_node.position, node.position
            )
            if new_cost < node.cost and not self._is_collision(
                new_node.position, node.position, obstacles
            ):
                node.parent = new_node
                node.cost = new_cost

    def _extract_path(self, goal_node: Node) -> List[np.ndarray]:
        """Extract the path from root to goal by traversing parents.

        Args:
            goal_node: The goal node.

        Returns:
            List of positions from start to goal.
        """
        path = []
        node = goal_node
        while node is not None:
            path.append(node.position.copy())
            node = node.parent
        path.reverse()
        return path
