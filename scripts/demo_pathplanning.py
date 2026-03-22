"""Standalone RRT* path planning demo in a randomly generated 2D forest.

No AirSim dependency — requires only numpy, matplotlib, scipy.
Generates 150 random trees, plans a path with RRT*, shows the raw and
smoothed result side by side, and prints planning statistics.
"""

import random
import time
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


# ---------------------------------------------------------------------------
# Forest generation
# ---------------------------------------------------------------------------

def generate_forest(
    num_trees: int = 150,
    area_size: float = 100.0,
    radius_range: Tuple[float, float] = (0.5, 2.5),
    start: np.ndarray = None,
    goal: np.ndarray = None,
    clearance: float = 4.0,
    seed: int = 42,
) -> List[Tuple[np.ndarray, float]]:
    """Generate random circular trees, keeping start/goal regions clear.

    Returns list of (center, radius) tuples.
    """
    rng = np.random.RandomState(seed)
    trees = []
    for _ in range(num_trees * 5):
        if len(trees) >= num_trees:
            break
        cx = rng.uniform(0, area_size)
        cy = rng.uniform(0, area_size)
        r = rng.uniform(*radius_range)
        center = np.array([cx, cy])

        # Keep start and goal areas clear
        if start is not None and np.linalg.norm(center - start) < clearance + r:
            continue
        if goal is not None and np.linalg.norm(center - goal) < clearance + r:
            continue

        # Avoid overlapping trees
        overlap = False
        for tc, tr in trees:
            if np.linalg.norm(center - tc) < r + tr + 0.3:
                overlap = True
                break
        if not overlap:
            trees.append((center, r))

    return trees


# ---------------------------------------------------------------------------
# 2‑D RRT* (standalone, no project imports)
# ---------------------------------------------------------------------------

class Node:
    __slots__ = ("pos", "parent", "cost", "children")

    def __init__(self, pos: np.ndarray, parent: "Node" = None, cost: float = 0.0):
        self.pos = pos
        self.parent = parent
        self.cost = cost
        self.children: List["Node"] = []


def _segment_hits_circle(
    p1: np.ndarray, p2: np.ndarray, center: np.ndarray, radius: float
) -> bool:
    """Return True if line segment p1→p2 intersects the circle."""
    d = p2 - p1
    f = p1 - center
    a = d @ d
    b = 2.0 * (f @ d)
    c = f @ f - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    disc_sqrt = np.sqrt(disc)
    t1 = (-b - disc_sqrt) / (2 * a)
    t2 = (-b + disc_sqrt) / (2 * a)
    # Intersection if any t in [0, 1]
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)


def _collision_free(
    p1: np.ndarray, p2: np.ndarray, trees: List[Tuple[np.ndarray, float]],
    margin: float = 0.8,
) -> bool:
    for center, r in trees:
        if _segment_hits_circle(p1, p2, center, r + margin):
            return False
    return True


def rrt_star(
    start: np.ndarray,
    goal: np.ndarray,
    trees: List[Tuple[np.ndarray, float]],
    area_size: float = 100.0,
    max_iter: int = 4000,
    step_size: float = 3.0,
    goal_sample_rate: float = 0.08,
    search_radius: float = 8.0,
    margin: float = 0.8,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[Node]]:
    """Run RRT* and return (path, full_tree).  path is [] if not found."""
    rng = np.random.RandomState(seed)
    root = Node(start.copy())
    all_nodes: List[Node] = [root]

    for _ in range(max_iter):
        # --- sample ---
        if rng.random() < goal_sample_rate:
            sample = goal.copy()
        else:
            sample = rng.uniform(0, area_size, size=2)

        # --- nearest ---
        dists = np.array([np.linalg.norm(n.pos - sample) for n in all_nodes])
        nearest = all_nodes[int(np.argmin(dists))]

        # --- steer ---
        diff = sample - nearest.pos
        d = np.linalg.norm(diff)
        new_pos = sample if d <= step_size else nearest.pos + diff / d * step_size

        # --- collision check ---
        if not _collision_free(nearest.pos, new_pos, trees, margin):
            continue

        # --- find near nodes ---
        near_nodes = [
            n for n in all_nodes
            if np.linalg.norm(n.pos - new_pos) < search_radius
        ]

        # --- choose best parent ---
        best_parent = nearest
        best_cost = nearest.cost + np.linalg.norm(nearest.pos - new_pos)
        for n in near_nodes:
            c = n.cost + np.linalg.norm(n.pos - new_pos)
            if c < best_cost and _collision_free(n.pos, new_pos, trees, margin):
                best_parent = n
                best_cost = c

        new_node = Node(new_pos, parent=best_parent, cost=best_cost)
        best_parent.children.append(new_node)
        all_nodes.append(new_node)

        # --- rewire ---
        for n in near_nodes:
            nc = new_node.cost + np.linalg.norm(new_node.pos - n.pos)
            if nc < n.cost and _collision_free(new_node.pos, n.pos, trees, margin):
                if n.parent is not None:
                    n.parent.children.remove(n)
                n.parent = new_node
                n.cost = nc
                new_node.children.append(n)

        # --- goal check ---
        if np.linalg.norm(new_pos - goal) < step_size:
            if _collision_free(new_pos, goal, trees, margin):
                goal_node = Node(
                    goal.copy(), parent=new_node,
                    cost=new_node.cost + np.linalg.norm(new_pos - goal),
                )
                new_node.children.append(goal_node)
                all_nodes.append(goal_node)
                path = _extract_path(goal_node)
                return path, all_nodes

    return [], all_nodes


def _extract_path(node: Node) -> List[np.ndarray]:
    path = []
    while node is not None:
        path.append(node.pos.copy())
        node = node.parent
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Path smoothing
# ---------------------------------------------------------------------------

def smooth_path(
    path: List[np.ndarray],
    trees: List[Tuple[np.ndarray, float]],
    margin: float = 0.8,
    num_points: int = 300,
) -> np.ndarray:
    """Shortcut + linear interpolation for a shorter, smoother path."""
    # --- greedy shortcutting: try to skip as many waypoints as possible ---
    wp = [p.copy() for p in path]
    i = 0
    while i < len(wp) - 2:
        j = len(wp) - 1
        while j > i + 1:
            if _collision_free(wp[i], wp[j], trees, margin):
                wp = wp[: i + 1] + wp[j:]
                break
            j -= 1
        i += 1

    # --- dense linear interpolation along the shortcut waypoints ---
    pts = np.array(wp)
    # Compute cumulative arc length
    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]

    if total_len < 1e-6:
        return pts

    t_fine = np.linspace(0, total_len, num_points)
    smooth = np.column_stack([
        np.interp(t_fine, cum_len, pts[:, 0]),
        np.interp(t_fine, cum_len, pts[:, 1]),
    ])
    return smooth


def path_length(pts) -> float:
    pts = np.asarray(pts)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_forest(ax, trees):
    for center, r in trees:
        circle = patches.Circle(center, r, fc="#b0b0b0", ec="#707070",
                                 linewidth=0.5, zorder=1)
        ax.add_patch(circle)


def draw_rrt_tree(ax, nodes: List[Node]):
    for node in nodes:
        if node.parent is not None:
            xs = [node.parent.pos[0], node.pos[0]]
            ys = [node.parent.pos[1], node.pos[1]]
            ax.plot(xs, ys, color="#a8d8ea", linewidth=0.35, zorder=2)


def main():
    area_size = 100.0
    start = np.array([5.0, 5.0])
    goal = np.array([95.0, 95.0])

    # --- generate forest ---
    print("Generating forest (150 trees)...")
    forest = generate_forest(
        num_trees=150, area_size=area_size,
        start=start, goal=goal, clearance=4.0, seed=42,
    )
    print(f"  Placed {len(forest)} trees")

    # --- plan ---
    print("Running RRT*...")
    t0 = time.perf_counter()
    raw_path, all_nodes = rrt_star(
        start, goal, forest, area_size=area_size,
        max_iter=6000, step_size=3.0, search_radius=5.0,
        goal_sample_rate=0.08, margin=0.8, seed=42,
    )
    planning_time = time.perf_counter() - t0

    if not raw_path:
        print("ERROR: No path found — try increasing max_iter or decreasing trees.")
        return

    raw_len = path_length(raw_path)
    smoothed = smooth_path(raw_path, forest, margin=0.8)
    smooth_len = path_length(smoothed)

    print(f"\n{'='*45}")
    print(f"  Planning time       : {planning_time:.3f} s")
    print(f"  Tree nodes explored : {len(all_nodes)}")
    print(f"  Raw path waypoints  : {len(raw_path)}")
    print(f"  Raw path length     : {raw_len:.2f} m")
    print(f"  Smoothed path length: {smooth_len:.2f} m")
    print(f"  Length reduction     : {(1 - smooth_len/raw_len)*100:.1f}%")
    print(f"{'='*45}\n")

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5), dpi=110)
    fig.suptitle("RRT* Path Planning in Dense Forest", fontsize=16, fontweight="bold")

    for ax, title, show_smooth in [
        (axes[0], "Raw RRT* Path", False),
        (axes[1], "Smoothed Path", True),
    ]:
        ax.set_xlim(-2, area_size + 2)
        ax.set_ylim(-2, area_size + 2)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True, alpha=0.15)

        # Area boundary
        ax.add_patch(patches.Rectangle(
            (0, 0), area_size, area_size,
            fill=False, ec="black", lw=1.5, ls="--", zorder=0,
        ))

        draw_forest(ax, forest)
        draw_rrt_tree(ax, all_nodes)

        if show_smooth:
            # Raw path in thin dashed orange for reference
            raw_arr = np.array(raw_path)
            ax.plot(raw_arr[:, 0], raw_arr[:, 1], color="orange",
                    lw=1.5, ls="--", zorder=4, label=f"Raw ({raw_len:.1f} m)")
            ax.plot(smoothed[:, 0], smoothed[:, 1], color="#d32f2f",
                    lw=3, zorder=5, label=f"Smoothed ({smooth_len:.1f} m)")
            ax.legend(loc="upper left", fontsize=9)
        else:
            raw_arr = np.array(raw_path)
            ax.plot(raw_arr[:, 0], raw_arr[:, 1], color="#d32f2f",
                    lw=2.5, zorder=5, label=f"Path ({raw_len:.1f} m)")
            ax.legend(loc="upper left", fontsize=9)

        # Start & goal
        ax.plot(*start, "o", color="#2e7d32", ms=14, zorder=6,
                markeredgecolor="black", markeredgewidth=1.5, label="Start")
        ax.plot(*goal, "o", color="#1565c0", ms=14, zorder=6,
                markeredgecolor="black", markeredgewidth=1.5, label="Goal")

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = "results/demo_rrt_pathplanning.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
