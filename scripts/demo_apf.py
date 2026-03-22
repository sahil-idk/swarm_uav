"""Standalone APF multi-drone collision avoidance demo with animation.

No AirSim dependency — requires only numpy, matplotlib.
Simulates 5 drones navigating through 50 obstacles using Artificial
Potential Fields, and saves the result as an animated GIF.
"""

import sys
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np


# ---------------------------------------------------------------------------
# APF engine (standalone 2‑D, mirrors src/collision_avoidance.py logic)
# ---------------------------------------------------------------------------

class APF2D:
    """2‑D Artificial Potential Field controller."""

    def __init__(
        self,
        attractive_gain: float = 1.5,
        repulsive_gain_obs: float = 6.0,
        repulsive_gain_drone: float = 8.0,
        influence_dist: float = 12.0,
        safe_dist_obs: float = 3.0,
        safe_dist_drone: float = 5.0,
        max_speed: float = 2.0,
    ):
        self.k_att = attractive_gain
        self.k_rep_obs = repulsive_gain_obs
        self.k_rep_drone = repulsive_gain_drone
        self.d_inf = influence_dist
        self.d_safe_obs = safe_dist_obs
        self.d_safe_drone = safe_dist_drone
        self.max_speed = max_speed

    def compute_velocity(
        self,
        pos: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
        obs_radii: np.ndarray,
        other_positions: List[np.ndarray],
    ) -> np.ndarray:
        f_att = self._attractive(pos, goal)
        f_rep = self._repulsive_obstacles(pos, obstacles, obs_radii)
        f_rep += self._repulsive_drones(pos, other_positions)

        f = f_att + f_rep

        # Local-minimum escape: if repulsive nearly cancels attractive,
        # add a tangential component to slide around the obstacle.
        rep_mag = np.linalg.norm(f_rep)
        if rep_mag > 0.3 and np.linalg.norm(f) < rep_mag * 0.4:
            tangent = np.array([-f_rep[1], f_rep[0]])  # 90° rotation
            f += tangent * 1.2

        speed = np.linalg.norm(f)
        if speed > self.max_speed:
            f = f / speed * self.max_speed
        return f

    def _attractive(self, pos, goal):
        diff = goal - pos
        d = np.linalg.norm(diff)
        if d < 0.1:
            return np.zeros(2)
        return self.k_att * diff / d

    def _repulsive_obstacles(self, pos, centers, radii):
        f = np.zeros(2)
        for c, r in zip(centers, radii):
            diff = pos - c
            d = np.linalg.norm(diff) - r  # distance to surface
            d = max(d, 0.05)
            if d < self.d_inf:
                mag = self.k_rep_obs * (1.0 / d - 1.0 / self.d_inf) / (d * d)
                f += mag * diff / np.linalg.norm(diff)
        return f

    def _repulsive_drones(self, pos, others):
        f = np.zeros(2)
        for op in others:
            diff = pos - op
            d = np.linalg.norm(diff)
            d = max(d, 0.05)
            if d < self.d_safe_drone * 3.0:
                mag = self.k_rep_drone * (
                    1.0 / d - 1.0 / (self.d_safe_drone * 3.0)
                ) / (d * d)
                f += mag * diff / d
        return f


# ---------------------------------------------------------------------------
# Scenario setup
# ---------------------------------------------------------------------------

def generate_obstacles(
    n: int, area: float, margin: float, drone_starts, drone_goals,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """Place circular obstacles avoiding drone start/goal zones."""
    centers, radii = [], []
    for _ in range(n * 10):
        if len(centers) >= n:
            break
        c = rng.uniform(margin, area - margin, size=2)
        r = rng.uniform(0.8, 2.5)
        # Keep clear of start/goal positions
        too_close = False
        for p in list(drone_starts) + list(drone_goals):
            if np.linalg.norm(c - p) < r + 4.0:
                too_close = True
                break
        if too_close:
            continue
        # Avoid overlap
        ok = True
        for oc, orr in zip(centers, radii):
            if np.linalg.norm(c - oc) < r + orr + 0.5:
                ok = False
                break
        if ok:
            centers.append(c)
            radii.append(r)
    return np.array(centers), np.array(radii)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(
    num_drones: int = 5,
    num_obstacles: int = 50,
    num_steps: int = 200,
    dt: float = 0.3,
    area_size: float = 60.0,
    seed: int = 123,
):
    rng = np.random.RandomState(seed)

    # Drones start on the left, goals on the right (interleaved to force
    # interesting crossings without total gridlock)
    y_slots = np.linspace(8, area_size - 8, num_drones)
    starts = np.column_stack([np.full(num_drones, 5.0), y_slots])
    # Shift goals: each drone's goal is offset by 2 positions (mod N)
    goal_y = np.roll(y_slots, 2)
    goals = np.column_stack([np.full(num_drones, area_size - 5.0), goal_y])

    obs_centers, obs_radii = generate_obstacles(
        num_obstacles, area_size, 3.0, starts, goals, rng,
    )

    apf = APF2D(
        attractive_gain=4.0,
        repulsive_gain_obs=3.0,
        repulsive_gain_drone=20.0,
        influence_dist=6.0,
        safe_dist_obs=2.0,
        safe_dist_drone=4.0,
        max_speed=3.0,
    )

    # State history:  positions[step][drone] = (x, y)
    positions = np.zeros((num_steps + 1, num_drones, 2))
    velocities = np.zeros((num_steps, num_drones, 2))
    positions[0] = starts.copy()

    min_inter_drone = []  # per step

    for t in range(num_steps):
        cur = positions[t]
        nxt = cur.copy()
        for i in range(num_drones):
            others = [cur[j] for j in range(num_drones) if j != i]
            vel = apf.compute_velocity(
                cur[i], goals[i], obs_centers, obs_radii, others,
            )
            velocities[t, i] = vel
            nxt[i] = cur[i] + vel * dt
            # Clamp to area
            nxt[i] = np.clip(nxt[i], 0.5, area_size - 0.5)

        positions[t + 1] = nxt

        # Min inter-drone distance
        min_d = float("inf")
        for i in range(num_drones):
            for j in range(i + 1, num_drones):
                d = np.linalg.norm(nxt[i] - nxt[j])
                min_d = min(min_d, d)
        min_inter_drone.append(min_d)

    return (
        positions, velocities, starts, goals,
        obs_centers, obs_radii, min_inter_drone, area_size,
    )


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

DRONE_COLORS = ["#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa"]


def animate_result(
    positions, velocities, starts, goals,
    obs_centers, obs_radii, min_inter_drone, area_size,
    save_path: str = "results/demo_apf_swarm.gif",
    fps: int = 20,
):
    num_steps = len(velocities)
    num_drones = positions.shape[1]

    fig, (ax_main, ax_dist) = plt.subplots(
        1, 2, figsize=(16, 7.5), dpi=100,
        gridspec_kw={"width_ratios": [2, 1]},
    )
    fig.suptitle(
        "APF Multi-Drone Collision Avoidance",
        fontsize=15, fontweight="bold",
    )

    # --- main view setup ---
    ax_main.set_xlim(-1, area_size + 1)
    ax_main.set_ylim(-1, area_size + 1)
    ax_main.set_aspect("equal")
    ax_main.set_xlabel("X (m)")
    ax_main.set_ylabel("Y (m)")
    ax_main.grid(True, alpha=0.15)

    # Obstacles (static)
    for c, r in zip(obs_centers, obs_radii):
        ax_main.add_patch(
            patches.Circle(c, r, fc="#c0c0c0", ec="#808080", lw=0.5, zorder=1)
        )

    # Goals (static X marks)
    for i in range(num_drones):
        ax_main.plot(
            goals[i, 0], goals[i, 1], "X",
            color=DRONE_COLORS[i % len(DRONE_COLORS)],
            ms=14, markeredgecolor="black", markeredgewidth=1.2, zorder=2,
        )

    # Trails (lines updated each frame)
    trail_lines = []
    for i in range(num_drones):
        (line,) = ax_main.plot([], [], color=DRONE_COLORS[i % len(DRONE_COLORS)],
                                lw=1.2, alpha=0.5, zorder=3)
        trail_lines.append(line)

    # Drone dots
    drone_dots = []
    for i in range(num_drones):
        (dot,) = ax_main.plot([], [], "o", color=DRONE_COLORS[i % len(DRONE_COLORS)],
                               ms=10, markeredgecolor="black", markeredgewidth=1.2,
                               zorder=5)
        drone_dots.append(dot)

    # Velocity arrows (quiver)
    quiver = ax_main.quiver(
        np.zeros(num_drones), np.zeros(num_drones),
        np.zeros(num_drones), np.zeros(num_drones),
        color=[DRONE_COLORS[i % len(DRONE_COLORS)] for i in range(num_drones)],
        scale=1.0, scale_units="xy", width=0.004, zorder=4,
    )

    step_text = ax_main.text(
        0.02, 0.97, "", transform=ax_main.transAxes,
        fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    # --- distance plot setup ---
    ax_dist.set_xlim(0, num_steps)
    ax_dist.set_ylim(0, max(min_inter_drone) * 1.3 if min_inter_drone else 20)
    ax_dist.set_xlabel("Time Step")
    ax_dist.set_ylabel("Min Inter-Drone Distance (m)")
    ax_dist.set_title("Safety Margin", fontsize=11)
    ax_dist.axhline(y=4.0, color="red", ls="--", lw=1.5, label="Safe dist (4 m)")
    ax_dist.legend(loc="upper right", fontsize=8)
    ax_dist.grid(True, alpha=0.2)

    (dist_line,) = ax_dist.plot([], [], color="#1565c0", lw=1.5)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    # --- animation function ---
    def update(frame):
        t = frame
        cur = positions[t]
        label_parts = [f"Step {t}/{num_steps}"]

        for i in range(num_drones):
            # Trail
            trail_lines[i].set_data(positions[: t + 1, i, 0],
                                     positions[: t + 1, i, 1])
            # Dot
            drone_dots[i].set_data([cur[i, 0]], [cur[i, 1]])

        # Velocity arrows
        if t < num_steps:
            vel = velocities[t]
            quiver.set_offsets(cur)
            quiver.set_UVC(vel[:, 0] * 2, vel[:, 1] * 2)

        # Distance plot
        if t > 0:
            dist_line.set_data(range(t), min_inter_drone[:t])

        # Inter-drone distance text
        if t > 0:
            label_parts.append(f"Min dist: {min_inter_drone[t-1]:.2f} m")
        step_text.set_text("\n".join(label_parts))

        artists = trail_lines + drone_dots + [quiver, dist_line, step_text]
        return artists

    anim = animation.FuncAnimation(
        fig, update, frames=num_steps + 1,
        interval=1000 // fps, blit=False, repeat=False,
    )

    # Save
    print(f"Saving animation to {save_path} ({num_steps + 1} frames) ...")
    try:
        anim.save(save_path, writer="pillow", fps=fps)
        print(f"  Saved: {save_path}")
    except Exception as e:
        print(f"  GIF save failed ({e}), trying mp4...")
        mp4_path = save_path.rsplit(".", 1)[0] + ".mp4"
        try:
            anim.save(mp4_path, writer="ffmpeg", fps=fps)
            print(f"  Saved: {mp4_path}")
        except Exception as e2:
            print(f"  mp4 save also failed ({e2}). Showing interactive window instead.")

    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("="*55)
    print("  APF Multi-Drone Collision Avoidance Demo")
    print("="*55)
    print()
    print("Scenario: 5 drones, 50 obstacles, 400 timesteps")
    print("Drones start on the left, goals on the right (crossing paths)")
    print()

    (
        positions, velocities, starts, goals,
        obs_centers, obs_radii, min_inter_drone, area_size,
    ) = simulate(
        num_drones=5, num_obstacles=50, num_steps=400,
        dt=0.2, area_size=80.0, seed=77,
    )

    num_steps = len(velocities)

    # Print summary statistics
    arrived = 0
    for i in range(positions.shape[1]):
        final = positions[-1, i]
        if np.linalg.norm(final - goals[i]) < 3.0:
            arrived += 1

    min_ever = min(min_inter_drone)
    avg_min = np.mean(min_inter_drone)

    print(f"  Drones reached goal : {arrived}/{positions.shape[1]}")
    print(f"  Minimum inter-drone distance ever : {min_ever:.2f} m")
    print(f"  Average min inter-drone distance   : {avg_min:.2f} m")
    print(f"  Safe distance threshold            : 4.00 m")
    violations = sum(1 for d in min_inter_drone if d < 4.0)
    print(f"  Safety violations                  : {violations}/{num_steps} steps")
    print()

    # Per-step printout of concerning distances
    print("Inter-drone distance log (closest encounters):")
    for t, d in enumerate(min_inter_drone):
        if d < 6.0:
            print(f"  Step {t:3d}: min distance = {d:.2f} m"
                  f"{'  *** VIOLATION ***' if d < 4.0 else ''}")
    print()

    # Also save a static final-state figure
    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
    ax.set_xlim(-1, area_size + 1)
    ax.set_ylim(-1, area_size + 1)
    ax.set_aspect("equal")
    ax.set_title("APF Drone Trajectories (Final State)", fontsize=14,
                  fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.15)

    for c, r in zip(obs_centers, obs_radii):
        ax.add_patch(
            patches.Circle(c, r, fc="#c0c0c0", ec="#808080", lw=0.5, zorder=1)
        )

    for i in range(positions.shape[1]):
        color = DRONE_COLORS[i % len(DRONE_COLORS)]
        traj = positions[:, i]
        ax.plot(traj[:, 0], traj[:, 1], color=color, lw=1.5, alpha=0.6, zorder=3)
        ax.plot(*starts[i], "o", color=color, ms=12,
                markeredgecolor="black", markeredgewidth=1.2, zorder=5)
        ax.plot(*goals[i], "X", color=color, ms=14,
                markeredgecolor="black", markeredgewidth=1.2, zorder=5)
        ax.plot(*positions[-1, i], "s", color=color, ms=10,
                markeredgecolor="black", markeredgewidth=1.5, zorder=6)

    static_path = "results/demo_apf_trajectories.png"
    fig.savefig(static_path, dpi=150, bbox_inches="tight")
    print(f"Static trajectory plot saved to {static_path}")

    # Animate
    animate_result(
        positions, velocities, starts, goals,
        obs_centers, obs_radii, min_inter_drone, area_size,
    )


if __name__ == "__main__":
    main()
