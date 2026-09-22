"""
Fast pure-numpy training environment for a LOCAL navigation policy, built on the REAL maze walls
(same MazeGenerator as the mission, wall segments from MazeGenerator.wall_segments()).

What changed versus the earlier version:
  * The train/held-out split is by maze LAYOUT (MazeGenerator.layout_split), not by seed range.
    A 3x3 grid has ~100 distinct layouts, so two different seeds very often built the identical
    maze and the old "held-out" mazes were mostly training mazes.
  * Lidar and wall-distance are vectorised over all wall segments (much faster training).
  * Constants come from one place and match the deployed controller (see rl/rl_controller.py).
Not modelled: crates (they sit below the drone's lidar plane) and the drone's dynamics (kinematic unicycle).
"""

import numpy as np

from maze_env.maze_generator import MazeGenerator


class MazeNavEnv:
    def __init__(self, num_rays=16, max_range=3.5, agent_radius=0.18, goal_radius=0.35, max_steps=250,
                 grid_n_range=(3, 4), seed=None, split="train"):
        assert split in ("train", "val")
        self.num_rays, self.max_range = num_rays, max_range
        self.agent_radius, self.goal_radius, self.max_steps = agent_radius, goal_radius, max_steps
        self.grid_n_range, self.split = grid_n_range, split
        self.rng = np.random.default_rng(seed)
        self.ray_angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False)
        self.obs_dim, self.act_dim = num_rays + 3, 2

    def reset(self):
        while True:                                           # draw mazes until one lands in our split
            grid_n = int(self.rng.integers(self.grid_n_range[0], self.grid_n_range[1] + 1))
            maze = MazeGenerator(grid_n=grid_n, seed=int(self.rng.integers(0, 10_000_000)))
            if maze.layout_split() == self.split:
                break
        self.maze = maze
        self.segments = maze.wall_segments()
        rooms = maze.all_room_centers()
        a, b = self.rng.choice(len(rooms), size=2, replace=False)
        self.pos = np.array(rooms[a]) + self.rng.uniform(-0.3, 0.3, 2)
        self.goal = np.array(rooms[b])
        self.yaw = self.rng.uniform(-np.pi, np.pi)
        self.steps = 0
        self.prev_dist = np.linalg.norm(self.goal - self.pos)
        self.norm_span = maze.grid_n * maze.cell_pitch * 1.5
        return self._obs()

    def _lidar(self):
        wa = self.ray_angles + self.yaw
        d = np.stack([np.cos(wa), np.sin(wa)], axis=1)                      # (R,2)
        s = self.segments
        sx, sy = s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]                        # (S,)
        denom = d[:, 0:1] * sy[None, :] - d[:, 1:2] * sx[None, :]           # (R,S)
        ok = np.abs(denom) > 1e-9
        den = np.where(ok, denom, 1.0)
        ox, oy = s[:, 0] - self.pos[0], s[:, 1] - self.pos[1]
        t = (ox[None, :] * sy[None, :] - oy[None, :] * sx[None, :]) / den
        u = (ox[None, :] * d[:, 1:2] - oy[None, :] * d[:, 0:1]) / den
        valid = ok & (t >= 0) & (u >= 0) & (u <= 1) & (t <= self.max_range)
        return np.where(valid, t, self.max_range).min(axis=1)

    def _min_wall_dist(self):
        s = self.segments
        a, ab = s[:, :2], s[:, 2:] - s[:, :2]
        t = np.clip(((self.pos - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
        return float(np.linalg.norm(self.pos - (a + t[:, None] * ab), axis=1).min())

    def _obs(self):
        to_goal = self.goal - self.pos
        bearing = np.arctan2(to_goal[1], to_goal[0]) - self.yaw
        return np.concatenate([self._lidar() / self.max_range,
                               [np.clip(np.linalg.norm(to_goal) / self.norm_span, 0, 1), np.sin(bearing), np.cos(bearing)]
                               ]).astype(np.float32)

    def step(self, action):
        speed = float(np.clip(action[0], 0, 1)) * 1.2
        turn = float(np.clip(action[1], -1, 1)) * 2.0
        dt = 0.15
        self.yaw += turn * dt
        self.pos = self.pos + speed * np.array([np.cos(self.yaw), np.sin(self.yaw)]) * dt
        self.steps += 1
        dist = np.linalg.norm(self.goal - self.pos)
        progress = (self.prev_dist - dist) * 5.0
        self.prev_dist = dist
        wall = self._min_wall_dist()
        collided = wall < self.agent_radius
        reached = dist < self.goal_radius
        reward = progress + (-0.3 * max(0.0, 0.35 - wall) if not collided else 0.0) - 0.02
        done = False
        if collided:
            reward -= 5.0
            done = True
        elif reached:
            reward += 10.0
            done = True
        elif self.steps >= self.max_steps:
            done = True
        return self._obs(), reward, done, {"collided": collided, "reached_goal": reached}

    def line_of_sight_clear(self):
        """True if the straight line start->goal crosses no wall (used to split results)."""
        p, q = self.pos, self.goal
        d = q - p
        s = self.segments
        sx, sy = s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]
        den = d[0] * sy - d[1] * sx
        ok = np.abs(den) > 1e-9
        den = np.where(ok, den, 1.0)
        t = ((s[:, 0] - p[0]) * sy - (s[:, 1] - p[1]) * sx) / den
        u = ((s[:, 0] - p[0]) * d[1] - (s[:, 1] - p[1]) * d[0]) / den
        return not bool(np.any(ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)))
