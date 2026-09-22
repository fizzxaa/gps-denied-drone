"""
A fast, pure-numpy 2D training environment for a LOCAL navigation policy:
given lidar-style range readings and a relative goal direction, output a
forward speed + turn rate that reaches the goal without hitting obstacles.

This is deliberately decoupled from the full PyBullet sim -- training
against real physics would be far too slow to iterate on (each PyBullet
episode costs real wall-clock time; this abstract env runs thousands of
steps per second in pure numpy). The trained policy is then dropped into
the PyBullet mission loop as a swappable local controller (see
rl/rl_controller.py) alongside the classical potential-field controller,
so you can directly compare "hand-tuned" vs "learned" local navigation --
a natural evaluation story for a project report.

Obstacles are circles (not the maze's rectangular walls) -- a standard
simplification in mobile-robot local-navigation RL literature that keeps
ray-obstacle intersection a closed-form quadratic instead of a polygon
raycast, which is what makes thousands-of-steps-per-second training
possible on CPU. The policy only ever sees lidar ranges + relative goal
vector, so it transfers reasonably to the polygonal maze walls at
deployment time (both just look like "something is r meters away in
direction theta" to the policy).
"""

import numpy as np


class LocalNavEnv:
    def __init__(self, num_obstacles=7, arena_half=4.0, num_rays=16, max_range=3.0,
                 agent_radius=0.2, goal_radius=0.3, max_steps=200, seed=None):
        self.num_obstacles = num_obstacles
        self.arena_half = arena_half
        self.num_rays = num_rays
        self.max_range = max_range
        self.agent_radius = agent_radius
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.ray_angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False)
        self.obs_dim = num_rays + 3   # lidar + [dist_norm, sin(bearing), cos(bearing)]
        self.act_dim = 2              # [forward_speed in [0,1], turn_rate in [-1,1]]

    def reset(self):
        self.obstacles = self.rng.uniform(-self.arena_half * 0.8, self.arena_half * 0.8,
                                           size=(self.num_obstacles, 2))
        self.obstacle_r = self.rng.uniform(0.25, 0.6, size=self.num_obstacles)

        while True:
            self.pos = self.rng.uniform(-self.arena_half, self.arena_half, size=2)
            if self._clear_of_obstacles(self.pos):
                break
        while True:
            self.goal = self.rng.uniform(-self.arena_half, self.arena_half, size=2)
            if self._clear_of_obstacles(self.goal) and np.linalg.norm(self.goal - self.pos) > 2.0:
                break

        self.yaw = self.rng.uniform(-np.pi, np.pi)
        self.steps = 0
        self.prev_dist = np.linalg.norm(self.goal - self.pos)
        return self._obs()

    def _clear_of_obstacles(self, xy, margin=0.4):
        d = np.linalg.norm(self.obstacles - xy, axis=1)
        return np.all(d > self.obstacle_r + margin)

    def _lidar(self):
        ranges = np.full(self.num_rays, self.max_range)
        for i, a in enumerate(self.ray_angles):
            world_a = a + self.yaw
            d = np.array([np.cos(world_a), np.sin(world_a)])
            best = self.max_range
            for (ox, oy), r in zip(self.obstacles, self.obstacle_r):
                # ray-circle intersection (closed form)
                f = self.pos - np.array([ox, oy])
                b = 2 * np.dot(f, d)
                c = np.dot(f, f) - r ** 2
                disc = b ** 2 - 4 * c
                if disc < 0:
                    continue
                sqrt_disc = np.sqrt(disc)
                t1 = (-b - sqrt_disc) / 2
                t2 = (-b + sqrt_disc) / 2
                for t in (t1, t2):
                    if 0 < t < best:
                        best = t
            # arena boundary as an implicit wall
            for bound, axis, sign in [(self.arena_half, 0, 1), (-self.arena_half, 0, -1),
                                        (self.arena_half, 1, 1), (-self.arena_half, 1, -1)]:
                if d[axis] * sign > 1e-6:
                    t = (bound - self.pos[axis]) / d[axis]
                    if 0 < t < best:
                        best = t
            ranges[i] = best
        return ranges

    def _obs(self):
        ranges = self._lidar()
        to_goal = self.goal - self.pos
        dist = np.linalg.norm(to_goal)
        bearing = np.arctan2(to_goal[1], to_goal[0]) - self.yaw
        obs = np.concatenate([
            ranges / self.max_range,
            [np.clip(dist / (2 * self.arena_half), 0, 1), np.sin(bearing), np.cos(bearing)],
        ])
        return obs.astype(np.float32)

    def step(self, action):
        speed = float(np.clip(action[0], 0, 1)) * 1.2       # m/s
        turn = float(np.clip(action[1], -1, 1)) * 2.0        # rad/s
        dt = 0.15

        self.yaw += turn * dt
        self.pos = self.pos + speed * np.array([np.cos(self.yaw), np.sin(self.yaw)]) * dt
        self.steps += 1

        dist = np.linalg.norm(self.goal - self.pos)
        progress_reward = (self.prev_dist - dist) * 5.0
        self.prev_dist = dist
        step_penalty = -0.02

        collided = not self._clear_of_obstacles(self.pos, margin=self.agent_radius)
        out_of_bounds = np.any(np.abs(self.pos) > self.arena_half)
        reached_goal = dist < self.goal_radius

        reward = progress_reward + step_penalty
        done = False
        if collided or out_of_bounds:
            reward -= 5.0
            done = True
        elif reached_goal:
            reward += 10.0
            done = True
        elif self.steps >= self.max_steps:
            done = True

        return self._obs(), reward, done, {"collided": collided, "reached_goal": reached_goal}
