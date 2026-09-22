"""
Wraps the ES-trained MLP policy as a LOCAL STEERING controller for the mission loop.

The policy was trained as a forward-driving unicycle (speed + turn rate) whose lidar and goal bearing
are relative to its heading. The mission drone is holonomic and spins continuously so its camera
sweeps 360 deg, so its body heading is meaningless for steering. The adapter therefore builds the
policy's observation in a VIRTUAL heading that always points at the waypoint: it resamples the lidar
around that direction, gives the policy "goal dead ahead", and reads the policy's turn output as a
sideways deflection. Speed is decided by the planner; the policy only chooses the direction.

Normalisation matches rl/env_maze.py exactly (goal distance / (grid_n * pitch * 1.5)); the old
version used a fixed 16 m, which is not what the policy was trained with.
Trained-policy limits (measured, see README): it cannot plan around a wall, which is fine here
because the waypoints it is given come from the clearance-aware planner and are in line of sight.
"""

import os
import numpy as np

from rl.policy import MLPPolicy

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "trained_policy.npz")
_NUM_RAYS = 16
_MAX_RANGE = 3.5


class RLController:
    def __init__(self, path=_DEFAULT_PATH, grid_n=4, cell_pitch=3.0):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No trained policy at {path}. Run `python -m rl.train_es` first.")
        self.policy = MLPPolicy.load(path)
        self.norm_span = grid_n * cell_pitch * 1.5

    def desired_direction(self, angles, ranges, est_pos, est_yaw, target_xy):
        """Unit direction (in the map frame) the policy wants to move in, or None if the target is here."""
        to = np.asarray(target_xy) - np.asarray(est_pos)
        dist = float(np.linalg.norm(to))
        if dist < 1e-3:
            return None
        bearing_world = np.arctan2(to[1], to[0])
        bearing_body = bearing_world - est_yaw
        rel = np.linspace(-np.pi, np.pi, _NUM_RAYS, endpoint=False)
        sample_at = np.arctan2(np.sin(bearing_body + rel), np.cos(bearing_body + rel))
        r = np.interp(sample_at, angles, ranges, period=2 * np.pi)
        obs = np.concatenate([np.clip(r, 0, _MAX_RANGE) / _MAX_RANGE,
                              [np.clip(dist / self.norm_span, 0, 1), 0.0, 1.0]]).astype(np.float32)
        speed, turn = self.policy.forward(obs)
        heading = bearing_world + float(turn) * 0.6         # positive turn = counter-clockwise, as in training
        return np.array([np.cos(heading), np.sin(heading)]), float(speed)
