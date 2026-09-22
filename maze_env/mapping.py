"""
Two pieces, both GPS-free:

1. OccupancyGridMap -- log-odds occupancy grid, updated from lidar (fully vectorised: one
   numpy pass per scan instead of a Python loop per ray), plus a "seen by the camera"
   coverage grid so the mission knows which floor area has actually been looked at.

2. StateEstimator -- a particle filter (Monte-Carlo localisation) fed by the noisy IMU.
   Fixes versus the earlier version:
     * Process noise is now specified PER SQRT-SECOND and scaled by the time step. It used to be
       0.03 m / 0.02 rad added on EVERY 60 Hz step -- about 30x more than the real sensor error,
       so the particle cloud spread 16 m in a minute and the filter fought itself.
     * Random-particle "kidnapped robot" injection is OFF by default. It fired on ordinary
       noise and teleported the estimate by metres. Nothing in this sim kidnaps the robot.
     * The estimate is smooth, so the controller can use it directly.
   Honest scope: this is localisation against ONE shared map built from the estimated pose (not
   per-particle FastSLAM). Replace with a production SLAM stack for real hardware.
"""

import numpy as np

from maze_env import config as C


class OccupancyGridMap:
    def __init__(self, half_size=9.0, resolution=0.15):
        self.res = resolution
        self.half_size = half_size
        n = int(2 * half_size / resolution)
        self.n = n
        self.log_odds = np.zeros((n, n), dtype=np.float32)
        self.seen = np.zeros((n, n), dtype=bool)      # cells the camera has had in view
        self.l_occ = 1.2
        self.l_free = -0.35
        self.l_min, self.l_max = -4.0, 4.0
        self.known_thresh = 0.15
        self._step = resolution * 0.7
        self._ray_t = np.arange(int(C.LIDAR_RANGE / self._step) + 2) * self._step

    # ---- coordinates ------------------------------------------------------
    def world_to_grid(self, xy):
        gx = int((xy[0] + self.half_size) / self.res)
        gy = int((xy[1] + self.half_size) / self.res)
        return min(max(gx, 0), self.n - 1), min(max(gy, 0), self.n - 1)

    def world_to_grid_frac(self, xy):
        return (xy[..., 0] + self.half_size) / self.res, (xy[..., 1] + self.half_size) / self.res

    def grid_to_world(self, gx, gy):
        return np.array([gx * self.res - self.half_size + self.res / 2,
                         gy * self.res - self.half_size + self.res / 2])

    def grid_box_label(self, xy):
        """Human-readable ~1 m grid box label, e.g. 'C4', for tagging survivors."""
        col = int((xy[0] + self.half_size) // 1.0)
        row = int((xy[1] + self.half_size) // 1.0)
        return f"{chr(ord('A') + min(max(col, 0), 25))}{max(row, 0)}"

    # ---- masks ------------------------------------------------------------
    def occupied_mask(self):
        return self.log_odds > 0.4

    def free_mask(self):
        return self.log_odds < -0.3

    def unknown_mask(self):
        return np.abs(self.log_odds) < 0.05

    def occupancy_prob(self):
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    # ---- lidar update -----------------------------------------------------
    def _ray_cells(self, pose_xy, world_angles, ranges, max_len=None, margin=None):
        """Grid cell indices (flattened) strictly before each ray's end, one pass, vectorised."""
        n, res = self.n, self.res
        r_end = ranges - (self.res * 0.5 if margin is None else margin)
        if max_len is not None:
            r_end = np.minimum(r_end, max_len)
        t = self._ray_t
        valid = t[None, :] < r_end[:, None]
        dx, dy = np.cos(world_angles), np.sin(world_angles)
        gx = ((pose_xy[0] + dx[:, None] * t[None, :] + self.half_size) / res).astype(np.int32)
        gy = ((pose_xy[1] + dy[:, None] * t[None, :] + self.half_size) / res).astype(np.int32)
        ok = valid & (gx >= 0) & (gx < n) & (gy >= 0) & (gy < n)
        return np.unique(gx[ok] * n + gy[ok])

    def update(self, pose_xy, yaw, angles, ranges, max_range):
        """Integrate one lidar scan taken at the ESTIMATED pose."""
        self._updates = getattr(self, "_updates", 0) + 1
        n = self.n
        world_a = angles + yaw
        flat = self.log_odds.reshape(-1)
        # stop marking free space a full cell short of the wall: rays that graze a wall used to
        # free the wall's own cells and punch phantom gaps in it (a fake shortcut that came and
        # went as the drone moved, making it flip between two routes forever)
        free_idx = self._ray_cells(pose_xy, world_a, ranges, margin=self.res * 1.1)
        # A wall must not be erased by near-parallel long-range rays: they free a wall's cells more
        # often than they hit them. Never free a cell that is, or touches, a known wall cell.
        occ = self.log_odds > 0.4
        prot = occ.copy()
        prot[1:, :] |= occ[:-1, :]; prot[:-1, :] |= occ[1:, :]
        prot[:, 1:] |= occ[:, :-1]; prot[:, :-1] |= occ[:, 1:]
        free_idx = free_idx[~prot.reshape(-1)[free_idx]]
        flat[free_idx] = np.maximum(flat[free_idx] + self.l_free, self.l_min)
        hit = ranges < (max_range - 1e-3)
        if np.any(hit):
            ex = ((pose_xy[0] + np.cos(world_a[hit]) * ranges[hit] + self.half_size) / self.res).astype(np.int32)
            ey = ((pose_xy[1] + np.sin(world_a[hit]) * ranges[hit] + self.half_size) / self.res).astype(np.int32)
            ok = (ex >= 0) & (ex < n) & (ey >= 0) & (ey < n)
            occ_idx = np.unique(ex[ok] * n + ey[ok])
            flat[occ_idx] = np.minimum(flat[occ_idx] + self.l_occ, self.l_max)

    def update_coverage(self, pose_xy, yaw, angles, ranges, half_fov=None, view_range=C.COVERAGE_RANGE):
        """Mark floor cells inside the camera's horizontal FOV, within detection range and with a
        clear line of sight (limited by the lidar return along that bearing) as 'seen'."""
        if half_fov is None:
            half_fov = np.radians(C.CAM_FOV_H_DEG / 2 - 6.0)
        sel = np.abs(angles) <= half_fov
        idx = self._ray_cells(pose_xy, angles[sel] + yaw, ranges[sel], max_len=view_range, margin=0.0)
        self.seen.reshape(-1)[idx] = True

    def occupied_distance_field(self, max_cells=10, refresh_every=8):
        """Distance (m) from every cell to the nearest occupied cell, capped. Cached: recomputed only
        every `refresh_every` map updates (walls barely move between scans)."""
        self._updates = getattr(self, "_updates", 0)
        if getattr(self, "_df", None) is None or self._updates - self._df_at >= refresh_every:
            occ = self.log_odds > 0.4
            n = self.n
            pad = np.pad(occ, max_cells, constant_values=False)
            dist = np.full((n, n), max_cells + 1.0, dtype=np.float32)
            for dx in range(-max_cells, max_cells + 1):
                for dy in range(-max_cells, max_cells + 1):
                    d = np.hypot(dx, dy)
                    if d > max_cells:
                        continue
                    sh = pad[max_cells + dx:max_cells + dx + n, max_cells + dy:max_cells + dy + n]
                    dist = np.where(sh & (d < dist), np.float32(d), dist)
            self._df = dist * self.res
            self._df_at = self._updates
            self._n_occ = int(occ.sum())
        return self._df

    # ---- bilinear query used by the particle filter --------------------------
    def bilinear_query(self, points_xy, prob=None, known=None):
        if prob is None:
            prob = self.occupancy_prob()
        if known is None:
            known = np.abs(self.log_odds) > self.known_thresh
        gx, gy = self.world_to_grid_frac(points_xy)
        x0 = np.floor(gx).astype(int)
        y0 = np.floor(gy).astype(int)
        fx, fy = gx - x0, gy - y0
        in_bounds = (x0 >= 0) & (x0 < self.n - 1) & (y0 >= 0) & (y0 < self.n - 1)
        x0c = np.clip(x0, 0, self.n - 2)
        y0c = np.clip(y0, 0, self.n - 2)
        M00, M10 = prob[x0c, y0c], prob[x0c + 1, y0c]
        M01, M11 = prob[x0c, y0c + 1], prob[x0c + 1, y0c + 1]
        all_known = (known[x0c, y0c] & known[x0c + 1, y0c] & known[x0c, y0c + 1]
                     & known[x0c + 1, y0c + 1] & in_bounds)
        value = fy * (fx * M11 + (1 - fx) * M01) + (1 - fy) * (fx * M10 + (1 - fx) * M00)
        d_dfx = fy * (M11 - M01) + (1 - fy) * (M10 - M00)
        d_dfy = fx * (M11 - M10) + (1 - fx) * (M01 - M00)
        return (np.where(all_known, value, 0.0), np.where(all_known, d_dfx / self.res, 0.0),
                np.where(all_known, d_dfy / self.res, 0.0), all_known)


class StateEstimator:
    """Particle-filter pose estimator (x, y, yaw). See module docstring for what changed."""

    def __init__(self, start_xy, start_yaw=0.0, bound_half_size=None, num_particles=150,
                 motion_noise_xy=0.03, motion_noise_yaw=0.015,
                 initial_spread_xy=0.05, initial_spread_yaw=0.03,
                 enable_injection=False, sharpness=8.0, lf_sigma=0.12, lf_floor=0.15, seed=None):
        self.n_particles = num_particles
        self.bound_half_size = bound_half_size
        self.motion_noise_xy = motion_noise_xy        # m / sqrt(s)
        self.motion_noise_yaw = motion_noise_yaw      # rad / sqrt(s)
        self.enable_injection = enable_injection
        self.sharpness = sharpness
        self.lf_sigma = lf_sigma
        self.lf_floor = lf_floor
        self.rng = np.random.default_rng(seed)
        base = np.array([start_xy[0], start_xy[1], start_yaw])
        noise = np.column_stack([self.rng.normal(0, initial_spread_xy, num_particles),
                                 self.rng.normal(0, initial_spread_xy, num_particles),
                                 self.rng.normal(0, initial_spread_yaw, num_particles)])
        self.particles = (base + noise).astype(float)
        self.weights = np.full(num_particles, 1.0 / num_particles)
        self._calls = 0

    @property
    def pos(self):
        return np.array([np.sum(self.weights * self.particles[:, 0]),
                         np.sum(self.weights * self.particles[:, 1])])

    @property
    def yaw(self):
        return np.arctan2(np.sum(self.weights * np.sin(self.particles[:, 2])),
                          np.sum(self.weights * np.cos(self.particles[:, 2])))

    def confidence_spread(self):
        m = self.pos
        vx = np.sum(self.weights * (self.particles[:, 0] - m[0]) ** 2)
        vy = np.sum(self.weights * (self.particles[:, 1] - m[1]) ** 2)
        return float(np.sqrt(vx + vy))

    def predict(self, noisy_vel_xy, noisy_yaw_rate, dt):
        n = self.n_particles
        yaw_mid = self.particles[:, 2] + 0.5 * noisy_yaw_rate * dt      # rotate by mid-interval heading
        c, s = np.cos(yaw_mid), np.sin(yaw_mid)
        wvx = c * noisy_vel_xy[0] - s * noisy_vel_xy[1]
        wvy = s * noisy_vel_xy[0] + c * noisy_vel_xy[1]
        sq = np.sqrt(dt)
        self.particles[:, 0] += wvx * dt + self.rng.normal(0, self.motion_noise_xy * sq, n)
        self.particles[:, 1] += wvy * dt + self.rng.normal(0, self.motion_noise_xy * sq, n)
        self.particles[:, 2] += noisy_yaw_rate * dt + self.rng.normal(0, self.motion_noise_yaw * sq, n)
        if self.bound_half_size is not None:
            m = self.bound_half_size - 0.5
            self.particles[:, :2] = np.clip(self.particles[:, :2], -m, m)

    def correct(self, occ_map, angles, ranges, max_range):
        """Weight particles by likelihood-field agreement: how close each scan endpoint lands to a
        wall that is already in the map. Unmatched rays (new wall not mapped yet) get a small floor,
        so seeing new territory neither rewards nor punishes a particle much."""
        df = occ_map.occupied_distance_field()
        if occ_map._n_occ < 40:
            return
        self._calls += 1
        hit_mask = ranges < (max_range - 1e-3)
        if np.sum(hit_mask) < 8:
            return
        a, r = angles[hit_mask][::2], ranges[hit_mask][::2]
        wa = self.particles[:, 2:3] + a[None, :]
        ex = self.particles[:, 0:1] + r[None, :] * np.cos(wa)
        ey = self.particles[:, 1:2] + r[None, :] * np.sin(wa)
        gx = np.clip(((ex + occ_map.half_size) / occ_map.res).astype(int), 0, occ_map.n - 1)
        gy = np.clip(((ey + occ_map.half_size) / occ_map.res).astype(int), 0, occ_map.n - 1)
        d = df[gx, gy]
        lik = np.maximum(np.exp(-0.5 * (d / self.lf_sigma) ** 2), self.lf_floor)
        score = lik.mean(axis=1)
        w = self.weights * np.exp(self.sharpness * (score - score.max()))
        s = w.sum()
        if s < 1e-12:
            return
        self.weights = w / s
        if 1.0 / np.sum(self.weights ** 2) < 0.5 * self.n_particles:
            self._resample(occ_map, None, None)

    def _resample(self, occ_map=None, prob=None, known=None):
        n = self.n_particles
        pos = (np.arange(n) + self.rng.uniform()) / n
        cs = np.cumsum(self.weights)
        cs[-1] = 1.0
        self.particles = self.particles[np.searchsorted(cs, pos)].copy()
        self.weights = np.full(n, 1.0 / n)
