"""
Onboard sensing only. The mission loop may use ONLY: Lidar2D.scan(), RGBDCamera.capture(),
NoisyIMU.sample() and AltitudeSensor.read(). (They are given the simulator's true pose
internally because that is how a simulated sensor produces its measurement -- the
navigation code never sees that pose.)
"""

import numpy as np
import pybullet as p

from maze_env import config as C


class Lidar2D:
    """Horizontal 2D lidar ring, mounted at the robot's body frame."""

    def __init__(self, num_rays=C.LIDAR_RAYS, max_range=C.LIDAR_RANGE):
        self.num_rays = num_rays
        self.max_range = max_range
        self.angles = np.linspace(-np.pi, np.pi, num_rays, endpoint=False)

    def scan(self, drone_pos, drone_yaw):
        world_a = self.angles + drone_yaw
        dirs = np.stack([np.cos(world_a), np.sin(world_a), np.zeros_like(world_a)], axis=1)
        origin = np.asarray(drone_pos, dtype=float)
        tos = origin[None, :] + dirs * self.max_range
        results = p.rayTestBatch([origin.tolist()] * self.num_rays, tos.tolist(), collisionFilterMask=3)
        ranges = np.full(self.num_rays, self.max_range)
        for i, res in enumerate(results):
            if res[0] != -1:
                ranges[i] = res[2] * self.max_range
        return self.angles.copy(), ranges


class RGBDCamera:
    """Forward-facing RGB-D camera (slightly pitched down, like a real nose camera)."""

    def __init__(self, width=C.CAM_W, height=C.CAM_H, fov=C.CAM_FOV_V_DEG,
                 near=C.CAM_NEAR, far=C.CAM_FAR):
        self.width, self.height, self.fov = width, height, fov
        self.near, self.far = near, far
        self.proj_matrix = p.computeProjectionMatrixFOV(fov, width / height, near, far)

    def capture(self, robot_pos, robot_yaw, tilt=C.CAM_TILT, with_segmentation=False):
        """Returns rgb (H,W,3 uint8), metric depth (H,W) in meters along the optical axis,
        and (optionally) the segmentation image (body id per pixel)."""
        fwd_xy = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
        forward = np.array([fwd_xy[0], fwd_xy[1], tilt])
        cam_eye = np.asarray(robot_pos, dtype=float) + np.array(
            [fwd_xy[0] * C.CAM_MOUNT_FORWARD, fwd_xy[1] * C.CAM_MOUNT_FORWARD, 0.05])
        view = p.computeViewMatrix(cam_eye.tolist(), (cam_eye + forward).tolist(), [0, 0, 1])
        flags = 0 if with_segmentation else p.ER_NO_SEGMENTATION_MASK
        w, h, rgb, depth, seg = p.getCameraImage(self.width, self.height, view, self.proj_matrix,
                                                 renderer=p.ER_TINY_RENDERER, flags=flags)
        rgb = np.reshape(rgb, (h, w, 4))[:, :, :3].astype(np.uint8)
        d = np.reshape(depth, (h, w))
        z = self.far * self.near / (self.far - (self.far - self.near) * d)   # buffer -> meters
        if with_segmentation:
            return rgb, z, np.reshape(seg, (h, w))
        return rgb, z


class NoisyIMU:
    """
    Simulated inertial/odometry sensor: a body-frame velocity estimate and a yaw rate, each with
    white noise plus a slowly wandering bias. `noise_scale` multiplies everything so you can test
    how the estimator copes with a worse sensor (1 = the original model, ~5 = a cheap IMU that
    really drifts). NOTE: this outputs VELOCITY with small noise, i.e. it behaves like good
    visual/optical-flow odometry, not a raw accelerometer. Treat drift numbers accordingly.
    """

    def __init__(self, accel_noise_std=0.02, gyro_noise_std=0.0015, bias_walk_std=0.00005,
                 vel_bias_std=0.01, gyro_bias_std=0.0008, noise_scale=1.0, seed=None):
        self.rng = np.random.default_rng(seed)
        s = noise_scale
        self.accel_noise_std = accel_noise_std * s
        self.gyro_noise_std = gyro_noise_std * s
        self.bias_walk_std = bias_walk_std * s
        # constant turn-on biases (the dominant real-world drift source): ~1 cm/s velocity bias and
        # ~0.05 deg/s gyro bias at scale 1 -- uncorrected, that is metres of error over a mission
        self.accel_bias = self.rng.normal(0, vel_bias_std * s, 3)
        self.gyro_bias = float(self.rng.normal(0, gyro_bias_std * s))

    def sample(self, true_vel, true_yaw_rate, dt):
        self.accel_bias += self.rng.normal(0, self.bias_walk_std, 3) * dt
        self.gyro_bias += self.rng.normal(0, self.bias_walk_std) * dt
        noisy_vel = true_vel + self.accel_bias + self.rng.normal(0, self.accel_noise_std, 3)
        noisy_yaw_rate = true_yaw_rate + self.gyro_bias + self.rng.normal(0, self.gyro_noise_std)
        return noisy_vel, noisy_yaw_rate


class AltitudeSensor:
    """Downward rangefinder / barometer stand-in, so altitude hold does not read ground truth."""

    def __init__(self, noise_std=0.01, seed=None):
        self.rng = np.random.default_rng(seed)
        self.noise_std = noise_std

    def read(self, true_z):
        return float(true_z + self.rng.normal(0, self.noise_std))
