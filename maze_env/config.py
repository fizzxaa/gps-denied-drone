"""
One place for every number that the mission, the training-data generator and
the RL environments must AGREE on. Several earlier bugs were exactly this kind
of disagreement (camera height in the dataset vs. in flight, FOV in the labels
vs. in the renderer, lidar range in training vs. deployment).
"""
import math

# ---- flight ----------------------------------------------------------------
CRUISE_ALT = 1.5          # m. Survivor markers are ~1.08 m tall, ceiling is 2.44 m
TAKEOFF_ALT = 0.12        # m. drone starts on the floor
MAX_SPEED = 0.9           # m/s cruise
MAX_ACCEL = 1.0           # m/s^2 velocity slew limit (keeps tilt well under the 25 deg limit)
ROBOT_RADIUS = 0.19       # m. planning clearance -- must clear a 1m corridor at 0.15m grid resolution;
                          # drone body radius is ~0.13m, so this still leaves real margin

# ---- lidar -----------------------------------------------------------------
LIDAR_RAYS = 180
LIDAR_RANGE = 8.0

# ---- camera (used by flight, dataset generation, detector) -------------------
CAM_W, CAM_H = 160, 120
CAM_FOV_V_DEG = 70.0                       # PyBullet's fov is the VERTICAL field of view
CAM_TILT = -0.20                           # optical axis z-component (~ -11 deg pitch)
CAM_MOUNT_FORWARD = 0.16
CAM_NEAR, CAM_FAR = 0.05, 10.0
_aspect = CAM_W / CAM_H
CAM_FOV_H_DEG = math.degrees(2 * math.atan(math.tan(math.radians(CAM_FOV_V_DEG / 2)) * _aspect))
CAM_FOCAL_PX = (CAM_W / 2) / math.tan(math.radians(CAM_FOV_H_DEG / 2))
DETECT_RANGE = 6.0        # m. farthest range the detector will accept a survivor at
COVERAGE_RANGE = 4.0      # m. a floor cell only counts as 'looked at' inside this range (margin vs DETECT_RANGE,
                          #    because a detection needs several consecutive frames, not one flash)
SCAN_SPIN_RATE = 0.7      # rad/s. constant yaw spin so the camera sweeps 360 deg while flying

# ---- survivor marker (capsule) ----------------------------------------------
SURVIVOR_RADIUS = 0.18
SURVIVOR_HEIGHT = 0.9     # cylinder part; capsule total = 0.9 + 2*0.18

MAP_RESOLUTION = 0.10     # m. finer than corridor width so an 8-connected clearance-inflated path
                          # through a 1m corridor is never just one fragile diagonal-blocked cell wide
ROOM_HALF = 1.0           # m. must match maze_env.maze_generator.ROOM_SIZE/2 -- used by survivor
                          # placement so a fallback spawn point can never land past the room's own wall
