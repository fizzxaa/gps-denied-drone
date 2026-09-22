"""
A motor-mixed quadrotor: an outer velocity-tracking loop produces a
desired (total thrust, roll torque, pitch torque), which gets allocated
to 4 individual motor thrusts via pseudo-inverse control allocation using
the motors' REAL geometric positions -- so each motor's thrust, applied
as a vertical force at its actual world-space arm-tip location, produces
the resulting roll/pitch torque from real physics (r x F), not an
injected torque term. Yaw is handled separately and explicitly (see
below) because that's genuinely how it works on a real quadrotor too.

Why yaw can't come from the same lever-arm trick as roll/pitch: a purely
VERTICAL force applied at an offset position produces torque = r x F,
which only has x/y (roll/pitch) components when F is along z -- it can
never produce a z-axis (yaw) torque no matter where you put it. Real
quadrotors get yaw authority from motor REACTION torque instead (each
spinning rotor's blades push air one way; Newton's third law pushes back
on the airframe the other way -- a pure z-axis torque, independent of
motor position, that alternates sign between clockwise- and
counterclockwise-spinning motors). This model applies that as an
explicitly injected z-torque from a yaw-rate controller, exactly like
real flight-controller firmware treats "differential motor RPM" as a
yaw-torque request rather than something a position-based lever arm can
produce. This is standard practice for simplified quadrotor sims (the
same choice PX4 SITL and most textbook models make), not a shortcut
specific to this project.

Collision shape: a box sized to the frame's arm span, not a sphere. This
matters for the same reason the sibling ground-robot's box chassis matters
for ITS inertia: PyBullet auto-computes a correct inertia tensor from a
collision shape's mass and dimensions, and a real quadrotor's mass is
spread out across 4 arms, giving it meaningfully higher roll/pitch inertia
than a sphere of the same mass and radius would imply. A box approximates
that spread; a sphere would understate how much torque it takes to
actually tilt the airframe.
"""

import numpy as np
import pybullet as p


def _quat_about_z(angle_rad):
    return p.getQuaternionFromEuler([0, 0, angle_rad])


def build_quadrotor_visual_shape(arm_span=0.26, arm_width=0.03, arm_thick=0.012,
                                  body_size=0.09, body_height=0.05,
                                  rotor_radius=0.11, rotor_thick=0.008):
    """Compound visual shape: central body box + 4 diagonal arms + 4 rotor
    discs at the arm tips."""
    shape_types, half_extents, radii, lengths = [], [], [], []
    frame_positions, frame_orns, colors = [], [], []

    shape_types.append(p.GEOM_BOX)
    half_extents.append([body_size / 2, body_size / 2, body_height / 2])
    radii.append(0); lengths.append(0)
    frame_positions.append([0, 0, 0]); frame_orns.append([0, 0, 0, 1])
    colors.append([0.15, 0.15, 0.18, 1.0])

    arm_center_dist = arm_span / 2
    for k in range(4):
        angle = np.pi / 4 + k * np.pi / 2  # 45, 135, 225, 315 deg
        cx, cy = np.cos(angle) * arm_center_dist, np.sin(angle) * arm_center_dist

        shape_types.append(p.GEOM_BOX)
        half_extents.append([arm_center_dist, arm_width / 2, arm_thick / 2])
        radii.append(0); lengths.append(0)
        frame_positions.append([cx, cy, 0])
        frame_orns.append(list(_quat_about_z(angle)))
        colors.append([0.25, 0.25, 0.28, 1.0])

        tip_x, tip_y = np.cos(angle) * arm_span, np.sin(angle) * arm_span
        shape_types.append(p.GEOM_CYLINDER)
        half_extents.append([0, 0, 0])
        radii.append(rotor_radius); lengths.append(rotor_thick)
        frame_positions.append([tip_x, tip_y, arm_thick])
        frame_orns.append([0, 0, 0, 1])
        colors.append([0.05, 0.05, 0.05, 0.55])

    return p.createVisualShapeArray(
        shapeTypes=shape_types, halfExtents=half_extents, radii=radii, lengths=lengths,
        visualFramePositions=frame_positions, visualFrameOrientations=frame_orns,
        rgbaColors=colors,
    )


class Drone:
    def __init__(self, start_xyz, mass=1.8, arm_span=0.26, max_tilt_deg=22,
                 max_climb_rate=1.5, max_horiz_speed=1.5, motor_tau=0.05,
                 max_motor_thrust=None, client=0):
        self.client = client
        self.mass = mass
        self.g = 9.81
        self.arm_span = arm_span
        self.max_tilt = np.radians(max_tilt_deg)
        self.max_climb_rate = max_climb_rate
        self.max_horiz_speed = max_horiz_speed
        self.motor_tau = motor_tau

        body_size, body_height = 0.09, 0.05
        # box collision sized to the frame's spread, not a point/sphere --
        # see module docstring for why this matters for roll/pitch inertia
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[arm_span / 2, arm_span / 2, body_height / 2])
        vis = build_quadrotor_visual_shape(arm_span=arm_span, body_size=body_size,
                                            body_height=body_height)
        self.body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col,
                                       baseVisualShapeIndex=vis,
                                       basePosition=list(start_xyz))
        p.changeDynamics(self.body, -1, linearDamping=0.3, angularDamping=0.4)
        # The drone lives in collision group 4 (static walls/floor are group 2, dynamic default 1). Rays use mask=3, so the
        # lidar never returns the drone's own (tilting) body as a 0 m obstacle. Physical contact
        # with walls is unaffected (group 2 still collides with group 1).
        p.setCollisionFilterGroupMask(self.body, -1, 4, -1)

        # motor geometry: standard X-configuration, alternating spin
        # direction (needed for the yaw-reaction-torque sign convention,
        # even though we apply yaw as one lumped torque rather than
        # tracking each motor's individual spin direction separately)
        self.motor_angles = np.array([np.pi / 4 + k * np.pi / 2 for k in range(4)])
        self.motor_x = np.cos(self.motor_angles) * (arm_span / 2)
        self.motor_y = np.sin(self.motor_angles) * (arm_span / 2)

        # pseudo-inverse control allocation: maps desired [total_thrust,
        # roll_torque, pitch_torque] -> 4 individual motor thrusts, using
        # each motor's REAL position (roll_i = y_i, pitch_i = -x_i, from
        # torque = r x F for a vertical force -- see module docstring).
        # A pseudo-inverse (rather than a hand-derived per-configuration
        # formula) is standard practice for control allocation and works
        # for any motor layout, not just a perfectly symmetric X-quad.
        effectiveness = np.stack([
            np.ones(4),
            self.motor_y,
            -self.motor_x,
        ])
        self.allocation = np.linalg.pinv(effectiveness)  # (4, 3)

        hover_thrust_each = mass * self.g / 4
        self.max_motor_thrust = max_motor_thrust or hover_thrust_each * 2.5
        self.motor_thrust_actual = np.full(4, hover_thrust_each)

        self.target_yaw = 0.0
        # PID integral accumulator for the outer velocity loop (see
        # command_velocity) -- the attitude inner loop below was already
        # PD; this adds the missing I term to the outer loop so sustained
        # tracking error (e.g. from a steady headwind-like disturbance, or
        # just steady-state droop under the proportional term alone) gets
        # driven out over time instead of persisting indefinitely.
        self._vel_integral = np.zeros(3)


    def command_velocity_body(self, vbx, vby, vz, yaw_rate, pose=None):
        """Velocity command given in the drone's BODY frame (what a real flight controller
        is told: 'fly forward 0.5 m/s'). The controller converts it with its own attitude."""
        pose = pose if pose is not None else self.get_ground_truth_pose()
        yaw = pose[1]
        c, s = np.cos(yaw), np.sin(yaw)
        self.command_velocity(c * vbx - s * vby, s * vbx + c * vby, vz, yaw_rate, pose=pose)

    def reset_controller_state(self):
        self._vel_integral = np.zeros(3)
        self._prev_vel_error = np.zeros(3)

    # ------------------------------------------------------------------
    def get_ground_truth_pose(self):
        pos, orn = p.getBasePositionAndOrientation(self.body)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        vel, ang_vel = p.getBaseVelocity(self.body)
        # PyBullet reports angular velocity in the WORLD frame. The attitude
        # damping term needs BODY-frame roll/pitch rates -- using world-frame
        # rates only works when yaw happens to be ~0 and flips the drone as
        # soon as it yaws (the instability found in testing).
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        w_body = R.T @ np.array(ang_vel)
        # 9th value: what the flight controller's heading (yaw) estimate actually integrates. A body gyro
        # measures body rates (p,q,r); the yaw ANGLE changes at (q*sin(roll) + r*cos(roll))/cos(pitch).
        # World-z angular velocity differs from that whenever the drone is tilted and rocking, which
        # made the yaw estimate drift.
        heading_rate = (w_body[1] * np.sin(roll) + w_body[2] * np.cos(roll)) / max(np.cos(pitch), 0.3)
        return (np.array(pos), yaw, np.array(vel), ang_vel[2], roll, pitch,
                float(w_body[0]), float(w_body[1]), float(heading_rate))

    def _get_orientation_matrix(self):
        """Full 3x3 body-to-world rotation matrix (not just a yaw-only
        approximation) -- needed to apply motor thrust along the vehicle's
        actual tilted body Z-axis, not a fixed world-vertical direction.
        See command_velocity's docstring for why this matters."""
        _, orn = p.getBasePositionAndOrientation(self.body)
        return np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

    def sensor_pose(self):
        """World-space (pos, yaw) of the sensor mount -- kept for interface
        parity with the ground robot's sensor_pose(), even though a drone's
        camera/lidar mount offset from the body center is small enough not
        to matter much (unlike the ground robot's tall mast)."""
        pos, yaw, *_ = self.get_ground_truth_pose()
        return pos + np.array([0, 0, 0.03]), yaw

    # ------------------------------------------------------------------
    def command_velocity(self, vx, vy, vz, yaw_rate, dt=1.0 / 240.0, pose=None):
        """
        World-frame velocity-tracking outer loop -> body attitude targets
        -> motor mixing inner loop -> 4 individual motor thrusts (with
        first-order spin-up lag) applied at their real world positions,
        plus a separately-injected yaw reaction torque. Call every physics
        tick (motor lag assumes a consistent dt).
        """
        vx = np.clip(vx, -self.max_horiz_speed, self.max_horiz_speed)
        vy = np.clip(vy, -self.max_horiz_speed, self.max_horiz_speed)
        vz = np.clip(vz, -self.max_climb_rate, self.max_climb_rate)

        pos, yaw, vel, yaw_rate_actual, roll, pitch, roll_rate, pitch_rate, _ = \
            (pose if pose is not None else self.get_ground_truth_pose())

        # ---- outer loop: desired world velocity -> desired accel -> desired tilt/thrust ----
        # Full PID here (was P-only): proportional term for immediate
        # response, integral term to eliminate steady-state tracking
        # error, derivative term (on the error's rate of change) to damp
        # overshoot -- the standard 3-term structure, not just naming.
        k_vel_p, k_vel_i, k_vel_d = 4.0, 1.5, 0.15
        vel_error = np.array([vx, vy, vz]) - vel

        self._vel_integral += vel_error * dt
        # anti-windup: clamp the integral term's contribution so a
        # persistently unreachable target (e.g. commanded speed beyond
        # max_horiz_speed while already saturated) can't wind up the
        # integral term into a huge value that then overshoots wildly
        # once the error finally starts to close
        max_integral_accel = 2.0
        self._vel_integral = np.clip(self._vel_integral,
                                       -max_integral_accel / max(k_vel_i, 1e-6),
                                       max_integral_accel / max(k_vel_i, 1e-6))

        accel_error_rate = (vel_error - getattr(self, "_prev_vel_error", vel_error)) / max(dt, 1e-6)
        self._prev_vel_error = vel_error

        desired_accel = (k_vel_p * vel_error
                          + k_vel_i * self._vel_integral
                          + k_vel_d * accel_error_rate)
        max_accel_horiz = self.g * np.tan(self.max_tilt)
        ax = np.clip(desired_accel[0], -max_accel_horiz, max_accel_horiz)
        ay = np.clip(desired_accel[1], -max_accel_horiz, max_accel_horiz)
        az = np.clip(desired_accel[2], -3.0, 3.0)

        # rotate desired world-frame horizontal accel into body frame to get
        # desired tilt angles (small-angle quadrotor approximation: pitch
        # forward to accelerate forward, roll to accelerate sideways)
        c, s = np.cos(yaw), np.sin(yaw)
        body_ax = c * ax + s * ay
        body_ay = -s * ax + c * ay
        desired_pitch = np.clip(body_ax / self.g, -self.max_tilt, self.max_tilt)
        desired_roll = np.clip(-body_ay / self.g, -self.max_tilt, self.max_tilt)

        total_thrust = self.mass * (self.g + az)
        # Tilt compensation: a tilted thrust vector's VERTICAL component is
        # only total_thrust*cos(tilt), so without compensating, sustained
        # tilt (i.e. any sustained horizontal-velocity command) causes a
        # slow altitude sag as gravity wins by a small margin every tick.
        # Dividing by the current tilt's cosine is the standard fix real
        # flight controllers use (boost total thrust enough that its
        # vertical component still matches what was actually intended).
        tilt_cos = max(np.cos(roll) * np.cos(pitch), 0.5)
        total_thrust = total_thrust / tilt_cos

        # ---- attitude inner loop: PD on roll/pitch to the desired tilt ----
        # Gains set for roughly 3x the natural frequency of the outer
        # velocity loop (k_vel below) -- cascaded control loops need that
        # separation to avoid the inner and outer loop fighting each other,
        # which without it showed up during validation testing here as a
        # persistent pitch/velocity oscillation that never settled.
        k_att, k_att_rate = 1.2, 0.11
        roll_torque = k_att * (desired_roll - roll) - k_att_rate * roll_rate
        pitch_torque = k_att * (desired_pitch - pitch) - k_att_rate * pitch_rate

        # ---- motor mixing: [thrust, roll_torque, pitch_torque] -> 4 motor thrusts ----
        motor_thrust_cmd = self.allocation @ np.array([total_thrust, roll_torque, pitch_torque])
        motor_thrust_cmd = np.clip(motor_thrust_cmd, 0.0, self.max_motor_thrust)

        alpha = np.clip(dt / max(self.motor_tau, 1e-3), 0.0, 1.0)
        self.motor_thrust_actual += alpha * (motor_thrust_cmd - self.motor_thrust_actual)

        # Apply each motor's thrust along the vehicle's ACTUAL body Z-axis
        # (via the full orientation matrix), not a fixed world-vertical
        # direction. A real propeller's thrust is always perpendicular to
        # its own rotor plane, so when the airframe tilts, the thrust
        # vector tilts with it -- that tilt-induced horizontal thrust
        # component is the ENTIRE mechanism by which a quadrotor moves
        # horizontally at all. Applying thrust as pure world-frame-vertical
        # regardless of attitude (an earlier bug caught during validation
        # testing here) means the vehicle can tilt correctly but never
        # actually translate, since a vertical force can't accelerate it
        # sideways no matter how the airframe is oriented.
        R = self._get_orientation_matrix()
        body_z_world = R[:, 2]
        for i in range(4):
            local_offset = np.array([self.motor_x[i], self.motor_y[i], 0.0])
            world_offset = R @ local_offset
            world_pos = pos + world_offset
            force = body_z_world * self.motor_thrust_actual[i]
            p.applyExternalForce(self.body, -1, force.tolist(), world_pos.tolist(), p.WORLD_FRAME)

        # ---- yaw: explicit reaction torque (see module docstring for why
        # this can't come from the position-based mixing above) ----
        self.target_yaw += yaw_rate * dt
        yaw_err = np.arctan2(np.sin(self.target_yaw - yaw), np.cos(self.target_yaw - yaw))
        yaw_torque = 3.5 * yaw_err - 0.4 * yaw_rate_actual
        p.applyExternalTorque(self.body, -1, [0, 0, yaw_torque], p.WORLD_FRAME)
