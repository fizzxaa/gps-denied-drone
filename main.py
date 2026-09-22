"""
Autonomous GPS-denied indoor search-and-rescue drone -- PyBullet simulation.

Mission:  TAKEOFF -> EXPLORE (map every corridor/room, look at every floor cell, detect survivors)
          -> RETURN_TO_EXIT -> LAND -> DONE

What the flight code is allowed to use: lidar, camera, IMU, altitude sensor. Pose comes from the
particle-filter estimate; altitude from the altitude sensor; velocity commands are BODY-frame (a
real flight controller's interface). The simulator's true pose is read only to (a) simulate the
sensors, (b) run the drone's own inner attitude loop, (c) SCORE the run (drift, collisions,
coverage, survivors found). It never feeds a navigation or detection decision.

Usage:
    python main.py --grid_n 4 --seed 1
    python main.py --grid_n 3 --seed 2 --dashboard --save_final_frame result.png
"""

import argparse
import sys
import traceback
import numpy as np
import pybullet as p
import pybullet_data

from maze_env import config as C
from maze_env.maze_generator import MazeGenerator
from maze_env.drone import Drone
from maze_env.sensors import Lidar2D, RGBDCamera, NoisyIMU, AltitudeSensor
from maze_env.mapping import OccupancyGridMap, StateEstimator
from maze_env.navigation import Navigator, safety_govern, slew_limit
from maze_env.survivors import SurvivorManager


def rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def ground_truth_free_points(maze, spacing=0.3, wall_clear=0.25):
    """Lattice of floor points inside the maze, away from walls: the area a complete search must look at."""
    lo, hi = maze.bounds()
    xs = np.arange(lo + spacing / 2, hi, spacing)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    d2 = np.full(len(pts), np.inf)
    for x0, y0, x1, y1 in maze.wall_segments():
        a, ab = np.array([x0, y0]), np.array([x1 - x0, y1 - y0])
        t = np.clip(((pts - a) @ ab) / (ab @ ab), 0, 1)
        d2 = np.minimum(d2, ((pts - (a + t[:, None] * ab)) ** 2).sum(1))
    return pts[np.sqrt(d2) > wall_clear]


def run_mission(grid_n=3, seed=0, use_gui=False, use_dashboard=False, max_minutes=8.0,
                num_survivors=4, imu_noise_scale=1.0, use_detector=True, dashboard_hz=1.0,
                save_final_frame=None, verbose=True, detector_path=None, detector_use_cnn=True,
                controller="path"):
    client = p.connect(p.GUI if use_gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    dt = 1.0 / 240.0
    p.setTimeStep(dt)
    if use_gui:
        p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-70,
                                     cameraTargetPosition=[0, 0, 0])

    maze = MazeGenerator(grid_n=grid_n, seed=seed)
    maze.build_in_pybullet(client)
    entry_xy, exit_xy = maze.entry_world_xy(), maze.exit_world_xy()

    drone = Drone(start_xyz=[entry_xy[0], entry_xy[1], C.TAKEOFF_ALT])
    lidar, camera = Lidar2D(), RGBDCamera()
    imu = NoisyIMU(noise_scale=imu_noise_scale, seed=seed)
    altsensor = AltitudeSensor(seed=seed)

    half_size = (grid_n * maze.cell_pitch) / 2.0 + 1.5
    occ_map = OccupancyGridMap(half_size=half_size, resolution=C.MAP_RESOLUTION)
    gt_cov = OccupancyGridMap(half_size=half_size, resolution=C.MAP_RESOLUTION)        # scoring only

    estimator = StateEstimator(start_xy=entry_xy, start_yaw=0.0, bound_half_size=half_size, seed=seed)
    nav = Navigator(occ_map, exit_xy, room_centers=maze.all_room_centers())
    survivors = SurvivorManager(maze.all_room_centers(), num_survivors=num_survivors, seed=seed,
                                keep_clear=[entry_xy, exit_xy])
    detector = None
    if use_detector:
        from vision.detector import SurvivorDetector
        detector = SurvivorDetector(occ_map, path=detector_path, use_cnn=detector_use_cnn)
    rl = None
    if controller == "rl":
        from rl.rl_controller import RLController
        rl = RLController(grid_n=grid_n, cell_pitch=maze.cell_pitch)
    dashboard = None
    if use_dashboard:
        from maze_env.gcs import GCSDashboard
        dashboard = GCSDashboard(occ_map, maze)
    dash_every = max(1, int(round(1.0 / dt / dashboard_hz))) if dashboard_hz > 0 else None

    ctrl_every, det_every = 4, 24                 # 60 Hz control, 10 Hz camera
    dt_c = dt * ctrl_every
    max_time = max_minutes * 60.0
    max_steps = int(max_time / dt)

    phase = "TAKEOFF"
    stable_alt = 0
    cyc = 0
    v_prev_est = np.zeros(2)
    cmd_body, cmd_vz, cmd_yr = np.zeros(2), 0.0, 0.0
    min_clear = 5.0
    angles = lidar.angles
    ranges = np.full(len(angles), lidar.max_range)
    detections = []
    trajectory = []                                   # (t, x, y, yaw) at 10 Hz, ground truth, for analysis only
    last_cam_rgb = np.zeros((C.CAM_H, C.CAM_W, 3), dtype=np.uint8)

    # scoring / diagnostics (ground truth, never fed back)
    collision_ticks = 0
    max_tilt = 0.0
    err_log, speed_log, tilt_log = [], [], []
    imu_vel_acc, imu_yr_acc, imu_n = np.zeros(3), 0.0, 0
    path_len, prev_pos_gt = 0.0, None
    moving_ticks = hover_ticks = 0
    t_explore_done = t_exit = t_done = None

    if verbose:
        print(f"[MISSION] grid {grid_n}x{grid_n} seed {seed} | entry {np.round(entry_xy, 1)} "
              f"exit {np.round(exit_xy, 1)} | {len(survivors)} survivors")

    _COVLOG = []
    step = 0
    while step < max_steps:
        pose = drone.get_ground_truth_pose()
        pos_gt, yaw_gt, vel_gt, yr_world, roll, pitch = pose[:6]
        yr_gt = pose[8]                       # heading rate as a body gyro + attitude estimate gives it
        t = step * dt

        # the IMU is sampled every physics tick and averaged over the control interval, like a real
        # IMU that runs faster than the estimator (sampling once per cycle aliased a small yaw
        # ripple into a 3% yaw-rate bias)
        imu_vel_acc += np.array([*(rot(-yaw_gt) @ vel_gt[:2]), vel_gt[2]])
        imu_yr_acc += yr_gt
        imu_n += 1

        if step % ctrl_every == 0:
            cyc += 1
            sensor_pos = pos_gt + np.array([0, 0, 0.03])
            angles, ranges = lidar.scan(sensor_pos, yaw_gt)
            alt = altsensor.read(pos_gt[2])
            noisy_vel, noisy_yr = imu.sample(imu_vel_acc / imu_n, imu_yr_acc / imu_n, dt_c)
            imu_vel_acc, imu_yr_acc, imu_n = np.zeros(3), 0.0, 0
            estimator.predict(noisy_vel[:2], noisy_yr, dt_c)
            pos_e, yaw_e = estimator.pos, estimator.yaw

            if alt > C.CRUISE_ALT - 0.3:                       # map only once at flight altitude
                if cyc % 2 == 0:
                    estimator.correct(occ_map, angles, ranges, lidar.max_range)
                    pos_e, yaw_e = estimator.pos, estimator.yaw
                    occ_map.update(pos_e, yaw_e, angles, ranges, lidar.max_range)
                    occ_map.update_coverage(pos_e, yaw_e, angles, ranges)
                    gt_cov.update_coverage(pos_gt[:2], yaw_gt, angles, ranges)     # scoring only
            min_clear = float(ranges.min())

            # ---------------- phase logic ----------------
            spin = 0.0
            if phase == "TAKEOFF":
                cmd_body = np.zeros(2)
                cmd_vz = float(np.clip(1.5 * (C.CRUISE_ALT - alt), -0.3, 0.7))
                stable_alt = stable_alt + 1 if alt >= C.CRUISE_ALT - 0.08 else 0
                if stable_alt >= 30:
                    phase = "EXPLORE"
                    if verbose:
                        print(f"[PHASE] EXPLORE at t={t:.1f}s")
            elif phase in ("EXPLORE", "RETURN_TO_EXIT"):
                spin = C.SCAN_SPIN_RATE
                # Corridor-axis alignment: continuous free-spin only glimpses a given corridor floor
                # point at whatever bearing the spin happens to be at while passing through -- measured
                # across a real run, the closest it ever got was 32-37 deg off-axis, right at a grazing
                # angle where a nearby wall corner blocks that specific ray. When lidar shows close walls
                # on both left and right (the signature of being inside a 1m corridor), briefly stop
                # free-spinning and turn to look straight down the corridor's own long axis instead --
                # guaranteed dead-centre view, rather than hoping the free spin lands there by chance.
                i_left = int(np.argmin(np.abs(angles - np.pi / 2)))
                i_right = int(np.argmin(np.abs(angles + np.pi / 2)))
                in_corridor = ranges[i_left] < 0.75 and ranges[i_right] < 0.75
                if in_corridor:
                    i_fwd = int(np.argmin(np.abs(angles)))
                    i_back = int(np.argmin(np.abs(np.abs(angles) - np.pi)))
                    axis_offset = 0.0 if ranges[i_fwd] >= ranges[i_back] else np.pi
                    yaw_err = np.arctan2(np.sin(axis_offset), np.cos(axis_offset))
                    spin = float(np.clip(1.5 * yaw_err, -C.SCAN_SPIN_RATE, C.SCAN_SPIN_RATE))
                v_des, status = nav.command(t, pos_e, min_clear)
                if rl is not None and nav.path is not None and np.linalg.norm(v_des) > 1e-3:
                    tgt, _ = nav._pursuit_target(pos_e)
                    out = rl.desired_direction(angles, ranges, pos_e, yaw_e, tgt) if tgt is not None else None
                    if out is not None and out[1] > 0.05:            # policy picks the DIRECTION, planner the speed
                        v_des = out[0] * np.linalg.norm(v_des)
                if phase == "EXPLORE":
                    time_left = max_time - t
                    # exit_cost is a ROUTING cost (soft-penalised up to 4x near walls, exactly where
                    # this maze's 1m corridors force the drone to fly), not seconds -- comparing it to
                    # time_left directly made the guard fire far too early in a corridor-heavy maze,
                    # abandoning rooms that were still genuinely reachable in the time left. Convert it
                    # to an actual time estimate using the drone's real cruise speed, with a wide margin.
                    est_return_s = nav.exit_cost / (1.6 * C.MAX_SPEED) if np.isfinite(nav.exit_cost) else float("inf")
                    if nav.explored_out or time_left < est_return_s + 25.0:
                        why = "everything explored and seen" if nav.explored_out else "time guard"
                        if verbose:
                            print(f"[PHASE] RETURN_TO_EXIT at t={t:.1f}s ({why})")
                        phase, t_explore_done = "RETURN_TO_EXIT", t
                        nav.start_exit()
                elif np.linalg.norm(pos_e - exit_xy) < 0.5:
                    phase, t_exit = "LAND", t
                    if verbose:
                        print(f"[PHASE] LAND at t={t:.1f}s (estimated position at exit)")
                v_ctl = slew_limit(v_prev_est, v_des, dt_c)
                v_body, _ = safety_govern(rot(-yaw_e) @ v_ctl, angles, ranges)
                v_prev_est = v_ctl            # governor output is NOT fed back (it would accumulate)
                cmd_body = v_body
                cmd_vz = float(np.clip(1.5 * (C.CRUISE_ALT - alt), -0.5, 0.8))
            if phase == "LAND":
                cmd_body = np.zeros(2)
                v_prev_est = np.zeros(2)
                cmd_vz = -0.35
                if alt < 0.22:
                    phase, t_done = "DONE", t
                    if verbose:
                        print(f"[PHASE] DONE at t={t:.1f}s")
            cmd_yr = spin

            # ---------------- detection (onboard, 10 Hz) ----------------
            if detector is not None and phase in ("EXPLORE", "RETURN_TO_EXIT") and cyc % (det_every // ctrl_every) == 0:
                rgb, depth = camera.capture(sensor_pos, yaw_gt)
                trajectory.append((t, float(pos_gt[0]), float(pos_gt[1]), float(yaw_gt)))
                last_cam_rgb = rgb
                for d in detector.process(rgb, depth, pos_e, yaw_e, t):
                    detections.append(d)
                    if verbose:
                        print(f"[DETECT] survivor #{len(detections)} at grid box {d['label']} "
                              f"(est. xy {np.round(d['xy'], 2)}, t={t:.1f}s)")

            if dashboard is not None and dash_every and step % dash_every < ctrl_every:
                dashboard.update(pos_e, yaw_e, pos_gt[:2], survivors, detections, last_cam_rgb, t, phase)

            # diagnostics
            err_log.append(np.linalg.norm(pos_e - pos_gt[:2]))
            if phase in ("EXPLORE", "RETURN_TO_EXIT"):
                sp = float(np.linalg.norm(vel_gt[:2]))
                speed_log.append(sp)
                moving_ticks += 1
                hover_ticks += sp < 0.1

            if __import__('os').environ.get('COVTRACE'):
                _COVLOG.append((t, float(pos_gt[0]), float(pos_gt[1]), float(yaw_gt), min_clear))
            if verbose and step % int(10 / dt) == 0 and step > 0:
                print(f"[HEARTBEAT] t={t:5.0f}s {phase:14s} found {len(detections)}/{len(survivors)} "
                      f"cover={occ_map.seen.sum():5d} est_err={err_log[-1]:.2f}m mode={nav.mode} stuck={nav.stuck_events}")

        if phase == "DONE":
            break

        drone.command_velocity_body(cmd_body[0], cmd_body[1], cmd_vz, cmd_yr, pose=pose)

        contacts = p.getContactPoints(bodyA=drone.body) or ()      # GUI mode returns None when nothing touches
        hit_ids = [c[2] for c in contacts if c[2] != maze.floor_id and c[2] != maze.ceiling_id]
        if hit_ids:
            collision_ticks += 1
            if collision_ticks == 1 and verbose:
                print(f"[COLLISION] first contact at t={t:.1f}s phase={phase} gt={np.round(pos_gt[:2],2)} "
                      f"est={np.round(estimator.pos,2)} vel={np.round(vel_gt[:2],2)} cmd={np.round(cmd_body,2)} "
                      f"min_range={min_clear:.2f} tilt=({np.degrees(roll):.0f},{np.degrees(pitch):.0f}) body={hit_ids[0]} at {np.round(p.getBasePositionAndOrientation(hit_ids[0])[0],2)} survivors={[int(b) for b in survivors.body_ids]} surv_xy={np.round(survivors.xy,1).tolist()}")
        if phase not in ("TAKEOFF", "LAND"):
            max_tilt = max(max_tilt, abs(roll), abs(pitch))
        if prev_pos_gt is not None and phase in ("EXPLORE", "RETURN_TO_EXIT"):
            path_len += float(np.linalg.norm(pos_gt[:2] - prev_pos_gt))
        prev_pos_gt = pos_gt[:2].copy()

        p.stepSimulation()
        step += 1

    elapsed = step * dt
    pos_gt, *_ = drone.get_ground_truth_pose()
    at_exit = bool(np.linalg.norm(pos_gt[:2] - exit_xy) < 0.7 and pos_gt[2] < 0.4)

    free_pts = ground_truth_free_points(maze)
    gx = np.clip(((free_pts[:, 0] + half_size) / C.MAP_RESOLUTION).astype(int), 0, gt_cov.n - 1)
    gy = np.clip(((free_pts[:, 1] + half_size) / C.MAP_RESOLUTION).astype(int), 0, gt_cov.n - 1)
    coverage = float(gt_cov.seen[gx, gy].mean())

    score = survivors.score(detections)
    res = {
        "elapsed": elapsed, "phase": phase, "completed": phase == "DONE" and at_exit,
        "found": score["found"], "total": score["total"], "false_positives": score["false_positives"],
        "mean_loc_error_m": score["mean_loc_error_m"], "detections": detections,
        "coverage_gt": coverage,
        "mean_drift": float(np.mean(err_log)) if err_log else 0.0,
        "max_drift": float(np.max(err_log)) if err_log else 0.0,
        "collision_ticks": collision_ticks, "total_ticks": step,
        "max_tilt_deg": float(np.degrees(max_tilt)),
        "hover_fraction": hover_ticks / max(moving_ticks, 1),
        "mean_speed": float(np.mean(speed_log)) if speed_log else 0.0,
        "path_len_m": path_len, "stuck_events": nav.stuck_events,
        "t_explore_done": t_explore_done, "t_exit": t_exit,
        "missed_xy": [survivors.xy[i].tolist() for i in range(len(survivors)) if i not in score["found_ids"]],
        "detector_stats": dict(detector.stats) if detector is not None else {},
        "trajectory": trajectory,
    }
    if __import__('os').environ.get('COVTRACE'):
        import numpy as _np
        _np.save('/tmp/covlog.npy', _np.array(_COVLOG))
    if verbose:
        print(f"\n[RESULT] phase={phase} completed={res['completed']} t={elapsed:.0f}s | survivors "
              f"{score['found']}/{score['total']} (false alarms {score['false_positives']}, "
              f"loc err {score['mean_loc_error_m']:.2f} m) | floor viewed {coverage:.0%}")
        print(f"[FLIGHT] collisions {collision_ticks} ticks | max tilt {res['max_tilt_deg']:.1f} deg | "
              f"hover {res['hover_fraction']:.1%} | mean speed {res['mean_speed']:.2f} m/s | "
              f"path {path_len:.0f} m | stuck events {nav.stuck_events}")
        print(f"[LOCALIZATION] mean error {res['mean_drift']:.2f} m, max {res['max_drift']:.2f} m")
    if dashboard is not None:
        if save_final_frame:
            dashboard.save_frame(save_final_frame)
        dashboard.close()
    p.disconnect()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_n", type=int, default=None, help="rooms per side (3-5); random if omitted")
    ap.add_argument("--seed", type=int, default=None, help="maze/survivor seed; random if omitted")
    ap.add_argument("--num_survivors", type=int, default=None)
    ap.add_argument("--max_minutes", type=float, default=8.0)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--dashboard", action="store_true")
    ap.add_argument("--save_final_frame", type=str, default=None)
    ap.add_argument("--imu_noise_scale", type=float, default=1.0, help="1=default sensor, 5=drifty cheap IMU")
    ap.add_argument("--no_detector", action="store_true")
    ap.add_argument("--controller", choices=["path", "rl"], default="path",
                    help="local steering: clearance-aware path follower, or the ES-trained RL policy choosing the direction")
    ap.add_argument("--no_cnn", action="store_true", help="ablation: skip the CNN check, trust the colour proposal + size check")
    args = ap.parse_args()
    seed = args.seed if args.seed is not None else int(np.random.randint(0, 1_000_000))
    rc = np.random.default_rng(seed)
    grid_n = args.grid_n if args.grid_n is not None else int(rc.integers(3, 5))
    n_surv = args.num_survivors if args.num_survivors is not None else int(rc.integers(3, 7))
    print(f"[CONFIG] --seed {seed} --grid_n {grid_n} --num_survivors {n_surv}  (rerun with these to reproduce)")
    try:
        run_mission(grid_n=grid_n, seed=seed, use_gui=args.gui, use_dashboard=args.dashboard,
                    max_minutes=args.max_minutes, num_survivors=n_surv, imu_noise_scale=args.imu_noise_scale,
                    use_detector=not args.no_detector, save_final_frame=args.save_final_frame)
    except Exception:
        print("\n[FATAL ERROR]\n", file=sys.stderr)
        traceback.print_exc()
        try:
            p.disconnect()
        except Exception:
            pass
        sys.exit(1)
