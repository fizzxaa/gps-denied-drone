# GPS-denied indoor search-and-rescue drone (PyBullet simulation)

A simulated quadrotor that takes off at the maze entry, explores an unknown maze using only lidar, camera and
IMU-style sensors, looks at every part of the floor, reports survivors with a map position and grid-box label,
returns to the exit and lands. It does not hover in place, and it does not stall at walls.

```
python main.py --grid_n 4 --seed 1                              # headless, prints a mission log
python main.py --grid_n 4 --seed 1 --dashboard --save_final_frame result.png
python main.py --grid_n 3 --seed 1 --gui                        # PyBullet 3D window (needs a display)
python main.py --grid_n 4 --seed 1 --controller rl              # RL policy picks the steering direction
python main.py --grid_n 4 --seed 1 --imu_noise_scale 4          # much worse IMU
python evaluate.py --grids 3 4 5 --seeds 3300 3301 3302         # many missions, mean +- 95% CI
```
`pip install -r requirements.txt` (pybullet, numpy, matplotlib; nothing else).

## Measured results
Full missions (takeoff -> explore -> exit -> land) on maze seeds 3300+ that were never used while developing.
Every metric is scored against ground truth the flight code never sees.

These numbers are from the FIRST maze version (uniform 3m open cells -- an easier maze than the
brief specifies; see "History" below). They are kept here for the record but are NOT what the
current code builds.

| Setup | Runs | Completed | Survivors found | False alarms | Wall contact | Floor viewed | Mission time |
|---|---|---|---|---|---|---|---|
| 3x3, wide-cell maze | 8 | 8/8 | 32/32 | 0 | 0/8 | 99.9% | 59 s |
| 4x4, wide-cell maze | 8 | 8/8 | 32/32 | 0 | 0/8 | 100% | 102 s |
| 5x5, wide-cell maze | 8 | 8/8 | 32/32 | 0 | 0/8 | 100% | 158 s |

## Current results: real 2m rooms, 1m corridors (matches the brief)
9 fresh 3x3-maze seeds and 2 fresh 4x4-maze seeds, none used while developing.

| Maze | Runs | Completed | Survivors found | False alarms | Wall contact | Floor viewed* | Mission time |
|---|---|---|---|---|---|---|---|
| 3x3 | 9 | 9/9 | 27/27 | 0 | 0/9 | 76-85% | 148-220 s |
| 4x4 | 2 | 2/2 | 8/8 | 0 | 0/2 | 68-71% | 302-353 s |

\* "Floor viewed" is scored against every square of true floor, including corridors. Broken down for
one traced run: **rooms 100% covered, corridors 63% covered.** Survivors are only ever placed in
rooms (see maze_env/survivors.py), so this gap does not cost detections in any run measured so far.

**Root cause of the corridor gap, found and confirmed with data:** the drone spins continuously to
scan 360 degrees. Across one full mission, a specific unseen corridor floor point was within camera
range and nominal field-of-view on 173 separate close passes -- but the closest bearing it ever
actually achieved was 32-37 degrees off dead-centre, right at the edge of the field of view, where a
nearby wall corner blocks that specific ray even though a ray aimed straight at the point (confirmed
in isolation) reaches it cleanly. Free-spinning only glimpses a given corridor point at whatever
bearing the spin happens to be at while passing through; it is not guided toward looking straight
down a corridor it has not scanned yet.

Tried: aligning the camera to the corridor's own axis whenever lidar shows close walls on both left
and right (maze_env's spin override in main.py). Measured result: no improvement (63% before and
after, tested on the same seed). Cause: that detector only fires once the drone is already facing
roughly along the corridor -- it reinforces an orientation the spin was passing through anyway,
rather than actively turning toward a corridor it has not looked down yet. A real fix needs the
drone to actively aim at a KNOWN-but-unseen corridor direction (available from the map, since a
corridor's doorway position is known as soon as its walls are), not just react to instantaneous
lidar symmetry. Not yet built.

Position-estimate error 5-13 cm (mean), survivor position error 8-14 cm, max tilt 22 deg (the controller limit).
"Completed 8/8" only proves the true rate is above ~68% (95% interval); it does not prove 100%.
Raw per-run rows are in `results/`.

## What was broken and what fixed it (each was measured)
| Symptom | Cause | Fix |
|---|---|---|
| Drone flipped upside down when yawing while moving | attitude damping used world-frame rates | body-frame rates |
| Lidar reported walls at 0 m, drone made wild moves | lidar rays hit the drone's own tilted body | drone in its own collision group |
| Drone stalled at walls | A* had no clearance (median 8 cm from walls) and the safety filter then blocked motion | wall inflation + clearance cost, path-distance frontier choice, smooth speed governor |
| Position estimate jumped metres | process noise 30x too big; scan-match rewarded poses that saw the *fewest* mapped walls; gyro sampled at the same phase as a yaw ripple; wrong rotation rate when tilted | dt-scaled noise, likelihood-field scoring, averaged IMU, correct heading rate |
| Drone flipped between two routes forever | long near-parallel lidar rays erased real walls in the map (phantom gap) | never free cells next to known walls |
| Explored for minutes after covering everything | un-observed sliver beside every wall looked like a frontier | ignore unknown cells hugging walls |
| "Detected" a survivor at t=0 behind a wall | detection gate used true survivor positions; nearest survivor credited on any CNN yes | onboard pipeline: colour proposals -> CNN -> depth localisation -> multi-frame confirmation |
| Survivors seen from 5 m were dropped | 4.5 m detection cap; coverage counted a one-frame flash as "viewed" | detect to 6 m, count coverage only within 4 m |
| Landed on a survivor | keep-clear rule could never be met inside the exit room | markers placed off the landing spot |
| RL "held-out" mazes were training mazes | a 3x3 grid has only ~100 distinct layouts; splitting by seed does not separate layouts | split by layout hash; policy retrained from scratch |
| `--env pybullet` crashed | unpacked 4 values from an 8-value pose | fixed |
| `--gui` mode crashed on contact check | `getContactPoints` returns `None`, not `[]`, when nothing touches, but only in GUI mode | treat `None` as no contact |
| Maze didn't match the brief | every "cell" was one uniform 3m open square, not a 2m room + 1m corridor | real room/corridor geometry, wall segments rebuilt from scratch |
| Explored for minutes then quit early in the new maze | "time to return" compared an inflated routing cost directly to seconds, with no unit conversion | convert to seconds using cruise speed |
| Wild tilting (32 deg) and stalling in 1m corridors | the wall-push safety term activated at exactly the wall distance of a centred drone in a 1m corridor, so any wobble triggered a push-overshoot-push loop | narrowed the push trigger distance below a corridor's half-width |
| Some rooms silently skipped, floor coverage stuck near 50% | a room could be marked "visited" by one lucky lidar ray touching its centre cell, without the drone ever actually flying in | require most of a room's floor, sampled at multiple points, to be seen before moving on |
| Floor-coverage score capped below 100% even with perfect exploration | `bounds()` still used the old (wider) maze's formula, counting space outside the real walls as floor to be viewed | corrected to the true wall extent |

## What the flight code may and may not use
Allowed: lidar, camera (RGB + depth), IMU-style velocity/yaw-rate, altitude sensor, and its own particle-filter pose.
Velocity commands are body-frame, like a real flight controller. The true pose is used only to (a) generate sensor
readings, (b) run the drone's inner attitude/velocity loop, (c) score the run. The start pose is assumed known
(it defines the map frame).

## Honest limits
- **Simulation only.** The survivor is a bright yellow capsule; the CNN's perfect score on held-out crops (1145 crops from unseen maze layouts: 483/483 survivors and 662/662
  non-survivors correct) says nothing about real people. The colour proposal stage does much of the work here; the CNN's measured job is rejecting look-alike blobs (crates).
- **The IMU model outputs velocity with small noise** (like good visual odometry, not a raw accelerometer). Constant
  biases are included and the scan-matcher removes the drift; a real MEMS IMU would be much worse.
- **Corridor floor coverage (~63% in the one traced run) is lower than room coverage (100%).** Corridors are flown
  through quickly rather than lingered in, and are never where a survivor is placed, so this has not cost a
  detection in any run measured so far -- but it is a real, open gap in the coverage metric.
- **Localisation** is one shared map with a particle filter, not FastSLAM, and there is no loop closure.
- **Layout overlap in the mission tests:** a 3x3 grid has ~100 distinct layouts, so "new seeds" can rebuild a maze seen during
  development. The RL and CNN evaluations are split by layout; the mission evaluations are not.
- **Small samples:** 8 runs per maze size. The x4-noise runs missed 2 of 40 survivors; the cause was not investigated.
- **The RL policy** succeeds on 52% of episodes on unseen layouts, 88% when the goal is in line of sight but only 32% when a
  wall is in the way (n=300, `python -m rl.evaluate_policy`). It cannot plan around walls, so in the mission it only picks the
  steering *direction* between waypoints the planner already made line-of-sight; the planner sets the speed.
- The mission takes 1-2.6 minutes of simulated time and runs about 5x faster than real time on one CPU core.
- Some flight is deliberately not "hover-free": the drone spins slowly (0.7 rad/s) so the camera sweeps 360 degrees.

## Files
```
main.py                  mission loop and CLI            evaluate.py   many-mission evaluation with CIs
maze_env/config.py       every shared constant (camera, altitude, ranges)
maze_env/maze_generator  maze + layout signature/split   maze_env/drone.py    quadrotor physics + controller
maze_env/sensors.py      lidar, RGB-D camera, IMU, altimeter
maze_env/mapping.py      occupancy grid, coverage grid, particle-filter estimator
maze_env/planning.py     clearance-aware Dijkstra, frontiers, unseen floor
maze_env/navigation.py   goal logic, path following, stall recovery, safety governor
maze_env/survivors.py    ground-truth markers (scoring only)      maze_env/gcs.py   dashboard
vision/                  proposals, dataset generator (segmentation labels), CNN (numpy), trainer, detector
rl/                      ES-trained steering policy, environments, held-out evaluator
results/                 raw CSVs from the evaluations above
```
Retrain the CNN: `python -m vision.generate_dataset --split train --frames 2800 --out vision/dataset_train.npz`,
same with `--split val --frames 350 --out vision/dataset_val.npz`, then `python -m vision.train_cnn`.
Retrain the policy: `python -m rl.train_es --generations 160 --population 48`.
