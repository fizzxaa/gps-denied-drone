"""
Generates labelled 32x32 RGB crops for the survivor verifier, from real generated mazes.

Labels come from the renderer's own SEGMENTATION MASK (which pixels belong to a survivor), not from
a hand-written field-of-view + raycast rule. (The old labeller used a 70 deg horizontal FOV while the
camera really sees 86 deg, so visible survivors were labelled "absent".) Crops are produced by the
SAME proposal code the flight detector uses (vision/proposals.py), so training and flight agree.

Train and validation mazes are split by LAYOUT (MazeGenerator.layout_split), never by seed: a 3x3
grid only has ~100 distinct layouts, so two different seeds often build the identical maze.

Usage:
    python -m vision.generate_dataset --split train --frames 1400 --out vision/dataset_train.npz
    python -m vision.generate_dataset --split val   --frames 400  --out vision/dataset_val.npz
"""

import argparse
import numpy as np
import pybullet as p
import pybullet_data

from maze_env import config as C
from maze_env.maze_generator import MazeGenerator
from maze_env.sensors import RGBDCamera
from maze_env.survivors import SurvivorManager
from vision.proposals import warm_mask, connected_blobs, crop_box


def generate(split, frames, seed=0, out_path=None, frames_per_maze=25):
    rng = np.random.default_rng(seed)
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    cam = RGBDCamera()
    crops, labels, layouts, dists = [], [], [], []
    maze_seed = int(rng.integers(0, 10_000_000))
    frames_done = 0
    while frames_done < frames:
        maze_seed += 1
        grid_n = int(rng.choice([3, 4, 5], p=[0.3, 0.4, 0.3]))
        maze = MazeGenerator(grid_n=grid_n, seed=maze_seed)
        if maze.layout_split() != split:
            continue
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        maze.build_in_pybullet(client)
        rooms = maze.all_room_centers()
        surv = SurvivorManager(rooms, num_survivors=int(rng.integers(2, 7)), seed=int(rng.integers(1 << 30)))
        surv_ids = set(surv.body_ids)
        layout_id = int(maze.layout_signature()[:8], 16)
        for f in range(frames_per_maze):
            if frames_done >= frames:
                break
            if f % 2 == 0:                                   # aimed near a survivor
                tgt = surv.xy[rng.integers(len(surv))]
                off = rng.uniform(0.8, C.DETECT_RANGE) * np.array([np.cos(a := rng.uniform(-np.pi, np.pi)), np.sin(a)])
                xy = tgt + off
                yaw = np.arctan2(tgt[1] - xy[1], tgt[0] - xy[0]) + rng.normal(0, 0.5)
            else:                                            # anywhere
                xy = rooms[rng.integers(len(rooms))] + rng.uniform(-1.2, 1.2, 2)
                yaw = rng.uniform(-np.pi, np.pi)
            lo, hi = maze.bounds()
            if not (lo + 0.3 < xy[0] < hi - 0.3 and lo + 0.3 < xy[1] < hi - 0.3):
                continue
            pos = np.array([xy[0], xy[1], C.CRUISE_ALT + rng.normal(0, 0.03)])
            rgb, z, seg = cam.capture(pos, yaw, with_segmentation=True)
            frames_done += 1
            body = seg & 0xFFFFFF
            surv_mask = np.isin(body, list(surv_ids))
            blobs = connected_blobs(warm_mask(rgb), min_area=10, depth=z)
            for b in blobs:                                   # the same proposals the drone will see
                pix = b["pix"]
                frac = surv_mask[pix[:, 0], pix[:, 1]].mean()
                lab = 1 if frac >= 0.6 else (0 if frac <= 0.1 else None)
                if lab is None:
                    continue
                boxes = [b["box"]]
                if lab == 1:                                  # jittered copies of positives
                    x0, y0, x1, y1 = b["box"]
                    w, h = x1 - x0 + 1, y1 - y0 + 1
                    for _ in range(2):
                        dx, dy = rng.uniform(-0.15, 0.15) * w, rng.uniform(-0.1, 0.1) * h
                        s = rng.uniform(0.85, 1.2)
                        boxes.append((int(x0 + dx - (s - 1) * w / 2), int(y0 + dy - (s - 1) * h / 2),
                                      int(x1 + dx + (s - 1) * w / 2), int(y1 + dy + (s - 1) * h / 2)))
                zz = float(np.median(z[pix[:, 0], pix[:, 1]]))
                for bx in boxes:
                    crops.append(crop_box(rgb, bx)); labels.append(lab); layouts.append(layout_id); dists.append(zz)
            for _ in range(2):                                # random background negatives
                w, h = int(rng.integers(6, 40)), int(rng.integers(15, 80))
                x0, y0 = int(rng.integers(0, C.CAM_W - w)), int(rng.integers(0, C.CAM_H - h))
                if surv_mask[y0:y0 + h, x0:x0 + w].mean() < 0.02:
                    crops.append(crop_box(rgb, (x0, y0, x0 + w, y0 + h)))
                    labels.append(0); layouts.append(layout_id); dists.append(float(np.median(z[y0:y0 + h, x0:x0 + w])))
    p.disconnect()
    crops = np.array(crops, dtype=np.uint8); labels = np.array(labels, dtype=np.float32)
    print(f"[{split}] {frames_done} frames -> {len(labels)} crops: {int(labels.sum())} positive, "
          f"{int(len(labels) - labels.sum())} negative, {len(set(layouts))} distinct maze layouts")
    if out_path:
        np.savez_compressed(out_path, images=crops, labels=labels, layouts=np.array(layouts, dtype=np.int64),
                            dists=np.array(dists, dtype=np.float32))
        print("saved", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], required=True)
    ap.add_argument("--frames", type=int, default=1400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()
    generate(a.split, a.frames, a.seed, a.out)
