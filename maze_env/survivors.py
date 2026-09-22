"""
Survivor markers in the world (GROUND TRUTH, for spawning and SCORING only).

The mission never reads this list to navigate or to decide what it has found. Detection happens
onboard in vision/detector.py from camera frames; `score()` then compares what the drone
reported against where the markers really are.
"""

import numpy as np
import pybullet as p

from maze_env import config as C


class SurvivorManager:
    def __init__(self, room_centers, num_survivors=4, seed=None, keep_clear=None, clear_radius=0.45):
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(room_centers), size=min(num_survivors, len(room_centers)), replace=False)
        self.xy = []
        self.body_ids = []
        # A fallback placement radius of clear_radius+0.2 used to be allowed to exceed the room's own
        # half-width (1.0m for a 2m room) -- confirmed spawning a survivor 0.17m PAST the room's own
        # wall, in dead space the drone can never reach. max_rad keeps every fallback placement
        # strictly inside the room, with margin, regardless of how clear_radius is set.
        max_rad = C.ROOM_HALF - 0.15
        for idx in chosen:
            cx, cy = room_centers[idx]
            ok = lambda q: keep_clear is None or all(np.linalg.norm(q - np.asarray(k)) >= clear_radius for k in keep_clear)
            xy = np.array([cx, cy]) + rng.uniform(-0.6, 0.6, 2)
            for _ in range(20):
                if ok(xy):
                    break
                xy = np.array([cx, cy]) + rng.uniform(-0.6, 0.6, 2)
            if not ok(xy):      # entry / exit room: stand the marker off to the side of the launch/landing spot
                for _ in range(50):
                    ang, rad = rng.uniform(0, 2 * np.pi), rng.uniform(min(clear_radius, max_rad), max_rad)
                    xy = np.array([cx, cy]) + rad * np.array([np.cos(ang), np.sin(ang)])
                    if ok(xy) and np.all(np.abs(xy - [cx, cy]) < max_rad):
                        break
            self.xy.append(xy)
            self.body_ids.append(self._spawn_marker(xy))
        self.xy = np.array(self.xy)

    @staticmethod
    def _spawn_marker(xy, height=C.SURVIVOR_HEIGHT, radius=C.SURVIVOR_RADIUS):
        col = p.createCollisionShape(p.GEOM_CAPSULE, radius=radius, height=height)
        vis = p.createVisualShape(p.GEOM_CAPSULE, radius=radius, length=height,
                                  rgbaColor=[0.95, 0.75, 0.1, 1.0])
        return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                 basePosition=[xy[0], xy[1], height / 2])

    def __len__(self):
        return len(self.xy)

    def score(self, detections, match_radius=1.0):
        """detections: list of dicts with 'xy' in the map frame. One-to-one nearest matching."""
        used, errs, found = set(), [], set()
        for d in detections:
            best, bd = None, match_radius
            for i, s in enumerate(self.xy):
                if i in used:
                    continue
                dist = float(np.linalg.norm(np.asarray(d["xy"]) - s))
                if dist < bd:
                    best, bd = i, dist
            if best is not None:
                used.add(best)
                found.add(best)
                errs.append(bd)
        return {"found": len(found), "total": len(self.xy),
                "false_positives": len(detections) - len(found),
                "mean_loc_error_m": float(np.mean(errs)) if errs else float("nan"),
                "found_ids": sorted(found)}
