"""
Onboard survivor detector. Uses ONLY what the drone has: an RGB-D frame and its own pose estimate.
It never sees where the survivors really are.

Pipeline per camera frame (10 Hz):
  1. PROPOSE  warm-coloured blobs (deliberately loose: brown crates are proposed too).
  2. CHECK    physical size from depth: a 0.36 m-wide person must not appear 3 m wide.
  3. VERIFY   the trained CNN looks at a 32x32 RGB crop and says survivor / not survivor.
  4. LOCATE   depth + pixel position + the estimated pose -> (x, y) in the map frame.
  5. CONFIRM  a survivor is only reported once several frames agree on roughly the same spot,
              so one bad frame cannot create a detection.
"""

import os
import numpy as np

from maze_env import config as C
from vision.cnn import SurvivorCNN
from vision.proposals import warm_mask, connected_blobs, crop_box

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "trained_cnn.npz")


class SurvivorDetector:
    def __init__(self, occ_map, path=None, prob_thresh=0.8, confirm_hits=4, min_span_s=0.3,
                 assoc_radius=1.0, use_cnn=True):
        path = path or _DEFAULT_PATH
        self.use_cnn = use_cnn
        if use_cnn:
            if not os.path.exists(path):
                raise FileNotFoundError(f"No trained CNN at {path}. Run `python -m vision.generate_dataset` "
                                        f"(train + val) and `python -m vision.train_cnn` first.")
            self.model = SurvivorCNN.load(path)
        self.map = occ_map
        self.prob_thresh = prob_thresh
        self.confirm_hits = confirm_hits
        self.min_span_s = min_span_s
        self.assoc_radius = assoc_radius
        self.tracks = []
        self.confirmed = []          # dicts: xy, label, t, n_hits, conf  (xy keeps refining in place)
        self.stats = {"frames": 0, "proposals": 0, "rejected_size": 0, "rejected_cnn": 0, "accepted": 0}

    # ------------------------------------------------------------------
    def _locate(self, pix, z, est_pos, est_yaw):
        H, W = C.CAM_H, C.CAM_W
        f = C.CAM_FOCAL_PX
        u = pix[:, 1].mean() + 0.5
        v = pix[:, 0].mean() + 0.5
        fwd = np.array([np.cos(est_yaw), np.sin(est_yaw), C.CAM_TILT])
        fwd /= np.linalg.norm(fwd)
        right = np.array([np.sin(est_yaw), -np.cos(est_yaw), 0.0])
        up = np.cross(right, fwd)
        x_cam = (u - W / 2) / f * z
        y_up = -(v - H / 2) / f * z
        P = fwd * z + right * x_cam + up * y_up
        eye_xy = np.asarray(est_pos) + np.array([np.cos(est_yaw), np.sin(est_yaw)]) * C.CAM_MOUNT_FORWARD
        d = P[:2] / max(np.linalg.norm(P[:2]), 1e-6)
        return eye_xy + P[:2] + C.SURVIVOR_RADIUS * d          # depth hits the front surface: step to the centre

    def process(self, rgb, depth, est_pos, est_yaw, t):
        """Returns the list of NEWLY confirmed survivors (dicts) for this frame."""
        self.stats["frames"] += 1
        cands = []
        for b in connected_blobs(warm_mask(rgb), min_area=10, depth=depth):
            self.stats["proposals"] += 1
            x0, y0, x1, y1 = b["box"]
            if x0 <= 0 or x1 >= C.CAM_W - 1:                   # cut by the image edge: size/position unreliable
                continue
            z = float(np.median(depth[b["pix"][:, 0], b["pix"][:, 1]]))
            if not (0.4 < z < C.DETECT_RANGE + 0.5):
                continue
            w_m = (x1 - x0 + 1) * z / C.CAM_FOCAL_PX
            if not (0.12 <= w_m <= 0.9):
                self.stats["rejected_size"] += 1
                continue
            cands.append((b, z))
        if cands and self.use_cnn:
            x = np.stack([crop_box(rgb, b["box"]) for b, _ in cands]).astype(np.float32) / 127.5 - 1.0
            probs = self.model.forward(x.transpose(0, 3, 1, 2)).flatten()
        else:
            probs = np.ones(len(cands))
        new = []
        for (b, z), pr in zip(cands, probs):
            if pr < self.prob_thresh:
                self.stats["rejected_cnn"] += 1
                continue
            self.stats["accepted"] += 1
            xy = self._locate(b["pix"], z, est_pos, est_yaw)
            new += self._update_tracks(xy, z, float(pr), t)
        self.tracks = [tr for tr in self.tracks if tr["confirmed"] or t - tr["t1"] < 4.0]
        return new

    def _update_tracks(self, xy, z, prob, t):
        best, bd = None, self.assoc_radius
        for tr in self.tracks:
            d = np.linalg.norm(tr["xy"] - xy)
            if d < bd:
                best, bd = tr, d
        w = 1.0 / max(z, 0.5) ** 2                              # nearer views are more accurate
        if best is None:
            best = {"xy": xy.copy(), "wsum": w, "n": 0, "t0": t, "t1": t, "psum": 0.0,
                    "confirmed": False, "rec": None}
            self.tracks.append(best)
        else:
            best["xy"] = (best["xy"] * best["wsum"] + xy * w) / (best["wsum"] + w)
            best["wsum"] += w
        best["n"] += 1
        best["t1"] = t
        best["psum"] += prob
        out = []
        if best["confirmed"]:
            best["rec"]["xy"] = best["xy"].copy()
            best["rec"]["n_hits"] = best["n"]
        elif (best["n"] >= self.confirm_hits and best["t1"] - best["t0"] >= self.min_span_s
              and best["psum"] / best["n"] >= self.prob_thresh):
            rec = {"xy": best["xy"].copy(), "label": self.map.grid_box_label(best["xy"]), "t": t,
                   "n_hits": best["n"], "conf": best["psum"] / best["n"]}
            best["confirmed"], best["rec"] = True, rec
            self.confirmed.append(rec)
            out.append(rec)
        return out
