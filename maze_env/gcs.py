"""
Mission Control dashboard (matplotlib stand-in for a real ground station): live 2D map, the drone's
ESTIMATED position, reported survivors with grid-box tags, live camera feed, phase/time.
Ground truth (true position, true survivor positions) is drawn faintly, clearly labelled, for
simulator debugging only -- a real GCS would never have it.
"""

import numpy as np
import matplotlib.pyplot as plt


class GCSDashboard:
    def __init__(self, occ_map, maze):
        self.map, self.maze = occ_map, maze
        plt.ion()
        self.fig, (self.ax_map, self.ax_cam) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.suptitle("Autonomous GPS-Denied SAR Drone - Mission Control (sim)")

    def update(self, pos_est, yaw_est, pos_gt, survivors, detections, cam_frame, t, phase):
        m = self.map
        ax = self.ax_map
        ax.clear()
        img = np.full(m.log_odds.shape + (3,), 0.55)
        img[m.free_mask()] = 1.0
        img[m.occupied_mask()] = 0.0
        h = m.half_size
        ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", extent=[-h, h, -h, h])
        ax.plot(*pos_est, "b^", ms=10, label="Estimated position")
        ax.arrow(pos_est[0], pos_est[1], 0.5 * np.cos(yaw_est), 0.5 * np.sin(yaw_est), color="blue", head_width=0.15)
        ax.plot(*pos_gt, "g.", ms=4, alpha=0.4, label="True position (debug only)")
        ex, ey = self.maze.entry_world_xy()
        xx, xy_ = self.maze.exit_world_xy()
        ax.plot(ex, ey, "gs", ms=9, label="Entry")
        ax.plot(xx, xy_, "ms", ms=9, label="Exit")
        ax.scatter(survivors.xy[:, 0], survivors.xy[:, 1], s=60, facecolors="none", edgecolors="orange",
                   linewidths=1, alpha=0.5, label="True survivors (debug only)")
        for d in detections:
            ax.scatter(*d["xy"], s=90, c="red", marker="x", linewidths=2)
            ax.annotate(d["label"], d["xy"], textcoords="offset points", xytext=(6, 6), color="red", weight="bold")
        ax.scatter([], [], c="red", marker="x", label="Reported survivors")
        ax.set_title(f"Live 2D map | t={t:5.1f}s | {phase} | found {len(detections)}")
        ax.set_xlim(-h, h); ax.set_ylim(-h, h); ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=6)
        self.ax_cam.clear()
        self.ax_cam.imshow(cam_frame.astype(np.uint8))
        self.ax_cam.set_title("Live camera feed")
        self.ax_cam.axis("off")
        self.fig.canvas.draw()

    def save_frame(self, path):
        self.fig.savefig(path, dpi=110)

    def close(self):
        plt.ioff()
        plt.close(self.fig)
