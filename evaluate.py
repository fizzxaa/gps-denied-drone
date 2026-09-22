"""
Batch evaluation: run the full mission over many seeds / maze sizes and report mean +- 95% CI.

Every metric is scored against ground truth that the flight code never sees:
  completed          landed at the exit (true position within 0.7 m, altitude < 0.4 m)
  survivors_found    reported detections matched one-to-one to real survivors (within 1 m)
  false_alarms       reported detections that match no real survivor
  coverage_gt        share of the true free floor the camera actually had in view
  collision_ticks    physics ticks (1/240 s) in contact with a wall/obstacle
  hover_fraction     share of flight time at true speed < 0.1 m/s (should be ~0)
  drift              estimated-vs-true position error

Usage:
    python evaluate.py --grids 3 4 5 --seeds 100 101 102 103 104
    python evaluate.py --grids 4 --seeds 100 101 102 --imu_noise_scale 4
    python evaluate.py --grids 4 --seeds 100 101 102 --no_cnn          # ablation
Choose seeds you did NOT tune on. Rows are appended to the CSV as they finish.
"""

import argparse
import csv
import io
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import run_mission

T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
       12: 2.18, 15: 2.13, 20: 2.09, 30: 2.04}


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float("nan")
    df = len(x) - 1
    t = T95.get(df) or T95[min(T95, key=lambda k: abs(k - df))]
    return float(x.mean()), float(t * x.std(ddof=1) / np.sqrt(len(x)))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - m), min(1.0, c + m)


FIELDS = ["grid_n", "seed", "imu_noise_scale", "cnn", "phase", "completed", "elapsed_s", "found", "total",
          "false_alarms", "loc_error_m", "coverage_gt", "collision_ticks", "hover_fraction", "mean_speed",
          "max_tilt_deg", "path_len_m", "stuck_events", "mean_drift_m", "max_drift_m"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+", type=int, default=[3, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    ap.add_argument("--num_survivors", type=int, default=4)
    ap.add_argument("--max_minutes", type=float, default=12.0)
    ap.add_argument("--imu_noise_scale", type=float, default=1.0)
    ap.add_argument("--no_cnn", action="store_true")
    ap.add_argument("--controller", choices=["path", "rl"], default="path")
    ap.add_argument("--out_csv", default="results.csv")
    ap.add_argument("--out_plot", default="results_summary.png")
    a = ap.parse_args()

    rows = []
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for g in a.grids:
            for s in a.seeds:
                with contextlib.redirect_stdout(io.StringIO()):
                    r = run_mission(grid_n=g, seed=s, use_gui=False, use_dashboard=False, max_minutes=a.max_minutes,
                                    num_survivors=a.num_survivors, imu_noise_scale=a.imu_noise_scale,
                                    detector_use_cnn=not a.no_cnn, verbose=False, controller=a.controller)
                row = {"grid_n": g, "seed": s, "imu_noise_scale": a.imu_noise_scale, "cnn": not a.no_cnn,
                       "phase": r["phase"], "completed": int(r["completed"]), "elapsed_s": round(r["elapsed"], 1),
                       "found": r["found"], "total": r["total"], "false_alarms": r["false_positives"],
                       "loc_error_m": round(r["mean_loc_error_m"], 3), "coverage_gt": round(r["coverage_gt"], 3),
                       "collision_ticks": r["collision_ticks"], "hover_fraction": round(r["hover_fraction"], 4),
                       "mean_speed": round(r["mean_speed"], 3), "max_tilt_deg": round(r["max_tilt_deg"], 1),
                       "path_len_m": round(r["path_len_m"], 1), "stuck_events": r["stuck_events"],
                       "mean_drift_m": round(r["mean_drift"], 3), "max_drift_m": round(r["max_drift"], 3)}
                rows.append(row)
                w.writerow(row); f.flush()
                print(row, flush=True)

    print("\n================ SUMMARY (mean +- 95% CI) ================")
    groups = sorted(set(r["grid_n"] for r in rows))
    summary = {}
    for g in groups + ["all"]:
        R = rows if g == "all" else [r for r in rows if r["grid_n"] == g]
        n = len(R)
        comp = sum(r["completed"] for r in R)
        lo, hi = wilson(comp, n)
        found = sum(r["found"] for r in R); total = sum(r["total"] for r in R)
        flo, fhi = wilson(found, total)
        no_coll = sum(r["collision_ticks"] == 0 for r in R)
        cov = mean_ci([r["coverage_gt"] for r in R])
        line = {
            "n": n, "completed": f"{comp}/{n} ({lo:.0%}-{hi:.0%})",
            "survivors_found": f"{found}/{total} ({flo:.0%}-{fhi:.0%})",
            "false_alarms_total": sum(r["false_alarms"] for r in R),
            "runs_with_zero_collisions": f"{no_coll}/{n}",
            "coverage": "%.3f +- %.3f" % cov,
            "hover": "%.1f%% +- %.1f%%" % tuple(100 * v for v in mean_ci([r["hover_fraction"] for r in R])),
            "time_s": "%.0f +- %.0f" % mean_ci([r["elapsed_s"] for r in R]),
            "loc_error_m": "%.3f +- %.3f" % mean_ci([r["loc_error_m"] for r in R]),
            "drift_m": "%.3f +- %.3f" % mean_ci([r["mean_drift_m"] for r in R]),
            "max_tilt_deg": "%.1f (worst %.1f)" % (np.mean([r["max_tilt_deg"] for r in R]), max(r["max_tilt_deg"] for r in R)),
        }
        summary[g] = line
        print(f"grid {g}: " + " | ".join(f"{k}={v}" for k, v in line.items()))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for i, (key, title) in enumerate([("coverage_gt", "Floor viewed"), ("found", "Survivors found"), ("elapsed_s", "Mission time (s)")]):
        for g in groups:
            R = [r for r in rows if r["grid_n"] == g]
            v = [r[key] / r["total"] if key == "found" else r[key] for r in R]
            m, c = mean_ci(v)
            ax[i].bar(str(g), m, yerr=0 if np.isnan(c) else c, capsize=5)
        ax[i].set_title(title); ax[i].set_xlabel("maze size (n x n rooms)")
    plt.tight_layout(); plt.savefig(a.out_plot, dpi=120)
    print("saved", a.out_csv, a.out_plot)


if __name__ == "__main__":
    main()
