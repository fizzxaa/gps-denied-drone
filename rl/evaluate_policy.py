"""
Held-out evaluation of the local-navigation policy on maze LAYOUTS it never trained on
(rl.train_es.make_env(..., held_out=True) draws only validation-split layouts).

Reports success / collision / timeout with 95% Wilson intervals, AND the same split by whether the
goal was in straight line of sight. That second split matters: a memoryless policy that only sees
lidar + a straight-line goal vector cannot plan around a wall, so headline success rates on
"room to room" tasks mostly measure how many episodes happened to have a clear line.

Usage:
    python -m rl.evaluate_policy --num_episodes 200
"""

import argparse
import numpy as np

from rl.train_es import make_env
from rl.policy import MLPPolicy


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    m = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (max(0.0, c - m), min(1.0, c + m))


def evaluate(policy_path, num_episodes, seed):
    policy = MLPPolicy.load(policy_path)
    outcomes, los = [], []
    for ep in range(num_episodes):
        env = make_env("maze", seed=seed + ep, held_out=True)
        obs = env.reset()
        los.append(env.line_of_sight_clear())
        done = False
        while not done:
            obs, _, done, info = env.step(policy.forward(obs))
        outcomes.append("success" if info["reached_goal"] else ("collision" if info["collided"] else "timeout"))
    return outcomes, np.array(los)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=str, default="rl/trained_policy.npz")
    ap.add_argument("--num_episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    outcomes, los = evaluate(a.policy, a.num_episodes, a.seed)
    outcomes = np.array(outcomes)
    print(f"Held-out layouts, {len(outcomes)} episodes")
    for name, mask in [("all episodes", np.ones(len(los), bool)), ("goal in line of sight", los), ("wall in the way", ~los)]:
        n = int(mask.sum())
        if n == 0:
            continue
        row = []
        for k in ("success", "collision", "timeout"):
            c = int((outcomes[mask] == k).sum())
            lo, hi = wilson_ci(c, n)
            row.append(f"{k} {c/n:.2f} [{lo:.2f}-{hi:.2f}]")
        print(f"  {name:22s} n={n:3d}  " + "  ".join(row))
