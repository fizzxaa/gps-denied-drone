"""
Trains the local-navigation MLP policy with Evolution Strategies (ES) --
the OpenAI-ES flavor: antithetic-sampled Gaussian perturbations of the
policy's parameter vector, fitness-shaped by rank, updated with a plain
gradient-ascent step (Adam). This is a legitimate reinforcement-learning
method (it optimizes a black-box, non-differentiable objective -- total
episode reward -- purely from reward signal, no labeled data) and needs
nothing beyond numpy, unlike policy-gradient methods that usually want
autodiff. Runs in well under a minute on a single CPU core for this
network size + environment.

Usage:
    python -m rl.train_es --generations 150 --population 48
Outputs:
    rl/trained_policy.npz   -- the trained policy weights
    rl/training_curve.png   -- mean episode reward per generation
"""

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl.env_fast import LocalNavEnv
from rl.env_maze import MazeNavEnv
from rl.policy import MLPPolicy


def make_env(env_name, seed, held_out=False):
    """held_out=True -> mazes whose LAYOUT falls in the validation split; training only ever draws
    layouts from the train split. The split is on the wall layout itself (not the seed), so the two
    sets cannot share a maze. Use held_out=True for anything you want to call generalisation."""
    if env_name == "maze":
        return MazeNavEnv(seed=seed, split="val" if held_out else "train")
    elif env_name == "circles":
        return LocalNavEnv(seed=seed)
    elif env_name == "pybullet":
        from rl.env_pybullet import RealMazeNavEnv
        return RealMazeNavEnv(grid_n=3, maze_seed=seed, seed=seed, split="val" if held_out else "train")
    raise ValueError(env_name)


def evaluate_policy(policy, env, num_episodes=3, seed_offset=0):
    total = 0.0
    successes = 0
    for ep in range(num_episodes):
        env.rng = np.random.default_rng(seed_offset + ep)
        obs = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = policy.forward(obs)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
        total += ep_reward
        successes += int(info.get("reached_goal", False))
    return total / num_episodes, successes / num_episodes


class AdamOptimizer:
    """Plain Adam, applied to the flat ES gradient estimate."""

    def __init__(self, dim, lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = np.zeros(dim)
        self.v = np.zeros(dim)
        self.t = 0

    def step(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def rank_shape(rewards):
    """Centered rank transform -- makes ES robust to reward scale/outliers."""
    order = np.argsort(rewards)
    ranks = np.empty(len(rewards))
    ranks[order] = np.arange(len(rewards))
    ranks = ranks / (len(rewards) - 1) - 0.5
    return ranks


def train(generations=150, population=32, sigma=0.08, lr=0.02, seed=0, log_every=10,
          init_from=None, env_name="maze", checkpoint_every=10, checkpoint_path="rl/trained_policy.npz"):
    env = make_env(env_name, seed)
    obs_dim, act_dim = env.obs_dim, env.act_dim
    policy = MLPPolicy(obs_dim, act_dim, hidden=32, seed=seed)
    if init_from:
        policy = MLPPolicy.load(init_from)
        print(f"Resuming from {init_from}")
    theta = policy.get_flat_params()
    n_params = len(theta)
    optimizer = AdamOptimizer(n_params, lr=lr)

    rng = np.random.default_rng(seed)
    history = []
    t0 = time.time()

    half_pop = population // 2
    for gen in range(generations):
        noise = rng.normal(0, 1, size=(half_pop, n_params))
        full_noise = np.concatenate([noise, -noise], axis=0)  # antithetic pairs
        rewards = np.zeros(2 * half_pop)

        for i in range(2 * half_pop):
            candidate = theta + sigma * full_noise[i]
            policy.set_flat_params(candidate.astype(np.float32))
            # every candidate in a generation is scored on the SAME mazes: otherwise a candidate can
            # look better just because it drew easier mazes, and ES follows that noise
            reward, _ = evaluate_policy(policy, env, num_episodes=3, seed_offset=gen * 1000)
            rewards[i] = reward

        shaped = rank_shape(rewards)
        grad_estimate = (full_noise.T @ shaped) / (2 * half_pop * sigma)
        theta = theta + optimizer.step(grad_estimate)

        policy.set_flat_params(theta.astype(np.float32))
        mean_r, success_rate = evaluate_policy(policy, env, num_episodes=8, seed_offset=999_000)   # fixed mazes: comparable across generations
        history.append(mean_r)

        if gen % log_every == 0 or gen == generations - 1:
            print(f"[ES] gen {gen:4d}/{generations}  mean_reward={mean_r:6.2f}  "
                  f"success_rate={success_rate:4.2f}  elapsed={time.time()-t0:5.1f}s")

        # checkpoint periodically so a timed-out/interrupted run doesn't
        # lose all progress -- only the final __main__ block used to save,
        # which meant a run cut short by a wall-clock limit saved nothing
        if gen % checkpoint_every == 0 and gen > 0:
            policy.save(checkpoint_path)

    policy.set_flat_params(theta.astype(np.float32))
    return policy, history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init_from", type=str, default=None, help="resume from a saved .npz policy")
    ap.add_argument("--env", type=str, default="maze", choices=["maze", "circles", "pybullet"],
                     help="'maze' trains against real maze wall geometry via fast analytic "
                          "raycasting (recommended default); 'circles' is the older abstract "
                          "task kept for comparison; 'pybullet' trains inside the actual "
                          "PyBullet physics sim (real differential-drive dynamics + real contact collisions, "
                          "but ~50x slower per episode and only one maze per run -- see "
                          "rl/env_pybullet.py's docstring)")
    ap.add_argument("--history_out", type=str, default=None, help="path to append/save reward history .npy")
    args = ap.parse_args()

    policy, history = train(generations=args.generations, population=args.population,
                             sigma=args.sigma, lr=args.lr, seed=args.seed, init_from=args.init_from,
                             env_name=args.env)

    policy.save("rl/trained_policy.npz")
    print("Saved trained policy to rl/trained_policy.npz")

    if args.history_out:
        prior = list(np.load(args.history_out)) if __import__("os").path.exists(args.history_out) else []
        full_history = prior + history
        np.save(args.history_out, np.array(full_history))
        history = full_history

    plt.figure(figsize=(6, 4))
    plt.plot(history)
    plt.xlabel("Generation")
    plt.ylabel("Mean episode reward")
    plt.title("ES training curve — local navigation policy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("rl/training_curve.png", dpi=120)
    print("Saved training curve to rl/training_curve.png")
