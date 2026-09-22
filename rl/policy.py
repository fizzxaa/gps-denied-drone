"""
A minimal MLP policy implemented in raw numpy -- no PyTorch/JAX dependency,
which keeps this project installable with just numpy+matplotlib+pybullet
(handy after the earlier Mac pip/venv trouble). Small enough that a
CPU-only Evolution Strategies training loop (see train_es.py) converges
in well under a minute.
"""

import numpy as np


class MLPPolicy:
    def __init__(self, obs_dim, act_dim, hidden=32, seed=None):
        rng = np.random.default_rng(seed)
        self.obs_dim, self.act_dim, self.hidden = obs_dim, act_dim, hidden
        scale1 = np.sqrt(2.0 / obs_dim)
        scale2 = np.sqrt(2.0 / hidden)
        self.W1 = rng.normal(0, scale1, size=(obs_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, scale2, size=(hidden, act_dim)).astype(np.float32)
        self.b2 = np.zeros(act_dim, dtype=np.float32)

    def forward(self, obs):
        h = np.tanh(obs @ self.W1 + self.b1)
        out = np.tanh(h @ self.W2 + self.b2)
        # action[0] (forward speed) squashed to [0, 1], action[1] (turn) stays in [-1, 1]
        out[0] = (out[0] + 1) / 2
        return out

    # ---- flat parameter vector helpers, used by the ES trainer ----
    def get_flat_params(self):
        return np.concatenate([self.W1.ravel(), self.b1.ravel(),
                                self.W2.ravel(), self.b2.ravel()])

    def set_flat_params(self, flat):
        i = 0
        n = self.W1.size; self.W1 = flat[i:i + n].reshape(self.W1.shape); i += n
        n = self.b1.size; self.b1 = flat[i:i + n].reshape(self.b1.shape); i += n
        n = self.W2.size; self.W2 = flat[i:i + n].reshape(self.W2.shape); i += n
        n = self.b2.size; self.b2 = flat[i:i + n].reshape(self.b2.shape); i += n

    def num_params(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                  obs_dim=self.obs_dim, act_dim=self.act_dim, hidden=self.hidden)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        policy = cls(int(d["obs_dim"]), int(d["act_dim"]), int(d["hidden"]))
        policy.W1, policy.b1 = d["W1"], d["b1"]
        policy.W2, policy.b2 = d["W2"], d["b2"]
        return policy
