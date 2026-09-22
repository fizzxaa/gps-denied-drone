"""
A small CNN implemented from scratch in numpy -- forward AND backward
passes, both hand-derived, no autodiff framework. Kept deliberately tiny
(two conv layers, ~35K parameters) so training on CPU with a few thousand
32x32 grayscale images finishes in a couple of minutes.

Why numpy instead of PyTorch: consistent with the rest of this project's
"no heavy DL framework required to run this" design (see the RL section
of the README for the same reasoning) -- and installing PyTorch turned
out to be genuinely painful even in a plain Linux dev sandbox during this
project (a GPU build pulled in several hundred MB of CUDA packages and
filled the disk), which only reinforces that choice.

Architecture: Conv(3->8, 3x3) -> ReLU -> MaxPool(2x2)
              -> Conv(8->16, 3x3) -> ReLU -> MaxPool(2x2)
              -> Flatten -> Dense(->32) -> ReLU -> Dense(->1) -> Sigmoid
Input: 32x32 RGB crop. Output: P(crop shows a survivor).

Convolution is implemented via im2col (unfold input patches into a matrix,
turn convolution into a single matmul) -- the standard trick for making
numpy convolution fast enough to be usable; a naive 4-nested-loop
convolution would be far too slow for even this small a network.
"""

import numpy as np


def im2col(x, kh, kw, stride=1, pad=1):
    """x: (N, C, H, W) -> (N, out_h, out_w, C*kh*kw) patch matrix."""
    N, C, H, W = x.shape
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1

    cols = np.zeros((N, out_h, out_w, C, kh, kw), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            cols[:, :, :, :, i, j] = x_padded[:, :, i:i_max:stride, j:j_max:stride].transpose(0, 2, 3, 1)
    return cols.reshape(N, out_h, out_w, C * kh * kw), out_h, out_w


def col2im(dcols, x_shape, kh, kw, stride=1, pad=1):
    """Inverse of im2col -- scatters gradient patches back onto the input shape."""
    N, C, H, W = x_shape
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1
    dcols = dcols.reshape(N, out_h, out_w, C, kh, kw)

    dx_padded = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=dcols.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            dx_padded[:, :, i:i_max:stride, j:j_max:stride] += dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    if pad == 0:
        return dx_padded
    return dx_padded[:, :, pad:-pad, pad:-pad]


class Conv2D:
    def __init__(self, in_ch, out_ch, k=3, stride=1, pad=1, seed=None):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (in_ch * k * k))
        self.W = rng.normal(0, scale, size=(out_ch, in_ch, k, k)).astype(np.float32)
        self.b = np.zeros(out_ch, dtype=np.float32)
        self.k, self.stride, self.pad = k, stride, pad
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_ch = self.W.shape[0]
        cols, out_h, out_w = im2col(x, self.k, self.k, self.stride, self.pad)
        cols_flat = cols.reshape(-1, C * self.k * self.k)               # (N*out_h*out_w, C*k*k)
        W_flat = self.W.reshape(out_ch, -1).T                            # (C*k*k, out_ch)
        out = cols_flat @ W_flat + self.b                                # (N*out_h*out_w, out_ch)
        out = out.reshape(N, out_h, out_w, out_ch).transpose(0, 3, 1, 2)  # (N, out_ch, out_h, out_w)
        self.cache = (x.shape, cols_flat)
        return out

    def backward(self, dout, lr):
        x_shape, cols_flat = self.cache
        N, out_ch, out_h, out_w = dout.shape
        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, out_ch)  # (N*out_h*out_w, out_ch)

        dW = (dout_flat.T @ cols_flat).reshape(self.W.shape)
        db = dout_flat.sum(axis=0)

        W_flat = self.W.reshape(out_ch, -1)
        dcols_flat = dout_flat @ W_flat  # (N*out_h*out_w, C*k*k)
        dx = col2im(dcols_flat, x_shape, self.k, self.k, self.stride, self.pad)

        self.W -= lr * dW
        self.b -= lr * db
        return dx


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout, lr=None):
        return dout * self.mask


class MaxPool2D:
    def __init__(self, size=2, stride=2):
        self.size, self.stride = size, stride

    def forward(self, x):
        N, C, H, W = x.shape
        s, st = self.size, self.stride
        out_h, out_w = H // st, W // st
        x_reshaped = x[:, :, :out_h * st, :out_w * st]
        x_reshaped = x_reshaped.reshape(N, C, out_h, st, out_w, st)
        out = x_reshaped.max(axis=(3, 5))
        # store argmax mask for backward
        self.cache = (x.shape, x_reshaped, out)
        return out

    def backward(self, dout, lr=None):
        x_shape, x_reshaped, out = self.cache
        N, C, out_h, st1, out_w, st2 = x_reshaped.shape
        mask = (x_reshaped == out[:, :, :, None, :, None])
        # split gradient evenly across ties (rare with random floats, but correct)
        counts = mask.sum(axis=(3, 5), keepdims=True)
        dout_expanded = (dout[:, :, :, None, :, None] * mask) / np.maximum(counts, 1)
        dx = np.zeros(x_shape, dtype=dout.dtype)
        dx[:, :, :out_h * self.stride, :out_w * self.stride] = dout_expanded.reshape(
            N, C, out_h * self.stride, out_w * self.stride)
        return dx


class Flatten:
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout, lr=None):
        return dout.reshape(self.shape)


class Dense:
    def __init__(self, in_dim, out_dim, seed=None):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0, scale, size=(in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout, lr):
        N = dout.shape[0]
        dW = self.x.T @ dout
        db = dout.sum(axis=0)
        dx = dout @ self.W.T
        self.W -= lr * dW
        self.b -= lr * db
        return dx


class Sigmoid:
    def forward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
        return self.out

    def backward(self, dout, lr=None):
        return dout * self.out * (1 - self.out)


class SurvivorCNN:
    """Conv(1->8)->ReLU->Pool -> Conv(8->16)->ReLU->Pool -> Flatten -> Dense(32)->ReLU -> Dense(1)->Sigmoid"""

    def __init__(self, input_size=32, in_ch=3, seed=0):
        self.input_size = input_size
        self.in_ch = in_ch
        pooled = input_size // 4  # two 2x2 pools
        self.layers = [
            Conv2D(in_ch, 8, k=3, pad=1, seed=seed),
            ReLU(),
            MaxPool2D(),
            Conv2D(8, 16, k=3, pad=1, seed=seed + 1),
            ReLU(),
            MaxPool2D(),
            Flatten(),
            Dense(16 * pooled * pooled, 32, seed=seed + 2),
            ReLU(),
            Dense(32, 1, seed=seed + 3),
            Sigmoid(),
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout, lr):
        for layer in reversed(self.layers):
            dout = layer.backward(dout, lr)
        return dout

    def predict(self, x):
        return self.forward(x)

    def get_params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (Conv2D, Dense)):
                params[f"W{i}"] = layer.W
                params[f"b{i}"] = layer.b
        return params

    def set_params(self, params):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (Conv2D, Dense)):
                layer.W = params[f"W{i}"]
                layer.b = params[f"b{i}"]

    def save(self, path):
        np.savez(path, input_size=self.input_size, in_ch=self.in_ch, **self.get_params())

    @classmethod
    def load(cls, path):
        d = np.load(path)
        model = cls(input_size=int(d["input_size"]), in_ch=int(d["in_ch"]) if "in_ch" in d.files else 1)
        params = {k: d[k] for k in d.files if k not in ("input_size", "in_ch")}
        model.set_params(params)
        return model


def bce_loss(pred, target):
    eps = 1e-7
    pred = np.clip(pred, eps, 1 - eps)
    loss = -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))
    dpred = (pred - target) / (pred * (1 - pred) * len(pred))
    return loss, dpred
