"""
Trains the from-scratch numpy CNN (vision/cnn.py) to verify survivor crops.

Data: two files generated from DISJOINT MAZE LAYOUTS (see generate_dataset.py); this script asserts
zero layout overlap. Reports precision / recall / false-alarm rate on the held-out layouts, because
plain accuracy hides how many false alarms a mission would generate.

Usage:
    python -m vision.train_cnn --epochs 20
"""

import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vision.cnn import SurvivorCNN, bce_loss


def prep(x):
    return (x.astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)      # (N,3,32,32)


def load_dataset(train_path="vision/dataset_train.npz", val_path="vision/dataset_val.npz"):
    tr, va = np.load(train_path), np.load(val_path)
    overlap = set(tr["layouts"].tolist()) & set(va["layouts"].tolist())
    assert not overlap, f"train/val share {len(overlap)} maze layouts -- regenerate with the layout split"
    return prep(tr["images"]), tr["labels"], prep(va["images"]), va["labels"]


def predict(model, x, bs=128):
    return np.concatenate([model.forward(x[i:i + bs]).flatten() for i in range(0, len(x), bs)])


def metrics(p, y, thr=0.5):
    pred, pos = p > thr, y > 0.5
    tp, fp = int(np.sum(pred & pos)), int(np.sum(pred & ~pos))
    fn, tn = int(np.sum(~pred & pos)), int(np.sum(~pred & ~pos))
    return {"acc": (tp + tn) / len(y), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "false_alarm_rate": fp / max(fp + tn, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def augment(x, rng):
    flip = rng.random(len(x)) < 0.5
    x = np.where(flip[:, None, None, None], x[..., ::-1], x)                 # mirror left/right
    x = x * rng.uniform(0.8, 1.2, (len(x), 1, 1, 1)).astype(np.float32)      # brightness
    return np.clip(x, -1, 1).astype(np.float32)


def train(epochs=20, batch_size=32, lr=0.03, seed=0, train_path="vision/dataset_train.npz",
          val_path="vision/dataset_val.npz"):
    xtr, ytr, xva, yva = load_dataset(train_path, val_path)
    print(f"train {len(xtr)} crops ({int(ytr.sum())} pos) | val {len(xva)} crops ({int(yva.sum())} pos, "
          f"majority-class accuracy {max(yva.mean(), 1 - yva.mean()):.3f})")
    model = SurvivorCNN(input_size=32, in_ch=3, seed=seed)
    rng = np.random.default_rng(seed)
    hist = {"train_loss": [], "val_loss": [], "val_f1": []}
    best, best_params = -1, None
    t0 = time.time()
    for ep in range(epochs):
        idx = rng.permutation(len(xtr))
        losses = []
        for i in range(0, len(idx), batch_size):
            b = idx[i:i + batch_size]
            pred = model.forward(augment(xtr[b], rng)).flatten()
            loss, dpred = bce_loss(pred, ytr[b])
            model.backward(dpred.reshape(-1, 1), lr=lr)
            losses.append(loss)
        pv = predict(model, xva)
        vloss, _ = bce_loss(pv, yva)
        m = metrics(pv, yva)
        f1 = 2 * m["precision"] * m["recall"] / max(m["precision"] + m["recall"], 1e-9)
        hist["train_loss"].append(float(np.mean(losses))); hist["val_loss"].append(float(vloss)); hist["val_f1"].append(f1)
        print(f"[epoch {ep+1:2d}/{epochs}] train_loss={np.mean(losses):.4f} val_loss={vloss:.4f} "
              f"prec={m['precision']:.3f} recall={m['recall']:.3f} false_alarm={m['false_alarm_rate']:.3f} "
              f"({time.time()-t0:.0f}s)")
        if f1 >= best:
            best, best_params = f1, {k: v.copy() for k, v in model.get_params().items()}
    model.set_params(best_params)                     # keep the best epoch, not just the last
    return model, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    model, hist = train(epochs=a.epochs, lr=a.lr, seed=a.seed)
    model.save("vision/trained_cnn.npz")
    _, _, xva, yva = load_dataset()
    m = metrics(predict(model, xva), yva)
    print("FINAL held-out (unseen maze layouts):", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val"); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot(hist["val_f1"]); ax[1].set_ylim(0, 1); ax[1].set_title("Validation F1 (unseen layouts)")
    plt.tight_layout(); plt.savefig("vision/training_curve.png", dpi=120)
