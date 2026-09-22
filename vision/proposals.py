"""
Cheap first stage of the survivor detector: find warm/yellow-ish blobs in a camera frame and cut a
32x32 colour crop around each. Shared by the dataset generator and the flight detector so that
training crops and flight crops come from exactly the same code.

The threshold is deliberately LOOSE (it also fires on brown crates), so the CNN in the second stage
has real work to do: telling a person-shaped, saturated-yellow blob from a crate.
"""

import numpy as np

CROP = 32


def warm_mask(rgb, thr=35):
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    return ((r - b) > thr) & ((g - b) > 0.5 * thr) & (r > 80)


def connected_blobs(mask, min_area=10, depth=None, dz=0.4):
    """4-connected components of a boolean mask -> list of dicts (box, area, pixel coords).
    If a depth image is given, neighbouring pixels are only joined when their depth differs by less
    than `dz` metres. Without this, two survivors standing one behind the other merged into ONE blob,
    got one position halfway between them, and one of them was never reported."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    H, W = mask.shape
    label = -np.ones((H, W), dtype=np.int32)
    blobs = []
    for y0, x0 in zip(ys, xs):
        if label[y0, x0] != -1:
            continue
        k = len(blobs)
        stack = [(y0, x0)]
        label[y0, x0] = k
        pix = []
        while stack:
            y, x = stack.pop()
            pix.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= ny < H and 0 <= nx < W and mask[ny, nx] and label[ny, nx] == -1
                        and (depth is None or abs(depth[ny, nx] - depth[y, x]) < dz)):
                    label[ny, nx] = k
                    stack.append((ny, nx))
        pix = np.array(pix)
        if len(pix) >= min_area:
            blobs.append({"box": (int(pix[:, 1].min()), int(pix[:, 0].min()),
                                  int(pix[:, 1].max()), int(pix[:, 0].max())),
                          "area": len(pix), "pix": pix})
        else:
            blobs.append(None)
    return [b for b in blobs if b is not None]


def crop_box(rgb, box, pad=0.25):
    """Square crop centred on the box (side = longest side * (1+pad)), edge-clamped, resized to 32x32."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1 + 1) / 2.0, (y0 + y1 + 1) / 2.0
    side = max(x1 - x0 + 1, y1 - y0 + 1) * (1.0 + pad)
    side = max(side, 12.0)
    H, W = rgb.shape[:2]
    t = (np.arange(CROP) + 0.5) / CROP
    xs = np.clip(np.round(cx - side / 2 + t * side).astype(int), 0, W - 1)
    ys = np.clip(np.round(cy - side / 2 + t * side).astype(int), 0, H - 1)
    return rgb[np.ix_(ys, xs)]
