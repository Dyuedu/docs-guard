"""Pre/post-processing shared by the Triton BLS model and offline evaluation.

Pure numpy/OpenCV so it runs inside the Triton Python backend and is unit-testable
without a GPU. Tensor layout is NCHW float32 throughout.
"""

from __future__ import annotations

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- decoding

def decode_image(data: bytes, max_long_side: int) -> tuple[np.ndarray, float]:
    """Decode JPEG/PNG bytes to RGB uint8. Returns (image, scale) where scale<1 means downscaled."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cannot decode image bytes")
    h, w = bgr.shape[:2]
    scale = min(1.0, max_long_side / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), scale


def to_chw(image: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    x = (x - mean) / std
    return x.transpose(2, 0, 1)


# ---------------------------------------------------------------- tampering (tiling at native resolution)

def _starts(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    s = list(range(0, length - patch + 1, stride))
    if s[-1] != length - patch:
        s.append(length - patch)
    return s


def tile(image: np.ndarray, patch: int, stride: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Split an RGB image into overlapping patches (N,3,P,P). Small images are zero-padded."""
    h, w = image.shape[:2]
    ph, pw = max(h, patch), max(w, patch)
    if (ph, pw) != (h, w):
        padded = np.zeros((ph, pw, 3), dtype=image.dtype)
        padded[:h, :w] = image
        image = padded
    coords = [(y, x) for y in _starts(ph, patch, stride) for x in _starts(pw, patch, stride)]
    patches = np.stack([to_chw(image[y : y + patch, x : x + patch]) for y, x in coords])
    return patches, coords


def stitch(prob_patches: np.ndarray, coords: list[tuple[int, int]], shape: tuple[int, int], patch: int) -> np.ndarray:
    """Average overlapping per-patch probability maps (N,P,P) back into an (H,W) map."""
    h, w = shape
    ph, pw = max(h, patch), max(w, patch)
    acc = np.zeros((ph, pw), dtype=np.float32)
    cnt = np.zeros((ph, pw), dtype=np.float32)
    for p, (y, x) in zip(prob_patches, coords):
        acc[y : y + patch, x : x + patch] += p
        cnt[y : y + patch, x : x + patch] += 1
    return (acc / np.maximum(cnt, 1))[:h, :w]


def mask_to_boxes(prob: np.ndarray, threshold: float, min_area: int) -> list[list[int]]:
    binary = (prob >= threshold).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append([int(x), int(y), int(x + w), int(y + h)])
    return boxes


# ---------------------------------------------------------------- text detection (DBNet-style)

def det_preprocess(image: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """Letterbox to size x size (top-left aligned). Returns (1,3,S,S) and the resize ratio."""
    h, w = image.shape[:2]
    r = size / max(h, w)
    resized = cv2.resize(image, (max(1, round(w * r)), max(1, round(h * r))))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return to_chw(canvas)[None], r


def det_postprocess(
    prob: np.ndarray, ratio: float, bin_threshold: float, box_threshold: float, unclip_ratio: float,
    image_shape: tuple[int, int],
) -> list[list[int]]:
    """Probability map (S,S) -> axis-aligned text boxes in image coordinates."""
    binary = (prob > bin_threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image_shape
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 3 or bh < 3:
            continue
        if float(prob[y : y + bh, x : x + bw][binary[y : y + bh, x : x + bw] > 0].mean()) < box_threshold:
            continue
        # DB shrinks text regions during training; expand back (approximation of polygon unclip).
        area, perim = bw * bh, 2 * (bw + bh)
        d = area * unclip_ratio / perim
        x1, y1 = (x - d) / ratio, (y - d) / ratio
        x2, y2 = (x + bw + d) / ratio, (y + bh + d) / ratio
        boxes.append([max(0, int(x1)), max(0, int(y1)), min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))])
    return sorted(boxes, key=lambda b: (b[1], b[0]))


# ---------------------------------------------------------------- text recognition (CRNN + CTC)

def crop_for_rec(image: np.ndarray, boxes: list[list[int]], height: int, width: int) -> np.ndarray:
    """Crop boxes, resize keeping aspect ratio to fixed height, right-pad to width. Returns (N,3,H,W)."""
    out = np.zeros((len(boxes), 3, height, width), dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        new_w = min(width, max(1, round(crop.shape[1] * height / crop.shape[0])))
        resized = cv2.resize(crop, (new_w, height)).astype(np.float32) / 255.0
        out[i, :, :, :new_w] = ((resized - 0.5) / 0.5).transpose(2, 0, 1)
    return out


def load_vocab(path: str) -> list[str]:
    """One symbol per line; the literal token <space> stands for ' '. Index 0 of the model is blank."""
    with open(path, encoding="utf-8") as f:
        return [" " if s == "<space>" else s for s in (line.rstrip("\n") for line in f) if s]


def ctc_greedy_decode(logits: np.ndarray, vocab: list[str], blank: int = 0) -> list[tuple[str, float]]:
    """logits (N,T,C) with class 0 = blank and class i>0 = vocab[i-1]. Returns (text, mean char prob)."""
    x = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)
    best = probs.argmax(-1)
    results = []
    for seq, p in zip(best, probs):
        chars, confs, prev = [], [], blank
        for t, k in enumerate(seq):
            if k != blank and k != prev:
                chars.append(vocab[k - 1])
                confs.append(p[t, k])
            prev = k
        results.append(("".join(chars), float(np.mean(confs)) if confs else 0.0))
    return results


# ---------------------------------------------------------------- decision

def box_intersects(a: list[int], b: list[int] | tuple[int, ...]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def decide(score: float, area_ratio: float, cfg: dict) -> tuple[bool, str]:
    if area_ratio > cfg["tampered_area_ratio"]:
        return False, f"tampered_area_ratio > {cfg['tampered_area_ratio']}"
    if score > cfg["tampered_score"]:
        return False, f"tampered_score > {cfg['tampered_score']}"
    return True, "no tampering signal above thresholds"
