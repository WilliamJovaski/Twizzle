"""
Twizzle shared preprocessing (used by the preprocessing notebook, the model notebook AND the web app).

Contract (Team Technical Spec, Section 3):
  * 11 Set-level classes, index = label (CLASS_NAMES below)
  * 16 frames per clip, 160x160, RGB, uint8, layout (T, H, W, 3)
  * resize = whole frame squashed to a square with cv2.INTER_AREA (no crop, no letterbox)
  * sampling = decode ALL frames of the clip file, then np.linspace(0, n-1, 16).astype(int)
  * normalization is NOT applied here; use frames_to_tensor() with the model's own mean/std

Do not edit constants without telling the whole team: training and serving must stay identical.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import cv2
import numpy as np

NUM_FRAMES = 16
IMG_SIZE = 160

CLASS_NAMES = [
    "Toeloop", "Salchow", "Flip", "Loop", "Lutz", "Axel",    # 0-5  jumps
    "CamelSpin", "SitSpin", "UprightSpin", "NoBasic",        # 6-9  spins
    "Sequence",                                               # 10   step/choreo sequence
]
NAME_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
_NORM_LOOKUP = {name.lower().replace(" ", ""): name for name in CLASS_NAMES}


# --------------------------------------------------------------------------- labels
def element_code_to_set_level(code):
    """Maps a dataset element code ('3Lutz', 'Camel Spin', 'No Basic', 'Sequence') to one of the
    11 Set-level class names. Case/space-insensitive. Returns None when the code is unknown:
    callers must treat None as an error instead of silently dropping rows.
    """
    if code is None:
        return None
    code = str(code).strip()
    norm = code.lower().replace(" ", "")
    if norm in _NORM_LOOKUP:                       # 'camelspin', 'nobasic', 'sequence', ...
        return _NORM_LOOKUP[norm]
    if norm == "null":                             # alias used in the paper for Sequence
        return "Sequence"
    m = re.match(r"^(\d+)([A-Za-z]+)$", code)      # '3Lutz' -> rotations + jump name
    if m and m.group(2).lower() in _NORM_LOOKUP:
        return _NORM_LOOKUP[m.group(2).lower()]
    return None


# --------------------------------------------------------------------------- files
def resolve_video_path(video_dir, name):
    """train.txt / answer.txt store bare clip ids (no extension); files on disk are '<id>.mp4'.
    Returns the existing Path, or None if neither '<name>' nor '<name>.mp4' exists."""
    video_dir = Path(video_dir)
    for candidate in (video_dir / name, video_dir / f"{name}.mp4"):
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------- frames
def sample_frames(video_path, num_frames: int = NUM_FRAMES, img_size: int = IMG_SIZE):
    """Returns a (num_frames, img_size, img_size, 3) uint8 RGB array, or None if unreadable.

    Decodes the whole clip (so it never depends on container frame-count metadata), squashes
    each frame to img_size x img_size with INTER_AREA, then picks num_frames indices uniformly.
    A clip shorter than num_frames simply repeats frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None
    idx = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    return np.stack([frames[i] for i in idx], axis=0)


def probe_video(video_path):
    """Returns dict(fps, n_frames, duration_s, width, height) or None. Falls back to counting
    frames when the container metadata is missing."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n <= 0:
        n = 0
        while cap.grab():
            n += 1
    cap.release()
    duration = n / fps if fps and fps > 0 else float("nan")
    return {"fps": fps, "n_frames": n, "duration_s": duration, "width": w, "height": h}


def frames_to_tensor(frames, mean, std):
    """uint8 (T,H,W,3) -> float32 torch tensor (3,T,H,W), scaled to 0-1 then normalised with the
    model's own mean/std (they differ between models; read them from deployment_meta.json)."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(frames)).float() / 255.0      # (T,H,W,3)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 1, 1, 3)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 1, 1, 3)
    x = (x - mean_t) / std_t
    return x.permute(3, 0, 1, 2).contiguous()                                # (3,T,H,W)


# --------------------------------------------------------------------------- caching worker
def cache_clip(task):
    """Worker for ProcessPoolExecutor (must live in a module so Windows can import it).
    task = (video_path, out_path, num_frames, img_size). Returns (out_path, status) where status
    is 'done', 'skipped' (already cached, makes the run resumable) or 'failed'."""
    video_path, out_path, num_frames, img_size = task
    out = Path(out_path)
    if out.is_file() and out.stat().st_size > 0:
        return out_path, "skipped"
    cv2.setNumThreads(1)
    arr = sample_frames(video_path, num_frames, img_size)
    if arr is None:
        return out_path, "failed"
    import torch
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pt.tmp")
    torch.save(torch.from_numpy(arr), tmp)
    os.replace(tmp, out)                       # atomic: an interrupted run never leaves half files
    return out_path, "done"
