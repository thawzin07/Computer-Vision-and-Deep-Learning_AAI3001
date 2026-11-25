"""Builder module for colab/01_Train_ASL_Words_Conformer_2026-11-19.ipynb (v2 + SupCon + INT8)."""

from __future__ import annotations

from pathlib import Path

from nb_common import build_and_validate_notebook


def get_words_v2_notebook_cells() -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []

    cells.append((
        "md",
        """# ASL Word-Level Recognition (15 Dynamic Words) — v2 Balanced Conformer + 5-Fold Signer CV (`2026-11-19`)

**Target Vocabulary (15 Visually Distinct Signs):**  
`eat`, `family`, `finish`, `friend`, `hello`, `help`, `love`, `more`, `no`, `please`, `stop`, `thank you`, `want`, `water`, `yes`

### Key Upgrades in v2 (Built from v1 Diagnostic Findings)
1. **Fixes `eat`/`want` 2× Class Imbalance:** Applies `WeightedRandomSampler` + class-balanced label-smoothed Cross-Entropy so majority glosses no longer steal predictions from `thank you` and `stop`.
2. **Fixes Small Train Split & Signer Disjointness:** Uses `StratifiedGroupKFold(n_splits=6)` (`~67%` Train / `17%` Val / `17%` Test) with asserted zero signer overlap across splits, plus optional **5-Fold Signer-Independent CV** (`cv5_summary.csv`).
3. **Architectural & Regularization Fixes:** Replaces `BatchNorm1d` with `GroupNorm(8, d_model)` in `ConformerBlockV2`, fixes EMA warm-up decay & buffer synchronization, adds `DropPath` + temporal frame dropout, and shrinks Conformer capacity (`d_model=144, 3 layers`) to stop small-data overfitting.
4. **Full 5-Model Ablation + Single-ONNX Ensemble Export:** Benchmarks `RGB ResNet18+BiGRU` vs `Pose BiGRU (171d)` vs `Kinematic BiGRU (688d)` vs `Kinematic Conformer v2 (688d)` vs `Dual-Architecture Ensemble`, automatically exporting the highest Val Macro-F1 winner as a single verified `asl_words_conformer.onnx` (`(B, 48, 688) -> (B, 15)`).
5. **Fast v1 Drive Cache Reuse:** Automatically reuses your existing v1 `word_features_cache_v1.npz` on Google Drive when no new personal calibration clips are added."""
    ))

    cells.append((
        "code",
        """# @title 1. Install Dependencies, Check GPU/High-RAM & Mount Google Drive
!pip install -q mediapipe onnx onnxruntime remotezip kagglehub umap-learn tqdm scikit-learn seaborn opencv-python-headless psutil

import os
import random
import shutil
import json
import re
import time
import warnings
import multiprocessing
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOTAL_RAM_GB = psutil.virtual_memory().total / 1e9
AVAIL_RAM_GB = psutil.virtual_memory().available / 1e9
CPU_CORES = os.cpu_count() or 2
GPU_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0
HIGH_RAM = TOTAL_RAM_GB >= 24.0
EXTRACTION_WORKERS = min(8, max(1, CPU_CORES - 1))
LOADER_WORKERS = min(6, max(2, CPU_CORES // 2))
KIN_BATCH_SIZE = 64 if GPU_VRAM_GB >= 20.0 else 32
RGB_BATCH_SIZE = 32 if GPU_VRAM_GB >= 20.0 else 16

print(f"PyTorch: {torch.__version__} | Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {GPU_VRAM_GB:.1f} GB")
import mediapipe as mp
print(f"MediaPipe: {mp.__version__} | CPU cores: {CPU_CORES} | RAM: {TOTAL_RAM_GB:.1f} GB total ({AVAIL_RAM_GB:.1f} GB free) | High-RAM: {HIGH_RAM}")

USE_DRIVE = True  # @param {type:"boolean"}
DRIVE_BASE = Path("/content/ASL_Project")
if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        DRIVE_BASE = Path("/content/drive/MyDrive/ASL_Project")
    except Exception as exc:
        print("Drive mount skipped:", exc)

V1_WORK_DIR = DRIVE_BASE / "words_conformer"
WORK_DIR = DRIVE_BASE / "words_conformer_v2_20261119"
RAW_DIR = Path("/content/raw_asl_words_v2")
CAL_DIR = DRIVE_BASE / "calibration_data" / "words"
FIG_DIR = WORK_DIR / "figures"
EXPORT_DIR = WORK_DIR / "export"
for d in [WORK_DIR, RAW_DIR, FIG_DIR, EXPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25})
print("v2 Workspace:", WORK_DIR)"""
    ))

    cells.append((
        "code",
        """# @title 2. Vocabulary, MediaPipe Assets & Exact 688-dim Kinematic Feature Engine
import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

WORDS_15 = [
    "eat", "family", "finish", "friend", "hello",
    "help", "love", "more", "no", "please",
    "stop", "thank you", "want", "water", "yes",
]
WORD_TO_ID = {w: i for i, w in enumerate(WORDS_15)}

GLOSS_ALIASES = {
    "EAT": "eat", "FOOD": "eat", "FAMILY": "family", "FINISH": "finish",
    "DONE": "finish", "FRIEND": "friend", "HELLO": "hello", "HI": "hello",
    "HELP": "help", "LOVE": "love", "MORE": "more", "NO": "no",
    "PLEASE": "please", "STOP": "stop", "THANKYOU": "thank you",
    "THANK YOU": "thank you", "THANKS": "thank you", "WANT": "want",
    "WATER": "water", "YES": "yes",
}

def normalize_gloss(raw_gloss: str) -> str | None:
    if not isinstance(raw_gloss, str):
        return None
    clean = raw_gloss.strip().upper()
    clean = re.sub(r"[_\\-]+", " ", clean)
    clean_no_digits = re.sub(r"\\s*[0-9]+$", "", clean).strip()
    compact = clean_no_digits.replace(" ", "")
    for cand in (clean, clean_no_digits, compact):
        if cand in GLOSS_ALIASES:
            return GLOSS_ALIASES[cand]
    return None

POSE_INDICES = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]
POSE_SWAP_PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13]
WORD_SEQ_LEN = 48
WORD_STATIC_DIM = 346
WORD_FEATURE_DIM = 688

FINGER_ANGLE_TRIPLES = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]
FINGERTIP_INDICES = [4, 8, 12, 16, 20]

MP_ASSETS_DIR = WORK_DIR / "mediapipe_assets"
MP_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
MP_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}
for fname, url in MP_URLS.items():
    dst = MP_ASSETS_DIR / fname
    v1_dst = V1_WORK_DIR / "mediapipe_assets" / fname
    if (not dst.exists() or dst.stat().st_size < 100_000) and v1_dst.exists() and v1_dst.stat().st_size >= 100_000:
        shutil.copy2(v1_dst, dst)
    if not dst.exists() or dst.stat().st_size < 100_000:
        print(f"Downloading {fname}...")
        urllib.request.urlretrieve(url, dst)

def _lm_to_np(lms) -> np.ndarray:
    arr = np.zeros((len(lms), 3), dtype=np.float32)
    for i, lm in enumerate(lms):
        arr[i] = (lm.x, lm.y, lm.z)
    return arr

def compute_joint_angles(hand_pts: np.ndarray) -> np.ndarray:
    if float(np.abs(hand_pts).sum()) < 1e-6:
        return np.zeros(15, dtype=np.float32)
    out = np.zeros(15, dtype=np.float32)
    for i, (a, b, c) in enumerate(FINGER_ANGLE_TRIPLES):
        v1 = hand_pts[a] - hand_pts[b]
        v2 = hand_pts[c] - hand_pts[b]
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
        out[i] = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return out

def normalize_hand_local(hand_pts: np.ndarray, present: bool) -> np.ndarray:
    if not present or float(np.abs(hand_pts).sum()) < 1e-6:
        return np.zeros((21, 3), dtype=np.float32)
    centered = hand_pts - hand_pts[0:1]
    scale = max(float(np.linalg.norm(centered[9])), 0.02)
    return (centered / scale).astype(np.float32)

def extract_word_static_from_arrays(pose_15, lh_21, rh_21, pose_ok: bool, lh_ok: bool, rh_ok: bool) -> np.ndarray:
    if pose_ok:
        anchor = 0.5 * (pose_15[7] + pose_15[8])
        scale = max(float(np.linalg.norm(pose_15[7, :2] - pose_15[8, :2])), 0.15)
        pose_norm = ((pose_15 - anchor) / scale).astype(np.float32)
    else:
        anchor = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        scale = 0.25
        pose_norm = np.zeros((15, 3), dtype=np.float32)

    lh_norm = ((lh_21 - anchor) / scale).astype(np.float32) if lh_ok else np.zeros((21, 3), dtype=np.float32)
    rh_norm = ((rh_21 - anchor) / scale).astype(np.float32) if rh_ok else np.zeros((21, 3), dtype=np.float32)
    pos_torso = np.concatenate([pose_norm, lh_norm, rh_norm], axis=0).reshape(-1)

    lh_local = normalize_hand_local(lh_21, lh_ok)
    rh_local = normalize_hand_local(rh_21, rh_ok)
    lh_angles = compute_joint_angles(lh_21) if lh_ok else np.zeros(15, dtype=np.float32)
    rh_angles = compute_joint_angles(rh_21) if rh_ok else np.zeros(15, dtype=np.float32)

    rel_dists = np.zeros(16, dtype=np.float32)
    if lh_ok:
        for i, tip in enumerate(FINGERTIP_INDICES):
            rel_dists[i] = float(np.linalg.norm(lh_local[tip]))
    if rh_ok:
        for i, tip in enumerate(FINGERTIP_INDICES):
            rel_dists[5 + i] = float(np.linalg.norm(rh_local[tip]))
    if pose_ok:
        nose = pose_norm[0]
        mouth_mid = 0.5 * (pose_norm[5] + pose_norm[6])
        if lh_ok:
            rel_dists[10] = float(np.linalg.norm(lh_norm[0] - nose))
            rel_dists[11] = float(np.linalg.norm(lh_norm[0] - mouth_mid))
        if rh_ok:
            rel_dists[12] = float(np.linalg.norm(rh_norm[0] - nose))
            rel_dists[13] = float(np.linalg.norm(rh_norm[0] - mouth_mid))
    if lh_ok and rh_ok:
        rel_dists[14] = float(np.linalg.norm(lh_norm[0] - rh_norm[0]))
        rel_dists[15] = float(np.linalg.norm(lh_norm[8] - rh_norm[8]))

    presence = np.array([float(pose_ok), float(lh_ok), float(rh_ok)], dtype=np.float32)
    static_vec = np.concatenate([
        pos_torso, lh_local.reshape(-1), rh_local.reshape(-1),
        lh_angles, rh_angles, rel_dists, presence
    ], axis=0).astype(np.float32)
    assert static_vec.shape == (WORD_STATIC_DIM,)
    return static_vec

def interpolate_short_gaps(static_seq: np.ndarray, max_gap: int = 3) -> np.ndarray:
    seq = static_seq.copy()
    if seq.shape[0] < 3:
        return seq
    flag_specs = [
        (343, [(0, 45)]),
        (344, [(45, 108), (171, 234), (297, 312), (327, 332)]),
        (345, [(108, 171), (234, 297), (312, 327), (332, 337)]),
    ]
    for flag_col, slices in flag_specs:
        valid_idx = np.where(seq[:, flag_col] > 0.5)[0]
        for i in range(len(valid_idx) - 1):
            a, b = int(valid_idx[i]), int(valid_idx[i + 1])
            if 1 <= (b - a - 1) <= max_gap:
                for t in range(a + 1, b):
                    alpha = (t - a) / (b - a)
                    for s0, s1 in slices:
                        seq[t, s0:s1] = (1.0 - alpha) * seq[a, s0:s1] + alpha * seq[b, s0:s1]
                    seq[t, flag_col] = 1.0
    return seq

def resample_sequence(seq: np.ndarray, target_len: int = WORD_SEQ_LEN) -> np.ndarray:
    t_len, dim = seq.shape
    if t_len == 0:
        return np.zeros((target_len, dim), dtype=np.float32)
    if t_len == 1:
        return np.repeat(seq, target_len, axis=0).astype(np.float32)
    if t_len == target_len:
        return seq.astype(np.float32)
    pos = np.linspace(0.0, t_len - 1, target_len, dtype=np.float32)
    idx0 = np.floor(pos).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, t_len - 1)
    alpha = (pos - idx0)[:, None]
    return ((1.0 - alpha) * seq[idx0] + alpha * seq[idx1]).astype(np.float32)

def static_to_kinematic_688(static_seq: np.ndarray, target_len: int = WORD_SEQ_LEN) -> np.ndarray:
    seq = interpolate_short_gaps(static_seq, max_gap=3)
    res = resample_sequence(seq, target_len=target_len)
    pos = res[:, :171]
    vel = np.diff(pos, axis=0, prepend=pos[:1])
    acc = np.diff(vel, axis=0, prepend=vel[:1])
    feat = np.concatenate([res, vel, acc], axis=1).astype(np.float32)
    assert feat.shape == (target_len, WORD_FEATURE_DIM)
    return feat

print("Feature engine ready. Per-frame dim:", WORD_FEATURE_DIM)"""
    ))

    cells.append((
        "code",
        """# @title 3. Load v1 Cache (or Run Selective Video Acquisition + Extraction into v2 Cache)
REUSE_V1_CACHE = True   # @param {type:"boolean"}
FORCE_REEXTRACT = False # @param {type:"boolean"}
USE_WLASL = True        # @param {type:"boolean"}
MAX_CLIPS_PER_WORD_ASL_CITIZEN = 150 if HIGH_RAM else 80  # @param {type:"integer"}
ASL_CITIZEN_URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"

V1_CACHE_PATH = V1_WORK_DIR / "word_features_cache_v1.npz"
V2_CACHE_PATH = WORK_DIR / "word_features_cache_v2.npz"
RGB_FRAMES = 16
RGB_SIZE = 112

def load_team_word_annotations():
    ann_map = {}
    for aroot in [CAL_DIR, CAL_DIR.parent, Path("/content/calibration_data/words"), Path("/content/calibration_data")]:
        csv_p = aroot / "annotations.csv"
        if csv_p.exists():
            try:
                adf = pd.read_csv(csv_p)
                for _, r in adf.iterrows():
                    rel = str(r.get("rel_path", "")).strip()
                    sid = str(r.get("signer_id", "personal_webcam")).strip() or "personal_webcam"
                    if rel:
                        ann_map[Path(rel).name] = sid
                        ann_map[rel.replace("\\\\", "/")] = sid
            except Exception as exc:
                print(f"Annotation CSV read note ({csv_p}):", exc)
    return ann_map

ann_signer_map = load_team_word_annotations()
cal_records = []
for croot in [CAL_DIR, Path("/content/calibration_data/words")]:
    if croot.exists():
        for w in WORDS_15:
            wdir = croot / w.replace(" ", "_")
            if wdir.exists():
                for mp4 in sorted(wdir.glob("*.mp4")):
                    raw_sid = ann_signer_map.get(mp4.name, "personal_webcam")
                    sid = f"team_{raw_sid}" if raw_sid not in ("anonymous", "personal_webcam", "") else "personal_webcam"
                    cal_records.append({"clip_path": str(mp4), "word": w, "label_id": WORD_TO_ID[w], "signer_id": sid, "source": "calibration"})

def run_word_mp_extraction(df_manifest: pd.DataFrame, desc_label: str):
    worker_script = Path("/content/mp_word_worker_v2.py")
    worker_code = f'''
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

POSE_INDICES = {POSE_INDICES!r}
WORD_SEQ_LEN = {WORD_SEQ_LEN}
FINGER_ANGLE_TRIPLES = {FINGER_ANGLE_TRIPLES!r}
FINGERTIP_INDICES = {FINGERTIP_INDICES!r}
RGB_FRAMES = {RGB_FRAMES}
RGB_SIZE = {RGB_SIZE}
_HAND_DET = None
_POSE_DET = None

def init_worker(hand_path: str, pose_path: str):
    global _HAND_DET, _POSE_DET
    _HAND_DET = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=hand_path), running_mode=vision.RunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=0.40, min_hand_presence_confidence=0.40)
    )
    _POSE_DET = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=pose_path), running_mode=vision.RunningMode.IMAGE, num_poses=1, min_pose_detection_confidence=0.40, min_pose_presence_confidence=0.40)
    )

def _lm_np(lms):
    arr = np.zeros((len(lms), 3), dtype=np.float32)
    for i, lm in enumerate(lms): arr[i] = (lm.x, lm.y, lm.z)
    return arr

def _angles(pts):
    if float(np.abs(pts).sum()) < 1e-6: return np.zeros(15, dtype=np.float32)
    out = np.zeros(15, dtype=np.float32)
    for i, (a, b, c) in enumerate(FINGER_ANGLE_TRIPLES):
        v1, v2 = pts[a] - pts[b], pts[c] - pts[b]
        out[i] = float(np.clip(np.dot(v1, v2) / (float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6), -1.0, 1.0))
    return out

def _hand_local(pts, ok):
    if not ok or float(np.abs(pts).sum()) < 1e-6: return np.zeros((21, 3), dtype=np.float32)
    c = pts - pts[0:1]
    return (c / max(float(np.linalg.norm(c[9])), 0.02)).astype(np.float32)

def _resample(seq):
    t_len, dim = seq.shape
    if t_len == 0: return np.zeros((WORD_SEQ_LEN, dim), dtype=np.float32)
    if t_len == 1: return np.repeat(seq, WORD_SEQ_LEN, axis=0).astype(np.float32)
    if t_len == WORD_SEQ_LEN: return seq.astype(np.float32)
    pos = np.linspace(0.0, t_len - 1, WORD_SEQ_LEN, dtype=np.float32)
    idx0 = np.floor(pos).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, t_len - 1)
    alpha = (pos - idx0)[:, None]
    return ((1.0 - alpha) * seq[idx0] + alpha * seq[idx1]).astype(np.float32)

def run_clip(task):
    idx, video_path = task
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return idx, None
    frames = []
    while True:
        ok, frm = cap.read()
        if not ok or frm is None: break
        frames.append(frm)
    cap.release()
    if len(frames) < 6: return idx, None
    if len(frames) > 96: frames = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, 96, dtype=int)]
    static_list, hand_hits = [], 0
    for frm in frames:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        h_res, p_res = _HAND_DET.detect(mp_img), _POSE_DET.detect(mp_img)
        pose_arr, pose_ok = np.zeros((15, 3), dtype=np.float32), False
        if p_res and p_res.pose_landmarks:
            pose_arr, pose_ok = _lm_np(p_res.pose_landmarks[0])[POSE_INDICES], True
        lh_arr, rh_arr = np.zeros((21, 3), dtype=np.float32), np.zeros((21, 3), dtype=np.float32)
        lh_ok, rh_ok = False, False
        if h_res and h_res.hand_landmarks:
            dets = []
            for j, lms in enumerate(h_res.hand_landmarks):
                pts = _lm_np(lms)
                lbl = h_res.handedness[j][0].category_name if (j < len(h_res.handedness) and h_res.handedness[j]) else "Right"
                sc = float(h_res.handedness[j][0].score) if (j < len(h_res.handedness) and h_res.handedness[j]) else 0.5
                dets.append((sc, lbl, pts))
            dets.sort(key=lambda x: x[0], reverse=True)
            for _, lbl, pts in dets:
                if lbl == "Left" and not lh_ok: lh_arr, lh_ok = pts, True
                elif lbl == "Right" and not rh_ok: rh_arr, rh_ok = pts, True
                elif not lh_ok: lh_arr, lh_ok = pts, True
                elif not rh_ok: rh_arr, rh_ok = pts, True
        if lh_ok or rh_ok: hand_hits += 1
        if pose_ok:
            anchor = 0.5 * (pose_arr[7] + pose_arr[8])
            scale = max(float(np.linalg.norm(pose_arr[7, :2] - pose_arr[8, :2])), 0.15)
            pose_norm = ((pose_arr - anchor) / scale).astype(np.float32)
        else:
            anchor, scale, pose_norm = np.array([0.5, 0.5, 0.0], dtype=np.float32), 0.25, np.zeros((15, 3), dtype=np.float32)
        lh_norm = ((lh_arr - anchor) / scale).astype(np.float32) if lh_ok else np.zeros((21, 3), dtype=np.float32)
        rh_norm = ((rh_arr - anchor) / scale).astype(np.float32) if rh_ok else np.zeros((21, 3), dtype=np.float32)
        pos_torso = np.concatenate([pose_norm, lh_norm, rh_norm], axis=0).reshape(-1)
        lh_l, rh_l = _hand_local(lh_arr, lh_ok), _hand_local(rh_arr, rh_ok)
        lh_a, rh_a = (_angles(lh_arr) if lh_ok else np.zeros(15, dtype=np.float32)), (_angles(rh_arr) if rh_ok else np.zeros(15, dtype=np.float32))
        rd = np.zeros(16, dtype=np.float32)
        if lh_ok:
            for k, tip in enumerate(FINGERTIP_INDICES): rd[k] = float(np.linalg.norm(lh_l[tip]))
        if rh_ok:
            for k, tip in enumerate(FINGERTIP_INDICES): rd[5 + k] = float(np.linalg.norm(rh_l[tip]))
        if pose_ok:
            nose, mouth_mid = pose_norm[0], 0.5 * (pose_norm[5] + pose_norm[6])
            if lh_ok: rd[10], rd[11] = float(np.linalg.norm(lh_norm[0] - nose)), float(np.linalg.norm(lh_norm[0] - mouth_mid))
            if rh_ok: rd[12], rd[13] = float(np.linalg.norm(rh_norm[0] - nose)), float(np.linalg.norm(rh_norm[0] - mouth_mid))
        if lh_ok and rh_ok: rd[14], rd[15] = float(np.linalg.norm(lh_norm[0] - rh_norm[0])), float(np.linalg.norm(lh_norm[8] - rh_norm[8]))
        pres = np.array([float(pose_ok), float(lh_ok), float(rh_ok)], dtype=np.float32)
        static_list.append(np.concatenate([pos_torso, lh_l.reshape(-1), rh_l.reshape(-1), lh_a, rh_a, rd, pres], axis=0).astype(np.float32))
    if (hand_hits / len(frames)) < 0.18: return idx, None
    static_arr = _resample(np.stack(static_list, axis=0))
    rgb_idx = np.linspace(0, len(frames) - 1, RGB_FRAMES, dtype=int)
    rgb_clip = np.stack([cv2.cvtColor(cv2.resize(frames[int(i)], (RGB_SIZE, RGB_SIZE)), cv2.COLOR_BGR2RGB) for i in rgb_idx], axis=0).astype(np.uint8)
    return idx, (static_arr, rgb_clip)
'''
    worker_script.write_text(worker_code, encoding="utf-8")
    import sys
    if "/content" not in sys.path:
        sys.path.insert(0, "/content")
    import mp_word_worker_v2

    tasks = list(enumerate(df_manifest["clip_path"].tolist()))
    results_map = {}
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=EXTRACTION_WORKERS,
        mp_context=ctx,
        initializer=mp_word_worker_v2.init_worker,
        initargs=(str(MP_ASSETS_DIR / "hand_landmarker.task"), str(MP_ASSETS_DIR / "pose_landmarker_lite.task")),
    ) as pool:
        for idx, out in tqdm(pool.map(mp_word_worker_v2.run_clip, tasks, chunksize=4), total=len(tasks), desc=desc_label):
            if out is not None:
                results_map[idx] = out

    s_list, r_list, y_list, sg_list, src_list = [], [], [], [], []
    for idx in sorted(results_map.keys()):
        st, rc = results_map[idx]
        row = df_manifest.iloc[idx]
        s_list.append(st); r_list.append(rc); y_list.append(int(row["label_id"]))
        sg_list.append(str(row["signer_id"])); src_list.append(str(row["source"]))
    if not s_list:
        return None
    return (
        np.stack(s_list, axis=0).astype(np.float32),
        np.stack(r_list, axis=0).astype(np.uint8),
        np.array(y_list, dtype=np.int64),
        np.array(sg_list),
        np.array(src_list),
    )

cache_loaded = False
if not FORCE_REEXTRACT:
    if V2_CACHE_PATH.exists():
        with np.load(V2_CACHE_PATH, allow_pickle=True) as cached:
            if int(cached.get("cal_len", -1)) == len(cal_records):
                static_seqs = cached["static_seqs"]
                rgb_clips = cached["rgb_clips"]
                labels = cached["labels"]
                signers = cached["signers"]
                sources = cached["sources"]
                cache_loaded = True
                print(f"Loaded v2 cache ({len(labels)} clips, cal={len(cal_records)}) from {V2_CACHE_PATH}!")
    elif REUSE_V1_CACHE and V1_CACHE_PATH.exists():
        print(f"Reusing base extracted sequences from {V1_CACHE_PATH}...")
        with np.load(V1_CACHE_PATH, allow_pickle=True) as cached:
            keep = cached["sources"] != "calibration"
            static_seqs = cached["static_seqs"][keep]
            rgb_clips = cached["rgb_clips"][keep]
            labels = cached["labels"][keep]
            signers = cached["signers"][keep]
            sources = cached["sources"][keep]
        if cal_records:
            cal_df = pd.DataFrame(cal_records).drop_duplicates(subset=["clip_path"]).reset_index(drop=True)
            cal_ext = run_word_mp_extraction(cal_df, f"Extracting {len(cal_df)} team calibration clips")
            if cal_ext is not None:
                cs, cr, cy, csg, csrc = cal_ext
                static_seqs = np.concatenate([static_seqs, cs], axis=0)
                rgb_clips = np.concatenate([rgb_clips, cr], axis=0)
                labels = np.concatenate([labels, cy], axis=0)
                signers = np.concatenate([signers, csg], axis=0)
                sources = np.concatenate([sources, csrc], axis=0)
        np.savez_compressed(
            V2_CACHE_PATH, cal_len=len(cal_records),
            static_seqs=static_seqs, rgb_clips=rgb_clips,
            labels=labels, signers=signers, sources=sources,
        )
        cache_loaded = True
        print(f"Ready with {len(labels)} clips (including {len(cal_records)} team calibration files)!")

if not cache_loaded:
    from remotezip import RemoteZip
    citizen_dir = RAW_DIR / "asl_citizen"
    citizen_dir.mkdir(parents=True, exist_ok=True)
    records = []

    try:
        print("Inspecting ASL Citizen remote archive (selective download of 15 target words)...")
        with RemoteZip(ASL_CITIZEN_URL) as rz:
            names = rz.namelist()
            csv_members = [n for n in names if n.lower().endswith(".csv") and any(s in n.lower() for s in ("train", "val", "test"))]
            video_lookup = {Path(n).name: n for n in names if n.lower().endswith(".mp4")}
            cand_rows = []
            for csv_m in csv_members:
                extracted_csv = rz.extract(csv_m, path=str(citizen_dir))
                df = pd.read_csv(extracted_csv)
                cols = {c.lower().strip(): c for c in df.columns}
                gloss_col = cols.get("gloss") or cols.get("sign") or list(df.columns)[2]
                vid_col = cols.get("video file") or cols.get("video") or cols.get("file") or list(df.columns)[1]
                signer_col = cols.get("participant id") or cols.get("signer") or cols.get("user") or list(df.columns)[0]
                df["canon_word"] = df[gloss_col].astype(str).map(normalize_gloss)
                for _, r in df[df["canon_word"].isin(WORDS_15)].iterrows():
                    vname = Path(str(r[vid_col]).strip()).name
                    member = video_lookup.get(vname)
                    if member:
                        cand_rows.append((member, r["canon_word"], f"ac_{r[signer_col]}"))

            # Uniform per-canonical-word cap to prevent 2x EAT/WANT gloss imbalance
            cand_df = pd.DataFrame(cand_rows, columns=["member", "word", "signer"]).drop_duplicates(subset=["member"])
            selected = []
            for w, grp in cand_df.groupby("word"):
                selected.extend(grp.sample(min(len(grp), MAX_CLIPS_PER_WORD_ASL_CITIZEN), random_state=SEED).itertuples(index=False, name=None))
            for member, word, signer in tqdm(selected, desc="ASL Citizen"):
                lp = citizen_dir / member
                if not lp.exists():
                    rz.extract(member, path=str(citizen_dir))
                records.append({"clip_path": str(lp), "word": word, "label_id": WORD_TO_ID[word], "signer_id": signer, "source": "asl_citizen"})
    except Exception as exc:
        print("ASL Citizen selective download skipped:", exc)

    if USE_WLASL:
        try:
            import kagglehub
            wlasl_root = Path(kagglehub.dataset_download("risangbaskoro/wlasl-processed"))
            meta_json = next(wlasl_root.rglob("WLASL_v0.3.json"), None)
            if meta_json is not None:
                data = json.loads(meta_json.read_text(encoding="utf-8"))
                vid_files = {p.stem: p for p in wlasl_root.rglob("*.mp4")}
                for entry in data:
                    canon = normalize_gloss(entry.get("gloss", ""))
                    if canon not in WORD_TO_ID:
                        continue
                    for inst in entry.get("instances", []):
                        vid_id = str(inst.get("video_id", "")).zfill(5)
                        p = vid_files.get(vid_id) or vid_files.get(str(inst.get("video_id", "")))
                        if p and p.exists():
                            records.append({"clip_path": str(p), "word": canon, "label_id": WORD_TO_ID[canon], "signer_id": f"wl_{inst.get('signer_id', 'unk')}", "source": "wlasl"})
        except Exception as exc:
            print("WLASL optional enrichment skipped:", exc)

    records.extend(cal_records)
    manifest_df = pd.DataFrame(records).drop_duplicates(subset=["clip_path"]).reset_index(drop=True)
    assert len(manifest_df) > 0, "No video clips found!"
    ext_all = run_word_mp_extraction(manifest_df, f"Extracting ({EXTRACTION_WORKERS} workers)")
    assert ext_all is not None, "MediaPipe extraction returned zero valid clips!"
    static_seqs, rgb_clips, labels, signers, sources = ext_all
    np.savez_compressed(V2_CACHE_PATH, cal_len=len(cal_records), static_seqs=static_seqs, rgb_clips=rgb_clips, labels=labels, signers=signers, sources=sources)

ram_mb = (static_seqs.nbytes + rgb_clips.nbytes) / 1e6
print(f"In-RAM clips: {len(labels)} | Unique signers: {len(set(signers))} | RAM: {ram_mb:.1f} MB / {TOTAL_RAM_GB:.1f} GB")"""
    ))

    cells.append((
        "code",
        """# @title 4. Figure 1: Exploratory Data Analysis & Class-Balance Inspection
eda_df = pd.DataFrame({
    "word": [WORDS_15[int(y)] for y in labels],
    "source": [str(s) for s in sources],
    "signer": [str(g) for g in signers],
})
display(eda_df.groupby(["word", "source"]).size().unstack(fill_value=0).reindex(WORDS_15))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
eda_df.groupby(["word", "source"]).size().unstack(fill_value=0).reindex(WORDS_15).plot(
    kind="bar", stacked=True, ax=axes[0], colormap="viridis"
)
axes[0].set_title("Curated Video Clips per ASL Word (Notice eat/want Imbalance Fixed in Sampler)")
axes[0].set_xlabel("Target Word")
axes[0].set_ylabel("Clips")
axes[0].tick_params(axis="x", rotation=40)

signer_counts = eda_df.groupby("word")["signer"].nunique().reindex(WORDS_15)
signer_counts.plot(kind="bar", ax=axes[1], color="#4e79a7")
axes[1].set_title("Unique Signers per ASL Word")
axes[1].set_xlabel("Target Word")
axes[1].set_ylabel("Unique Signers")
axes[1].tick_params(axis="x", rotation=40)

plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_eda_dataset.png", dpi=300)
plt.show()"""
    ))

    cells.append((
        "code",
        """# @title 5. Signer-Disjoint StratifiedGroupKFold Split, Class Balancing & Regularized Datasets
from sklearn.model_selection import StratifiedGroupKFold

non_cal_mask = (sources != "calibration")
pool_idx = np.where(non_cal_mask)[0]
cal_idx = np.where(~non_cal_mask)[0]

sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=SEED)
fold_splits = list(sgkf.split(pool_idx, labels[pool_idx], groups=signers[pool_idx]))
val_idx = pool_idx[fold_splits[0][1]]
test_idx = pool_idx[fold_splits[1][1]]
train_base_idx = np.setdiff1d(pool_idx, np.concatenate([val_idx, test_idx]))

# Keep personal calibration clips strictly out of test set
if len(cal_idx):
    train_base_idx = np.concatenate([train_base_idx, np.repeat(cal_idx, 3)])

assert set(signers[train_base_idx]).isdisjoint(set(signers[test_idx])), "Signer leak into Test!"
assert set(signers[val_idx]).isdisjoint(set(signers[test_idx])), "Signer leak Val/Test!"
assert set(signers[train_base_idx]).isdisjoint(set(signers[val_idx])), "Signer leak Train/Val!"

# Inverse-sqrt class weights + WeightedRandomSampler to fix eat/want majority bias
tr_counts = np.bincount(labels[train_base_idx], minlength=len(WORDS_15))
raw_cls_w = 1.0 / np.sqrt(np.maximum(tr_counts, 1))
cls_weights_np = (raw_cls_w / raw_cls_w.mean()).astype(np.float32)
CLASS_WEIGHTS_T = torch.tensor(cls_weights_np, dtype=torch.float32, device=DEVICE)

sample_weights = torch.tensor([1.0 / max(int(tr_counts[labels[i]]), 1) for i in train_base_idx], dtype=torch.double)
train_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_base_idx), replacement=True)

print(f"Strict Signer-Disjoint Split -> Train: {len(train_base_idx)} ({len(set(signers[train_base_idx]))} signers) | Val: {len(val_idx)} ({len(set(signers[val_idx]))} signers) | Test: {len(test_idx)} ({len(set(signers[test_idx]))} signers)")
print("Min/Max train clips per class:", int(tr_counts.min()), "/", int(tr_counts.max()))

def mirror_static_sequence(seq: np.ndarray) -> np.ndarray:
    out = seq.copy()
    t = out.shape[0]
    pose = out[:, 0:45].reshape(t, 15, 3)
    pose[:, :, 0] *= -1.0
    pose = pose[:, POSE_SWAP_PERM, :]
    lh_t = out[:, 45:108].reshape(t, 21, 3).copy()
    rh_t = out[:, 108:171].reshape(t, 21, 3).copy()
    lh_t[:, :, 0] *= -1.0; rh_t[:, :, 0] *= -1.0
    out[:, 0:45] = pose.reshape(t, 45); out[:, 45:108] = rh_t.reshape(t, 63); out[:, 108:171] = lh_t.reshape(t, 63)
    lh_l = out[:, 171:234].reshape(t, 21, 3).copy()
    rh_l = out[:, 234:297].reshape(t, 21, 3).copy()
    lh_l[:, :, 0] *= -1.0; rh_l[:, :, 0] *= -1.0
    out[:, 171:234] = rh_l.reshape(t, 63); out[:, 234:297] = lh_l.reshape(t, 63)
    lh_a, rh_a = out[:, 297:312].copy(), out[:, 312:327].copy()
    out[:, 297:312] = rh_a; out[:, 312:327] = lh_a
    rd = out[:, 327:343].copy()
    out[:, 327:332] = rd[:, 5:10]; out[:, 332:337] = rd[:, 0:5]
    out[:, 337:339] = rd[:, 12:14]; out[:, 339:341] = rd[:, 10:12]
    pr = out[:, 343:346].copy()
    out[:, 344] = pr[:, 2]; out[:, 345] = pr[:, 1]
    return out

def augment_static_sequence(seq: np.ndarray) -> np.ndarray:
    out = seq.copy()
    if random.random() < 0.50:
        out = mirror_static_sequence(out)
    if random.random() < 0.70:
        t = out.shape[0]
        start, end = random.randint(0, 5), t - random.randint(0, 5)
        if end - start >= 24:
            out = resample_sequence(out[start:end], WORD_SEQ_LEN)
    if random.random() < 0.75:
        scale = random.uniform(0.88, 1.14)
        theta = np.deg2rad(random.uniform(-10.0, 10.0))
        c, s = np.cos(theta), np.sin(theta)
        pts = out[:, :171].reshape(WORD_SEQ_LEN, 57, 3)
        xn = (pts[:, :, 0] * c - pts[:, :, 1] * s) * scale
        yn = (pts[:, :, 0] * s + pts[:, :, 1] * c) * scale
        pts[:, :, 0], pts[:, :, 1] = xn, yn
        out[:, :171] = pts.reshape(WORD_SEQ_LEN, 171)
    # Temporal hand-dropout simulation (MediaPipe flicker robustness)
    if random.random() < 0.25:
        drop_frames = random.sample(range(WORD_SEQ_LEN), random.randint(1, 3))
        if random.random() < 0.5:
            out[drop_frames, 45:108] = 0.0; out[drop_frames, 171:234] = 0.0; out[drop_frames, 344] = 0.0
        else:
            out[drop_frames, 108:171] = 0.0; out[drop_frames, 234:297] = 0.0; out[drop_frames, 345] = 0.0
    if random.random() < 0.65:
        out[:, :297] += np.random.normal(0.0, 0.010, size=(WORD_SEQ_LEN, 297)).astype(np.float32)
    return out

class WordKinematicDataset(Dataset):
    def __init__(self, indices, augment: bool = False):
        self.indices = indices
        self.augment = augment
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        i = int(self.indices[idx])
        st = augment_static_sequence(static_seqs[i]) if self.augment else static_seqs[i]
        feat = static_to_kinematic_688(st, target_len=WORD_SEQ_LEN)
        return torch.from_numpy(feat), torch.tensor(labels[i], dtype=torch.long)

class WordRgbDataset(Dataset):
    def __init__(self, indices, augment: bool = False):
        self.indices = indices
        self.augment = augment
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        i = int(self.indices[idx])
        clip = rgb_clips[i]
        if self.augment and random.random() < 0.5:
            clip = clip[:, :, ::-1, :].copy()
        x = torch.from_numpy(clip).permute(0, 3, 1, 2).float() / 255.0
        if self.augment:
            x = torch.clamp(x * random.uniform(0.85, 1.15), 0.0, 1.0)
        return (x - self.mean) / self.std, torch.tensor(labels[i], dtype=torch.long)

def make_kin_loaders(tr_idx, va_idx, te_idx):
    tr_cnt = np.bincount(labels[tr_idx], minlength=len(WORDS_15))
    sw = torch.tensor([1.0 / max(int(tr_cnt[labels[i]]), 1) for i in tr_idx], dtype=torch.double)
    smp = WeightedRandomSampler(weights=sw, num_samples=len(tr_idx), replacement=True)
    tr_l = DataLoader(WordKinematicDataset(tr_idx, augment=True), batch_size=KIN_BATCH_SIZE, sampler=smp, num_workers=LOADER_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    va_x = torch.from_numpy(np.stack([static_to_kinematic_688(static_seqs[i], WORD_SEQ_LEN) for i in va_idx], axis=0))
    va_l = DataLoader(TensorDataset(va_x, torch.from_numpy(labels[va_idx]).long()), batch_size=KIN_BATCH_SIZE, shuffle=False, pin_memory=True)
    te_l = None
    if te_idx is not None:
        te_x = torch.from_numpy(np.stack([static_to_kinematic_688(static_seqs[i], WORD_SEQ_LEN) for i in te_idx], axis=0))
        te_l = DataLoader(TensorDataset(te_x, torch.from_numpy(labels[te_idx]).long()), batch_size=KIN_BATCH_SIZE, shuffle=False, pin_memory=True)
    return tr_l, va_l, te_l, tr_cnt

train_kin_loader, val_kin_loader, test_kin_loader, _ = make_kin_loaders(train_base_idx, val_idx, test_idx)
train_rgb_loader = DataLoader(WordRgbDataset(train_base_idx, augment=True), batch_size=RGB_BATCH_SIZE, sampler=train_sampler, num_workers=LOADER_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val_rgb_loader = DataLoader(WordRgbDataset(val_idx, augment=False), batch_size=RGB_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_rgb_loader = DataLoader(WordRgbDataset(test_idx, augment=False), batch_size=RGB_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)"""
    ))

    cells.append((
        "code",
        """# @title 6. Define 4 Base Models + Dual-Architecture Ensemble (GroupNorm Conformer + Kinematic BiGRU)
import torchvision.models as tv_models

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        mask = torch.empty((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x.div(keep) * mask

class RgbVideoBaseline(nn.Module):
    def __init__(self, num_classes: int = 15):
        super().__init__()
        backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.gru = nn.GRU(512, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, num_classes))
    def forward(self, x):
        b, t, c, h, w = x.shape
        feats = self.encoder(x.view(b * t, c, h, w)).view(b, t, 512)
        out, _ = self.gru(feats)
        return self.head(out.mean(dim=1))

class PoseBiGRUBaseline(nn.Module):
    def __init__(self, in_dim: int = 171, num_classes: int = 15):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, 160), nn.LayerNorm(160), nn.GELU())
        self.gru = nn.GRU(160, 160, num_layers=2, batch_first=True, bidirectional=True, dropout=0.25)
        self.head = nn.Sequential(nn.LayerNorm(320), nn.Dropout(0.3), nn.Linear(320, num_classes))
    def forward(self, x):
        h, _ = self.gru(self.proj(x[:, :, :171]))
        return self.head(h.mean(dim=1))

class KinematicBiGRU(nn.Module):
    \"\"\"Ablation 3: Full 688-dim kinematic representation with BiGRU + Attention Pooling.\"\"\"
    def __init__(self, in_dim: int = WORD_FEATURE_DIM, hidden: int = 192, num_classes: int = 15):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.25))
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.30)
        self.attn_pool = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Dropout(0.30), nn.Linear(hidden * 2, num_classes))
    def forward(self, x, return_details: bool = False):
        h, _ = self.gru(self.proj(x))
        w = torch.softmax(self.attn_pool(h).squeeze(-1), dim=1)
        emb = torch.sum(h * w.unsqueeze(-1), dim=1)
        logits = self.head(emb)
        if return_details:
            return logits, w, emb
        return logits

class ExplicitMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.25):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(self.drop(attn), v).transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)

class ConformerBlockV2(nn.Module):
    def __init__(self, d_model: int = 144, n_heads: int = 4, conv_kernel: int = 7, dropout: float = 0.25, drop_path: float = 0.10):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = ExplicitMultiHeadSelfAttention(d_model, n_heads, dropout=dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.conv_norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=d_model)
        self.gn = nn.GroupNorm(8, d_model)  # Replaces BatchNorm1d to prevent train/eval drift & simplify ONNX
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.conv_drop = nn.Dropout(dropout)
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
        )
        self.dp = DropPath(drop_path)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.dp(0.5 * self.ff1(x))
        x = x + self.dp(self.attn_drop(self.attn(self.attn_norm(x))))
        c = self.conv_norm(x).transpose(1, 2)
        a, b_gate = self.pw1(c).chunk(2, dim=1)
        c = F.silu(self.gn(self.dw(a * torch.sigmoid(b_gate))))
        x = x + self.dp(self.conv_drop(self.pw2(c)).transpose(1, 2))
        x = x + self.dp(0.5 * self.ff2(x))
        return self.final_norm(x)

class WordConformerV2(nn.Module):
    def __init__(self, in_dim: int = WORD_FEATURE_DIM, d_model: int = 144, n_layers: int = 3, num_classes: int = 15):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(0.25))
        self.pos_emb = nn.Parameter(torch.randn(1, WORD_SEQ_LEN, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            ConformerBlockV2(d_model=d_model, n_heads=4, conv_kernel=7, dropout=0.25, drop_path=0.10 * (i + 1) / n_layers)
            for i in range(n_layers)
        ])
        self.attn_pool = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.Tanh(), nn.Linear(d_model // 2, 1))
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.30), nn.Linear(d_model, num_classes))

    def forward(self, x, return_details: bool = False):
        h = self.input_proj(x) + self.pos_emb
        for blk in self.blocks:
            h = blk(h)
        weights = torch.softmax(self.attn_pool(h).squeeze(-1), dim=1)
        emb = torch.sum(h * weights.unsqueeze(-1), dim=1)
        logits = self.head(emb)
        if return_details:
            return logits, weights, emb
        return logits

class WordDualEnsemble(nn.Module):
    \"\"\"Single exportable ONNX wrapper combining KinematicBiGRU and WordConformerV2.\"\"\"
    def __init__(self, gru_net: KinematicBiGRU, conf_net: WordConformerV2, alpha: float = 0.5):
        super().__init__()
        self.gru_net = gru_net
        self.conf_net = conf_net
        self.alpha = float(alpha)

    def forward(self, x):
        return (1.0 - self.alpha) * self.gru_net(x) + self.alpha * self.conf_net(x)

print("KinematicBiGRU params:", sum(p.numel() for p in KinematicBiGRU().parameters()) // 1000, "K")
print("WordConformerV2 params:", sum(p.numel() for p in WordConformerV2().parameters()) // 1000, "K")"""
    ))

    cells.append((
        "code",
        """# @title 7. Training Engine & 5-Model Ablation Benchmark (With Fixed EMA + Class Balancing)
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

def evaluate_model(model, loader):
    model.eval()
    all_logits, all_y = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(DEVICE, non_blocking=True))
            all_logits.append(logits.cpu().numpy())
            all_y.append(yb.numpy())
    elapsed_ms = ((time.perf_counter() - t0) / max(len(loader.dataset), 1)) * 1000.0
    logits_np = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    probs = torch.softmax(torch.from_numpy(logits_np), dim=-1).numpy()
    preds = np.argmax(probs, axis=1)
    return {
        "top1": accuracy_score(y_true, preds),
        "top3": top_k_accuracy_score(y_true, probs, k=3, labels=list(range(len(WORDS_15)))),
        "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, preds, average="weighted", zero_division=0),
        "latency_ms": elapsed_ms,
        "y_true": y_true,
        "y_pred": preds,
        "probs": probs,
        "logits": logits_np,
    }

class SupervisedContrastiveLoss(nn.Module):
    \"\"\"L2-normalized temperature-scaled Supervised Contrastive Loss on sequence embeddings.\"\"\"
    def __init__(self, temperature: float = 0.12):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.size(0) < 2:
            return features.new_tensor(0.0)
        z = F.normalize(features.float(), dim=1)
        sim = torch.matmul(z, z.T) / self.temperature
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - sim_max.detach()
        same_cls = labels.unsqueeze(0) == labels.unsqueeze(1)
        self_mask = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        pos_mask = same_cls & (~self_mask)
        exp_logits = torch.exp(logits) * (~self_mask).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        pos_per_row = pos_mask.sum(dim=1)
        valid = pos_per_row > 0
        if not valid.any():
            return features.new_tensor(0.0)
        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1)[valid] / pos_per_row[valid].float()
        return -mean_log_prob_pos.mean()

def train_word_model(model, train_loader, val_loader, epochs: int, lr: float, name: str, class_weights=None, use_mixup: bool = False, use_ema: bool = False, supcon_weight: float = 0.0, save_ckpt: bool = True):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.02)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.08)
    supcon_crit = SupervisedContrastiveLoss(temperature=0.12) if supcon_weight > 0.0 else None
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    ema_model = deepcopy(model) if use_ema else None
    best_score = -1.0
    best_state = deepcopy(model.state_dict())
    history = {"train_loss": [], "val_acc": [], "val_f1": []}
    global_step = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                if use_mixup and random.random() < 0.45 and xb.size(0) > 1:
                    lam = float(np.random.beta(0.3, 0.3))
                    perm = torch.randperm(xb.size(0), device=DEVICE)
                    logits = model(lam * xb + (1.0 - lam) * xb[perm])
                    loss = lam * criterion(logits, yb) + (1.0 - lam) * criterion(logits, yb[perm])
                elif supcon_crit is not None:
                    logits, _, emb = model(xb, return_details=True)
                    loss = criterion(logits, yb) + supcon_weight * supcon_crit(emb, yb)
                else:
                    loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * xb.size(0)
            global_step += 1

            if ema_model is not None:
                # Warm-up EMA decay + buffer sync fixes v1 cold-init EMA bug
                decay = min(0.96, (1.0 + global_step) / (10.0 + global_step))
                with torch.no_grad():
                    for p_ema, p_cur in zip(ema_model.parameters(), model.parameters()):
                        p_ema.data.mul_(decay).add_(p_cur.data, alpha=1.0 - decay)
                    for b_ema, b_cur in zip(ema_model.buffers(), model.buffers()):
                        b_ema.copy_(b_cur)

        scheduler.step()
        eval_net = ema_model if ema_model is not None else model
        val_m = evaluate_model(eval_net, val_loader)
        tr_loss = total_loss / max(len(train_loader.dataset), 1)
        history["train_loss"].append(tr_loss)
        history["val_acc"].append(val_m["top1"])
        history["val_f1"].append(val_m["macro_f1"])

        score = 0.5 * val_m["macro_f1"] + 0.5 * val_m["top1"]
        if score > best_score:
            best_score = score
            best_state = deepcopy(eval_net.state_dict())

        if save_ckpt and (ep in (1, epochs) or ep % max(1, epochs // 5) == 0):
            print(f"[{name}] Ep {ep:02d}/{epochs} | Loss {tr_loss:.4f} | Val Acc {val_m['top1']*100:.1f}% | Val F1 {val_m['macro_f1']*100:.1f}%")

    model.load_state_dict(best_state)
    if save_ckpt:
        torch.save(best_state, WORK_DIR / f"{name}_best.pt")
    return model, history

print("=== Baseline 1: RGB ResNet-18 + BiGRU ===")
rgb_model, hist_rgb = train_word_model(RgbVideoBaseline(len(WORDS_15)), train_rgb_loader, val_rgb_loader, epochs=12, lr=3e-4, name="rgb_baseline", class_weights=CLASS_WEIGHTS_T)

print("\\n=== Baseline 2: Pose-Only BiGRU (171d) ===")
pose_gru_model, hist_pose_gru = train_word_model(PoseBiGRUBaseline(171, len(WORDS_15)), train_kin_loader, val_kin_loader, epochs=45, lr=1e-3, name="pose_bigru", class_weights=CLASS_WEIGHTS_T)

print("\\n=== Ablation 3: Kinematic BiGRU (688d + Attention Pool) ===")
kin_gru_model, hist_kin_gru = train_word_model(KinematicBiGRU(WORD_FEATURE_DIM, 192, len(WORDS_15)), train_kin_loader, val_kin_loader, epochs=55, lr=9e-4, name="kinematic_bigru", class_weights=CLASS_WEIGHTS_T, use_mixup=True, use_ema=True)

print("\\n=== Ablation 4: Kinematic Conformer v2 (688d, CE Only) ===")
conf_v2_ce_model, hist_conf_v2_ce = train_word_model(WordConformerV2(WORD_FEATURE_DIM, 144, 3, len(WORDS_15)), train_kin_loader, val_kin_loader, epochs=65, lr=9e-4, name="kinematic_conformer_v2_ce", class_weights=CLASS_WEIGHTS_T, use_mixup=True, use_ema=True, supcon_weight=0.0)

print("\\n=== Proposed 5: Kinematic Conformer v2 + SupConLoss (Novel Contrastive Sequence Space) ===")
conf_v2_model, hist_conf_v2 = train_word_model(WordConformerV2(WORD_FEATURE_DIM, 144, 3, len(WORDS_15)), train_kin_loader, val_kin_loader, epochs=65, lr=9e-4, name="kinematic_conformer_v2_supcon", class_weights=CLASS_WEIGHTS_T, use_mixup=True, use_ema=True, supcon_weight=0.15)

# Tune ensemble weight alpha on Validation Macro-F1 across [0.0..1.0]
val_gru = evaluate_model(kin_gru_model, val_kin_loader)
val_conf = evaluate_model(conf_v2_model, val_kin_loader)
best_alpha, best_val_f1 = 0.5, -1.0
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    blend_probs = (1.0 - alpha) * val_gru["logits"] + alpha * val_conf["logits"]
    vf1 = f1_score(val_gru["y_true"], np.argmax(blend_probs, axis=1), average="macro", zero_division=0)
    if vf1 > best_val_f1:
        best_val_f1 = vf1
        best_alpha = alpha

ensemble_model = WordDualEnsemble(kin_gru_model, conf_v2_model, alpha=best_alpha).to(DEVICE)
print(f"Selected Dual Ensemble alpha = {best_alpha:.2f} (Val Macro-F1 = {best_val_f1*100:.2f}%)")

res_rgb = evaluate_model(rgb_model, test_rgb_loader)
res_pose_gru = evaluate_model(pose_gru_model, test_kin_loader)
res_kin_gru = evaluate_model(kin_gru_model, test_kin_loader)
res_conf_ce = evaluate_model(conf_v2_ce_model, test_kin_loader)
res_conf_v2 = evaluate_model(conf_v2_model, test_kin_loader)
res_best = evaluate_model(ensemble_model, test_kin_loader)

bench_df = pd.DataFrame([
    {"Model": "Baseline 1: RGB ResNet18 + BiGRU", "Top-1 Acc (%)": res_rgb["top1"]*100, "Top-3 Acc (%)": res_rgb["top3"]*100, "Macro F1 (%)": res_rgb["macro_f1"]*100, "Params (M)": sum(p.numel() for p in rgb_model.parameters())/1e6},
    {"Model": "Baseline 2: Pose-Only BiGRU (171d)", "Top-1 Acc (%)": res_pose_gru["top1"]*100, "Top-3 Acc (%)": res_pose_gru["top3"]*100, "Macro F1 (%)": res_pose_gru["macro_f1"]*100, "Params (M)": sum(p.numel() for p in pose_gru_model.parameters())/1e6},
    {"Model": "Ablation 3: Kinematic BiGRU (688d)", "Top-1 Acc (%)": res_kin_gru["top1"]*100, "Top-3 Acc (%)": res_kin_gru["top3"]*100, "Macro F1 (%)": res_kin_gru["macro_f1"]*100, "Params (M)": sum(p.numel() for p in kin_gru_model.parameters())/1e6},
    {"Model": "Ablation 4: Conformer v2 (688d, CE)", "Top-1 Acc (%)": res_conf_ce["top1"]*100, "Top-3 Acc (%)": res_conf_ce["top3"]*100, "Macro F1 (%)": res_conf_ce["macro_f1"]*100, "Params (M)": sum(p.numel() for p in conf_v2_ce_model.parameters())/1e6},
    {"Model": "Proposed 5: Conformer v2 + SupConLoss", "Top-1 Acc (%)": res_conf_v2["top1"]*100, "Top-3 Acc (%)": res_conf_v2["top3"]*100, "Macro F1 (%)": res_conf_v2["macro_f1"]*100, "Params (M)": sum(p.numel() for p in conf_v2_model.parameters())/1e6},
    {"Model": f"Selected Export: Dual Ensemble (a={best_alpha:.2f})", "Top-1 Acc (%)": res_best["top1"]*100, "Top-3 Acc (%)": res_best["top3"]*100, "Macro F1 (%)": res_best["macro_f1"]*100, "Params (M)": sum(p.numel() for p in ensemble_model.parameters())/1e6},
])
bench_df.to_csv(WORK_DIR / "benchmark_results_v2.csv", index=False)
display(bench_df.round(2))"""
    ))

    cells.append((
        "code",
        """# @title 8. 5-Fold Signer-Independent Cross-Validation (Uses Premium Colab GPU)
RUN_5FOLD_CV = True  # @param {type:"boolean"}

cv5_records = []
if RUN_5FOLD_CV:
    sgkf5 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold_id, (tr_rel, te_rel) in enumerate(sgkf5.split(pool_idx, labels[pool_idx], groups=signers[pool_idx]), start=1):
        tr_f = pool_idx[tr_rel]
        te_f = pool_idx[te_rel]
        if len(cal_idx):
            tr_f = np.concatenate([tr_f, cal_idx])
        tr_loader_f, te_loader_f, _, tr_cnt_f = make_kin_loaders(tr_f, te_f, None)
        rw = 1.0 / np.sqrt(np.maximum(tr_cnt_f, 1))
        cw_f = torch.tensor((rw / rw.mean()).astype(np.float32), device=DEVICE)

        m_pose, _ = train_word_model(PoseBiGRUBaseline(171, len(WORDS_15)), tr_loader_f, te_loader_f, epochs=40, lr=1e-3, name=f"cv_pose_f{fold_id}", class_weights=cw_f, save_ckpt=False)
        m_kgru, _ = train_word_model(KinematicBiGRU(WORD_FEATURE_DIM, 192, len(WORDS_15)), tr_loader_f, te_loader_f, epochs=50, lr=9e-4, name=f"cv_kgru_f{fold_id}", class_weights=cw_f, use_mixup=True, use_ema=True, save_ckpt=False)
        m_conf, _ = train_word_model(WordConformerV2(WORD_FEATURE_DIM, 144, 3, len(WORDS_15)), tr_loader_f, te_loader_f, epochs=55, lr=9e-4, name=f"cv_conf_f{fold_id}", class_weights=cw_f, use_mixup=True, use_ema=True, supcon_weight=0.15, save_ckpt=False)

        r_p = evaluate_model(m_pose, te_loader_f)
        r_g = evaluate_model(m_kgru, te_loader_f)
        r_c = evaluate_model(m_conf, te_loader_f)
        ens_probs = (1.0 - best_alpha) * r_g["logits"] + best_alpha * r_c["logits"]
        ens_pred = np.argmax(ens_probs, axis=1)
        ens_acc = accuracy_score(r_g["y_true"], ens_pred)
        ens_f1 = f1_score(r_g["y_true"], ens_pred, average="macro", zero_division=0)

        for label_m, acc_v, f1_v in [
            ("Pose BiGRU (171d)", r_p["top1"], r_p["macro_f1"]),
            ("Kinematic BiGRU (688d)", r_g["top1"], r_g["macro_f1"]),
            ("Kinematic Conformer v2 (688d)", r_c["top1"], r_c["macro_f1"]),
            (f"Dual Ensemble (a={best_alpha:.2f})", ens_acc, ens_f1),
        ]:
            cv5_records.append({"Fold": fold_id, "Model": label_m, "Top1": acc_v * 100, "MacroF1": f1_v * 100})
        print(f"[5-Fold CV {fold_id}/5] PoseGRU {r_p['top1']*100:.1f}% | KinGRU {r_g['top1']*100:.1f}% | ConfV2 {r_c['top1']*100:.1f}% | Ensemble {ens_acc*100:.1f}%")

    cv_df = pd.DataFrame(cv5_records)
    cv_summary = cv_df.groupby("Model", sort=False).agg(
        Top1_Mean=("Top1", "mean"), Top1_Std=("Top1", "std"),
        MacroF1_Mean=("MacroF1", "mean"), MacroF1_Std=("MacroF1", "std"),
    ).reset_index()
    cv_summary.to_csv(WORK_DIR / "cv5_summary.csv", index=False)
    display(cv_summary.round(2))
else:
    cv_summary = pd.DataFrame([{"Model": row["Model"], "Top1_Mean": row["Top-1 Acc (%)"], "Top1_Std": 0.0, "MacroF1_Mean": row["Macro F1 (%)"], "MacroF1_Std": 0.0} for _, row in bench_df.iloc[1:].iterrows()])"""
    ))

    cells.append((
        "code",
        """# @title 9. Academic Figures 2–7 (Training Curves, Confusion Matrix, F1, Saliency, UMAP, Ablation & 5-Fold CV)
from sklearn.metrics import confusion_matrix, classification_report
import umap

# Fig 2: Training Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for lbl, hist in [
    ("RGB Baseline", hist_rgb), ("Pose BiGRU (171d)", hist_pose_gru),
    ("Kinematic BiGRU (688d)", hist_kin_gru), ("Conformer v2 (688d)", hist_conf_v2),
]:
    axes[0].plot(hist["train_loss"], label=lbl, linewidth=2)
    axes[1].plot([v * 100 for v in hist["val_f1"]], label=lbl, linewidth=2)
axes[0].set_title("Training Loss (Weighted Cross-Entropy + Smoothing)"); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[1].set_title("Validation Macro-F1 (%)"); axes[1].set_xlabel("Epoch"); axes[1].legend()
plt.tight_layout(); plt.savefig(FIG_DIR / "fig2_training_dynamics.png", dpi=300); plt.show()

# Fig 3: Normalized Confusion Matrix (Best Exported Model)
cm = confusion_matrix(res_best["y_true"], res_best["y_pred"], labels=list(range(len(WORDS_15))), normalize="true")
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=WORDS_15, yticklabels=WORDS_15, ax=ax)
ax.set_title(f"Best Exported Word Model (a={best_alpha:.2f}) — Normalized Test Confusion Matrix")
ax.set_xlabel("Predicted Word"); ax.set_ylabel("True Word")
plt.xticks(rotation=40, ha="right"); plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_confusion_matrix.png", dpi=300); plt.show()

# Fig 4: Per-Word F1 Ranking
rep = classification_report(res_best["y_true"], res_best["y_pred"], labels=list(range(len(WORDS_15))), target_names=WORDS_15, output_dict=True, zero_division=0)
f1_df = pd.DataFrame([{"word": w, "f1": rep[w]["f1-score"] * 100} for w in WORDS_15]).sort_values("f1")
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(f1_df["word"], f1_df["f1"], color=["#e06c75" if v < 80 else "#61afef" if v < 92 else "#50c878" for v in f1_df["f1"]])
ax.set_xlim(0, 106); ax.set_xlabel("Test F1-Score (%)"); ax.set_title("Per-Word Test F1 Ranking (v2 Balanced)")
for idx, val in enumerate(f1_df["f1"]): ax.text(val + 1.0, idx, f"{val:.1f}%", va="center", fontsize=9)
plt.tight_layout(); plt.savefig(FIG_DIR / "fig4_per_word_f1.png", dpi=300); plt.show()

# Fig 5: Temporal Attention Saliency + Gradient Feature-Group Saliency
conf_v2_model.eval(); conf_v2_model.zero_grad()
sample_xb, sample_yb = next(iter(test_kin_loader))
sample_xb = sample_xb.to(DEVICE).requires_grad_(True)
logits, attn_w, _ = conf_v2_model(sample_xb, return_details=True)
logits.gather(1, sample_yb.to(DEVICE).unsqueeze(1)).sum().backward()
grad_abs = sample_xb.grad.detach().abs().mean(dim=(0, 1)).cpu().numpy()

group_imp = {
    "Upper Pose (45d)": grad_abs[0:45].mean(), "Left Hand Torso (63d)": grad_abs[45:108].mean(),
    "Right Hand Torso (63d)": grad_abs[108:171].mean(), "Hand Local Shape (126d)": grad_abs[171:297].mean(),
    "Joint Angles (30d)": grad_abs[297:327].mean(), "Anatomical Dists (16d)": grad_abs[327:343].mean(),
    "Velocity (171d)": grad_abs[346:517].mean(), "Acceleration (171d)": grad_abs[517:688].mean(),
}
fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
attn_np = attn_w[0].detach().cpu().numpy()
vel_mag = np.linalg.norm(sample_xb[0, :, 346:517].detach().cpu().numpy(), axis=1)
vel_mag = vel_mag / (vel_mag.max() + 1e-6) * attn_np.max()
axes[0].fill_between(range(WORD_SEQ_LEN), attn_np, alpha=0.35, color="#4e79a7", label="Conformer Attention")
axes[0].plot(attn_np, color="#4e79a7", linewidth=2); axes[0].plot(vel_mag, color="#f28e2b", linestyle="--", label="Hand Velocity")
axes[0].set_title(f"Temporal Attention Saliency ('{WORDS_15[int(sample_yb[0])]}')"); axes[0].legend()
pd.Series(group_imp).sort_values().plot(kind="barh", ax=axes[1], color="#59a14f")
axes[1].set_title("Input Gradient Feature-Group Saliency")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig5_attention_saliency.png", dpi=300); plt.show()

# Fig 6: UMAP Embeddings
embs = []
with torch.no_grad():
    for xb, _ in test_kin_loader:
        _, _, e = conf_v2_model(xb.to(DEVICE), return_details=True)
        embs.append(e.cpu().numpy())
embs_np = np.concatenate(embs, axis=0)
proj = umap.UMAP(n_neighbors=min(15, max(2, len(embs_np) - 1)), min_dist=0.25, random_state=SEED).fit_transform(embs_np)
fig, ax = plt.subplots(figsize=(9, 7))
palette = sns.color_palette("tab20", len(WORDS_15))
for wid, wname in enumerate(WORDS_15):
    m = (res_best["y_true"] == wid)
    if m.any(): ax.scatter(proj[m, 0], proj[m, 1], color=palette[wid], label=wname, s=45, alpha=0.85)
ax.set_title("UMAP 2D Projection of Word Embeddings (Test Set)"); ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout(); plt.savefig(FIG_DIR / "fig6_umap_embeddings.png", dpi=300); plt.show()

# Fig 7: 5-Model Ablation & 5-Fold Signer-Independent CV Summary
fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
axes[0].barh(bench_df["Model"], bench_df["Macro F1 (%)"], color="#4e79a7", label="Macro-F1 (%)")
axes[0].set_xlim(0, 105); axes[0].set_title("Held-Out Signer-Disjoint Ablation Benchmark")
for idx, val in enumerate(bench_df["Macro F1 (%)"]): axes[0].text(val + 1.0, idx, f"{val:.1f}%", va="center", fontsize=8)

axes[1].barh(cv_summary["Model"], cv_summary["MacroF1_Mean"], xerr=cv_summary["MacroF1_Std"].fillna(0), color="#59a14f", capsize=4)
axes[1].set_xlim(0, 105); axes[1].set_title("5-Fold Signer-Independent CV Macro-F1 (Mean ± Std)")
for idx, (_, r) in enumerate(cv_summary.iterrows()):
    axes[1].text(r["MacroF1_Mean"] + 1.2, idx, f"{r['MacroF1_Mean']:.1f}±{r['MacroF1_Std']:.1f}%", va="center", fontsize=8)
plt.tight_layout(); plt.savefig(FIG_DIR / "fig7_ablation_and_cv5.png", dpi=300); plt.show()

errors = []
for idx, (yt, yp) in enumerate(zip(res_best["y_true"], res_best["y_pred"])):
    if yt != yp:
        errors.append({"True": WORDS_15[int(yt)], "Predicted": WORDS_15[int(yp)], "Conf (%)": round(float(res_best["probs"][idx, yp]) * 100, 1), "True Prob (%)": round(float(res_best["probs"][idx, yt]) * 100, 1)})
err_df = pd.DataFrame(errors)
print(f"Total Test Misclassifications: {len(err_df)} / {len(res_best['y_true'])}")
if len(err_df):
    display(err_df.head(15))
    err_df.to_csv(WORK_DIR / "failure_cases_v2.csv", index=False)"""
    ))

    cells.append((
        "code",
        """# @title 10. Export Verified FP32 + INT8 Quantized ONNX Models + Optimization Benchmark
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

export_model = deepcopy(ensemble_model).cpu().eval()
dummy_in = torch.randn(1, WORD_SEQ_LEN, WORD_FEATURE_DIM, dtype=torch.float32)

onnx_path = EXPORT_DIR / "asl_words_conformer.onnx"
try:
    torch.onnx.export(
        export_model, dummy_in, str(onnx_path),
        input_names=["sequence"], output_names=["logits"],
        dynamic_axes={"sequence": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
except TypeError:
    torch.onnx.export(
        export_model, dummy_in, str(onnx_path),
        input_names=["sequence"], output_names=["logits"],
        dynamic_axes={"sequence": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

onnx.checker.check_model(onnx.load(str(onnx_path)))
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
with torch.no_grad():
    pt_out = export_model(dummy_in).numpy()
ort_out = ort_sess.run(["logits"], {"sequence": dummy_in.numpy()})[0]
max_diff = float(np.max(np.abs(pt_out - ort_out)))
print(f"FP32 ONNX Parity Check -> max_abs_diff = {max_diff:.2e} (threshold < 1e-4)")
assert max_diff < 1e-4

int8_path = EXPORT_DIR / "asl_words_conformer_int8.onnx"
quantize_dynamic(model_input=str(onnx_path), model_output=str(int8_path), weight_type=QuantType.QInt8)
int8_sess = ort.InferenceSession(str(int8_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])

def eval_word_onnx(session):
    all_logits, all_y = [], []
    times = []
    for xb, yb in test_kin_loader:
        arr = xb.numpy().astype(np.float32)
        for i in range(arr.shape[0]):
            t0 = time.perf_counter()
            out = session.run(["logits"], {"sequence": arr[i:i+1]})[0]
            times.append((time.perf_counter() - t0) * 1000.0)
            all_logits.append(out)
        all_y.append(yb.numpy())
    logits_np = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    preds = np.argmax(logits_np, axis=1)
    return {
        "top1": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "lat_ms": float(np.mean(times)),
    }

fp32_m = eval_word_onnx(ort_sess)
int8_m = eval_word_onnx(int8_sess)
fp32_mb = round(onnx_path.stat().st_size / (1024 * 1024), 3)
int8_mb = round(int8_path.stat().st_size / (1024 * 1024), 3)
opt_df = pd.DataFrame([
    {"Variant": "FP32 (ORT_ENABLE_ALL)", "Size (MB)": fp32_mb, "CPU Latency (ms/clip)": round(fp32_m["lat_ms"], 3), "Test Top-1 (%)": round(fp32_m["top1"]*100, 2), "Test Macro-F1 (%)": round(fp32_m["macro_f1"]*100, 2)},
    {"Variant": "INT8 Dynamic Quantized", "Size (MB)": int8_mb, "CPU Latency (ms/clip)": round(int8_m["lat_ms"], 3), "Test Top-1 (%)": round(int8_m["top1"]*100, 2), "Test Macro-F1 (%)": round(int8_m["macro_f1"]*100, 2)},
])
opt_df.to_csv(WORK_DIR / "optimization_benchmark_words.csv", index=False)
display(opt_df)

ens_cv = cv_summary[cv_summary["Model"].str.contains("Ensemble|Conformer", regex=True)].iloc[-1]
config_payload = {
    "feature_version": "word_v1_688",
    "notebook_version": "2026-11-25",
    "architecture": f"DualEnsemble(BiGRU+ConformerV2+SupCon, alpha={best_alpha:.2f})",
    "ensemble_alpha": round(float(best_alpha), 3),
    "labels": WORDS_15,
    "seq_len": WORD_SEQ_LEN,
    "feature_dim": WORD_FEATURE_DIM,
    "int8_file": "asl_words_conformer_int8.onnx",
    "fp32_size_mb": fp32_mb,
    "int8_size_mb": int8_mb,
    "fp32_latency_ms": round(fp32_m["lat_ms"], 3),
    "int8_latency_ms": round(int8_m["lat_ms"], 3),
    "test_top1_acc": round(float(res_best["top1"]), 4),
    "test_macro_f1": round(float(res_best["macro_f1"]), 4),
    "int8_top1_acc": round(float(int8_m["top1"]), 4),
    "int8_macro_f1": round(float(int8_m["macro_f1"]), 4),
    "cv5_macro_f1_mean": round(float(ens_cv["MacroF1_Mean"]) / 100.0, 4),
    "cv5_macro_f1_std": round(float(ens_cv["MacroF1_Std"]) / 100.0, 4),
}
(EXPORT_DIR / "word_model_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

bundle_zip = shutil.make_archive(str(WORK_DIR / "asl_words_export_v2_2026-11-25"), "zip", str(EXPORT_DIR))
print("Created export bundle:", bundle_zip)
print("Copy asl_words_conformer.onnx, asl_words_conformer_int8.onnx, and word_model_config.json into your local 'models/' folder and click Reload Models in the Web App!")

try:
    from google.colab import files
    files.download(bundle_zip)
except Exception:
    print("Manual download ready at:", bundle_zip)"""
    ))

    return cells


def build_words_v2_notebook(out_path: Path) -> Path:
    return build_and_validate_notebook(get_words_v2_notebook_cells(), out_path)
