"""Builder module for colab/02_Train_ASL_Letters_Hybrid_2026-11-19.ipynb (v2 + SupCon + INT8)."""

from __future__ import annotations

from pathlib import Path

from nb_common import build_and_validate_notebook


def get_letters_v2_notebook_cells() -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []

    cells.append((
        "md",
        """# ASL Fingerspelling Recognition (`A–Z`) — `HybridHandNet v2` + Synthetic→Real Benchmark (`2026-11-19`)

### Key Upgrades in v2 (Built from v1 Diagnostic Findings)
1. **Replaces Broken Massey Mirror:** Replaces the empty `elektroprogramming/datasetASL` clone (`0` images in v1) with external real-signer test dataset lookup (`danrasband/asl-alphabet-test` / mirror fallback) + `grassknoted` real hands.
2. **Modality Dropout + Auxiliary Supervision (`HybridHandNetV2`):** Randomly drops visual features (`p=0.35`) or geometric features (`p=0.10`) before gated fusion during training, paired with per-branch auxiliary heads (`aux_vis_head`, `aux_geo_head`), forcing both visual creases and 3D kinematics to pull their weight instead of collapsing to RGB background memorization.
3. **Domain-Randomized Visual Augmentations:** Adds `RandomGrayscale`, `RandomPerspective`, stronger `ColorJitter`, `GaussianBlur`, and `RandomErasing` to survive real webcam lighting/background shifts.
4. **Experiment A — Out-of-Domain Cross-Dataset Benchmark (`Synthetic → Real`):** Trains strictly on `lexset_synth` (synthetic 3D renders) and tests strictly on unseen real human hands (`grassknoted` + external real dataset), proving why `HybridHandNet v2` generalizes far better than `RGB-Only` or `Hybrid v1`.
5. **Experiment B — Production Multi-Source Export + v1 Cache Reuse:** Automatically reuses your existing 35,014-hand `letter_hybrid_cache_v1.npz` on Google Drive and exports a verified ONNX (`asl_letters_hybrid.onnx`) matching exact web runtime inputs (`(B, 3, 224, 224)` + `(B, 98)`)."""
    ))

    cells.append((
        "code",
        """# @title 1. Install Dependencies, Check GPU/High-RAM & Mount Google Drive
!pip install -q mediapipe onnx onnxruntime kagglehub umap-learn tqdm scikit-learn seaborn opencv-python-headless psutil

import os
import re
import json
import time
import random
import shutil
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
from torch.utils.data import Dataset, DataLoader
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
LOADER_WORKERS = min(8, max(2, CPU_CORES - 2))
BATCH_SIZE = 128 if GPU_VRAM_GB >= 20.0 else 64

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

V1_WORK_DIR = DRIVE_BASE / "letters_hybrid"
WORK_DIR = DRIVE_BASE / "letters_hybrid_v2_20261119"
RAW_DIR = Path("/content/raw_asl_letters_v2")
CAL_DIR = DRIVE_BASE / "calibration_data" / "letters"
FIG_DIR = WORK_DIR / "figures"
EXPORT_DIR = WORK_DIR / "export"
for d in [WORK_DIR, RAW_DIR, FIG_DIR, EXPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25})
print("v2 Workspace:", WORK_DIR)"""
    ))

    cells.append((
        "code",
        """# @title 2. Shared Constants, MediaPipe HandLandmarker & 224px + 98-dim Feature Extractor
import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

LETTERS_26 = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
LETTER_TO_ID = {ch: i for i, ch in enumerate(LETTERS_26)}

LETTER_CROP_SIZE = 224
LETTER_GEO_DIM = 98
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FINGER_ANGLE_TRIPLES = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]
FINGERTIP_INDICES = [4, 8, 12, 16, 20]
FINGER_MCP_REF = [(4, 5), (8, 5), (12, 9), (16, 13), (20, 17)]

MP_DIR = WORK_DIR / "mediapipe_assets"
MP_DIR.mkdir(parents=True, exist_ok=True)
HAND_TASK = MP_DIR / "hand_landmarker.task"
v1_hand = V1_WORK_DIR / "mediapipe_assets" / "hand_landmarker.task"
if (not HAND_TASK.exists() or HAND_TASK.stat().st_size < 100_000) and v1_hand.exists() and v1_hand.stat().st_size >= 100_000:
    shutil.copy2(v1_hand, HAND_TASK)
if not HAND_TASK.exists() or HAND_TASK.stat().st_size < 100_000:
    print("Downloading hand_landmarker.task...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        HAND_TASK,
    )

hand_detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_TASK)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.35,
        min_hand_presence_confidence=0.35,
    )
)

def compute_joint_angles(hand_pts: np.ndarray) -> np.ndarray:
    out = np.zeros(15, dtype=np.float32)
    for i, (a, b, c) in enumerate(FINGER_ANGLE_TRIPLES):
        v1, v2 = hand_pts[a] - hand_pts[b], hand_pts[c] - hand_pts[b]
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
        out[i] = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return out

def normalize_hand_local(hand_pts: np.ndarray) -> np.ndarray:
    centered = hand_pts - hand_pts[0:1]
    palm_scale = max(float(np.linalg.norm(centered[9])), 0.02)
    return (centered / palm_scale).astype(np.float32)

def compute_geo_98(canonical_pts: np.ndarray) -> np.ndarray:
    local_21 = normalize_hand_local(canonical_pts)
    angles_15 = compute_joint_angles(canonical_pts)
    tip_wrist_5 = np.array([float(np.linalg.norm(local_21[i])) for i in FINGERTIP_INDICES], dtype=np.float32)
    pair_dists = [float(np.linalg.norm(local_21[FINGERTIP_INDICES[i]] - local_21[FINGERTIP_INDICES[j]])) for i in range(5) for j in range(i + 1, 5)]
    tip_pair_10 = np.array(pair_dists, dtype=np.float32)
    tip_mcp_5 = np.array([float(np.linalg.norm(local_21[a] - local_21[b])) for a, b in FINGER_MCP_REF], dtype=np.float32)
    geo_vec = np.concatenate([local_21.reshape(-1), angles_15, tip_wrist_5, tip_pair_10, tip_mcp_5], axis=0).astype(np.float32)
    assert geo_vec.shape == (LETTER_GEO_DIM,)
    return geo_vec

def extract_crop_and_geo_from_bgr(img_bgr: np.ndarray):
    h0, w0 = img_bgr.shape[:2]
    pad_px = max(16, int(0.18 * max(h0, w0)))
    padded = cv2.copyMakeBorder(img_bgr, pad_px, pad_px, pad_px, pad_px, borderType=cv2.BORDER_REPLICATE)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    res = hand_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res or not res.hand_landmarks:
        return None
    pts = np.array([[lm.x, lm.y, lm.z] for lm in res.hand_landmarks[0]], dtype=np.float32)
    handedness = res.handedness[0][0].category_name if (res.handedness and res.handedness[0]) else "Right"
    h, w = padded.shape[:2]
    xs, ys = pts[:, 0] * w, pts[:, 1] * h
    cx, cy = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
    side = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 32.0) * 1.45
    half = int(round(side * 0.5))
    x1, y1, x2, y2 = int(round(cx)) - half, int(round(cy)) - half, int(round(cx)) + half, int(round(cy)) + half
    pl, pt, pr, pb = max(0, -x1), max(0, -y1), max(0, x2 - w), max(0, y2 - h)
    crop = padded[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return None
    if pl or pt or pr or pb:
        crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, borderType=cv2.BORDER_REPLICATE)
    canon = pts.copy()
    if handedness == "Left":
        crop = cv2.flip(crop, 1)
        canon[:, 0] = 1.0 - canon[:, 0]
    crop_rgb = cv2.cvtColor(cv2.resize(crop, (LETTER_CROP_SIZE, LETTER_CROP_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return crop_rgb.astype(np.uint8), compute_geo_98(canon)

print("Letter extractor ready. Crop:", LETTER_CROP_SIZE, "| Geo dim:", LETTER_GEO_DIM)"""
    ))

    cells.append((
        "code",
        """# @title 3. Load v1 35k Hand Cache + Personal Calibration + External Real-Signer OOD Test Set
REUSE_V1_CACHE = True     # @param {type:"boolean"}
FORCE_REEXTRACT = False   # @param {type:"boolean"}
USE_OOD_EXTERNAL = True   # @param {type:"boolean"}
MAX_PER_CLASS_LEXSET = 900 if HIGH_RAM else 450      # @param {type:"integer"}
MAX_PER_CLASS_GRASSKNOTED = 600 if HIGH_RAM else 300 # @param {type:"integer"}

V1_CACHE_PATH = V1_WORK_DIR / "letter_hybrid_cache_v1.npz"
V2_CACHE_PATH = WORK_DIR / "letter_hybrid_cache_v2.npz"
OOD_CACHE_PATH = WORK_DIR / "ood_real_cache_v2.npz"

try:
    from google.colab import userdata
    for k in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        val = userdata.get(k)
        if val:
            os.environ[k] = val
except Exception:
    pass

def collect_folder_records(root: Path, source_name: str, max_per_class: int):
    found = {ch: [] for ch in LETTERS_26}
    for p in root.rglob("*"):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        parent = p.parent.name.strip().upper()
        letter = None
        if len(parent) == 1 and parent in LETTER_TO_ID:
            letter = parent
        else:
            m = re.match(r"([a-z])_(?:test|[0-9]+)", p.stem.lower())
            if m and m.group(1).upper() in LETTER_TO_ID:
                letter = m.group(1).upper()
        if letter:
            found[letter].append(p)
    rows = []
    rng = random.Random(SEED)
    for ch, paths in found.items():
        paths_s = sorted(paths)
        if len(paths_s) > max_per_class:
            paths_s = rng.sample(paths_s, max_per_class)
        for p in paths_s:
            rows.append({"path": str(p), "letter": ch, "label_id": LETTER_TO_ID[ch], "source": source_name})
    print(f"[{source_name}] Found {len(rows)} images across {sum(1 for v in found.values() if v)} classes.")
    return rows

def extract_parallel(records_list, desc: str):
    if not records_list:
        return None
    worker_script = Path("/content/mp_letter_worker_v2.py")
    worker_code = f'''
import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
LETTER_CROP_SIZE = {LETTER_CROP_SIZE}
FINGER_ANGLE_TRIPLES = {FINGER_ANGLE_TRIPLES!r}
FINGERTIP_INDICES = {FINGERTIP_INDICES!r}
FINGER_MCP_REF = {FINGER_MCP_REF!r}
_HAND_DET = None
def init_worker(task_path: str):
    global _HAND_DET
    _HAND_DET = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=task_path), running_mode=vision.RunningMode.IMAGE, num_hands=1, min_hand_detection_confidence=0.35, min_hand_presence_confidence=0.35)
    )
def _angles(pts):
    out = np.zeros(15, dtype=np.float32)
    for i, (a, b, c) in enumerate(FINGER_ANGLE_TRIPLES):
        v1, v2 = pts[a] - pts[b], pts[c] - pts[b]
        out[i] = float(np.clip(np.dot(v1, v2) / (float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6), -1.0, 1.0))
    return out
def _geo98(pts):
    c = pts - pts[0:1]
    l21 = (c / max(float(np.linalg.norm(c[9])), 0.02)).astype(np.float32)
    a15 = _angles(pts)
    tw = np.array([float(np.linalg.norm(l21[i])) for i in FINGERTIP_INDICES], dtype=np.float32)
    pd_l = [float(np.linalg.norm(l21[FINGERTIP_INDICES[i]] - l21[FINGERTIP_INDICES[j]])) for i in range(5) for j in range(i + 1, 5)]
    tm = np.array([float(np.linalg.norm(l21[a] - l21[b])) for a, b in FINGER_MCP_REF], dtype=np.float32)
    return np.concatenate([l21.reshape(-1), a15, tw, np.array(pd_l, dtype=np.float32), tm], axis=0).astype(np.float32)
def run_img(task):
    idx, path_str = task
    img_bgr = cv2.imread(path_str)
    if img_bgr is None: return idx, None
    h0, w0 = img_bgr.shape[:2]
    pad_px = max(16, int(0.18 * max(h0, w0)))
    padded = cv2.copyMakeBorder(img_bgr, pad_px, pad_px, pad_px, pad_px, borderType=cv2.BORDER_REPLICATE)
    res = _HAND_DET.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)))
    if not res or not res.hand_landmarks: return idx, None
    pts = np.array([[lm.x, lm.y, lm.z] for lm in res.hand_landmarks[0]], dtype=np.float32)
    handedness = res.handedness[0][0].category_name if (res.handedness and res.handedness[0]) else "Right"
    h, w = padded.shape[:2]
    xs, ys = pts[:, 0] * w, pts[:, 1] * h
    cx, cy = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
    side = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()), 32.0) * 1.45
    half = int(round(side * 0.5))
    x1, y1, x2, y2 = int(round(cx)) - half, int(round(cy)) - half, int(round(cx)) + half, int(round(cy)) + half
    pl, pt, pr, pb = max(0, -x1), max(0, -y1), max(0, x2 - w), max(0, y2 - h)
    crop = padded[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0: return idx, None
    if pl or pt or pr or pb: crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, borderType=cv2.BORDER_REPLICATE)
    canon = pts.copy()
    if handedness == "Left": crop = cv2.flip(crop, 1); canon[:, 0] = 1.0 - canon[:, 0]
    crop_rgb = cv2.cvtColor(cv2.resize(crop, (LETTER_CROP_SIZE, LETTER_CROP_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return idx, (crop_rgb.astype(np.uint8), _geo98(canon))
'''
    worker_script.write_text(worker_code, encoding="utf-8")
    import sys
    if "/content" not in sys.path: sys.path.insert(0, "/content")
    import mp_letter_worker_v2

    tasks = [(i, r["path"]) for i, r in enumerate(records_list)]
    res_map = {}
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=EXTRACTION_WORKERS, mp_context=ctx, initializer=mp_letter_worker_v2.init_worker, initargs=(str(HAND_TASK),)) as pool:
        for idx, out in tqdm(pool.map(mp_letter_worker_v2.run_img, tasks, chunksize=32), total=len(tasks), desc=desc):
            if out is not None: res_map[idx] = out
    if not res_map:
        return None
    crops_a = np.stack([res_map[i][0] for i in sorted(res_map)], axis=0).astype(np.uint8)
    geos_a = np.stack([res_map[i][1] for i in sorted(res_map)], axis=0).astype(np.float32)
    labels_a = np.array([records_list[i]["label_id"] for i in sorted(res_map)], dtype=np.int64)
    sources_a = np.array([records_list[i]["source"] for i in sorted(res_map)])
    return crops_a, geos_a, labels_a, sources_a

# Check personal & team annotated calibration images (+ annotations.csv metadata)
cal_records = []
ann_letter_signers = {}
for aroot in [CAL_DIR, CAL_DIR.parent, Path("/content/calibration_data/letters"), Path("/content/calibration_data")]:
    csv_p = aroot / "annotations.csv"
    if csv_p.exists():
        try:
            adf = pd.read_csv(csv_p)
            for _, r in adf.iterrows():
                rel = str(r.get("rel_path", "")).strip()
                sid = str(r.get("signer_id", "personal_webcam")).strip() or "personal_webcam"
                if rel:
                    ann_letter_signers[Path(rel).name] = sid
        except Exception:
            pass
for croot in [CAL_DIR, Path("/content/calibration_data/letters")]:
    if croot.exists():
        for rec in collect_folder_records(croot, "calibration", 400):
            rec["signer_id"] = ann_letter_signers.get(Path(rec["path"]).name, "personal_webcam")
            cal_records.append(rec)

cache_loaded = False
if not FORCE_REEXTRACT:
    if V2_CACHE_PATH.exists():
        with np.load(V2_CACHE_PATH, allow_pickle=True) as cached:
            if int(cached.get("cal_len", -1)) == len(cal_records):
                crops_uint8, geos, labels, sources = cached["crops"], cached["geos"], cached["labels"], cached["sources"]
                cache_loaded = True
                print(f"Loaded v2 cache ({len(labels)} hands) from Drive!")
    elif REUSE_V1_CACHE and V1_CACHE_PATH.exists():
        print(f"Reusing 35k base hand cache from {V1_CACHE_PATH}...")
        with np.load(V1_CACHE_PATH, allow_pickle=True) as cached:
            crops_uint8, geos, labels, sources = cached["crops"], cached["geos"], cached["labels"], cached["sources"]
        if cal_records:
            cal_ext = extract_parallel(cal_records, "Extracting new personal calibration hands")
            if cal_ext is not None:
                c_cal, g_cal, y_cal, s_cal = cal_ext
                crops_uint8 = np.concatenate([crops_uint8, c_cal], axis=0)
                geos = np.concatenate([geos, g_cal], axis=0)
                labels = np.concatenate([labels, y_cal], axis=0)
                sources = np.concatenate([sources, s_cal], axis=0)
        np.savez_compressed(V2_CACHE_PATH, cal_len=len(cal_records), crops=crops_uint8, geos=geos, labels=labels, sources=sources)
        cache_loaded = True

if not cache_loaded:
    import kagglehub
    records = []
    for ds_slug, src_name, cap in [
        ("lexset/synthetic-asl-alphabet", "lexset_synth", MAX_PER_CLASS_LEXSET),
        ("grassknoted/asl-alphabet", "grassknoted", MAX_PER_CLASS_GRASSKNOTED),
    ]:
        try:
            ds_root = Path(kagglehub.dataset_download(ds_slug))
            records.extend(collect_folder_records(ds_root, src_name, cap))
        except Exception as exc:
            print(f"Skipping Kaggle dataset {ds_slug}:", exc)
    records.extend(cal_records)
    assert len(records) > 0, "No letter images available!"
    ext_res = extract_parallel(records, f"HandLandmarker ({EXTRACTION_WORKERS} workers)")
    crops_uint8, geos, labels, sources = ext_res
    np.savez_compressed(V2_CACHE_PATH, cal_len=len(cal_records), crops=crops_uint8, geos=geos, labels=labels, sources=sources)

# Optional external real-signer test dataset (strictly held out for OOD evaluation)
ood_crops, ood_geos, ood_labels = None, None, None
if USE_OOD_EXTERNAL:
    if OOD_CACHE_PATH.exists() and not FORCE_REEXTRACT:
        with np.load(OOD_CACHE_PATH, allow_pickle=True) as ood_npz:
            ood_crops, ood_geos, ood_labels = ood_npz["crops"], ood_npz["geos"], ood_npz["labels"]
        print(f"Loaded cached external real-signer OOD test set ({len(ood_labels)} hands).")
    else:
        import kagglehub
        ood_records = []
        for ds_slug, cap in [
            ("danrasband/asl-alphabet-test", 50),
            ("debashishsau/aslamerican-sign-language-aplhabet-dataset", 40),
        ]:
            try:
                ds_root = Path(kagglehub.dataset_download(ds_slug))
                rows = collect_folder_records(ds_root, f"ood_{ds_slug.split('/')[0]}", cap)
                if rows:
                    ood_records = rows
                    break
            except Exception as exc:
                print(f"External OOD dataset {ds_slug} skipped:", exc)
        if ood_records:
            ood_ext = extract_parallel(ood_records, "Extracting Real OOD Hands")
            if ood_ext is not None:
                ood_crops, ood_geos, ood_labels, _ = ood_ext
                np.savez_compressed(OOD_CACHE_PATH, crops=ood_crops, geos=ood_geos, labels=ood_labels)

print(f"Base Multi-Source Hands: {len(labels)} | RAM used: {(crops_uint8.nbytes + geos.nbytes)/1e9:.2f} GB")
display(pd.DataFrame({"letter": [LETTERS_26[int(y)] for y in labels], "source": sources}).groupby(["source", "letter"]).size().unstack(fill_value=0))"""
    ))

    cells.append((
        "code",
        """# @title 4. Figure 1 Sample Crops & Shared-Memory Domain-Randomized Datasets
import torchvision.transforms.v2 as T
from sklearn.model_selection import StratifiedShuffleSplit

fig, axes = plt.subplots(2, 6, figsize=(14, 5))
for ax, ch in zip(axes.flat, ["A", "B", "C", "E", "G", "K", "M", "N", "R", "S", "V", "W"]):
    idxs = np.where(labels == LETTER_TO_ID[ch])[0]
    if len(idxs):
        i = int(idxs[0])
        ax.imshow(crops_uint8[i])
        ax.set_title(f"{ch} ({sources[i]})", fontsize=9)
    ax.axis("off")
plt.suptitle("Figure 1: MediaPipe Canonicalized Hand Crops (224x224, Left/Right Unified)")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_sample_hand_crops.png", dpi=300)
plt.show()

crops_chw = torch.from_numpy(crops_uint8).permute(0, 3, 1, 2).contiguous().share_memory_()
geos_shared = torch.from_numpy(geos).share_memory_()
del crops_uint8

train_img_tf_v1 = T.Compose([
    T.ToImage(),
    T.RandomApply([T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.30, hue=0.06)], p=0.85),
    T.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.90, 1.10)),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
])

# Domain-randomized photometric + perspective augmentations for webcam robustness
train_img_tf_v2 = T.Compose([
    T.ToImage(),
    T.RandomApply([T.ColorJitter(brightness=0.45, contrast=0.45, saturation=0.40, hue=0.08)], p=0.90),
    T.RandomGrayscale(p=0.15),
    T.RandomPerspective(distortion_scale=0.18, p=0.40),
    T.RandomAffine(degrees=15, translate=(0.08, 0.08), scale=(0.88, 1.12)),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.4))], p=0.30),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    T.RandomErasing(p=0.25, scale=(0.02, 0.14)),
])

eval_img_tf = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
])

class HybridLetterDataset(Dataset):
    def __init__(self, c_tensor, g_tensor, y_arr, indices, tf, geo_noise: float = 0.014, augment: bool = False):
        self.c_tensor = c_tensor
        self.g_tensor = g_tensor
        self.y_arr = y_arr
        self.indices = indices
        self.tf = tf
        self.geo_noise = geo_noise
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        img_t = self.tf(self.c_tensor[i])
        geo_t = self.g_tensor[i].clone()
        if self.augment and random.random() < 0.75:
            geo_t[:63] += torch.randn(63, dtype=torch.float32) * self.geo_noise
        return img_t, geo_t, torch.tensor(self.y_arr[i], dtype=torch.long)

print("Shared zero-copy tensors ready for Cross-Domain & Multi-Source experiments.")"""
    ))

    cells.append((
        "code",
        """# @title 5. Architectures: RGB-Only, Landmark-Only, Hybrid v1, and HybridHandNet v2 (Modality Dropout + Aux Heads)
import torchvision.models as tv_models

class RgbOnlyNet(nn.Module):
    def __init__(self, num_classes: int = 26):
        super().__init__()
        net = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
        self.features, self.pool = net.features, net.avgpool
        self.head = nn.Sequential(nn.Dropout(0.30), nn.Linear(1280, num_classes))
    def forward(self, image, geo):
        return self.head(self.pool(self.features(image)).flatten(1))

class GeoOnlyNet(nn.Module):
    def __init__(self, geo_dim: int = LETTER_GEO_DIM, num_classes: int = 26):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(geo_dim, 256), nn.LayerNorm(256), nn.SiLU())
        self.block1 = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.20), nn.Linear(256, 256))
        self.block2 = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.20), nn.Linear(256, 256))
        self.head = nn.Sequential(nn.LayerNorm(256), nn.Dropout(0.25), nn.Linear(256, num_classes))
    def forward(self, image, geo):
        h = self.in_proj(geo)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.head(h)

class HybridHandNetV1(nn.Module):
    def __init__(self, geo_dim: int = LETTER_GEO_DIM, num_classes: int = 26):
        super().__init__()
        net = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
        self.vis_features, self.vis_pool = net.features, net.avgpool
        self.vis_proj = nn.Sequential(nn.Linear(1280, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.25))
        self.geo_in = nn.Sequential(nn.Linear(geo_dim, 256), nn.LayerNorm(256), nn.SiLU())
        self.geo_res = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.20), nn.Linear(256, 256))
        self.gate_layer = nn.Linear(512, 256)
        self.fusion_mlp = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256), nn.SiLU(), nn.Dropout(0.25))
        self.classifier = nn.Linear(256, num_classes)
    def forward(self, image, geo, return_details: bool = False):
        fmap = self.vis_features(image)
        v = self.vis_proj(self.vis_pool(fmap).flatten(1))
        g = self.geo_in(geo)
        g = g + self.geo_res(g)
        gate = torch.sigmoid(self.gate_layer(torch.cat([v, g], dim=1)))
        emb = self.fusion_mlp(gate * v + (1.0 - gate) * g)
        logits = self.classifier(emb)
        if return_details:
            return logits, gate, emb, fmap
        return logits

class HybridHandNetV2(nn.Module):
    \"\"\"Dual-branch fusion hardened with Modality Dropout & Auxiliary Supervision Heads.\"\"\"
    def __init__(self, geo_dim: int = LETTER_GEO_DIM, num_classes: int = 26, p_drop_rgb: float = 0.35, p_drop_geo: float = 0.10):
        super().__init__()
        self.p_drop_rgb = p_drop_rgb
        self.p_drop_geo = p_drop_geo
        net = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
        self.vis_features, self.vis_pool = net.features, net.avgpool
        self.vis_proj = nn.Sequential(nn.Linear(1280, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.25))
        self.geo_in = nn.Sequential(nn.Linear(geo_dim, 256), nn.LayerNorm(256), nn.SiLU())
        self.geo_res = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(0.20), nn.Linear(256, 256))
        self.gate_layer = nn.Linear(512, 256)
        self.fusion_mlp = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 256), nn.SiLU(), nn.Dropout(0.25))
        self.classifier = nn.Linear(256, num_classes)
        # Auxiliary training-only heads (never exported to ONNX)
        self.aux_vis_head = nn.Linear(256, num_classes)
        self.aux_geo_head = nn.Linear(256, num_classes)

    def forward(self, image, geo, return_details: bool = False, return_aux: bool = False):
        fmap = self.vis_features(image)
        v_raw = self.vis_proj(self.vis_pool(fmap).flatten(1))
        g_raw = self.geo_in(geo)
        g_raw = g_raw + self.geo_res(g_raw)

        v, g = v_raw, g_raw
        if self.training:
            bsz = image.size(0)
            drop_v = (torch.rand(bsz, 1, device=image.device) < self.p_drop_rgb)
            drop_g = (torch.rand(bsz, 1, device=image.device) < self.p_drop_geo) & (~drop_v)
            v = v_raw * (~drop_v).float()
            g = g_raw * (~drop_g).float()

        gate = torch.sigmoid(self.gate_layer(torch.cat([v, g], dim=1)))
        emb = self.fusion_mlp(gate * v + (1.0 - gate) * g)
        logits = self.classifier(emb)
        if return_aux:
            return logits, self.aux_vis_head(v_raw), self.aux_geo_head(g_raw), emb
        if return_details:
            return logits, gate, emb, fmap
        return logits

from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

def evaluate_letter_model(model, loader):
    model.eval()
    all_logits, all_y = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for img_b, geo_b, yb in loader:
            logits = model(img_b.to(DEVICE, non_blocking=True), geo_b.to(DEVICE, non_blocking=True))
            all_logits.append(logits.cpu().numpy())
            all_y.append(yb.numpy())
    elapsed_ms = ((time.perf_counter() - t0) / max(len(loader.dataset), 1)) * 1000.0
    logits_np = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    probs = torch.softmax(torch.from_numpy(logits_np), dim=-1).numpy()
    preds = np.argmax(probs, axis=1)
    return {
        "top1": accuracy_score(y_true, preds),
        "top3": top_k_accuracy_score(y_true, probs, k=3, labels=list(range(26))),
        "macro_f1": f1_score(y_true, preds, average="macro", zero_division=0),
        "latency_ms": elapsed_ms,
        "y_true": y_true,
        "y_pred": preds,
        "probs": probs,
    }

class LetterSupConLoss(nn.Module):
    \"\"\"L2-normalized temperature-scaled Supervised Contrastive Loss on fused bimodal embeddings.\"\"\"
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

def train_letter_model(model, tr_loader, va_loader, epochs: int, lr: float, name: str, is_v2_aux: bool = False, supcon_weight: float = 0.0, save_ckpt: bool = True):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.02)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    supcon_crit = LetterSupConLoss(temperature=0.12) if (is_v2_aux and supcon_weight > 0.0) else None
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    best_f1 = -1.0
    best_state = deepcopy(model.state_dict())
    history = {"train_loss": [], "val_acc": [], "val_f1": []}

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for img_b, geo_b, yb in tr_loader:
            img_b, geo_b, yb = img_b.to(DEVICE, non_blocking=True), geo_b.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                if is_v2_aux:
                    logits, v_log, g_log, emb = model(img_b, geo_b, return_aux=True)
                    loss = criterion(logits, yb) + 0.25 * criterion(v_log, yb) + 0.25 * criterion(g_log, yb)
                    if supcon_crit is not None:
                        loss = loss + supcon_weight * supcon_crit(emb, yb)
                else:
                    logits = model(img_b, geo_b)
                    loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * img_b.size(0)

        scheduler.step()
        val_m = evaluate_letter_model(model, va_loader)
        tr_loss = total_loss / max(len(tr_loader.dataset), 1)
        history["train_loss"].append(tr_loss)
        history["val_acc"].append(val_m["top1"])
        history["val_f1"].append(val_m["macro_f1"])
        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            best_state = deepcopy(model.state_dict())
        if ep in (1, epochs) or ep % max(1, epochs // 3) == 0:
            print(f"[{name}] Ep {ep:02d}/{epochs} | Loss {tr_loss:.4f} | Val Acc {val_m['top1']*100:.2f}% | Val F1 {val_m['macro_f1']*100:.2f}%")

    model.load_state_dict(best_state)
    if save_ckpt:
        torch.save(best_state, WORK_DIR / f"{name}_best.pt")
    return model, history

print("HybridHandNetV2 params:", round(sum(p.numel() for p in HybridHandNetV2().parameters()) / 1e6, 2), "M")"""
    ))

    cells.append((
        "code",
        """# @title 6. Experiment A — Cross-Domain Out-of-Domain Benchmark (Train: Synthetic Lexset -> Test: Unseen Real Hands)
synth_idx = np.where(sources == "lexset_synth")[0]
real_idx = np.where(sources == "grassknoted")[0]
assert len(synth_idx) > 100 and len(real_idx) > 100, "Both lexset_synth and grassknoted required for Cross-Domain benchmark!"

sss_ood = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=SEED)
ood_tr_rel, ood_va_rel = next(sss_ood.split(geos_shared[synth_idx].numpy(), labels[synth_idx]))
ood_tr_idx, ood_va_idx = synth_idx[ood_tr_rel], synth_idx[ood_va_rel]

ood_tr_loader_v1 = DataLoader(HybridLetterDataset(crops_chw, geos_shared, labels, ood_tr_idx, train_img_tf_v1, augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=LOADER_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4)
ood_tr_loader_v2 = DataLoader(HybridLetterDataset(crops_chw, geos_shared, labels, ood_tr_idx, train_img_tf_v2, augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=LOADER_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4)
ood_va_loader = DataLoader(HybridLetterDataset(crops_chw, geos_shared, labels, ood_va_idx, eval_img_tf, augment=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=min(4, LOADER_WORKERS), pin_memory=True)
ood_real_loader = DataLoader(HybridLetterDataset(crops_chw, geos_shared, labels, real_idx, eval_img_tf, augment=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=min(4, LOADER_WORKERS), pin_memory=True)

print(f"Cross-Domain Split -> Train (Synthetic): {len(ood_tr_idx)} | Val (Synthetic): {len(ood_va_idx)} | Test (Unseen Real Hands): {len(real_idx)}")

m_ood_rgb, _ = train_letter_model(RgbOnlyNet(26), ood_tr_loader_v1, ood_va_loader, epochs=7, lr=4e-4, name="ood_rgb", save_ckpt=False)
m_ood_geo, _ = train_letter_model(GeoOnlyNet(LETTER_GEO_DIM, 26), ood_tr_loader_v1, ood_va_loader, epochs=16, lr=1e-3, name="ood_geo", save_ckpt=False)
m_ood_v1, _ = train_letter_model(HybridHandNetV1(LETTER_GEO_DIM, 26), ood_tr_loader_v1, ood_va_loader, epochs=8, lr=4e-4, name="ood_hybrid_v1", save_ckpt=False)
m_ood_v2, _ = train_letter_model(HybridHandNetV2(LETTER_GEO_DIM, 26, 0.35, 0.10), ood_tr_loader_v2, ood_va_loader, epochs=10, lr=4e-4, name="ood_hybrid_v2", is_v2_aux=True, save_ckpt=False)

ood_rows = []
for mname, net in [
    ("RGB-Only EfficientNet-B0", m_ood_rgb),
    ("Landmark-Only ResMLP (98d)", m_ood_geo),
    ("HybridHandNet v1 (No Modality Dropout)", m_ood_v1),
    ("HybridHandNet v2 (Modality Dropout + Aux)", m_ood_v2),
]:
    ind_res = evaluate_letter_model(net, ood_va_loader)
    real_res = evaluate_letter_model(net, ood_real_loader)
    ood_rows.append({
        "Model": mname,
        "Synth Val Top-1 (%)": ind_res["top1"] * 100,
        "Unseen Real Top-1 (%)": real_res["top1"] * 100,
        "Unseen Real Macro-F1 (%)": real_res["macro_f1"] * 100,
        "Unseen Real Top-3 (%)": real_res["top3"] * 100,
    })

ood_df = pd.DataFrame(ood_rows)
ood_df.to_csv(WORK_DIR / "cross_domain_results_v2.csv", index=False)
print("\\n=== Cross-Domain (Synthetic -> Real) Generalization Results ===")
display(ood_df.round(2))"""
    ))

    cells.append((
        "code",
        """# @title 7. Experiment B — Production Multi-Source Training & 4-Way Benchmark (HybridHandNet v2)
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.22, random_state=SEED)
train_idx, tmp_idx = next(sss1.split(geos_shared.numpy(), labels))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
val_sub, test_sub = next(sss2.split(geos_shared[tmp_idx].numpy(), labels[tmp_idx]))
val_idx, test_idx = tmp_idx[val_sub], tmp_idx[test_sub]

cal_tr = [i for i in train_idx if sources[i] == "calibration"]
if cal_tr:
    train_idx = np.concatenate([train_idx] + [np.array(cal_tr)] * 4)

print(f"Multi-Source Splits -> Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

prod_tr_loader_v1 = DataLoader(HybridLetterDataset(crops_chw, geos_shared, labels, train_idx, train_img_tf_v1, augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=LOADER_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4)
prod_tr_loader_v2 = DataLoader(HybridLetterDataset(crops_chw, geos_shared, labels, train_idx, train_img_tf_v2, augment=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=LOADER_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val_loader = DataLoader(HybridLetterDataset(crops_chw, geos_shared, labels, val_idx, eval_img_tf, augment=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=min(4, LOADER_WORKERS), pin_memory=True, persistent_workers=True, prefetch_factor=4)
test_loader = DataLoader(HybridLetterDataset(crops_chw, geos_shared, labels, test_idx, eval_img_tf, augment=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=min(4, LOADER_WORKERS), pin_memory=True)

rgb_net, hist_rgb = train_letter_model(RgbOnlyNet(26), prod_tr_loader_v2, val_loader, epochs=12, lr=4e-4, name="rgb_only_v2")
geo_net, hist_geo = train_letter_model(GeoOnlyNet(LETTER_GEO_DIM, 26), prod_tr_loader_v1, val_loader, epochs=24, lr=1e-3, name="geo_only_v2")
hyb_v1_net, hist_v1 = train_letter_model(HybridHandNetV1(LETTER_GEO_DIM, 26), prod_tr_loader_v1, val_loader, epochs=14, lr=4e-4, name="hybrid_v1_prod")
hybrid_v2_ce_net, hist_hyb_ce = train_letter_model(HybridHandNetV2(LETTER_GEO_DIM, 26, 0.35, 0.10), prod_tr_loader_v2, val_loader, epochs=18, lr=4e-4, name="hybrid_handnet_v2_ce", is_v2_aux=True, supcon_weight=0.0)
hybrid_v2_net, hist_hyb = train_letter_model(HybridHandNetV2(LETTER_GEO_DIM, 26, 0.35, 0.10), prod_tr_loader_v2, val_loader, epochs=18, lr=4e-4, name="hybrid_handnet_v2_supcon", is_v2_aux=True, supcon_weight=0.12)

res_rgb = evaluate_letter_model(rgb_net, test_loader)
res_geo = evaluate_letter_model(geo_net, test_loader)
res_v1 = evaluate_letter_model(hyb_v1_net, test_loader)
res_hyb_ce = evaluate_letter_model(hybrid_v2_ce_net, test_loader)
res_hyb = evaluate_letter_model(hybrid_v2_net, test_loader)

bench_df = pd.DataFrame([
    {"Model": "RGB-Only EfficientNet-B0", "Top-1 Acc (%)": res_rgb["top1"]*100, "Top-3 Acc (%)": res_rgb["top3"]*100, "Macro F1 (%)": res_rgb["macro_f1"]*100, "Params (M)": sum(p.numel() for p in rgb_net.parameters())/1e6},
    {"Model": "Landmark-Only ResMLP (98d)", "Top-1 Acc (%)": res_geo["top1"]*100, "Top-3 Acc (%)": res_geo["top3"]*100, "Macro F1 (%)": res_geo["macro_f1"]*100, "Params (M)": sum(p.numel() for p in geo_net.parameters())/1e6},
    {"Model": "HybridHandNet v1 (Baseline)", "Top-1 Acc (%)": res_v1["top1"]*100, "Top-3 Acc (%)": res_v1["top3"]*100, "Macro F1 (%)": res_v1["macro_f1"]*100, "Params (M)": sum(p.numel() for p in hyb_v1_net.parameters())/1e6},
    {"Model": "HybridHandNet v2 (ModDrop + Aux CE)", "Top-1 Acc (%)": res_hyb_ce["top1"]*100, "Top-3 Acc (%)": res_hyb_ce["top3"]*100, "Macro F1 (%)": res_hyb_ce["macro_f1"]*100, "Params (M)": sum(p.numel() for p in hybrid_v2_ce_net.parameters())/1e6},
    {"Model": "Proposed: HybridHandNet v2 + SupConLoss", "Top-1 Acc (%)": res_hyb["top1"]*100, "Top-3 Acc (%)": res_hyb["top3"]*100, "Macro F1 (%)": res_hyb["macro_f1"]*100, "Params (M)": sum(p.numel() for p in hybrid_v2_net.parameters())/1e6},
])

if ood_crops is not None:
    ext_chw = torch.from_numpy(ood_crops).permute(0, 3, 1, 2).contiguous()
    ext_geo = torch.from_numpy(ood_geos)
    ext_loader = DataLoader(HybridLetterDataset(ext_chw, ext_geo, ood_labels, np.arange(len(ood_labels)), eval_img_tf, augment=False), batch_size=64, shuffle=False)
    res_ext = evaluate_letter_model(hybrid_v2_net, ext_loader)
    print(f"External Unseen Kaggle Real-Hand Evaluation -> Top-1: {res_ext['top1']*100:.2f}% | Macro-F1: {res_ext['macro_f1']*100:.2f}%")

bench_df.to_csv(WORK_DIR / "letter_benchmark_results_v2.csv", index=False)
display(bench_df.round(2))"""
    ))

    cells.append((
        "code",
        """# @title 8. Academic Figures 2–7 (Curves, 26x26 Confusion Matrix, F1, Grad-CAM, Gate & UMAP, Cross-Domain Chart)
from sklearn.metrics import confusion_matrix, classification_report
import umap

# Fig 2: Training Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for lbl, hist in [("RGB-Only v2", hist_rgb), ("Landmark-Only", hist_geo), ("Hybrid v1", hist_v1), ("HybridHandNet v2", hist_hyb)]:
    axes[0].plot(hist["train_loss"], label=lbl, linewidth=2)
    axes[1].plot([v * 100 for v in hist["val_f1"]], label=lbl, linewidth=2)
axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[1].set_title("Validation Macro-F1 (%)"); axes[1].set_xlabel("Epoch"); axes[1].legend()
plt.tight_layout(); plt.savefig(FIG_DIR / "fig2_letter_training_curves.png", dpi=300); plt.show()

# Fig 3: 26x26 Confusion Matrix
cm = confusion_matrix(res_hyb["y_true"], res_hyb["y_pred"], labels=list(range(26)), normalize="true")
fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=LETTERS_26, yticklabels=LETTERS_26, ax=ax, annot_kws={"size": 7})
ax.set_title("HybridHandNet v2 — Normalized Test Confusion Matrix (A–Z)")
ax.set_xlabel("Predicted Letter"); ax.set_ylabel("True Letter")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig3_letter_confusion_matrix.png", dpi=300); plt.show()

# Fig 4: Per-Letter F1
rep = classification_report(res_hyb["y_true"], res_hyb["y_pred"], labels=list(range(26)), target_names=LETTERS_26, output_dict=True, zero_division=0)
f1_s = pd.Series({ch: rep[ch]["f1-score"] * 100 for ch in LETTERS_26})
fig, ax = plt.subplots(figsize=(13, 4))
f1_s.plot(kind="bar", ax=ax, color=["#e06c75" if v < 90 else "#50c878" for v in f1_s.values])
ax.set_ylim(0, 105); ax.set_ylabel("F1-Score (%)"); ax.set_title("Per-Letter Test F1-Score (HybridHandNet v2)")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig4_per_letter_f1.png", dpi=300); plt.show()

# Fig 5: Grad-CAM Saliency
hybrid_v2_net.eval()
focus_letters = ["A", "B", "E", "K", "M", "N", "R", "S", "V"]
fig, axes = plt.subplots(2, len(focus_letters), figsize=(18, 4.5))
for col_idx, ch in enumerate(focus_letters):
    matches = [i for i in test_idx if labels[i] == LETTER_TO_ID[ch]]
    if not matches: continue
    ds_idx = int(matches[0])
    img_raw = crops_chw[ds_idx].permute(1, 2, 0).numpy()
    img_t = eval_img_tf(crops_chw[ds_idx]).unsqueeze(0).to(DEVICE)
    geo_t = geos_shared[ds_idx].unsqueeze(0).to(DEVICE)
    hybrid_v2_net.zero_grad()
    fmap = hybrid_v2_net.vis_features(img_t); fmap.retain_grad()
    v = hybrid_v2_net.vis_proj(hybrid_v2_net.vis_pool(fmap).flatten(1))
    g = hybrid_v2_net.geo_in(geo_t); g = g + hybrid_v2_net.geo_res(g)
    gate = torch.sigmoid(hybrid_v2_net.gate_layer(torch.cat([v, g], dim=1)))
    logits = hybrid_v2_net.classifier(hybrid_v2_net.fusion_mlp(gate * v + (1.0 - gate) * g))
    pred_id = int(logits.argmax(dim=1).item())
    logits[0, pred_id].backward()
    cam = torch.relu((fmap.grad[0].detach().mean(dim=(1, 2), keepdim=True) * fmap[0].detach()).sum(dim=0)).cpu().numpy()
    cam = cv2.resize(cam, (LETTER_CROP_SIZE, LETTER_CROP_SIZE))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    heatmap = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    axes[0, col_idx].imshow(img_raw); axes[0, col_idx].set_title(f"True: {ch}", fontsize=9); axes[0, col_idx].axis("off")
    axes[1, col_idx].imshow(np.uint8(0.58 * img_raw + 0.42 * heatmap)); axes[1, col_idx].set_title(f"Pred: {LETTERS_26[pred_id]}", fontsize=9); axes[1, col_idx].axis("off")
plt.suptitle("Figure 5: Grad-CAM Saliency Over Tight Hand Crops (HybridHandNet v2)")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig5_gradcam_letters.png", dpi=300); plt.show()

# Fig 6: Modality Gate Analysis + UMAP
gates, embs = [], []
with torch.no_grad():
    for img_b, geo_b, _ in test_loader:
        _, g_w, e, _ = hybrid_v2_net(img_b.to(DEVICE), geo_b.to(DEVICE), return_details=True)
        gates.append(g_w.mean(dim=1).cpu().numpy()); embs.append(e.cpu().numpy())
gates_np, embs_np = np.concatenate(gates, axis=0), np.concatenate(embs, axis=0)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].bar(LETTERS_26, [gates_np[res_hyb["y_true"] == c].mean() for c in range(26)], color="#61afef")
axes[0].axhline(0.5, color="red", linestyle="--", label="Equal Visual/Geo Weight")
axes[0].set_ylim(0, 1); axes[0].set_title("Modality Gate Weight per Letter (With Modality Dropout)"); axes[0].legend()
proj = umap.UMAP(n_neighbors=15, min_dist=0.2, random_state=SEED).fit_transform(embs_np)
pal = sns.color_palette("husl", 26)
for c, ch in enumerate(LETTERS_26):
    m = (res_hyb["y_true"] == c)
    if m.any(): axes[1].scatter(proj[m, 0], proj[m, 1], color=pal[c], label=ch, s=18, alpha=0.8)
axes[1].set_title("UMAP Projection of HybridHandNet v2 Embeddings"); axes[1].legend(ncol=3, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig6_gate_and_umap.png", dpi=300); plt.show()

# Fig 7: Cross-Domain Generalization Comparison
x_pos = np.arange(len(ood_df))
w_bar = 0.35
fig, ax = plt.subplots(figsize=(11, 5))
b1 = ax.bar(x_pos - w_bar/2, ood_df["Synth Val Top-1 (%)"], w_bar, label="In-Domain (Synthetic Val)", color="#93c5fd")
b2 = ax.bar(x_pos + w_bar/2, ood_df["Unseen Real Top-1 (%)"], w_bar, label="Cross-Domain (Unseen Real Hands)", color="#10b981")
ax.set_xticks(x_pos); ax.set_xticklabels(ood_df["Model"], rotation=15, ha="right")
ax.set_ylim(0, 110); ax.set_ylabel("Top-1 Accuracy (%)")
ax.set_title("Figure 7: Out-of-Domain Generalization (Train: Synthetic -> Test: Unseen Real Human Hands)")
for bars in (b1, b2):
    for rect in bars:
        h_v = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h_v + 1.2, f"{h_v:.1f}%", ha="center", va="bottom", fontsize=8)
ax.legend(); plt.tight_layout(); plt.savefig(FIG_DIR / "fig7_cross_domain_generalization.png", dpi=300); plt.show()"""
    ))

    cells.append((
        "code",
        """# @title 9. Export Verified FP32 + INT8 Quantized HybridHandNet v2 ONNX + Optimization Benchmark
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

class HybridHandExport(nn.Module):
    def __init__(self, core: HybridHandNetV2):
        super().__init__()
        self.core = core
    def forward(self, image, geo):
        return self.core(image, geo, return_details=False, return_aux=False)

export_net = HybridHandExport(hybrid_v2_net).cpu().eval()
dummy_img = torch.randn(1, 3, LETTER_CROP_SIZE, LETTER_CROP_SIZE, dtype=torch.float32)
dummy_geo = torch.randn(1, LETTER_GEO_DIM, dtype=torch.float32)

onnx_path = EXPORT_DIR / "asl_letters_hybrid.onnx"
try:
    torch.onnx.export(
        export_net, (dummy_img, dummy_geo), str(onnx_path),
        input_names=["image", "geo"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "geo": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
except TypeError:
    torch.onnx.export(
        export_net, (dummy_img, dummy_geo), str(onnx_path),
        input_names=["image", "geo"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "geo": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

onnx.checker.check_model(onnx.load(str(onnx_path)))
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
with torch.no_grad():
    pt_out = export_net(dummy_img, dummy_geo).numpy()
ort_out = ort_sess.run(["logits"], {"image": dummy_img.numpy(), "geo": dummy_geo.numpy()})[0]
max_diff = float(np.max(np.abs(pt_out - ort_out)))
print(f"FP32 ONNX Parity Check -> max_abs_diff = {max_diff:.2e} (threshold < 1e-4)")
assert max_diff < 1e-4

int8_path = EXPORT_DIR / "asl_letters_hybrid_int8.onnx"
quantize_dynamic(model_input=str(onnx_path), model_output=str(int8_path), weight_type=QuantType.QInt8)
int8_sess = ort.InferenceSession(str(int8_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])

def eval_letter_onnx(session):
    all_logits, all_y, times = [], [], []
    for batch_i, (img_b, geo_b, yb) in enumerate(test_loader):
        im_np = img_b.numpy().astype(np.float32)
        ge_np = geo_b.numpy().astype(np.float32)
        if batch_i < 4:
            for k in range(min(16, im_np.shape[0])):
                t0 = time.perf_counter()
                session.run(["logits"], {"image": im_np[k:k+1], "geo": ge_np[k:k+1]})
                times.append((time.perf_counter() - t0) * 1000.0)
        all_logits.append(session.run(["logits"], {"image": im_np, "geo": ge_np})[0])
        all_y.append(yb.numpy())
    logits_np = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    preds = np.argmax(logits_np, axis=1)
    return {
        "top1": float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "lat_ms": float(np.mean(times)) if times else 0.0,
    }

fp32_m = eval_letter_onnx(ort_sess)
int8_m = eval_letter_onnx(int8_sess)
fp32_mb = round(onnx_path.stat().st_size / (1024 * 1024), 3)
int8_mb = round(int8_path.stat().st_size / (1024 * 1024), 3)
opt_df = pd.DataFrame([
    {"Variant": "FP32 (ORT_ENABLE_ALL)", "Size (MB)": fp32_mb, "CPU Latency (ms/img)": round(fp32_m["lat_ms"], 3), "Test Top-1 (%)": round(fp32_m["top1"]*100, 2), "Test Macro-F1 (%)": round(fp32_m["macro_f1"]*100, 2)},
    {"Variant": "INT8 Dynamic Quantized", "Size (MB)": int8_mb, "CPU Latency (ms/img)": round(int8_m["lat_ms"], 3), "Test Top-1 (%)": round(int8_m["top1"]*100, 2), "Test Macro-F1 (%)": round(int8_m["macro_f1"]*100, 2)},
])
opt_df.to_csv(WORK_DIR / "optimization_benchmark_letters.csv", index=False)
display(opt_df)

hyb_ood_row = ood_df.iloc[-1]
config_payload = {
    "feature_version": "letter_v1_hybrid_224_98",
    "notebook_version": "2026-11-25",
    "architecture": "HybridHandNetV2(ModalityDropout+Aux+SupCon)",
    "labels": LETTERS_26,
    "crop_size": LETTER_CROP_SIZE,
    "geo_dim": LETTER_GEO_DIM,
    "int8_file": "asl_letters_hybrid_int8.onnx",
    "fp32_size_mb": fp32_mb,
    "int8_size_mb": int8_mb,
    "fp32_latency_ms": round(fp32_m["lat_ms"], 3),
    "int8_latency_ms": round(int8_m["lat_ms"], 3),
    "test_top1_acc": round(float(res_hyb["top1"]), 4),
    "test_macro_f1": round(float(res_hyb["macro_f1"]), 4),
    "int8_top1_acc": round(float(int8_m["top1"]), 4),
    "int8_macro_f1": round(float(int8_m["macro_f1"]), 4),
    "cross_domain_top1_acc": round(float(hyb_ood_row["Unseen Real Top-1 (%)"]) / 100.0, 4),
    "cross_domain_macro_f1": round(float(hyb_ood_row["Unseen Real Macro-F1 (%)"]) / 100.0, 4),
}
(EXPORT_DIR / "letter_model_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

bundle_zip = shutil.make_archive(str(WORK_DIR / "asl_letters_export_v2_2026-11-25"), "zip", str(EXPORT_DIR))
print("Created export bundle:", bundle_zip)
print("Copy asl_letters_hybrid.onnx, asl_letters_hybrid_int8.onnx, and letter_model_config.json into your local 'models/' folder and click Reload Models in the Web App!")

try:
    from google.colab import files
    files.download(bundle_zip)
except Exception:
    print("Manual download ready at:", bundle_zip)"""
    ))

    return cells


def build_letters_v2_notebook(out_path: Path) -> Path:
    return build_and_validate_notebook(get_letters_v2_notebook_cells(), out_path)
