"""Deterministic feature engineering for ASL Words (688-dim) and Letters (224px + 98-dim)."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .config import (
    FINGER_ANGLE_TRIPLES,
    FINGER_MCP_REF,
    FINGERTIP_INDICES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LETTER_CROP_SIZE,
    LETTER_GEO_DIM,
    WORD_FEATURE_DIM,
    WORD_SEQ_LEN,
    WORD_STATIC_DIM,
)
from .landmarks import FrameLandmarks


def compute_joint_angles(hand_pts: np.ndarray) -> np.ndarray:
    """Compute 15 cosine joint angles along the five fingers of a (21, 3) hand."""
    if float(np.abs(hand_pts).sum()) < 1e-6:
        return np.zeros(15, dtype=np.float32)

    angles = np.zeros(15, dtype=np.float32)
    for idx, (a, b, c) in enumerate(FINGER_ANGLE_TRIPLES):
        v1 = hand_pts[a] - hand_pts[b]
        v2 = hand_pts[c] - hand_pts[b]
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
        cos_val = float(np.dot(v1, v2)) / denom
        angles[idx] = float(np.clip(cos_val, -1.0, 1.0))
    return angles


def normalize_hand_local(hand_pts: np.ndarray, present: bool) -> np.ndarray:
    """Center a (21, 3) hand at wrist (idx 0) and scale by wrist-to-middle-MCP distance."""
    if not present or float(np.abs(hand_pts).sum()) < 1e-6:
        return np.zeros((21, 3), dtype=np.float32)
    centered = hand_pts - hand_pts[0:1]
    palm_scale = max(float(np.linalg.norm(centered[9])), 0.02)
    return (centered / palm_scale).astype(np.float32)


def extract_word_static_frame(lm: FrameLandmarks) -> np.ndarray:
    """Build the 346-dim static per-frame vector for ASL Word recognition."""
    if lm.pose_present:
        # Indices 7 & 8 inside POSE_INDICES correspond to left_shoulder (11) & right_shoulder (12)
        l_sh = lm.pose[7]
        r_sh = lm.pose[8]
        anchor = 0.5 * (l_sh + r_sh)
        scale = max(float(np.linalg.norm(l_sh[:2] - r_sh[:2])), 0.15)
        pose_norm = ((lm.pose - anchor) / scale).astype(np.float32)
    else:
        anchor = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        scale = 0.25
        pose_norm = np.zeros((15, 3), dtype=np.float32)

    lh_norm = (
        ((lm.left_hand - anchor) / scale).astype(np.float32)
        if lm.left_present
        else np.zeros((21, 3), dtype=np.float32)
    )
    rh_norm = (
        ((lm.right_hand - anchor) / scale).astype(np.float32)
        if lm.right_present
        else np.zeros((21, 3), dtype=np.float32)
    )

    pos_torso = np.concatenate([pose_norm, lh_norm, rh_norm], axis=0).reshape(-1)  # 171

    lh_local_21 = normalize_hand_local(lm.left_hand, lm.left_present)
    rh_local_21 = normalize_hand_local(lm.right_hand, lm.right_present)

    lh_angles = compute_joint_angles(lm.left_hand) if lm.left_present else np.zeros(15, dtype=np.float32)
    rh_angles = compute_joint_angles(lm.right_hand) if lm.right_present else np.zeros(15, dtype=np.float32)

    # 16 anatomical & cross-hand distances
    rel_dists = np.zeros(16, dtype=np.float32)
    if lm.left_present:
        for i, tip_idx in enumerate(FINGERTIP_INDICES):
            rel_dists[i] = float(np.linalg.norm(lh_local_21[tip_idx]))
    if lm.right_present:
        for i, tip_idx in enumerate(FINGERTIP_INDICES):
            rel_dists[5 + i] = float(np.linalg.norm(rh_local_21[tip_idx]))

    if lm.pose_present:
        nose = pose_norm[0]
        mouth_mid = 0.5 * (pose_norm[5] + pose_norm[6])
        if lm.left_present:
            rel_dists[10] = float(np.linalg.norm(lh_norm[0] - nose))
            rel_dists[11] = float(np.linalg.norm(lh_norm[0] - mouth_mid))
        if lm.right_present:
            rel_dists[12] = float(np.linalg.norm(rh_norm[0] - nose))
            rel_dists[13] = float(np.linalg.norm(rh_norm[0] - mouth_mid))

    if lm.left_present and lm.right_present:
        rel_dists[14] = float(np.linalg.norm(lh_norm[0] - rh_norm[0]))
        rel_dists[15] = float(np.linalg.norm(lh_norm[8] - rh_norm[8]))

    presence = np.array(
        [float(lm.pose_present), float(lm.left_present), float(lm.right_present)],
        dtype=np.float32,
    )

    static_vec = np.concatenate(
        [
            pos_torso,                  # 171
            lh_local_21.reshape(-1),    # 63
            rh_local_21.reshape(-1),    # 63
            lh_angles,                  # 15
            rh_angles,                  # 15
            rel_dists,                  # 16
            presence,                   # 3
        ],
        axis=0,
    ).astype(np.float32)

    assert static_vec.shape == (WORD_STATIC_DIM,), f"Expected {WORD_STATIC_DIM}, got {static_vec.shape}"
    return static_vec


def interpolate_short_gaps(static_seq: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """Linearly interpolate short missing-hand/pose dropouts up to max_gap frames."""
    seq = static_seq.copy()
    t_len = seq.shape[0]
    if t_len < 3:
        return seq

    # Slices within static_vec corresponding to pose (0:45), lh (45:108, 171:234, 297:312), rh (108:171, 234:297, 312:327)
    flag_specs = [
        (343, [(0, 45)]),
        (344, [(45, 108), (171, 234), (297, 312), (327, 332)]),
        (345, [(108, 171), (234, 297), (312, 327), (332, 337)]),
    ]
    for flag_col, slices in flag_specs:
        present = seq[:, flag_col] > 0.5
        valid_idx = np.where(present)[0]
        if len(valid_idx) < 2:
            continue
        for i in range(len(valid_idx) - 1):
            a, b = int(valid_idx[i]), int(valid_idx[i + 1])
            gap = b - a - 1
            if 1 <= gap <= max_gap:
                for t in range(a + 1, b):
                    alpha = (t - a) / (b - a)
                    for s0, s1 in slices:
                        seq[t, s0:s1] = (1.0 - alpha) * seq[a, s0:s1] + alpha * seq[b, s0:s1]
                    seq[t, flag_col] = 1.0
    return seq


def resample_sequence(seq: np.ndarray, target_len: int = WORD_SEQ_LEN) -> np.ndarray:
    """Temporally resample a (T, D) sequence cleanly to (target_len, D) via vectorized linear interpolation."""
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


def build_word_sequence_features(
    frames: Sequence[FrameLandmarks] | np.ndarray,
    target_len: int = WORD_SEQ_LEN,
) -> np.ndarray:
    """Convert raw FrameLandmarks list or (T, 346) static array into (target_len, 688) kinematic tensor."""
    if isinstance(frames, np.ndarray):
        static_seq = frames.astype(np.float32)
    else:
        if len(frames) == 0:
            return np.zeros((target_len, WORD_FEATURE_DIM), dtype=np.float32)
        static_seq = np.stack([extract_word_static_frame(f) for f in frames], axis=0)

    static_seq = interpolate_short_gaps(static_seq, max_gap=3)
    resampled = resample_sequence(static_seq, target_len=target_len)

    pos = resampled[:, :171]
    vel = np.diff(pos, axis=0, prepend=pos[:1])
    acc = np.diff(vel, axis=0, prepend=vel[:1])

    features = np.concatenate([resampled, vel, acc], axis=1).astype(np.float32)
    assert features.shape == (target_len, WORD_FEATURE_DIM)
    return features


def compute_hand_motion_energy(prev_lm: FrameLandmarks | None, curr_lm: FrameLandmarks) -> float:
    """Compute shoulder-normalized hand kinetic energy between consecutive frames."""
    if prev_lm is None:
        return 0.0
    scale = 0.25
    if curr_lm.pose_present:
        scale = max(float(np.linalg.norm(curr_lm.pose[7, :2] - curr_lm.pose[8, :2])), 0.15)

    deltas = []
    if prev_lm.left_present and curr_lm.left_present:
        pts_p = prev_lm.left_hand[[0, 4, 8, 12, 16, 20], :2]
        pts_c = curr_lm.left_hand[[0, 4, 8, 12, 16, 20], :2]
        deltas.append(float(np.mean(np.linalg.norm(pts_c - pts_p, axis=1))) / scale)
    if prev_lm.right_present and curr_lm.right_present:
        pts_p = prev_lm.right_hand[[0, 4, 8, 12, 16, 20], :2]
        pts_c = curr_lm.right_hand[[0, 4, 8, 12, 16, 20], :2]
        deltas.append(float(np.mean(np.linalg.norm(pts_c - pts_p, axis=1))) / scale)

    if not deltas:
        return 0.0
    return float(max(deltas))


def extract_letter_features(
    frame_bgr: np.ndarray,
    hand_pts: np.ndarray,
    handedness: str = "Right",
    crop_size: int = LETTER_CROP_SIZE,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """
    Extract canonicalized (3, 224, 224) hand crop + (98,) 3D geometric vector + pixel bbox.

    Both Left and Right hands are canonicalized into a unified orientation by horizontally
    mirroring Left-handed detections in both the RGB crop and landmark x-coordinates.
    """
    h, w = frame_bgr.shape[:2]
    xs = hand_pts[:, 0] * w
    ys = hand_pts[:, 1] * h

    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    side = max(max_x - min_x, max_y - min_y, 32.0) * 1.45
    half = int(round(side * 0.5))

    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x2 = int(round(cx)) + half
    y2 = int(round(cy)) + half

    # Aspect-ratio-preserving crop with edge border replication when near image edges
    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - w)
    pad_b = max(0, y2 - h)

    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(w, x2), min(h, y2)
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    elif pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(
            crop, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_REPLICATE
        )

    canonical_pts = hand_pts.astype(np.float32).copy()
    if handedness == "Left":
        crop = cv2.flip(crop, 1)
        canonical_pts[:, 0] = 1.0 - canonical_pts[:, 0]

    crop_resized = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    crop_tensor = np.transpose((crop_rgb - mean) / std, (2, 0, 1)).astype(np.float32)

    # Geometric features (98 dims)
    local_21 = normalize_hand_local(canonical_pts, present=True)
    angles_15 = compute_joint_angles(canonical_pts)
    tip_wrist_5 = np.array(
        [float(np.linalg.norm(local_21[i])) for i in FINGERTIP_INDICES],
        dtype=np.float32,
    )
    pair_dists = []
    for i in range(len(FINGERTIP_INDICES)):
        for j in range(i + 1, len(FINGERTIP_INDICES)):
            a, b = FINGERTIP_INDICES[i], FINGERTIP_INDICES[j]
            pair_dists.append(float(np.linalg.norm(local_21[a] - local_21[b])))
    tip_pair_10 = np.array(pair_dists, dtype=np.float32)

    tip_mcp_5 = np.array(
        [float(np.linalg.norm(local_21[a] - local_21[b])) for a, b in FINGER_MCP_REF],
        dtype=np.float32,
    )

    geo_vec = np.concatenate(
        [local_21.reshape(-1), angles_15, tip_wrist_5, tip_pair_10, tip_mcp_5],
        axis=0,
    ).astype(np.float32)
    assert geo_vec.shape == (LETTER_GEO_DIM,), f"Expected {LETTER_GEO_DIM}, got {geo_vec.shape}"

    bbox = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
    return crop_tensor, geo_vec, bbox
