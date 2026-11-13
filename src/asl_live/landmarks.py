"""MediaPipe Tasks (v1.0+) landmark extraction, asset downloading, and drawing helpers."""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .config import DEFAULT_MEDIAPIPE_DIR, MEDIAPIPE_MODELS, POSE_INDICES

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Upper-body pose skeleton edges within the 15-point POSE_INDICES subset
# POSE_INDICES = [0:nose, 1:l_eye, 2:r_eye, 3:l_ear, 4:r_ear, 5:m_l, 6:m_r,
#                 7:l_sh, 8:r_sh, 9:l_elb, 10:r_elb, 11:l_wri, 12:r_wri, 13:l_hip, 14:r_hip]
UPPER_POSE_EDGES = [
    (7, 8), (7, 9), (9, 11), (8, 10), (10, 12),
    (7, 13), (8, 14), (13, 14), (5, 6),
]


@dataclass
class FrameLandmarks:
    """Structured landmarks extracted from a single RGB frame."""

    pose: np.ndarray            # (15, 3) float32
    left_hand: np.ndarray       # (21, 3) float32
    right_hand: np.ndarray      # (21, 3) float32
    pose_present: bool
    left_present: bool
    right_present: bool
    primary_hand: Optional[np.ndarray] = None       # (21, 3) float32
    primary_handedness: str = "Right"               # "Left" or "Right"
    primary_hand_score: float = 0.0


def ensure_mediapipe_assets(models_dir: Path = DEFAULT_MEDIAPIPE_DIR) -> dict[str, Path]:
    """Ensure required MediaPipe .task bundles exist locally, downloading if needed."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for filename, url in MEDIAPIPE_MODELS.items():
        target = models_dir / filename
        if not target.exists() or target.stat().st_size < 100_000:
            print(f"[MediaPipe] Downloading {filename}...")
            tmp = target.with_suffix(".tmp")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(target)
            print(f"[MediaPipe] Saved {target}")
        paths[filename] = target
    return paths


def _landmarks_to_array(landmarks) -> np.ndarray:
    arr = np.zeros((len(landmarks), 3), dtype=np.float32)
    for idx, lm in enumerate(landmarks):
        arr[idx, 0] = lm.x
        arr[idx, 1] = lm.y
        arr[idx, 2] = lm.z
    return arr


class HolisticExtractor:
    """Thread-safe MediaPipe Tasks wrapper using HandLandmarker + PoseLandmarker."""

    def __init__(
        self,
        mediapipe_dir: Path = DEFAULT_MEDIAPIPE_DIR,
        video_mode: bool = True,
        min_detection_conf: float = 0.45,
        min_tracking_conf: float = 0.45,
    ) -> None:
        assets = ensure_mediapipe_assets(mediapipe_dir)
        self.video_mode = video_mode
        mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE

        hand_opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(assets["hand_landmarker.task"])
            ),
            running_mode=mode,
            num_hands=2,
            min_hand_detection_confidence=min_detection_conf,
            min_hand_presence_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        pose_opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(assets["pose_landmarker_lite.task"])
            ),
            running_mode=mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_conf,
            min_pose_presence_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_opts)
        self.pose_detector = vision.PoseLandmarker.create_from_options(pose_opts)
        self._last_ts_ms = -1

    def close(self) -> None:
        self.hand_detector.close()
        self.pose_detector.close()

    def __enter__(self) -> "HolisticExtractor":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def process_bgr(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: Optional[int] = None,
        extract_pose: bool = True,
    ) -> FrameLandmarks:
        """Run MediaPipe landmark extraction on a BGR OpenCV image."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.video_mode:
            if timestamp_ms is None:
                timestamp_ms = max(0, self._last_ts_ms + 33)
            if timestamp_ms <= self._last_ts_ms:
                timestamp_ms = self._last_ts_ms + 1
            self._last_ts_ms = timestamp_ms
            hand_res = self.hand_detector.detect_for_video(mp_image, timestamp_ms)
            pose_res = (
                self.pose_detector.detect_for_video(mp_image, timestamp_ms)
                if extract_pose
                else None
            )
        else:
            hand_res = self.hand_detector.detect(mp_image)
            pose_res = self.pose_detector.detect(mp_image) if extract_pose else None

        pose_arr = np.zeros((len(POSE_INDICES), 3), dtype=np.float32)
        pose_present = False
        if pose_res and pose_res.pose_landmarks:
            full_pose = _landmarks_to_array(pose_res.pose_landmarks[0])
            pose_arr = full_pose[POSE_INDICES]
            pose_present = True

        left_arr = np.zeros((21, 3), dtype=np.float32)
        right_arr = np.zeros((21, 3), dtype=np.float32)
        left_present = False
        right_present = False

        primary_hand: Optional[np.ndarray] = None
        primary_handedness = "Right"
        primary_score = -1.0

        if hand_res and hand_res.hand_landmarks:
            detections = []
            for idx, lms in enumerate(hand_res.hand_landmarks):
                pts = _landmarks_to_array(lms)
                label = "Right"
                score = 0.5
                if idx < len(hand_res.handedness) and hand_res.handedness[idx]:
                    cat = hand_res.handedness[idx][0]
                    label = cat.category_name or "Right"
                    score = float(cat.score or 0.5)
                detections.append((score, label, pts))

                # Primary hand prefers highest confidence (and larger visible span on ties)
                span = float(np.ptp(pts[:, :2]))
                rank = score + 0.25 * span
                if rank > primary_score:
                    primary_score = rank
                    primary_hand = pts
                    primary_handedness = label

            # Assign up to two hands to Left/Right slots without collisions
            detections.sort(key=lambda d: d[0], reverse=True)
            for _, label, pts in detections:
                if label == "Left" and not left_present:
                    left_arr = pts
                    left_present = True
                elif label == "Right" and not right_present:
                    right_arr = pts
                    right_present = True
                elif not left_present:
                    left_arr = pts
                    left_present = True
                elif not right_present:
                    right_arr = pts
                    right_present = True

        return FrameLandmarks(
            pose=pose_arr,
            left_hand=left_arr,
            right_hand=right_arr,
            pose_present=pose_present,
            left_present=left_present,
            right_present=right_present,
            primary_hand=primary_hand,
            primary_handedness=primary_handedness,
            primary_hand_score=max(0.0, primary_score),
        )


def draw_landmarks_overlay(
    canvas_bgr: np.ndarray,
    landmarks: FrameLandmarks,
    mirrored: bool = True,
    show_pose: bool = True,
    hand_bbox: Optional[tuple[int, int, int, int]] = None,
) -> None:
    """Draw hand & upper-body landmarks onto the display canvas in-place."""
    h, w = canvas_bgr.shape[:2]

    def to_px(pt: np.ndarray) -> tuple[int, int]:
        x = (1.0 - float(pt[0])) if mirrored else float(pt[0])
        y = float(pt[1])
        return int(np.clip(x * w, 0, w - 1)), int(np.clip(y * h, 0, h - 1))

    if show_pose and landmarks.pose_present:
        pts_px = [to_px(landmarks.pose[i]) for i in range(len(POSE_INDICES))]
        for a, b in UPPER_POSE_EDGES:
            cv2.line(canvas_bgr, pts_px[a], pts_px[b], (90, 140, 210), 2, cv2.LINE_AA)
        for pt in pts_px:
            cv2.circle(canvas_bgr, pt, 3, (140, 195, 255), -1, cv2.LINE_AA)

    for present, hand_pts, line_col, dot_col in [
        (landmarks.left_present, landmarks.left_hand, (70, 210, 150), (140, 255, 200)),
        (landmarks.right_present, landmarks.right_hand, (240, 175, 70), (255, 220, 130)),
    ]:
        if not present:
            continue
        pts_px = [to_px(hand_pts[i]) for i in range(21)]
        for a, b in HAND_CONNECTIONS:
            cv2.line(canvas_bgr, pts_px[a], pts_px[b], line_col, 2, cv2.LINE_AA)
        for i, pt in enumerate(pts_px):
            radius = 5 if i in (4, 8, 12, 16, 20) else 3
            cv2.circle(canvas_bgr, pt, radius, dot_col, -1, cv2.LINE_AA)

    if hand_bbox is not None:
        x1, y1, x2, y2 = hand_bbox
        if mirrored:
            x1, x2 = w - x2, w - x1
        cv2.rectangle(canvas_bgr, (x1, y1), (x2, y2), (0, 225, 255), 2, cv2.LINE_AA)


def open_camera(camera_idx: int, width: int = 1280, height: int = 720) -> cv2.VideoCapture:
    """Open Windows webcam via DirectShow first for fast startup, falling back to default."""
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_idx)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
    return cap
