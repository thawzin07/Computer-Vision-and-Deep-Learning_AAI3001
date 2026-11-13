"""ONNX Runtime inference engines for ASL Words, Hybrid Letters, and dynamic J/Z."""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from .config import (
    DEFAULT_MODELS_DIR,
    JZ_TRAJECTORY_WINDOW,
    LETTER_CROP_SIZE,
    LETTER_GEO_DIM,
    LETTERS_26,
    WORD_FEATURE_DIM,
    WORD_SEQ_LEN,
    WORDS_15,
)


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class WordOnnxPredictor:
    """ONNX inference wrapper for the 15-word Conformer sequence model."""

    def __init__(self, models_dir: Path = DEFAULT_MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)
        self.onnx_path = self.models_dir / "asl_words_conformer.onnx"
        self.config_path = self.models_dir / "word_model_config.json"
        self.labels = list(WORDS_15)
        self.seq_len = WORD_SEQ_LEN
        self.feature_dim = WORD_FEATURE_DIM
        self.meta: dict = {}
        self.session = None
        self.load_error: Optional[str] = None
        self._load()

    @property
    def available(self) -> bool:
        return self.session is not None

    def _load(self) -> None:
        if not self.onnx_path.exists():
            self.load_error = f"Missing {self.onnx_path.name} (train & export from Colab Notebook 1)"
            return

        if self.config_path.exists():
            cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
            self.meta = cfg
            self.labels = list(cfg.get("labels", self.labels))
            self.seq_len = int(cfg.get("seq_len", self.seq_len))
            self.feature_dim = int(cfg.get("feature_dim", self.feature_dim))
            if self.feature_dim != WORD_FEATURE_DIM or self.seq_len != WORD_SEQ_LEN:
                self.load_error = (
                    f"Config mismatch: expected ({WORD_SEQ_LEN}, {WORD_FEATURE_DIM}), "
                    f"got ({self.seq_len}, {self.feature_dim})"
                )
                return

        try:
            import onnxruntime as ort
        except ImportError as exc:
            self.load_error = f"onnxruntime not installed: {exc}"
            return

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.onnx_path), sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.load_error = None

    def predict(self, seq_features: np.ndarray) -> tuple[np.ndarray, list[tuple[str, float]], float]:
        """Run inference on (seq_len, 688) or (B, seq_len, 688) features."""
        if not self.available or self.session is None:
            zeros = np.zeros(len(self.labels), dtype=np.float32)
            return zeros, [], 0.0

        batch = seq_features[None, ...] if seq_features.ndim == 2 else seq_features
        batch = batch.astype(np.float32, copy=False)

        t0 = time.perf_counter()
        logits = self.session.run([self.output_name], {self.input_name: batch})[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        probs = _softmax(logits, axis=-1)
        mean_probs = np.mean(probs, axis=0)
        order = np.argsort(mean_probs)[::-1][:3]
        topk = [(self.labels[int(i)], float(mean_probs[int(i)])) for i in order]
        return mean_probs.astype(np.float32), topk, latency_ms


class DynamicJZDetector:
    """Detects dynamic ASL motion letters 'J' (pinky hook) and 'Z' (index zig-zag)."""

    def __init__(self, window: int = JZ_TRAJECTORY_WINDOW) -> None:
        self.window = window
        self.pinky_hist: deque[np.ndarray] = deque(maxlen=window)
        self.index_hist: deque[np.ndarray] = deque(maxlen=window)
        self.scale_hist: deque[float] = deque(maxlen=window)
        self.cooldown_until = 0.0

    def reset(self) -> None:
        self.pinky_hist.clear()
        self.index_hist.clear()
        self.scale_hist.clear()

    def update(self, hand_pts: np.ndarray, static_probs: Optional[np.ndarray] = None) -> Optional[tuple[str, float]]:
        """Update trajectory history and return ('J'|'Z', confidence) if stroke matched."""
        palm_scale = max(float(np.linalg.norm(hand_pts[9, :2] - hand_pts[0, :2])), 0.025)
        self.pinky_hist.append(hand_pts[20, :2].copy())
        self.index_hist.append(hand_pts[8, :2].copy())
        self.scale_hist.append(palm_scale)

        now = time.perf_counter()
        if now < self.cooldown_until or len(self.pinky_hist) < 10:
            return None

        avg_scale = max(float(np.mean(self.scale_hist)), 0.025)

        # Finger extension checks in current frame
        pinky_ext = float(np.linalg.norm(hand_pts[20, :2] - hand_pts[0, :2])) / palm_scale
        index_ext = float(np.linalg.norm(hand_pts[8, :2] - hand_pts[0, :2])) / palm_scale
        middle_ext = float(np.linalg.norm(hand_pts[12, :2] - hand_pts[0, :2])) / palm_scale
        ring_ext = float(np.linalg.norm(hand_pts[16, :2] - hand_pts[0, :2])) / palm_scale

        # Check Z stroke first (index extended, middle/ring curled)
        if index_ext > 1.20 and middle_ext < 1.15 and ring_ext < 1.15:
            pts = np.array(self.index_hist, dtype=np.float32) / avg_scale
            if self._is_z_stroke(pts):
                self.cooldown_until = now + 0.9
                self.reset()
                return ("Z", 0.92)

        # Check J stroke (pinky extended, index/middle/ring curled, I or J plausible)
        i_or_j_prob = 1.0
        if static_probs is not None and len(static_probs) == 26:
            i_or_j_prob = float(static_probs[ord("I") - ord("A")] + static_probs[ord("J") - ord("A")])

        if pinky_ext > 1.15 and index_ext < 1.15 and middle_ext < 1.15 and i_or_j_prob > 0.12:
            pts = np.array(self.pinky_hist, dtype=np.float32) / avg_scale
            if self._is_j_stroke(pts):
                self.cooldown_until = now + 0.9
                self.reset()
                return ("J", 0.90)

        return None

    @staticmethod
    def _is_j_stroke(pts: np.ndarray) -> bool:
        deltas = np.diff(pts, axis=0)
        arc_len = float(np.sum(np.linalg.norm(deltas, axis=1)))
        chord = float(np.linalg.norm(pts[-1] - pts[0]))
        max_drop = float(np.max(pts[:, 1]) - pts[0, 1])  # downward extent of the J
        dx = abs(float(pts[-1, 0] - pts[0, 0]))
        # J moves downward first and curves laterally in the lower half
        mid = len(pts) // 2
        v1 = pts[mid] - pts[0]
        v2 = pts[-1] - pts[mid]
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        cos_turn = float(np.dot(v1, v2) / (n1 * n2 + 1e-6))
        return (
            max_drop > 0.50
            and float(v1[1]) > 0.25
            and dx > 0.30
            and abs(float(v2[0])) > 0.22
            and arc_len > 0.90
            and (arc_len / (chord + 1e-5)) > 1.06
            and cos_turn < 0.85
        )

    @staticmethod
    def _is_z_stroke(pts: np.ndarray) -> bool:
        deltas = np.diff(pts, axis=0)
        arc_len = float(np.sum(np.linalg.norm(deltas, axis=1)))
        dy = float(pts[-1, 1] - pts[0, 1])
        if arc_len < 1.75 or dy < 0.38:
            return False

        # Smooth step deltas and count horizontal sign reversals (expects 2 reversals for Z)
        dx_steps = deltas[:, 0]
        significant = [float(v) for v in dx_steps if abs(v) > 0.045]
        if len(significant) < 4:
            return False
        signs = [1 if v > 0 else -1 for v in significant]
        reversals = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
        return reversals >= 2


class LetterOnnxPredictor:
    """ONNX inference wrapper for the Dual-Branch HybridHandNet (A-Z)."""

    def __init__(self, models_dir: Path = DEFAULT_MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)
        self.onnx_path = self.models_dir / "asl_letters_hybrid.onnx"
        self.config_path = self.models_dir / "letter_model_config.json"
        self.labels = list(LETTERS_26)
        self.crop_size = LETTER_CROP_SIZE
        self.geo_dim = LETTER_GEO_DIM
        self.meta: dict = {}
        self.session = None
        self.load_error: Optional[str] = None
        self.jz_detector = DynamicJZDetector()
        self._load()

    @property
    def available(self) -> bool:
        return self.session is not None

    def _load(self) -> None:
        if not self.onnx_path.exists():
            self.load_error = f"Missing {self.onnx_path.name} (train & export from Colab Notebook 2)"
            return

        if self.config_path.exists():
            cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
            self.meta = cfg
            self.labels = list(cfg.get("labels", self.labels))
            self.crop_size = int(cfg.get("crop_size", self.crop_size))
            self.geo_dim = int(cfg.get("geo_dim", self.geo_dim))
            if self.crop_size != LETTER_CROP_SIZE or self.geo_dim != LETTER_GEO_DIM:
                self.load_error = (
                    f"Config mismatch: expected crop={LETTER_CROP_SIZE}, geo={LETTER_GEO_DIM}; "
                    f"got crop={self.crop_size}, geo={self.geo_dim}"
                )
                return

        try:
            import onnxruntime as ort
        except ImportError as exc:
            self.load_error = f"onnxruntime not installed: {exc}"
            return

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            str(self.onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        self.image_input_name = inputs[0].name
        self.geo_input_name = inputs[1].name
        self.output_name = self.session.get_outputs()[0].name
        self.load_error = None

    def predict(
        self,
        crop_tensor: np.ndarray,
        geo_vec: np.ndarray,
        hand_pts: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, list[tuple[str, float]], float, Optional[str]]:
        """Run static hybrid inference + dynamic J/Z trajectory check."""
        static_probs = np.zeros(len(self.labels), dtype=np.float32)
        latency_ms = 0.0

        if self.available and self.session is not None:
            img_b = crop_tensor[None, ...].astype(np.float32, copy=False)
            geo_b = geo_vec[None, ...].astype(np.float32, copy=False)
            t0 = time.perf_counter()
            logits = self.session.run(
                [self.output_name],
                {self.image_input_name: img_b, self.geo_input_name: geo_b},
            )[0]
            latency_ms = (time.perf_counter() - t0) * 1000.0
            static_probs = _softmax(logits[0], axis=-1).astype(np.float32)

        dynamic_match = None
        if hand_pts is not None:
            dynamic_match = self.jz_detector.update(hand_pts, static_probs if self.available else None)

        probs = static_probs.copy()
        dynamic_tag: Optional[str] = None
        if dynamic_match is not None:
            letter, conf = dynamic_match
            idx = ord(letter) - ord("A")
            if 0 <= idx < len(probs):
                probs = probs * 0.15
                probs[idx] = conf
                probs = probs / max(float(np.sum(probs)), 1e-6)
                dynamic_tag = f"{letter} (motion)"

        if self.available or dynamic_match is not None:
            order = np.argsort(probs)[::-1][:3]
            topk = [(self.labels[int(i)], float(probs[int(i)])) for i in order]
        else:
            topk = []
        return probs, topk, latency_ms, dynamic_tag
