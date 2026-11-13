"""Motion-gated word segmentation, letter EMA smoothing, and transcript buffer."""

from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import (
    LETTER_CONFIDENCE_THRESHOLD,
    LETTER_EMA_ALPHA,
    LETTER_HOLD_CONFIRM_SEC,
    WORD_CONFIDENCE_THRESHOLD,
    WORD_COOLDOWN_SEC,
    WORD_MARGIN_THRESHOLD,
    WORD_MIN_ACTIVE_FRAMES,
    WORD_MOTION_ENERGY_THRESHOLD,
    WORD_SEQ_LEN,
)
from .features import build_word_sequence_features, compute_hand_motion_energy, extract_word_static_frame
from .landmarks import FrameLandmarks
from .predictors import WordOnnxPredictor


@dataclass
class PredictionState:
    """Unified display state returned to the OpenCV HUD each frame."""

    status: str = "READY"
    predicted_label: str = "---"
    confidence: float = 0.0
    topk: list[tuple[str, float]] = field(default_factory=list)
    latency_ms: float = 0.0
    motion_energy: float = 0.0
    confirmed_token: Optional[str] = None
    hold_progress: float = 0.0


class WordStreamStabilizer:
    """Active-signing motion gate + rolling sequence evaluator for 15 dynamic ASL words."""

    def __init__(self, predictor: WordOnnxPredictor) -> None:
        self.predictor = predictor
        self.auto_mode = True
        self.manual_recording = False
        self.prev_lm: Optional[FrameLandmarks] = None
        self.lead_in_buffer: deque[np.ndarray] = deque(maxlen=6)
        self.active_static: list[np.ndarray] = []
        self.state = "IDLE"  # IDLE | SIGNING | COOLDOWN
        self.idle_streak = 0
        self.cooldown_until = 0.0
        self.frame_counter = 0
        self.smoothed_energy = 0.0
        self.last_prediction = PredictionState(status="READY")

    def reset(self) -> None:
        self.active_static.clear()
        self.lead_in_buffer.clear()
        self.state = "IDLE"
        self.idle_streak = 0
        self.manual_recording = False
        self.last_prediction = PredictionState(status="READY")

    def toggle_auto_mode(self) -> bool:
        self.auto_mode = not self.auto_mode
        self.reset()
        return self.auto_mode

    def trigger_manual(self) -> Optional[PredictionState]:
        """Toggle manual start/stop sign capture when auto_mode is off."""
        if not self.manual_recording:
            self.active_static = list(self.lead_in_buffer)
            self.manual_recording = True
            self.state = "SIGNING"
            self.last_prediction = PredictionState(status="MANUAL SIGNING...")
            return None

        self.manual_recording = False
        res = self._evaluate_active(final=True)
        self.active_static.clear()
        self.state = "IDLE"
        return res

    def update(self, lm: FrameLandmarks) -> PredictionState:
        now = time.perf_counter()
        static_vec = extract_word_static_frame(lm)
        energy = compute_hand_motion_energy(self.prev_lm, lm)
        self.prev_lm = lm
        self.smoothed_energy = 0.55 * energy + 0.45 * self.smoothed_energy
        hands_visible = lm.left_present or lm.right_present

        if not self.auto_mode:
            if self.manual_recording:
                self.active_static.append(static_vec)
                return PredictionState(
                    status=f"RECORDING SIGN ({len(self.active_static)}f) - Press [G] to predict",
                    predicted_label=self.last_prediction.predicted_label,
                    confidence=self.last_prediction.confidence,
                    topk=self.last_prediction.topk,
                    motion_energy=self.smoothed_energy,
                )
            self.lead_in_buffer.append(static_vec)
            out = PredictionState(
                status="MANUAL READY - Press [G] to start/stop",
                predicted_label=self.last_prediction.predicted_label,
                confidence=self.last_prediction.confidence,
                topk=self.last_prediction.topk,
                latency_ms=self.last_prediction.latency_ms,
                motion_energy=self.smoothed_energy,
            )
            return out

        # Auto Segmenter mode
        if self.state == "COOLDOWN":
            self.lead_in_buffer.append(static_vec)
            if now < self.cooldown_until:
                out = PredictionState(
                    status=f"CONFIRMED: {self.last_prediction.predicted_label.upper()}",
                    predicted_label=self.last_prediction.predicted_label,
                    confidence=self.last_prediction.confidence,
                    topk=self.last_prediction.topk,
                    latency_ms=self.last_prediction.latency_ms,
                    motion_energy=self.smoothed_energy,
                )
                return out
            self.state = "IDLE"
            self.last_prediction.status = "READY"

        if self.state == "IDLE":
            self.lead_in_buffer.append(static_vec)
            if hands_visible and self.smoothed_energy >= WORD_MOTION_ENERGY_THRESHOLD:
                self.state = "SIGNING"
                self.active_static = list(self.lead_in_buffer)
                self.idle_streak = 0
                self.frame_counter = 0
            else:
                status = "READY (Show hands & sign)" if hands_visible else "WAITING FOR HANDS"
                return PredictionState(
                    status=status,
                    predicted_label=self.last_prediction.predicted_label,
                    confidence=self.last_prediction.confidence,
                    topk=self.last_prediction.topk,
                    latency_ms=self.last_prediction.latency_ms,
                    motion_energy=self.smoothed_energy,
                )

        # State == SIGNING
        self.active_static.append(static_vec)
        self.frame_counter += 1

        if (not hands_visible) or (self.smoothed_energy < WORD_MOTION_ENERGY_THRESHOLD * 0.65):
            self.idle_streak += 1
        else:
            self.idle_streak = 0

        should_finalize = self.idle_streak >= 6 or len(self.active_static) >= 54
        if should_finalize:
            if len(self.active_static) >= WORD_MIN_ACTIVE_FRAMES:
                result = self._evaluate_active(final=True)
                self.active_static.clear()
                if result.confirmed_token is not None:
                    self.state = "COOLDOWN"
                    self.cooldown_until = now + WORD_COOLDOWN_SEC
                else:
                    self.state = "IDLE"
                self.last_prediction = result
                return result
            self.active_static.clear()
            self.state = "IDLE"
            return PredictionState(
                status="TOO SHORT - Sign clearly",
                predicted_label=self.last_prediction.predicted_label,
                confidence=self.last_prediction.confidence,
                topk=self.last_prediction.topk,
                motion_energy=self.smoothed_energy,
            )

        # Live preview update every 4 active frames once minimum length reached
        if len(self.active_static) >= WORD_MIN_ACTIVE_FRAMES and self.frame_counter % 4 == 0:
            preview = self._evaluate_active(final=False)
            self.last_prediction = preview
            if preview.confirmed_token is not None:
                self.active_static.clear()
                self.state = "COOLDOWN"
                self.cooldown_until = now + WORD_COOLDOWN_SEC
            return preview

        return PredictionState(
            status=f"SIGNING... ({len(self.active_static)}f)",
            predicted_label=self.last_prediction.predicted_label,
            confidence=self.last_prediction.confidence,
            topk=self.last_prediction.topk,
            latency_ms=self.last_prediction.latency_ms,
            motion_energy=self.smoothed_energy,
        )

    def _evaluate_active(self, final: bool) -> PredictionState:
        if not self.predictor.available:
            return PredictionState(
                status="PREVIEW ONLY (No word ONNX weights loaded)",
                motion_energy=self.smoothed_energy,
            )

        arr = np.stack(self.active_static, axis=0)
        windows = [build_word_sequence_features(arr, target_len=WORD_SEQ_LEN)]
        if len(arr) >= 34:
            windows.append(build_word_sequence_features(arr[-32:], target_len=WORD_SEQ_LEN))

        batch = np.stack(windows, axis=0)
        probs, topk, latency_ms = self.predictor.predict(batch)
        if not topk:
            return PredictionState(status="READY", motion_energy=self.smoothed_energy)

        top1_label, top1_conf = topk[0]
        top2_conf = topk[1][1] if len(topk) > 1 else 0.0
        margin = top1_conf - top2_conf

        confirmed: Optional[str] = None
        if top1_conf >= WORD_CONFIDENCE_THRESHOLD and margin >= WORD_MARGIN_THRESHOLD:
            confirmed = top1_label
            status = f"DETECTED: {top1_label.upper()}"
        elif final and top1_conf >= 0.52:
            status = f"UNCERTAIN ({top1_label}? {top1_conf * 100:.0f}%)"
        else:
            status = "SIGNING..."

        return PredictionState(
            status=status,
            predicted_label=top1_label,
            confidence=top1_conf,
            topk=topk,
            latency_ms=latency_ms,
            motion_energy=self.smoothed_energy,
            confirmed_token=confirmed,
        )


class LetterStreamStabilizer:
    """EMA probability smoother + hold-to-confirm dwell detector for A-Z fingerspelling."""

    def __init__(self, alpha: float = LETTER_EMA_ALPHA, hold_sec: float = LETTER_HOLD_CONFIRM_SEC) -> None:
        self.alpha = alpha
        self.hold_sec = hold_sec
        self.ema_probs: Optional[np.ndarray] = None
        self.candidate_label: Optional[str] = None
        self.candidate_since: float = 0.0
        self.last_confirmed_label: Optional[str] = None
        self.last_confirmed_at: float = 0.0

    def reset(self) -> None:
        self.ema_probs = None
        self.candidate_label = None
        self.candidate_since = 0.0

    def update(
        self,
        probs: np.ndarray,
        labels: list[str],
        latency_ms: float,
        dynamic_tag: Optional[str] = None,
        hand_present: bool = True,
    ) -> PredictionState:
        now = time.perf_counter()
        if not hand_present:
            self.reset()
            self.last_confirmed_label = None
            return PredictionState(status="WAITING FOR HAND")

        if float(np.sum(probs)) <= 1e-6:
            return PredictionState(
                status="PREVIEW ONLY (No letter ONNX weights loaded)" if dynamic_tag is None else f"MOTION: {dynamic_tag}",
                predicted_label=dynamic_tag.split()[0] if dynamic_tag else "---",
                confidence=0.90 if dynamic_tag else 0.0,
                topk=[(dynamic_tag.split()[0], 0.90)] if dynamic_tag else [],
                confirmed_token=dynamic_tag.split()[0] if dynamic_tag else None,
            )

        if dynamic_tag is not None:
            # Immediate confirmation for explicit J / Z motion strokes
            letter = dynamic_tag.split()[0]
            self.ema_probs = probs.copy()
            self.candidate_label = letter
            self.candidate_since = now
            self.last_confirmed_label = letter
            self.last_confirmed_at = now
            order = np.argsort(probs)[::-1][:3]
            topk = [(labels[int(i)], float(probs[int(i)])) for i in order]
            return PredictionState(
                status=f"CONFIRMED MOTION: {letter}",
                predicted_label=letter,
                confidence=float(probs[ord(letter) - ord("A")]),
                topk=topk,
                latency_ms=latency_ms,
                confirmed_token=letter,
                hold_progress=1.0,
            )

        if self.ema_probs is None or self.ema_probs.shape != probs.shape:
            self.ema_probs = probs.copy()
        else:
            self.ema_probs = self.alpha * probs + (1.0 - self.alpha) * self.ema_probs

        order = np.argsort(self.ema_probs)[::-1][:3]
        topk = [(labels[int(i)], float(self.ema_probs[int(i)])) for i in order]
        top1_label, top1_conf = topk[0]

        confirmed_token: Optional[str] = None
        hold_progress = 0.0
        status = "SIGNING LETTER"

        if top1_conf >= LETTER_CONFIDENCE_THRESHOLD:
            if self.candidate_label != top1_label:
                self.candidate_label = top1_label
                self.candidate_since = now
            elapsed = now - self.candidate_since
            hold_progress = min(1.0, elapsed / max(self.hold_sec, 0.1))

            can_retrigger = (
                self.last_confirmed_label != top1_label
                or (now - self.last_confirmed_at) > 1.35
            )
            if elapsed >= self.hold_sec and can_retrigger:
                confirmed_token = top1_label
                self.last_confirmed_label = top1_label
                self.last_confirmed_at = now
                self.candidate_since = now + 0.35
                status = f"LOCKED: {top1_label}"
            else:
                status = f"HOLDING {top1_label} ({int(hold_progress * 100)}%)"
        else:
            self.candidate_label = None
            status = f"LOW CONF ({top1_label}? {top1_conf * 100:.0f}%)"

        return PredictionState(
            status=status,
            predicted_label=top1_label,
            confidence=top1_conf,
            topk=topk,
            latency_ms=latency_ms,
            confirmed_token=confirmed_token,
            hold_progress=hold_progress,
        )


class TranscriptBuffer:
    """Builds clean English text from confirmed ASL words and fingerspelled letters."""

    def __init__(self, max_chars: int = 80) -> None:
        self.tokens: list[tuple[str, str]] = []  # (text, kind: "letter" | "word")
        self.max_chars = max_chars
        self._join_letters = False

    def add_letter(self, letter: str) -> None:
        clean = letter.strip().upper()
        if not clean:
            return
        if self._join_letters and self.tokens and self.tokens[-1][1] == "letter":
            prev_text, _ = self.tokens[-1]
            self.tokens[-1] = (prev_text + clean, "letter")
        else:
            self.tokens.append((clean, "letter"))
        self._join_letters = True
        self._trim()

    def add_word(self, word: str) -> None:
        clean = word.strip().lower()
        if not clean:
            return
        self.tokens.append((clean, "word"))
        self._join_letters = False
        self._trim()

    def add_space(self) -> None:
        self._join_letters = False

    def backspace(self) -> None:
        if not self.tokens:
            return
        text, kind = self.tokens[-1]
        if kind == "letter" and len(text) > 1:
            self.tokens[-1] = (text[:-1], "letter")
            self._join_letters = True
        else:
            self.tokens.pop()
            self._join_letters = bool(self.tokens and self.tokens[-1][1] == "letter")

    def clear(self) -> None:
        self.tokens.clear()
        self._join_letters = False

    def get_text(self) -> str:
        return " ".join(text for text, _ in self.tokens)

    def _trim(self) -> None:
        while len(self.get_text()) > self.max_chars and len(self.tokens) > 1:
            self.tokens.pop(0)

    def speak_async(self) -> None:
        text = self.get_text().strip()
        if not text:
            return

        def _worker(msg: str) -> None:
            safe = msg.replace("'", " ")
            cmd = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Speak('{safe}')"
            )
            try:
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", cmd],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            except Exception:
                pass

        threading.Thread(target=_worker, args=(text,), daemon=True).start()
