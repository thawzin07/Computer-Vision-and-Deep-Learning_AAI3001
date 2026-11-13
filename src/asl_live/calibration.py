"""Built-in personal webcam calibration recorder for Colab fine-tuning."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import DEFAULT_CALIBRATION_DIR, LETTERS_26, WORDS_15


class CalibrationRecorder:
    """Records short word MP4 clips or letter JPG frames directly from the webcam."""

    def __init__(self, root_dir: Path = DEFAULT_CALIBRATION_DIR) -> None:
        self.root_dir = Path(root_dir)
        self.enabled = False
        self.mode = "words"  # "words" | "letters"
        self.word_idx = 0
        self.letter_idx = 0
        self.state = "IDLE"  # IDLE | COUNTDOWN | RECORDING
        self.countdown_end = 0.0
        self.record_end = 0.0
        self.frames: list[np.ndarray] = []
        self.last_saved_msg = ""

    @property
    def current_label(self) -> str:
        if self.mode == "words":
            return WORDS_15[self.word_idx % len(WORDS_15)]
        return LETTERS_26[self.letter_idx % len(LETTERS_26)]

    def set_mode(self, mode: str) -> None:
        self.mode = "words" if mode == "words" else "letters"
        self.state = "IDLE"
        self.frames.clear()

    def toggle(self, mode: str) -> bool:
        self.enabled = not self.enabled
        self.set_mode(mode)
        self.last_saved_msg = ""
        return self.enabled

    def step_label(self, delta: int) -> str:
        if self.mode == "words":
            self.word_idx = (self.word_idx + delta) % len(WORDS_15)
        else:
            self.letter_idx = (self.letter_idx + delta) % len(LETTERS_26)
        self.state = "IDLE"
        self.frames.clear()
        return self.current_label

    def start_capture(self) -> None:
        if not self.enabled or self.state != "IDLE":
            return
        self.state = "COUNTDOWN"
        self.countdown_end = time.perf_counter() + 1.5
        self.frames.clear()
        self.last_saved_msg = ""

    def step(self, raw_bgr: np.ndarray, fps: float = 30.0) -> Optional[str]:
        """Advance countdown/recording and save when finished. Returns HUD status banner."""
        if not self.enabled:
            return None

        now = time.perf_counter()
        label = self.current_label

        if self.state == "IDLE":
            count = self._count_existing(label)
            return f"CAL [{label.upper()}] ({count} saved) | [N/P] label | [Enter] record"

        if self.state == "COUNTDOWN":
            rem = self.countdown_end - now
            if rem > 0:
                return f"GET READY [{label.upper()}]... {rem:.1f}s"
            self.state = "RECORDING"
            duration = 2.0 if self.mode == "words" else 0.7
            self.record_end = now + duration
            self.frames = [raw_bgr.copy()]
            return f"RECORDING [{label.upper()}]..."

        if self.state == "RECORDING":
            self.frames.append(raw_bgr.copy())
            rem = self.record_end - now
            if rem > 0:
                return f"REC [{label.upper()}] {rem:.1f}s ({len(self.frames)}f)"
            saved_path = self._save_capture(label, fps=max(15.0, min(30.0, fps)))
            self.state = "IDLE"
            self.frames.clear()
            self.last_saved_msg = f"Saved {saved_path.name}"
            return f"SAVED {self.last_saved_msg} ({self._count_existing(label)} total)"

        return None

    def _target_dir(self, label: str) -> Path:
        folder = "words" if self.mode == "words" else "letters"
        safe_label = label.replace(" ", "_")
        out = self.root_dir / folder / safe_label
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _count_existing(self, label: str) -> int:
        folder = "words" if self.mode == "words" else "letters"
        safe_label = label.replace(" ", "_")
        d = self.root_dir / folder / safe_label
        if not d.exists():
            return 0
        ext = "*.mp4" if self.mode == "words" else "*.jpg"
        return len(list(d.glob(ext)))

    def _save_capture(self, label: str, fps: float) -> Path:
        out_dir = self._target_dir(label)
        ts = int(time.time() * 1000)

        if self.mode == "words":
            out_path = out_dir / f"clip_{ts}.mp4"
            h, w = self.frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )
            for frm in self.frames:
                writer.write(frm)
            writer.release()
            return out_path

        # Letter burst: save up to 5 evenly spaced crisp frames
        indices = np.linspace(0, len(self.frames) - 1, min(5, len(self.frames)), dtype=int)
        last_path = out_dir / f"img_{ts}_0.jpg"
        for k, idx in enumerate(indices):
            last_path = out_dir / f"img_{ts}_{k}.jpg"
            cv2.imwrite(str(last_path), self.frames[int(idx)])
        return last_path
