"""Built-in annotated dataset collection & webcam calibration helpers for Colab fine-tuning."""

from __future__ import annotations

import csv
import io
import json
import re
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .config import DEFAULT_CALIBRATION_DIR, LETTERS_26, WORDS_15

ANNOTATION_FIELDS = [
    "sample_id",
    "rel_path",
    "mode",
    "label",
    "signer_id",
    "lighting",
    "background",
    "handedness",
    "trim_start_pct",
    "trim_end_pct",
    "num_frames",
    "is_custom_label",
    "verified",
    "created_at_utc",
]
_CUSTOM_LABEL_RE = re.compile(r"^[A-Za-z0-9 _-]{1,32}$")


def validate_calibration_label(mode: str, label: str, allow_custom: bool = False) -> str:
    """Validate and canonicalize a calibration target label for words or letters."""
    clean = re.sub(r"\s+", " ", str(label or "").strip())
    if mode == "words":
        norm = clean.replace("_", " ").lower()
        for w in WORDS_15:
            if w.lower() == norm:
                return w
        if allow_custom and _CUSTOM_LABEL_RE.match(norm):
            return norm
        raise ValueError(f"Unsupported word label: {label!r}")
    if mode == "letters":
        upper = clean.upper()
        if upper in LETTERS_26:
            return upper
        if allow_custom and _CUSTOM_LABEL_RE.match(upper):
            return upper
        raise ValueError(f"Unsupported letter label: {label!r}")
    raise ValueError(f"Unsupported calibration mode: {mode!r}")


def _calibration_target_dir(root_dir: Path, mode: str, label: str, allow_custom: bool = False) -> tuple[Path, str, bool]:
    canon = validate_calibration_label(mode, label, allow_custom=allow_custom)
    is_custom = (canon not in WORDS_15) if mode == "words" else (canon not in LETTERS_26)
    folder = "words" if mode == "words" else "letters"
    safe = canon.replace(" ", "_")
    out = Path(root_dir) / folder / safe
    out.mkdir(parents=True, exist_ok=True)
    return out, canon, is_custom


def _clean_tag(val: Optional[str], default: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(val or "").strip()).strip("_")
    return (cleaned[:40] if cleaned else default)


def _load_annotations_raw(root_dir: Path) -> list[dict[str, Any]]:
    jsonl_path = Path(root_dir) / "annotations.jsonl"
    if not jsonl_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    return records


def _rewrite_annotations(root_dir: Path, records: list[dict[str, Any]]) -> None:
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    jsonl_path = root / "annotations.jsonl"
    csv_path = root / "annotations.csv"

    jsonl_text = "".join(json.dumps(rec) + "\n" for rec in records)
    jsonl_path.write_text(jsonl_text, encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ANNOTATION_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def _append_annotation(root_dir: Path, record: dict[str, Any]) -> None:
    records = _load_annotations_raw(root_dir)
    records.append(record)
    _rewrite_annotations(root_dir, records)


def list_annotations(root_dir: Path, limit: int = 100) -> list[dict[str, Any]]:
    """Return existing annotation records ordered newest-first."""
    root = Path(root_dir)
    valid = [
        rec
        for rec in _load_annotations_raw(root)
        if rec.get("rel_path") and (root / str(rec["rel_path"])).is_file()
    ]
    return list(reversed(valid))[: max(1, int(limit))]


def delete_calibration_sample(root_dir: Path, sample_id: str) -> bool:
    """Delete a recorded sample file and remove its annotation entry."""
    root = Path(root_dir)
    records = _load_annotations_raw(root)
    kept: list[dict[str, Any]] = []
    removed = False
    for rec in records:
        if rec.get("sample_id") == sample_id:
            rel = rec.get("rel_path")
            if rel:
                target = root / str(rel)
                if target.is_file():
                    target.unlink(missing_ok=True)
            removed = True
        else:
            kept.append(rec)
    if removed:
        _rewrite_annotations(root, kept)
    return removed


def save_annotated_letter(
    root_dir: Path,
    label: str,
    frame_bgr: np.ndarray,
    *,
    signer_id: str = "anonymous",
    lighting: str = "normal",
    background: str = "plain",
    handedness: str = "right",
    allow_custom: bool = False,
) -> dict[str, Any]:
    """Save a letter snapshot and append structured annotation metadata."""
    root = Path(root_dir)
    out_dir, canon, is_custom = _calibration_target_dir(root, "letters", label, allow_custom=allow_custom)
    ts = int(time.time() * 1000)
    sample_id = f"ltr_{ts}_{uuid.uuid4().hex[:6]}"
    out_path = out_dir / f"{sample_id}.jpg"
    if not cv2.imwrite(str(out_path), frame_bgr):
        raise ValueError("Failed to encode calibration image.")

    record: dict[str, Any] = {
        "sample_id": sample_id,
        "rel_path": out_path.relative_to(root).as_posix(),
        "mode": "letters",
        "label": canon,
        "signer_id": _clean_tag(signer_id, "anonymous"),
        "lighting": _clean_tag(lighting, "normal"),
        "background": _clean_tag(background, "plain"),
        "handedness": _clean_tag(handedness, "right"),
        "trim_start_pct": 0.0,
        "trim_end_pct": 1.0,
        "num_frames": 1,
        "is_custom_label": bool(is_custom),
        "verified": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _append_annotation(root, record)
    return {**record, "abs_path": out_path}


def save_calibration_letter_bgr(
    root_dir: Path,
    label: str,
    frame_bgr: np.ndarray,
    *,
    signer_id: str = "anonymous",
    lighting: str = "normal",
    background: str = "plain",
    handedness: str = "right",
    allow_custom: bool = False,
) -> Path:
    """Save a single BGR frame for letter calibration and log annotation metadata."""
    res = save_annotated_letter(
        root_dir,
        label,
        frame_bgr,
        signer_id=signer_id,
        lighting=lighting,
        background=background,
        handedness=handedness,
        allow_custom=allow_custom,
    )
    return Path(res["abs_path"])


def save_annotated_word(
    root_dir: Path,
    label: str,
    frames_bgr: list[np.ndarray],
    fps: float = 15.0,
    *,
    signer_id: str = "anonymous",
    lighting: str = "normal",
    background: str = "plain",
    handedness: str = "right",
    trim_start_pct: float = 0.0,
    trim_end_pct: float = 1.0,
    allow_custom: bool = False,
) -> dict[str, Any]:
    """Trim and save a word clip (.mp4) and append structured annotation metadata."""
    if len(frames_bgr) < 4:
        raise ValueError("At least 4 frames are required to save a word calibration clip.")

    t0 = float(np.clip(trim_start_pct, 0.0, 0.90))
    t1 = float(np.clip(trim_end_pct, 0.10, 1.0))
    if t1 <= t0:
        t0, t1 = 0.0, 1.0
    start_idx = int(round(t0 * len(frames_bgr)))
    end_idx = int(round(t1 * len(frames_bgr)))
    trimmed = frames_bgr[start_idx:end_idx]
    if len(trimmed) < 4:
        trimmed = frames_bgr

    root = Path(root_dir)
    out_dir, canon, is_custom = _calibration_target_dir(root, "words", label, allow_custom=allow_custom)
    ts = int(time.time() * 1000)
    sample_id = f"wrd_{ts}_{uuid.uuid4().hex[:6]}"
    out_path = out_dir / f"{sample_id}.mp4"
    h, w = trimmed[0].shape[:2]
    safe_fps = float(np.clip(fps, 5.0, 60.0))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), safe_fps, (w, h))
    try:
        for frm in trimmed:
            writer.write(frm)
    finally:
        writer.release()

    record: dict[str, Any] = {
        "sample_id": sample_id,
        "rel_path": out_path.relative_to(root).as_posix(),
        "mode": "words",
        "label": canon,
        "signer_id": _clean_tag(signer_id, "anonymous"),
        "lighting": _clean_tag(lighting, "normal"),
        "background": _clean_tag(background, "plain"),
        "handedness": _clean_tag(handedness, "right"),
        "trim_start_pct": round(t0, 3),
        "trim_end_pct": round(t1, 3),
        "num_frames": len(trimmed),
        "is_custom_label": bool(is_custom),
        "verified": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _append_annotation(root, record)
    return {**record, "abs_path": out_path}


def save_calibration_word_frames(
    root_dir: Path,
    label: str,
    frames_bgr: list[np.ndarray],
    fps: float = 15.0,
    *,
    signer_id: str = "anonymous",
    lighting: str = "normal",
    background: str = "plain",
    handedness: str = "right",
    trim_start_pct: float = 0.0,
    trim_end_pct: float = 1.0,
    allow_custom: bool = False,
) -> Path:
    """Save a sequence of BGR frames as an MP4 clip and log annotation metadata."""
    res = save_annotated_word(
        root_dir,
        label,
        frames_bgr,
        fps=fps,
        signer_id=signer_id,
        lighting=lighting,
        background=background,
        handedness=handedness,
        trim_start_pct=trim_start_pct,
        trim_end_pct=trim_end_pct,
        allow_custom=allow_custom,
    )
    return Path(res["abs_path"])


def get_calibration_counts(root_dir: Path) -> dict[str, Any]:
    """Count saved word clips (.mp4) and letter snapshots (.jpg), including custom labels."""
    root = Path(root_dir)
    word_counts: dict[str, int] = {w: 0 for w in WORDS_15}
    words_dir = root / "words"
    if words_dir.exists():
        for d in sorted(words_dir.iterdir()):
            if d.is_dir():
                label = d.name.replace("_", " ")
                word_counts[label] = len(list(d.glob("*.mp4")))

    letter_counts: dict[str, int] = {ch: 0 for ch in LETTERS_26}
    letters_dir = root / "letters"
    if letters_dir.exists():
        for d in sorted(letters_dir.iterdir()):
            if d.is_dir():
                letter_counts[d.name] = len(list(d.glob("*.jpg")))

    anns = list_annotations(root, limit=10000)
    signers = {str(r.get("signer_id", "")) for r in anns if r.get("signer_id")}
    return {
        "words": word_counts,
        "letters": letter_counts,
        "total_words": sum(word_counts.values()),
        "total_letters": sum(letter_counts.values()),
        "annotated_total": len(anns),
        "unique_signers": len(signers),
    }


def build_calibration_zip_bytes(root_dir: Path) -> bytes:
    """Create an in-memory ZIP archive with all saved media files and annotation manifests."""
    root = Path(root_dir)
    files: list[Path] = []
    for sub, pattern in [("words", "*.mp4"), ("letters", "*.jpg")]:
        base = root / sub
        if base.exists():
            files.extend(sorted(p for p in base.rglob(pattern) if p.is_file()))
    for manifest_name in ("annotations.csv", "annotations.jsonl"):
        mp = root / manifest_name
        if mp.is_file():
            files.append(mp)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in files:
            zf.write(fpath, arcname=fpath.relative_to(root).as_posix())
    return buf.getvalue()


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
