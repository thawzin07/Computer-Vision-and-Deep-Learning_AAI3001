"""Minimalist FastAPI + WebSocket web server for live ASL Words & Letters translation."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asl_live.calibration import (
    build_calibration_zip_bytes,
    delete_calibration_sample,
    get_calibration_counts,
    list_annotations,
    save_annotated_letter,
    save_annotated_word,
)
from asl_live.config import (
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_MEDIAPIPE_DIR,
    DEFAULT_MODELS_DIR,
    LETTER_CONFIDENCE_THRESHOLD,
    LETTER_HOLD_CONFIRM_SEC,
    WORD_CONFIDENCE_THRESHOLD,
    WORD_MARGIN_THRESHOLD,
    WORD_MOTION_ENERGY_THRESHOLD,
    WORD_SEQ_LEN,
)
from asl_live.features import (
    build_word_sequence_features,
    extract_letter_features,
    extract_word_static_frame,
)
from asl_live.landmarks import FrameLandmarks, HolisticExtractor, draw_landmarks_overlay
from asl_live.predictors import DynamicJZDetector, LetterOnnxPredictor, WordOnnxPredictor
from asl_live.stabilizer import LetterStreamStabilizer, PredictionState, WordStreamStabilizer

STATIC_DIR = PROJECT_ROOT / "web" / "static"
DEMO_IMAGES_DIR = PROJECT_ROOT / "demo_images"

DEMO_SIGN_GUIDE: list[dict[str, str]] = [
    {
        "word": "eat",
        "filename": "eat.png",
        "how_to": "Pinch your thumb and fingertips together (flat O hand) and tap gently toward your mouth once or twice.",
    },
    {
        "word": "family",
        "filename": "family.png",
        "how_to": "Form F hands (thumb and index touching, three fingers up) in front of your chest, then circle both hands outward until your pinkies meet.",
    },
    {
        "word": "finish",
        "filename": "finish.png",
        "how_to": "Hold both open five-finger hands at chest height with palms facing you, then flick both wrists outward so palms face away.",
    },
    {
        "word": "friend",
        "filename": "friend.png",
        "how_to": "Hook your left and right index fingers together like a link, then flip your hands over and hook them the opposite way.",
    },
    {
        "word": "hello",
        "filename": "helllo.png",
        "how_to": "Bring a flat open hand to your temple/forehead like a crisp salute, then sweep it outward and slightly away from your head.",
    },
    {
        "word": "help",
        "filename": "help.jpg",
        "how_to": "Rest a thumbs-up fist on the flat palm of your opposite hand, then lift both hands together slightly upward.",
    },
    {
        "word": "love",
        "filename": "love.png",
        "how_to": "Close both hands into gentle fists and cross your forearms in an X across your chest as if hugging yourself.",
    },
    {
        "word": "more",
        "filename": "more.png",
        "how_to": "Pinch both hands into flattened O shapes at chest height and tap your fingertips together twice.",
    },
    {
        "word": "no",
        "filename": "no.png",
        "how_to": "Extend your index and middle fingers toward your thumb and snap them closed against your thumb once or twice.",
    },
    {
        "word": "please",
        "filename": "please.png",
        "how_to": "Place your flat open palm against the center of your chest and rub it in a smooth circle.",
    },
    {
        "word": "stop",
        "filename": "stop.png",
        "how_to": "Hold one flat palm facing up, then bring the pinky side of your other open hand cleanly down onto the center of that palm.",
    },
    {
        "word": "thank you",
        "filename": "thankyou.jpg",
        "how_to": "Touch the fingertips of a flat open hand to your chin or lips, then move your hand smoothly forward and down.",
    },
    {
        "word": "want",
        "filename": "want.png",
        "how_to": "Hold both open palms facing upward in front of you, curl your fingers inward into claws, and pull both hands slightly toward you.",
    },
    {
        "word": "water",
        "filename": "water.png",
        "how_to": "Make a W handshape (index, middle, and ring fingers raised) and tap your index finger gently against your chin twice.",
    },
    {
        "word": "yes",
        "filename": "yes.png",
        "how_to": "Form a closed fist (S hand) at shoulder height and nod your wrist firmly up and down like nodding your head.",
    },
]


def get_demo_signs_payload(demo_dir: Path = DEMO_IMAGES_DIR) -> list[dict[str, str]]:
    """Return available demo signs paired with image URLs and beginner instructions."""
    items: list[dict[str, str]] = []
    for entry in DEMO_SIGN_GUIDE:
        img_path = Path(demo_dir) / entry["filename"]
        items.append(
            {
                "word": entry["word"],
                "image_url": f"/demo_images/{entry['filename']}" if img_path.is_file() else "",
                "how_to": entry["how_to"],
            }
        )
    return items


DEMO_LETTER_GUIDE: list[dict[str, Any]] = [
    {"letter": "A", "how_to": "Close all fingers into a compact fist with your thumb resting upright along the side of your index finger.", "dynamic": False},
    {"letter": "B", "how_to": "Hold four fingers straight up side by side, palm facing forward, with your thumb tucked across your palm.", "dynamic": False},
    {"letter": "C", "how_to": "Curve all fingers and thumb together into a clear C shape facing sideways.", "dynamic": False},
    {"letter": "D", "how_to": "Point your index finger straight up while touching your middle, ring, and pinky tips to your thumb.", "dynamic": False},
    {"letter": "E", "how_to": "Curl all four fingertips down so they rest cleanly against the top edge of your tucked thumb.", "dynamic": False},
    {"letter": "F", "how_to": "Touch your index tip to your thumb tip in an OK ring, spreading the other three fingers straight up.", "dynamic": False},
    {"letter": "G", "how_to": "Point your index finger and parallel thumb horizontally to the side with other fingers curled.", "dynamic": False},
    {"letter": "H", "how_to": "Extend your index and middle fingers together horizontally sideways while curling ring, pinky, and thumb.", "dynamic": False},
    {"letter": "I", "how_to": "Make a closed fist and raise only your pinky finger straight up.", "dynamic": False},
    {"letter": "J", "how_to": "Start with the pinky-up I handshape, then swoop your pinky downward and curve up in a J motion.", "dynamic": True},
    {"letter": "K", "how_to": "Raise index and middle fingers in a V and tuck your thumb upright between their base knuckles.", "dynamic": False},
    {"letter": "L", "how_to": "Extend your index finger straight up and your thumb straight out at 90 degrees to form an L.", "dynamic": False},
    {"letter": "M", "how_to": "Tuck your thumb underneath your first three curled fingers (index, middle, and ring) with pinky curled.", "dynamic": False},
    {"letter": "N", "how_to": "Tuck your thumb underneath only your index and middle curled fingers, with ring and pinky curled.", "dynamic": False},
    {"letter": "O", "how_to": "Curve all four fingertips smoothly around to meet your thumb tip in a clean round O.", "dynamic": False},
    {"letter": "P", "how_to": "Form the K handshape (thumb between index and middle) and point the wrist downward.", "dynamic": False},
    {"letter": "Q", "how_to": "Form the G handshape (index and thumb parallel) and point both fingers straight downward.", "dynamic": False},
    {"letter": "R", "how_to": "Raise your index and middle fingers and cross your middle finger snugly behind/over your index.", "dynamic": False},
    {"letter": "S", "how_to": "Make a firm closed fist with your thumb folded horizontally across the front of your curled fingers.", "dynamic": False},
    {"letter": "T", "how_to": "Make a fist with your thumb tucked snugly underneath only your curled index finger.", "dynamic": False},
    {"letter": "U", "how_to": "Raise your index and middle fingers straight up pressed tightly together side by side.", "dynamic": False},
    {"letter": "V", "how_to": "Raise your index and middle fingers straight up spread apart cleanly in a V.", "dynamic": False},
    {"letter": "W", "how_to": "Raise your index, middle, and ring fingers spread into three prongs while thumb holds pinky down.", "dynamic": False},
    {"letter": "X", "how_to": "Raise your index finger and bend its top joint into a hook while other fingers stay curled.", "dynamic": False},
    {"letter": "Y", "how_to": "Extend your thumb and pinky outward in a hang-loose shape with the three middle fingers curled.", "dynamic": False},
    {"letter": "Z", "how_to": "Point your index finger forward and trace a sharp Z zig-zag shape in the air.", "dynamic": True},
]


def _encode_preview_b64(bgr: np.ndarray, max_width: int = 860) -> str:
    h, w = bgr.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        bgr = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _landmarks_to_overlay(
    lm: FrameLandmarks,
    include_pose: bool,
    hand_bbox: Optional[tuple[int, int, int, int]] = None,
    frame_shape: Optional[tuple[int, int]] = None,
) -> dict[str, Any]:
    def pts_list(arr: np.ndarray) -> list[list[float]]:
        return [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in arr]

    bbox_norm = None
    if hand_bbox is not None and frame_shape is not None:
        fh, fw = max(frame_shape[0], 1), max(frame_shape[1], 1)
        x1, y1, x2, y2 = hand_bbox
        bbox_norm = [
            round(x1 / fw, 4),
            round(y1 / fh, 4),
            round(x2 / fw, 4),
            round(y2 / fh, 4),
        ]

    return {
        "pose": pts_list(lm.pose) if (include_pose and lm.pose_present) else None,
        "left_hand": pts_list(lm.left_hand) if lm.left_present else None,
        "right_hand": pts_list(lm.right_hand) if lm.right_present else None,
        "hand_bbox": bbox_norm,
    }


def _pred_to_dict(pred: PredictionState, kind: str) -> dict[str, Any]:
    return {
        "status": pred.status,
        "label": pred.predicted_label,
        "confidence": round(float(pred.confidence), 4),
        "topk": [[lbl, round(float(prob), 4)] for lbl, prob in pred.topk],
        "latency_ms": round(float(pred.latency_ms), 2),
        "motion_energy": round(float(pred.motion_energy), 4),
        "hold_progress": round(float(pred.hold_progress), 4),
        "confirmed_token": pred.confirmed_token,
        "confirmed_kind": kind if pred.confirmed_token else None,
    }


class _PackSelectBody(BaseModel):
    pack_id: str


class _PrecisionBody(BaseModel):
    prefer_int8: bool


class ModelRegistry:
    """Thread-safe container holding ONNX predictors, runtime precision, and pluggable domain packs."""

    def __init__(self, models_dir: Path = DEFAULT_MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)
        self.mediapipe_dir = (
            self.models_dir / "mediapipe" if self.models_dir != DEFAULT_MODELS_DIR else DEFAULT_MEDIAPIPE_DIR
        )
        self.lock = threading.RLock()
        self.deployment_env = os.getenv("DEPLOYMENT_ENV", "local").strip() or "local"
        self.prefer_int8 = os.getenv("ASL_PREFER_INT8", "0").strip().lower() in ("1", "true", "yes")
        self.active_pack_id = "default"
        self.word_predictor = WordOnnxPredictor(models_dir=self.models_dir, prefer_int8=self.prefer_int8)
        self.letter_predictor = LetterOnnxPredictor(models_dir=self.models_dir, prefer_int8=self.prefer_int8)

    def _pack_paths(self, pack_id: str) -> dict[str, Any]:
        if pack_id == "default":
            return {
                "id": "default",
                "name": "General ASL (Default)",
                "domain": "Everyday ASL (15 Words + A-Z)",
                "description": "Balanced Conformer+BiGRU Word model and HybridHandNet Letter model.",
                "word_onnx": self.models_dir / "asl_words_conformer.onnx",
                "word_config": self.models_dir / "word_model_config.json",
                "letter_onnx": self.models_dir / "asl_letters_hybrid.onnx",
                "letter_config": self.models_dir / "letter_model_config.json",
            }
        pack_dir = self.models_dir / "packs" / pack_id
        if not pack_dir.is_dir():
            raise ValueError(f"Model pack '{pack_id}' does not exist.")
        manifest_path = pack_dir / "pack_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        w_onnx = pack_dir / manifest.get("word_onnx", "asl_words_conformer.onnx")
        w_cfg = pack_dir / manifest.get("word_config", "word_model_config.json")
        l_onnx = pack_dir / manifest.get("letter_onnx", "asl_letters_hybrid.onnx")
        l_cfg = pack_dir / manifest.get("letter_config", "letter_model_config.json")
        return {
            "id": pack_id,
            "name": str(manifest.get("name", pack_id.replace("_", " ").title())),
            "domain": str(manifest.get("domain", "Custom Domain Pack")),
            "description": str(manifest.get("description", "Custom domain or sign-language pack.")),
            "word_onnx": w_onnx if w_onnx.is_file() else (self.models_dir / "asl_words_conformer.onnx"),
            "word_config": w_cfg if w_cfg.is_file() else (self.models_dir / "word_model_config.json"),
            "letter_onnx": l_onnx if l_onnx.is_file() else (self.models_dir / "asl_letters_hybrid.onnx"),
            "letter_config": l_cfg if l_cfg.is_file() else (self.models_dir / "letter_model_config.json"),
        }

    def discover_packs(self) -> list[dict[str, Any]]:
        packs_out: list[dict[str, Any]] = []
        candidate_ids = ["default"]
        packs_root = self.models_dir / "packs"
        if packs_root.is_dir():
            for child in sorted(packs_root.iterdir()):
                if child.is_dir() and ((child / "pack_manifest.json").is_file() or any(child.glob("*.onnx"))):
                    candidate_ids.append(child.name)
        for pid in candidate_ids:
            info = self._pack_paths(pid)
            packs_out.append(
                {
                    "id": info["id"],
                    "name": info["name"],
                    "domain": info["domain"],
                    "description": info["description"],
                    "has_words": Path(info["word_onnx"]).is_file(),
                    "has_letters": Path(info["letter_onnx"]).is_file(),
                    "active": pid == self.active_pack_id,
                }
            )
        return packs_out

    def _load_active(self) -> dict[str, Any]:
        info = self._pack_paths(self.active_pack_id)
        self.word_predictor = WordOnnxPredictor(
            models_dir=self.models_dir,
            onnx_path=info["word_onnx"],
            config_path=info["word_config"],
            prefer_int8=self.prefer_int8,
        )
        self.letter_predictor = LetterOnnxPredictor(
            models_dir=self.models_dir,
            onnx_path=info["letter_onnx"],
            config_path=info["letter_config"],
            prefer_int8=self.prefer_int8,
        )
        return self.status_dict()

    def reload(self) -> dict[str, Any]:
        with self.lock:
            return self._load_active()

    def select_pack(self, pack_id: str) -> dict[str, Any]:
        with self.lock:
            self._pack_paths(pack_id)
            self.active_pack_id = pack_id
            return self._load_active()

    def set_precision(self, prefer_int8: bool) -> dict[str, Any]:
        with self.lock:
            self.prefer_int8 = bool(prefer_int8)
            return self._load_active()

    @staticmethod
    def _has_int8_sibling(predictor: Any) -> bool:
        if not predictor.available:
            return False
        base = predictor.onnx_path
        stem = base.stem[:-5] if base.stem.endswith("_int8") else base.stem
        int8_name = str(predictor.meta.get("int8_file") or f"{stem}_int8.onnx")
        return base.with_name(int8_name).is_file() or base.with_name(f"{stem}_int8.onnx").is_file()

    def status_dict(self) -> dict[str, Any]:
        with self.lock:
            w = self.word_predictor
            l = self.letter_predictor
            return {
                "deployment": {
                    "env": self.deployment_env,
                    "prefer_int8": self.prefer_int8,
                    "active_pack": self.active_pack_id,
                    "packs": self.discover_packs(),
                },
                "words": {
                    "available": w.available,
                    "error": w.load_error,
                    "labels": list(w.labels),
                    "precision": w.precision,
                    "model_size_mb": w.model_size_mb,
                    "int8_available": self._has_int8_sibling(w),
                    "architecture": w.meta.get("architecture"),
                    "test_top1_acc": w.meta.get("test_top1_acc"),
                    "test_macro_f1": w.meta.get("test_macro_f1"),
                    "confidence_threshold": WORD_CONFIDENCE_THRESHOLD,
                    "motion_threshold": WORD_MOTION_ENERGY_THRESHOLD,
                },
                "letters": {
                    "available": l.available,
                    "error": l.load_error,
                    "labels": list(l.labels),
                    "precision": l.precision,
                    "model_size_mb": l.model_size_mb,
                    "int8_available": self._has_int8_sibling(l),
                    "architecture": l.meta.get("architecture"),
                    "test_top1_acc": l.meta.get("test_top1_acc"),
                    "test_macro_f1": l.meta.get("test_macro_f1"),
                    "confidence_threshold": LETTER_CONFIDENCE_THRESHOLD,
                    "hold_sec": LETTER_HOLD_CONFIRM_SEC,
                },
            }


def _evaluate_video_bytes(registry: ModelRegistry, video_bytes: bytes, suffix: str) -> dict[str, Any]:
    tmp_path: Optional[Path] = None
    cap: Optional[cv2.VideoCapture] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".mp4") as tmp:
            tmp.write(video_bytes)
            tmp_path = Path(tmp.name)

        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not decode the uploaded video file.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        raw_frames: list[np.ndarray] = []
        while len(raw_frames) < 400:
            ok, frm = cap.read()
            if not ok or frm is None:
                break
            raw_frames.append(frm)

        if not raw_frames:
            raise HTTPException(status_code=400, detail="Uploaded video contains zero readable frames.")

        # Uniformly subsample if clip is long so processing finishes quickly
        if len(raw_frames) > 150:
            idxs = np.linspace(0, len(raw_frames) - 1, num=150, dtype=int)
            sampled = [raw_frames[int(i)] for i in idxs]
        else:
            sampled = raw_frames

        static_vecs: list[np.ndarray] = []
        annotated_frames: list[np.ndarray] = []
        with HolisticExtractor(mediapipe_dir=registry.mediapipe_dir, video_mode=True) as extractor:
            for idx, frm in enumerate(sampled):
                lm = extractor.process_bgr(frm, timestamp_ms=idx * 33, extract_pose=True)
                if lm.pose_present or lm.left_present or lm.right_present:
                    static_vecs.append(extract_word_static_frame(lm))
                    vis = frm.copy()
                    draw_landmarks_overlay(vis, lm, mirrored=False, show_pose=True)
                    annotated_frames.append(vis)

        preview_src = annotated_frames[len(annotated_frames) // 2] if annotated_frames else sampled[len(sampled) // 2]
        preview_url = _encode_preview_b64(preview_src)

        if len(static_vecs) < 4:
            return {
                "ok": False,
                "available": registry.word_predictor.available,
                "message": "No clear hand or upper-body landmarks detected in clip.",
                "active_frames": len(static_vecs),
                "total_frames": total_frames or len(raw_frames),
                "preview_url": preview_url,
            }

        with registry.lock:
            wp = registry.word_predictor
            if not wp.available:
                return {
                    "ok": False,
                    "available": False,
                    "message": wp.load_error or "Word ONNX model not loaded.",
                    "active_frames": len(static_vecs),
                    "total_frames": total_frames or len(raw_frames),
                    "preview_url": preview_url,
                }
            seq = build_word_sequence_features(np.stack(static_vecs, axis=0), target_len=WORD_SEQ_LEN)
            _, topk, latency_ms = wp.predict(seq)

        top1_label, top1_conf = topk[0]
        top2_conf = topk[1][1] if len(topk) > 1 else 0.0
        confident = top1_conf >= WORD_CONFIDENCE_THRESHOLD and (top1_conf - top2_conf) >= WORD_MARGIN_THRESHOLD
        return {
            "ok": True,
            "available": True,
            "label": top1_label,
            "confidence": round(float(top1_conf), 4),
            "confident": confident,
            "topk": [[lbl, round(float(p), 4)] for lbl, p in topk],
            "latency_ms": round(float(latency_ms), 2),
            "active_frames": len(static_vecs),
            "total_frames": total_frames or len(raw_frames),
            "preview_url": preview_url,
        }
    finally:
        if cap is not None:
            cap.release()
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _evaluate_image_bytes(registry: ModelRegistry, image_bytes: bytes) -> dict[str, Any]:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image file.")

    with HolisticExtractor(mediapipe_dir=registry.mediapipe_dir, video_mode=False) as extractor:
        lm = extractor.process_bgr(bgr, extract_pose=False)

    vis = bgr.copy()
    if lm.primary_hand is None:
        return {
            "ok": False,
            "available": registry.letter_predictor.available,
            "hand_present": False,
            "message": "No hand detected in the uploaded image.",
            "preview_url": _encode_preview_b64(vis),
        }

    crop_tensor, geo_vec, hand_bbox = extract_letter_features(bgr, lm.primary_hand, lm.primary_handedness)
    draw_landmarks_overlay(vis, lm, mirrored=False, show_pose=False, hand_bbox=hand_bbox)
    preview_url = _encode_preview_b64(vis)

    with registry.lock:
        lp = registry.letter_predictor
        if not lp.available:
            return {
                "ok": False,
                "available": False,
                "hand_present": True,
                "message": lp.load_error or "Letter model not loaded yet (add Notebook 2 export files into models/).",
                "preview_url": preview_url,
            }
        # Static single-image check bypasses trajectory buffer
        _, topk, latency_ms, _ = lp.predict(crop_tensor, geo_vec, hand_pts=None)

    top1_label, top1_conf = topk[0]
    return {
        "ok": True,
        "available": True,
        "hand_present": True,
        "label": top1_label,
        "confidence": round(float(top1_conf), 4),
        "topk": [[lbl, round(float(p), 4)] for lbl, p in topk],
        "latency_ms": round(float(latency_ms), 2),
        "preview_url": preview_url,
    }


class _LiveWebSession:
    """Per-WebSocket stateful inference pipeline (isolated landmarks, stabilizers & J/Z state)."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.mode = "words"
        self.return_overlay = True
        with registry.lock:
            self.word_stab = WordStreamStabilizer(registry.word_predictor)
        self.letter_stab = LetterStreamStabilizer()
        self.jz_detector = DynamicJZDetector()
        self.extractor = HolisticExtractor(mediapipe_dir=registry.mediapipe_dir, video_mode=True)
        self.proc_lock = threading.Lock()
        self.t0 = time.perf_counter()

    def close(self) -> None:
        with self.proc_lock:
            self.extractor.close()

    def handle_control(self, msg: dict[str, Any]) -> dict[str, Any]:
        with self.proc_lock:
            mtype = msg.get("type")
            if mtype == "config":
                new_mode = msg.get("mode", self.mode)
                if new_mode in ("words", "letters") and new_mode != self.mode:
                    self.mode = new_mode
                    self.word_stab.reset()
                    self.letter_stab.reset()
                    self.jz_detector.reset()
                if "auto_word_mode" in msg:
                    want_auto = bool(msg["auto_word_mode"])
                    if want_auto != self.word_stab.auto_mode:
                        self.word_stab.toggle_auto_mode()
                if "return_overlay" in msg:
                    self.return_overlay = bool(msg["return_overlay"])
                if "word_conf" in msg:
                    self.word_stab.confidence_threshold = float(np.clip(float(msg["word_conf"]), 0.35, 0.95))
                if "word_motion" in msg:
                    self.word_stab.motion_threshold = float(np.clip(float(msg["word_motion"]), 0.004, 0.045))
                if "letter_conf" in msg:
                    self.letter_stab.confidence_threshold = float(np.clip(float(msg["letter_conf"]), 0.40, 0.95))
                if "letter_hold" in msg:
                    self.letter_stab.hold_sec = float(np.clip(float(msg["letter_hold"]), 0.20, 1.80))
                return {
                    "type": "ack",
                    "mode": self.mode,
                    "auto_word_mode": self.word_stab.auto_mode,
                    "word_conf": round(self.word_stab.confidence_threshold, 3),
                    "word_motion": round(self.word_stab.motion_threshold, 4),
                    "letter_conf": round(self.letter_stab.confidence_threshold, 3),
                    "letter_hold": round(self.letter_stab.hold_sec, 2),
                }

            if mtype == "reset":
                self.word_stab.reset()
                self.letter_stab.reset()
                self.jz_detector.reset()
                return {"type": "ack", "reset": True}

            if mtype == "manual_capture" and self.mode == "words":
                with self.registry.lock:
                    self.word_stab.predictor = self.registry.word_predictor
                    res = self.word_stab.trigger_manual()
                    wp = self.registry.word_predictor
                if res is not None:
                    self.word_stab.last_prediction = res
                    pred_dict = _pred_to_dict(res, "word")
                else:
                    pred_dict = _pred_to_dict(self.word_stab.last_prediction, "word")
                return {
                    "type": "prediction",
                    "mode": "words",
                    "model_ready": wp.available,
                    "model_error": wp.load_error,
                    "pred": pred_dict,
                    "segmenter": {
                        "auto_mode": self.word_stab.auto_mode,
                        "manual_recording": self.word_stab.manual_recording,
                        "active_frames": len(self.word_stab.active_static),
                        "word_state": self.word_stab.state,
                    },
                    "overlay": None,
                }

            return {"type": "ack"}

    def process_frame(self, jpeg_bytes: bytes) -> Optional[dict[str, Any]]:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        raw_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if raw_bgr is None:
            return None

        with self.proc_lock:
            ts_ms = int((time.perf_counter() - self.t0) * 1000.0)
            lm = self.extractor.process_bgr(raw_bgr, timestamp_ms=ts_ms, extract_pose=(self.mode == "words"))

            hand_bbox = None
            if self.mode == "words":
                with self.registry.lock:
                    wp = self.registry.word_predictor
                    self.word_stab.predictor = wp
                    pred = self.word_stab.update(lm)
                    ready, err = wp.available, wp.load_error
                pred_dict = _pred_to_dict(pred, "word")
                seg_info = {
                    "auto_mode": self.word_stab.auto_mode,
                    "manual_recording": self.word_stab.manual_recording,
                    "active_frames": len(self.word_stab.active_static),
                    "word_state": self.word_stab.state,
                }
            else:
                with self.registry.lock:
                    lp = self.registry.letter_predictor
                    ready, err = lp.available, lp.load_error
                    if lm.primary_hand is not None:
                        crop_tensor, geo_vec, hand_bbox = extract_letter_features(
                            raw_bgr, lm.primary_hand, lm.primary_handedness
                        )
                        probs, _, lat_ms, dyn_tag = lp.predict(
                            crop_tensor,
                            geo_vec,
                            hand_pts=lm.primary_hand,
                            jz_detector=self.jz_detector,
                        )
                        pred = self.letter_stab.update(
                            probs=probs,
                            labels=lp.labels,
                            latency_ms=lat_ms,
                            dynamic_tag=dyn_tag,
                            hand_present=True,
                        )
                    else:
                        self.jz_detector.reset()
                        pred = self.letter_stab.update(
                            probs=np.zeros(len(lp.labels), dtype=np.float32),
                            labels=lp.labels,
                            latency_ms=0.0,
                            hand_present=False,
                        )
                pred_dict = _pred_to_dict(pred, "letter")
                seg_info = {"auto_mode": True, "manual_recording": False, "active_frames": 0, "word_state": "IDLE"}

            overlay = (
                _landmarks_to_overlay(
                    lm,
                    include_pose=(self.mode == "words"),
                    hand_bbox=hand_bbox,
                    frame_shape=raw_bgr.shape[:2],
                )
                if self.return_overlay
                else None
            )

            return {
                "type": "prediction",
                "mode": self.mode,
                "model_ready": ready,
                "model_error": err,
                "pred": pred_dict,
                "segmenter": seg_info,
                "overlay": overlay,
            }


def _save_cal_letter_bytes(
    cal_dir: Path,
    label: str,
    image_bytes: bytes,
    *,
    signer_id: str = "anonymous",
    lighting: str = "normal",
    background: str = "plain",
    handedness: str = "right",
    allow_custom: bool = False,
) -> dict[str, Any]:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode calibration snapshot.")
    try:
        record = save_annotated_letter(
            cal_dir,
            label,
            bgr,
            signer_id=signer_id,
            lighting=lighting,
            background=background,
            handedness=handedness,
            allow_custom=allow_custom,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "label": record["label"],
        "path": Path(record["rel_path"]).name,
        "sample": record,
        "counts": get_calibration_counts(cal_dir),
        "annotations": list_annotations(cal_dir, limit=30),
    }


def _save_cal_word_bytes(
    cal_dir: Path,
    label: str,
    frames_bytes: list[bytes],
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
    frames: list[np.ndarray] = []
    for raw in frames_bytes:
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            frames.append(bgr)
    try:
        record = save_annotated_word(
            cal_dir,
            label,
            frames,
            fps=fps,
            signer_id=signer_id,
            lighting=lighting,
            background=background,
            handedness=handedness,
            trim_start_pct=trim_start_pct,
            trim_end_pct=trim_end_pct,
            allow_custom=allow_custom,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "label": record["label"],
        "path": Path(record["rel_path"]).name,
        "frames_saved": int(record["num_frames"]),
        "sample": record,
        "counts": get_calibration_counts(cal_dir),
        "annotations": list_annotations(cal_dir, limit=30),
    }


def create_app(
    models_dir: Path = DEFAULT_MODELS_DIR,
    calibration_dir: Path = DEFAULT_CALIBRATION_DIR,
) -> FastAPI:
    """Create the FastAPI web application bound to the given models and calibration directories."""
    app = FastAPI(title="ASL Live Web Translator", version="2.0.0")
    cors_env = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
    allow_origins = ["*"] if cors_env == "*" else [o.strip() for o in cors_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    registry = ModelRegistry(models_dir=models_dir)
    cal_dir = Path(calibration_dir)
    app.state.registry = registry
    app.state.calibration_dir = cal_dir

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    if DEMO_IMAGES_DIR.exists():
        app.mount("/demo_images", StaticFiles(directory=str(DEMO_IMAGES_DIR)), name="demo_images")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "env": registry.deployment_env,
                "active_pack": registry.active_pack_id,
                "prefer_int8": registry.prefer_int8,
                "words_ready": registry.word_predictor.available,
                "letters_ready": registry.letter_predictor.available,
            }
        )

    @app.get("/api/status")
    async def api_status() -> JSONResponse:
        return JSONResponse(registry.status_dict())

    @app.get("/api/packs")
    async def api_packs() -> JSONResponse:
        return JSONResponse(
            {
                "active_pack": registry.active_pack_id,
                "prefer_int8": registry.prefer_int8,
                "packs": registry.discover_packs(),
            }
        )

    @app.post("/api/packs/select")
    async def api_packs_select(body: _PackSelectBody) -> JSONResponse:
        try:
            status = await run_in_threadpool(registry.select_pack, body.pack_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(status)

    @app.post("/api/precision")
    async def api_precision(body: _PrecisionBody) -> JSONResponse:
        status = await run_in_threadpool(registry.set_precision, body.prefer_int8)
        return JSONResponse(status)

    @app.get("/api/demo-signs")
    async def api_demo_signs() -> JSONResponse:
        chart_path = DEMO_IMAGES_DIR / "asl_alphabet_chart.jpg"
        return JSONResponse(
            {
                "signs": get_demo_signs_payload(DEMO_IMAGES_DIR),
                "letters": DEMO_LETTER_GUIDE,
                "letter_chart_url": "/demo_images/asl_alphabet_chart.jpg" if chart_path.is_file() else "",
            }
        )

    @app.post("/api/reload")
    async def api_reload() -> JSONResponse:
        return JSONResponse(await run_in_threadpool(registry.reload))

    @app.get("/api/calibration/status")
    async def api_calibration_status() -> JSONResponse:
        return JSONResponse(await run_in_threadpool(get_calibration_counts, cal_dir))

    @app.get("/api/calibration/annotations")
    async def api_calibration_annotations(limit: int = 50) -> JSONResponse:
        counts, rows = await run_in_threadpool(
            lambda: (get_calibration_counts(cal_dir), list_annotations(cal_dir, limit=limit))
        )
        return JSONResponse({"counts": counts, "annotations": rows})

    @app.delete("/api/calibration/sample/{sample_id}")
    async def api_calibration_delete(sample_id: str) -> JSONResponse:
        removed = await run_in_threadpool(delete_calibration_sample, cal_dir, sample_id)
        if not removed:
            raise HTTPException(status_code=404, detail=f"Annotation sample '{sample_id}' not found.")
        counts, rows = await run_in_threadpool(
            lambda: (get_calibration_counts(cal_dir), list_annotations(cal_dir, limit=50))
        )
        return JSONResponse({"ok": True, "counts": counts, "annotations": rows})

    @app.post("/api/calibration/letter")
    async def api_calibration_letter(
        label: str = Form(...),
        file: UploadFile = File(...),
        signer_id: str = Form("anonymous"),
        lighting: str = Form("normal"),
        background: str = Form("plain"),
        handedness: str = Form("right"),
        allow_custom: bool = Form(False),
    ) -> JSONResponse:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty calibration snapshot.")
        res = await run_in_threadpool(
            lambda: _save_cal_letter_bytes(
                cal_dir,
                label,
                raw,
                signer_id=signer_id,
                lighting=lighting,
                background=background,
                handedness=handedness,
                allow_custom=allow_custom,
            )
        )
        return JSONResponse(res)

    @app.post("/api/calibration/word")
    async def api_calibration_word(
        label: str = Form(...),
        files: list[UploadFile] = File(...),
        fps: float = Form(15.0),
        signer_id: str = Form("anonymous"),
        lighting: str = Form("normal"),
        background: str = Form("plain"),
        handedness: str = Form("right"),
        trim_start_pct: float = Form(0.0),
        trim_end_pct: float = Form(1.0),
        allow_custom: bool = Form(False),
    ) -> JSONResponse:
        frames_bytes = [await f.read() for f in files]
        res = await run_in_threadpool(
            lambda: _save_cal_word_bytes(
                cal_dir,
                label,
                frames_bytes,
                fps=fps,
                signer_id=signer_id,
                lighting=lighting,
                background=background,
                handedness=handedness,
                trim_start_pct=trim_start_pct,
                trim_end_pct=trim_end_pct,
                allow_custom=allow_custom,
            )
        )
        return JSONResponse(res)

    @app.get("/api/calibration/export")
    async def api_calibration_export() -> Response:
        counts = await run_in_threadpool(get_calibration_counts, cal_dir)
        if counts["total_words"] == 0 and counts["total_letters"] == 0:
            raise HTTPException(status_code=404, detail="No calibration samples recorded yet.")
        payload = await run_in_threadpool(build_calibration_zip_bytes, cal_dir)
        return Response(
            content=payload,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=calibration_data.zip"},
        )

    @app.post("/api/predict/video")
    async def api_predict_video(file: UploadFile = File(...)) -> JSONResponse:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty video upload.")
        suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
        res = await run_in_threadpool(_evaluate_video_bytes, registry, raw, suffix)
        return JSONResponse(res)

    @app.post("/api/predict/image")
    async def api_predict_image(file: UploadFile = File(...)) -> JSONResponse:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty image upload.")
        res = await run_in_threadpool(_evaluate_image_bytes, registry, raw)
        return JSONResponse(res)

    @app.websocket("/ws/live")
    async def ws_live(websocket: WebSocket) -> None:
        await websocket.accept()
        session = await run_in_threadpool(_LiveWebSession, registry)
        try:
            while True:
                msg = await websocket.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("text"):
                    payload = json.loads(msg["text"])
                    reply = await run_in_threadpool(session.handle_control, payload)
                    await websocket.send_json(reply)
                elif msg.get("bytes"):
                    reply = await run_in_threadpool(session.process_frame, msg["bytes"])
                    if reply is not None:
                        await websocket.send_json(reply)
        except WebSocketDisconnect:
            pass
        finally:
            await run_in_threadpool(session.close)

    return app


def parse_args() -> argparse.Namespace:
    default_host = os.getenv("HOST", "127.0.0.1")
    default_port = int(os.getenv("PORT", "8000"))
    parser = argparse.ArgumentParser(description="Minimalist ASL Live Web App (Words + Letters)")
    parser.add_argument("--host", default=default_host, help="Bind host (default: $HOST or 127.0.0.1)")
    parser.add_argument("--port", type=int, default=default_port, help="Bind port (default: $PORT or 8000)")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Directory holding ONNX models")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser window on startup")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = create_app(models_dir=args.models_dir)
    url = f"http://{args.host}:{args.port}"
    print(f"[ASL Live Web] Serving minimalist UI at {url}")
    if not args.no_browser and args.host in ("127.0.0.1", "localhost"):
        threading.Timer(0.9, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
