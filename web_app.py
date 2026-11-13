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
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asl_live.config import (
    DEFAULT_MEDIAPIPE_DIR,
    DEFAULT_MODELS_DIR,
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
MAX_UPLOAD_BYTES = 3 * 1024 * 1024

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


class ModelRegistry:
    """Thread-safe container holding both ONNX predictors and hot-reload support."""

    def __init__(self, models_dir: Path = DEFAULT_MODELS_DIR) -> None:
        self.models_dir = Path(models_dir)
        self.mediapipe_dir = (
            self.models_dir / "mediapipe" if self.models_dir != DEFAULT_MODELS_DIR else DEFAULT_MEDIAPIPE_DIR
        )
        self.lock = threading.RLock()
        self.word_predictor = WordOnnxPredictor(models_dir=self.models_dir)
        self.letter_predictor = LetterOnnxPredictor(models_dir=self.models_dir)

    def reload(self) -> dict[str, Any]:
        with self.lock:
            self.word_predictor = WordOnnxPredictor(models_dir=self.models_dir)
            self.letter_predictor = LetterOnnxPredictor(models_dir=self.models_dir)
            return self.status_dict()

    def status_dict(self) -> dict[str, Any]:
        with self.lock:
            w = self.word_predictor
            l = self.letter_predictor
            return {
                "words": {
                    "available": w.available,
                    "error": w.load_error,
                    "labels": list(w.labels),
                    "test_top1_acc": w.meta.get("test_top1_acc"),
                    "test_macro_f1": w.meta.get("test_macro_f1"),
                    "motion_threshold": WORD_MOTION_ENERGY_THRESHOLD,
                },
                "letters": {
                    "available": l.available,
                    "error": l.load_error,
                    "labels": list(l.labels),
                    "test_top1_acc": l.meta.get("test_top1_acc"),
                    "test_macro_f1": l.meta.get("test_macro_f1"),
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
    """Per-WebSocket stateful inference pipeline (isolated landmarks & stabilizers)."""

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
                return {"type": "ack", "mode": self.mode, "auto_word_mode": self.word_stab.auto_mode}

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
                            crop_tensor, geo_vec, hand_pts=lm.primary_hand,
                            dynamic_detector=self.jz_detector,
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
                seg_info = {"auto_mode": True, "manual_recording": False, "active_frames": 0}

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


def create_app(models_dir: Path = DEFAULT_MODELS_DIR) -> FastAPI:
    """Create the FastAPI web application bound to the given models directory."""
    app = FastAPI(title="ASL Live Web Translator", version="1.0.0")
    registry = ModelRegistry(models_dir=models_dir)
    app.state.registry = registry

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        status = registry.status_dict()
        ready = all(status[mode]["available"] for mode in ("words", "letters"))
        ready = ready and all((registry.mediapipe_dir / name).is_file() for name in (
            "hand_landmarker.task", "pose_landmarker_lite.task",
        ))
        return JSONResponse({"ready": ready, "revision": os.getenv("VERCEL_GIT_COMMIT_SHA", "local")},
                            status_code=200 if ready else 503,
                            headers={"Cache-Control": "no-store"})

    async def read_upload(file: UploadFile) -> bytes:
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Choose a file smaller than 3 MiB.")
        return raw

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    if DEMO_IMAGES_DIR.exists():
        app.mount("/demo_images", StaticFiles(directory=str(DEMO_IMAGES_DIR)), name="demo_images")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    async def api_status() -> JSONResponse:
        return JSONResponse(registry.status_dict())

    @app.get("/api/demo-signs")
    async def api_demo_signs() -> JSONResponse:
        return JSONResponse({"signs": get_demo_signs_payload(DEMO_IMAGES_DIR)})

    @app.post("/api/reload")
    async def api_reload() -> JSONResponse:
        return JSONResponse(await run_in_threadpool(registry.reload))

    @app.post("/api/predict/video")
    async def api_predict_video(file: UploadFile = File(...)) -> JSONResponse:
        raw = await read_upload(file)
        if not raw:
            raise HTTPException(status_code=400, detail="Empty video upload.")
        suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
        res = await run_in_threadpool(_evaluate_video_bytes, registry, raw, suffix)
        return JSONResponse(res)

    @app.post("/api/predict/image")
    async def api_predict_image(file: UploadFile = File(...)) -> JSONResponse:
        raw = await read_upload(file)
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
    parser = argparse.ArgumentParser(description="Minimalist ASL Live Web App (Words + Letters)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="Directory holding ONNX models")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser window on startup")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = create_app(models_dir=args.models_dir)
    url = f"http://{args.host}:{args.port}"
    print(f"[ASL Live Web] Serving minimalist UI at {url}")
    if not args.no_browser:
        threading.Timer(0.9, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
