"""Unit tests for feature shapes, canonicalization invariances, J/Z motion, stabilizer, and notebooks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from asl_live.calibration import CalibrationRecorder
from asl_live.config import (
    LETTER_CROP_SIZE,
    LETTER_GEO_DIM,
    LETTERS_26,
    POSE_INDICES,
    WORD_FEATURE_DIM,
    WORD_SEQ_LEN,
    WORD_STATIC_DIM,
    WORDS_15,
)
from asl_live.features import (
    build_word_sequence_features,
    compute_hand_motion_energy,
    extract_letter_features,
    extract_word_static_frame,
    interpolate_short_gaps,
)
from asl_live.landmarks import FrameLandmarks, draw_landmarks_overlay
from asl_live.predictors import DynamicJZDetector, LetterOnnxPredictor, WordOnnxPredictor
from asl_live.stabilizer import LetterStreamStabilizer, PredictionState, TranscriptBuffer, WordStreamStabilizer
from asl_live.ui import render_hud


def _sample_hand(offset_x: float = 0.0, offset_y: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(7)
    pts = rng.uniform(0.35, 0.55, size=(21, 3)).astype(np.float32)
    pts[0] = np.array([0.45, 0.70, 0.0], dtype=np.float32)
    pts[9] = np.array([0.46, 0.56, 0.0], dtype=np.float32)
    pts[:, 0] += offset_x
    pts[:, 1] += offset_y
    return pts


def _sample_pose() -> np.ndarray:
    pose = np.zeros((15, 3), dtype=np.float32)
    pose[0] = [0.50, 0.20, 0.0]   # nose
    pose[5] = [0.48, 0.25, 0.0]   # mouth L
    pose[6] = [0.52, 0.25, 0.0]   # mouth R
    pose[7] = [0.38, 0.40, 0.0]   # left shoulder
    pose[8] = [0.62, 0.40, 0.0]   # right shoulder
    return pose


def test_word_feature_dimensions_and_gap_interpolation() -> None:
    pose = _sample_pose()
    lh = _sample_hand(0.0, 0.0)
    rh = _sample_hand(0.15, -0.05)

    frames = []
    for i in range(12):
        missing_lh = (i == 4)
        frames.append(
            FrameLandmarks(
                pose=pose,
                left_hand=np.zeros((21, 3), dtype=np.float32) if missing_lh else lh + i * 0.004,
                right_hand=rh + i * 0.003,
                pose_present=True,
                left_present=not missing_lh,
                right_present=True,
            )
        )

    static_seq = np.stack([extract_word_static_frame(f) for f in frames], axis=0)
    assert static_seq.shape == (12, WORD_STATIC_DIM)
    assert static_seq[4, 344] == 0.0

    filled = interpolate_short_gaps(static_seq, max_gap=3)
    assert filled[4, 344] == 1.0  # short 1-frame left-hand dropout interpolated

    kin = build_word_sequence_features(frames, target_len=WORD_SEQ_LEN)
    assert kin.shape == (WORD_SEQ_LEN, WORD_FEATURE_DIM)
    assert np.isfinite(kin).all()


def test_motion_energy_positive_on_moving_hands() -> None:
    pose = _sample_pose()
    f0 = FrameLandmarks(pose, _sample_hand(), np.zeros((21, 3), np.float32), True, True, False)
    f1 = FrameLandmarks(pose, _sample_hand(0.04, -0.02), np.zeros((21, 3), np.float32), True, True, False)
    assert compute_hand_motion_energy(f0, f0) == 0.0
    assert compute_hand_motion_energy(f0, f1) > 0.05


def test_letter_canonicalization_left_right_symmetry() -> None:
    img = np.full((240, 240, 3), 120, dtype=np.uint8)
    pts_right = _sample_hand(0.0, 0.0)
    crop_r, geo_r, bbox_r = extract_letter_features(img, pts_right, handedness="Right")
    assert crop_r.shape == (3, LETTER_CROP_SIZE, LETTER_CROP_SIZE)
    assert geo_r.shape == (LETTER_GEO_DIM,)
    assert bbox_r[2] > bbox_r[0] and bbox_r[3] > bbox_r[1]

    # Mirror both image and hand landmarks, and mark as "Left" -> should produce identical canonical geo
    img_flipped = np.ascontiguousarray(img[:, ::-1, :])
    pts_left = pts_right.copy()
    pts_left[:, 0] = 1.0 - pts_left[:, 0]
    crop_l, geo_l, _ = extract_letter_features(img_flipped, pts_left, handedness="Left")
    assert crop_l.shape == (3, LETTER_CROP_SIZE, LETTER_CROP_SIZE)
    assert np.allclose(geo_r, geo_l, atol=1e-5)


def test_dynamic_j_and_z_detectors() -> None:
    # Build synthetic extended-pinky hand tracing a downward J-hook
    det_j = DynamicJZDetector(window=16)
    base_hand = np.tile(np.array([[0.50, 0.46, 0.0]], dtype=np.float32), (21, 1))
    base_hand[0] = [0.50, 0.50, 0.0]
    base_hand[9] = [0.50, 0.40, 0.0]   # palm scale = 0.10; curled fingers ~0.04 (ratio 0.4 < 1.15)
    base_hand[20] = [0.54, 0.34, 0.0]  # pinky extended (~0.165 > 1.15 * 0.10)

    detected_j = None
    for step in range(14):
        pts = base_hand.copy()
        t = step / 13.0
        # Realistic J stroke: drop down in first half, curve sideways & slightly up at hook tip
        if t <= 0.5:
            dy = 0.16 * t
            dx = 0.0
        else:
            u = (t - 0.5) / 0.5
            dy = 0.08 - 0.02 * u
            dx = -0.07 * u
        pts[:, 0] += dx
        pts[:, 1] += dy
        res = det_j.update(pts, static_probs=None)
        if res is not None:
            detected_j = res[0]
            break
    assert detected_j == "J"

    # Build synthetic extended-index hand tracing a Z zig-zag
    det_z = DynamicJZDetector(window=18)
    z_base = np.tile(np.array([[0.50, 0.46, 0.0]], dtype=np.float32), (21, 1))
    z_base[0] = [0.50, 0.50, 0.0]
    z_base[9] = [0.50, 0.40, 0.0]
    z_base[8] = [0.50, 0.33, 0.0]  # index extended

    waypoints = np.array([
        [0.00, 0.00], [0.04, 0.00], [0.08, 0.00], [0.12, 0.00], [0.16, 0.00],
        [0.12, 0.03], [0.08, 0.06], [0.04, 0.09], [0.00, 0.12],
        [0.04, 0.12], [0.08, 0.12], [0.12, 0.12], [0.16, 0.12],
    ], dtype=np.float32)

    detected_z = None
    for dx, dy in waypoints:
        pts = z_base.copy()
        pts[:, 0] += dx
        pts[:, 1] += dy
        res = det_z.update(pts, static_probs=None)
        if res is not None:
            detected_z = res[0]
            break
    assert detected_z == "Z"


def test_transcript_buffer_and_preview_stabilizers(tmp_path: Path) -> None:
    buf = TranscriptBuffer()
    buf.add_letter("h")
    buf.add_letter("i")
    buf.add_word("hello")
    assert buf.get_text() == "HI hello"
    buf.backspace()
    assert buf.get_text() == "HI"
    buf.backspace()
    assert buf.get_text() == "H"
    buf.clear()
    assert buf.get_text() == ""

    # Predictors safely report unavailable when directory has no ONNX weights
    wp = WordOnnxPredictor(models_dir=tmp_path)
    lp = LetterOnnxPredictor(models_dir=tmp_path)
    assert not wp.available and not lp.available

    ws = WordStreamStabilizer(wp)
    ls = LetterStreamStabilizer()
    pose = _sample_pose()
    lm = FrameLandmarks(pose, _sample_hand(), _sample_hand(0.1, 0.0), True, True, True)
    st = ws.update(lm)
    assert st.predicted_label == "---"
    lst = ls.update(np.zeros(26, dtype=np.float32), LETTERS_26, 0.0, hand_present=True)
    assert "PREVIEW" in lst.status


def test_notebooks_valid_and_constants_match() -> None:
    assert len(WORDS_15) == 15
    assert len(LETTERS_26) == 26
    assert len(POSE_INDICES) == 15

    words_nb_path = PROJECT_ROOT / "colab" / "01_Train_ASL_Words_Conformer.ipynb"
    letters_nb_path = PROJECT_ROOT / "colab" / "02_Train_ASL_Letters_Hybrid.ipynb"
    assert words_nb_path.exists()
    assert letters_nb_path.exists()

    nb_w = nbformat.read(words_nb_path, as_version=4)
    nb_l = nbformat.read(letters_nb_path, as_version=4)
    nbformat.validate(nb_w)
    nbformat.validate(nb_l)

    w_src = "\n".join(c.source for c in nb_w.cells if c.cell_type == "code")
    l_src = "\n".join(c.source for c in nb_l.cells if c.cell_type == "code")
    assert "WORD_FEATURE_DIM = 688" in w_src
    assert "WORD_SEQ_LEN = 48" in w_src
    assert "LETTER_GEO_DIM = 98" in l_src
    assert "LETTER_CROP_SIZE = 224" in l_src


def test_ui_hud_overlay_and_calibration_recorder(tmp_path: Path) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    pose = _sample_pose()
    lm = FrameLandmarks(pose, _sample_hand(), _sample_hand(0.1, 0.0), True, True, True)
    draw_landmarks_overlay(frame, lm, mirrored=True, show_pose=True, hand_bbox=(100, 100, 220, 220))

    pred = PredictionState(
        status="DETECTED: HELLO",
        predicted_label="hello",
        confidence=0.94,
        topk=[("hello", 0.94), ("please", 0.04), ("yes", 0.02)],
        latency_ms=3.2,
        motion_energy=0.03,
        confirmed_token="hello",
        hold_progress=0.8,
    )
    hud_words = render_hud(
        camera_canvas_bgr=frame,
        mode="words",
        pred=pred,
        model_ready=True,
        model_status_msg="Word Conformer ONNX: ACTIVE",
        fps=29.5,
        buffer_enabled=True,
        transcript_text="hello world",
        auto_word_mode=True,
        show_skeleton=True,
        calibration_banner="CAL [HELLO]",
    )
    hud_letters = render_hud(
        camera_canvas_bgr=frame,
        mode="letters",
        pred=pred,
        model_ready=False,
        model_status_msg="Preview mode",
        fps=30.0,
        buffer_enabled=False,
        transcript_text="",
        auto_word_mode=True,
        show_skeleton=False,
        calibration_banner=None,
    )
    assert hud_words.ndim == 3 and hud_words.shape == hud_letters.shape

    cal = CalibrationRecorder(root_dir=tmp_path)
    cal.toggle("letters")
    cal.frames = [frame.copy(), frame.copy()]
    saved_jpg = cal._save_capture("A", fps=30.0)
    assert saved_jpg.exists()

    cal.set_mode("words")
    cal.frames = [frame.copy(), frame.copy()]
    saved_mp4 = cal._save_capture("thank you", fps=30.0)
    assert saved_mp4.exists()

    from asl_live.landmarks import open_camera
    assert callable(open_camera)


def test_web_app_endpoints_and_websocket(tmp_path: Path) -> None:
    import cv2

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import web_app

    # 1. Verify empty models dir status & reload cleanly
    empty_app = web_app.create_app(models_dir=tmp_path)
    empty_status = empty_app.state.registry.status_dict()
    assert empty_status["words"]["available"] is False
    assert empty_status["letters"]["available"] is False
    assert empty_app.state.registry.reload()["words"]["available"] is False

    # 2. Verify static UI assets, Demo Signs payload, & FastAPI route wiring
    app = web_app.create_app()
    routes = {getattr(r, "path", None) for r in app.routes}
    assert {"/", "/api/status", "/api/demo-signs", "/api/reload", "/api/predict/video", "/api/predict/image", "/ws/live"} <= routes
    assert (web_app.STATIC_DIR / "index.html").is_file()
    assert (web_app.STATIC_DIR / "app.css").is_file()
    assert (web_app.STATIC_DIR / "app.js").is_file()

    demo_signs = web_app.get_demo_signs_payload()
    assert len(demo_signs) == 15
    assert [s["word"] for s in demo_signs] == WORDS_15
    assert all(s["image_url"].startswith("/demo_images/") and len(s["how_to"]) > 15 for s in demo_signs)

    # 3. Verify image & video upload inference handlers + live session pipeline
    blank = np.zeros((180, 240, 3), dtype=np.uint8)
    ok_jpg, jpg_buf = cv2.imencode(".jpg", blank)
    assert ok_jpg
    jpg_bytes = jpg_buf.tobytes()

    img_res = web_app._evaluate_image_bytes(app.state.registry, jpg_bytes)
    assert img_res["hand_present"] is False
    assert img_res["preview_url"].startswith("data:image/jpeg;base64,")

    vid_path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (240, 180))
    for _ in range(6):
        writer.write(blank)
    writer.release()

    vid_res = web_app._evaluate_video_bytes(app.state.registry, vid_path.read_bytes(), ".mp4")
    assert "preview_url" in vid_res
    assert vid_res["preview_url"].startswith("data:image/jpeg;base64,")

    sess = web_app._LiveWebSession(app.state.registry)
    try:
        ack = sess.handle_control({"type": "config", "mode": "words", "auto_word_mode": False, "return_overlay": True})
        assert ack["type"] == "ack"
        assert ack["auto_word_mode"] is False

        pred_word = sess.process_frame(jpg_bytes)
        assert pred_word is not None and pred_word["type"] == "prediction" and pred_word["mode"] == "words"

        manual_ack = sess.handle_control({"type": "manual_capture"})
        assert manual_ack["type"] == "prediction" and manual_ack["segmenter"]["manual_recording"] is True

        ack2 = sess.handle_control({"type": "config", "mode": "letters"})
        assert ack2["mode"] == "letters"

        pred_letter = sess.process_frame(jpg_bytes)
        assert pred_letter is not None and pred_letter["type"] == "prediction" and pred_letter["mode"] == "letters"
    finally:
        sess.close()

