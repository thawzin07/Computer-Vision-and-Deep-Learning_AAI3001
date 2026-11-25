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

from asl_live.calibration import (
    CalibrationRecorder,
    build_calibration_zip_bytes,
    delete_calibration_sample,
    get_calibration_counts,
    list_annotations,
    save_annotated_letter,
    save_annotated_word,
    save_calibration_letter_bgr,
    save_calibration_word_frames,
)
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

    nb_pairs = [
        ("01_Train_ASL_Words_Conformer.ipynb", "02_Train_ASL_Letters_Hybrid.ipynb"),
        ("01_Train_ASL_Words_Conformer_2026-11-19.ipynb", "02_Train_ASL_Letters_Hybrid_2026-11-19.ipynb"),
    ]
    for w_name, l_name in nb_pairs:
        words_nb_path = PROJECT_ROOT / "colab" / w_name
        letters_nb_path = PROJECT_ROOT / "colab" / l_name
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
    app = web_app.create_app(calibration_dir=tmp_path / "cal")
    routes = {getattr(r, "path", None) for r in app.routes}
    expected_routes = {
        "/",
        "/api/status",
        "/api/demo-signs",
        "/api/reload",
        "/api/predict/video",
        "/api/predict/image",
        "/api/calibration/status",
        "/api/calibration/letter",
        "/api/calibration/word",
        "/api/calibration/export",
        "/ws/live",
    }
    assert expected_routes <= routes
    assert (web_app.STATIC_DIR / "index.html").is_file()
    assert (web_app.STATIC_DIR / "app.css").is_file()
    assert (web_app.STATIC_DIR / "app.js").is_file()

    demo_signs = web_app.get_demo_signs_payload()
    assert len(demo_signs) == 15
    assert [s["word"] for s in demo_signs] == WORDS_15
    assert all(s["image_url"].startswith("/demo_images/") and len(s["how_to"]) > 15 for s in demo_signs)
    assert [d["letter"] for d in web_app.DEMO_LETTER_GUIDE] == LETTERS_26
    assert {d["letter"] for d in web_app.DEMO_LETTER_GUIDE if d["dynamic"]} == {"J", "Z"}

    # Calibration save, count & ZIP export helpers
    blank = np.zeros((180, 240, 3), dtype=np.uint8)
    cal_root = tmp_path / "cal_helpers"
    save_calibration_letter_bgr(cal_root, "a", blank)
    save_calibration_word_frames(cal_root, "thank you", [blank.copy() for _ in range(5)], fps=15.0)
    counts = get_calibration_counts(cal_root)
    assert counts["total_letters"] == 1 and counts["letters"]["A"] == 1
    assert counts["total_words"] == 1 and counts["words"]["thank you"] == 1
    zip_bytes = build_calibration_zip_bytes(cal_root)
    assert len(zip_bytes) > 50 and zip_bytes[:2] == b"PK"

    # 3. Verify image & video upload inference handlers + live session pipeline + tuning + J/Z isolation
    ok_jpg, jpg_buf = cv2.imencode(".jpg", blank)
    assert ok_jpg
    jpg_bytes = jpg_buf.tobytes()

    cal_save_res = web_app._save_cal_letter_bytes(cal_root, "B", jpg_bytes)
    assert cal_save_res["ok"] and cal_save_res["counts"]["letters"]["B"] == 1
    cal_word_res = web_app._save_cal_word_bytes(cal_root, "hello", [jpg_bytes] * 5, fps=15.0)
    assert cal_word_res["ok"] and cal_word_res["counts"]["words"]["hello"] == 1

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
    sess2 = web_app._LiveWebSession(app.state.registry)
    try:
        assert sess.jz_detector is not sess2.jz_detector
        ack = sess.handle_control({
            "type": "config",
            "mode": "words",
            "auto_word_mode": False,
            "return_overlay": True,
            "word_conf": 0.55,
            "word_motion": 0.02,
            "letter_conf": 0.80,
            "letter_hold": 0.40,
        })
        assert ack["type"] == "ack"
        assert ack["auto_word_mode"] is False
        assert sess.word_stab.confidence_threshold == 0.55
        assert sess.word_stab.motion_threshold == 0.02
        assert sess.letter_stab.confidence_threshold == 0.80
        assert sess.letter_stab.hold_sec == 0.40

        pred_word = sess.process_frame(jpg_bytes)
        assert pred_word is not None and pred_word["type"] == "prediction" and pred_word["mode"] == "words"
        assert "word_state" in pred_word["segmenter"]

        manual_ack = sess.handle_control({"type": "manual_capture"})
        assert manual_ack["type"] == "prediction" and manual_ack["segmenter"]["manual_recording"] is True

        ack2 = sess.handle_control({"type": "config", "mode": "letters"})
        assert ack2["mode"] == "letters"

        pred_letter = sess.process_frame(jpg_bytes)
        assert pred_letter is not None and pred_letter["type"] == "prediction" and pred_letter["mode"] == "letters"
    finally:
        sess.close()
        sess2.close()


def test_exported_onnx_models_smoke() -> None:
    wp = WordOnnxPredictor()
    assert wp.available
    w_probs, w_topk, w_ms = wp.predict(np.zeros((WORD_SEQ_LEN, WORD_FEATURE_DIM), dtype=np.float32))
    assert w_probs.shape == (len(WORDS_15),)
    assert np.isfinite(w_probs).all()
    assert abs(float(w_probs.sum()) - 1.0) < 1e-4
    assert len(w_topk) == 3
    assert w_ms >= 0.0

    lp = LetterOnnxPredictor()
    assert lp.available
    l_probs, l_topk, l_ms, _ = lp.predict(
        np.zeros((3, LETTER_CROP_SIZE, LETTER_CROP_SIZE), dtype=np.float32),
        np.zeros(LETTER_GEO_DIM, dtype=np.float32),
    )
    assert l_probs.shape == (len(LETTERS_26),)
    assert np.isfinite(l_probs).all()
    assert abs(float(l_probs.sum()) - 1.0) < 1e-4
    assert len(l_topk) == 3
    assert l_ms >= 0.0


def test_annotated_calibration_trimming_custom_labels_and_delete(tmp_path: Path) -> None:
    import zipfile
    import io
    import cv2

    cal_root = tmp_path / "annotated_cal"
    blank = np.zeros((160, 200, 3), dtype=np.uint8)

    # Custom sign label + structured metadata
    rec_letter = save_annotated_letter(
        cal_root,
        "custom_ok",
        blank,
        signer_id="teammate_01",
        lighting="bright",
        background="cluttered",
        handedness="right",
        allow_custom=True,
    )
    assert rec_letter["label"] == "CUSTOM_OK"
    assert rec_letter["signer_id"] == "teammate_01"

    # Trimmed custom word clip (12 frames -> 25%..75% -> 6 frames)
    rec_word = save_annotated_word(
        cal_root,
        "clinic help",
        [blank.copy() for _ in range(12)],
        fps=15.0,
        signer_id="teammate_02",
        lighting="low_light",
        background="plain",
        handedness="both",
        trim_start_pct=0.25,
        trim_end_pct=0.75,
        allow_custom=True,
    )
    assert rec_word["label"] == "clinic help"
    assert rec_word["num_frames"] == 6

    counts = get_calibration_counts(cal_root)
    assert counts["annotated_total"] == 2
    assert counts["unique_signers"] == 2
    assert counts["letters"]["CUSTOM_OK"] == 1
    assert counts["words"]["clinic help"] == 1

    rows = list_annotations(cal_root, limit=10)
    assert len(rows) == 2

    ok_jpg, jpg_buf = cv2.imencode(".jpg", blank)
    assert ok_jpg
    import web_app

    api_res = web_app._save_cal_letter_bytes(
        cal_root,
        "C",
        jpg_buf.tobytes(),
        signer_id="teammate_01",
        lighting="normal",
        background="plain",
        handedness="left",
    )
    assert api_res["ok"] and api_res["counts"]["letters"]["C"] == 1

    # ZIP includes both annotations.csv and annotations.jsonl
    zbytes = build_calibration_zip_bytes(cal_root)
    with zipfile.ZipFile(io.BytesIO(zbytes), "r") as zf:
        names = set(zf.namelist())
    assert "annotations.csv" in names
    assert "annotations.jsonl" in names

    # Delete one sample cleanly
    assert delete_calibration_sample(cal_root, rec_letter["sample_id"]) is True
    assert delete_calibration_sample(cal_root, "nonexistent_id") is False
    counts_after = get_calibration_counts(cal_root)
    assert counts_after["annotated_total"] == 2
    assert counts_after["letters"]["CUSTOM_OK"] == 0

    # 4. DOM ID parity between index.html and app.js
    import re

    html_src = (web_app.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    js_src = (web_app.STATIC_DIR / "app.js").read_text(encoding="utf-8")
    html_ids = set(re.findall(r'id="([^"]+)"', html_src))
    js_refs = set(re.findall(r'getElementById\("([^"]+)"\)', js_src))
    assert not (js_refs - html_ids), f"Missing HTML IDs referenced in app.js: {sorted(js_refs - html_ids)}"


def test_model_packs_int8_fallback_and_optimization_tools(tmp_path: Path) -> None:
    import shutil
    import web_app
    from scripts import optimize_models

    # Set up temporary models dir with a pluggable domain pack reusing local weights
    src_models = PROJECT_ROOT / "models"
    models_root = tmp_path / "models"
    models_root.mkdir(parents=True)
    for fname in ("asl_words_conformer.onnx", "word_model_config.json", "asl_letters_hybrid.onnx", "letter_model_config.json"):
        shutil.copy2(src_models / fname, models_root / fname)

    pack_dir = models_root / "packs" / "medical_asl"
    pack_dir.mkdir(parents=True)
    (pack_dir / "pack_manifest.json").write_text(
        json.dumps({"name": "Medical ASL Pack", "domain": "Clinical Triage", "description": "Test pack"}),
        encoding="utf-8",
    )
    shutil.copy2(models_root / "asl_words_conformer.onnx", pack_dir / "asl_words_conformer.onnx")
    shutil.copy2(models_root / "word_model_config.json", pack_dir / "word_model_config.json")

    app = web_app.create_app(models_dir=models_root, calibration_dir=tmp_path / "cal")
    route_paths = {getattr(r, "path", None) for r in app.routes}
    for expected in (
        "/health",
        "/api/packs",
        "/api/packs/select",
        "/api/precision",
        "/api/calibration/annotations",
        "/api/calibration/sample/{sample_id}",
    ):
        assert expected in route_paths

    reg = app.state.registry
    packs = reg.discover_packs()
    assert {p["id"] for p in packs} == {"default", "medical_asl"}

    st_pack = reg.select_pack("medical_asl")
    assert st_pack["deployment"]["active_pack"] == "medical_asl"
    assert st_pack["words"]["available"] and st_pack["letters"]["available"]

    # prefer_int8=True falls back safely to FP32 when *_int8.onnx is not present, and selects *_int8.onnx when present
    st_int8_missing = reg.set_precision(True)
    assert st_int8_missing["words"]["precision"] == "FP32"

    shutil.copy2(models_root / "asl_words_conformer.onnx", models_root / "asl_words_conformer_int8.onnx")
    reg.select_pack("default")
    st_int8_active = reg.set_precision(True)
    assert st_int8_active["words"]["precision"] == "INT8"
    assert st_int8_active["words"]["int8_available"] is True

    bench = optimize_models.benchmark_onnx_model(models_root / "asl_words_conformer.onnx", models_root, warmup=1, runs=3)
    assert bench["precision"] == "fp32" and bench["size_mb"] > 0 and bench["mean_ms"] > 0

    for artifact in ("Dockerfile", ".dockerignore", "docker-compose.yml", "docs/DEPLOYMENT.md", "docs/ARCHITECTURE_AND_SCALABILITY.md", "scripts/optimize_models.py"):
        assert (PROJECT_ROOT / artifact).is_file(), f"Missing artifact: {artifact}"


