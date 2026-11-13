"""Preflight verification script for the Windows local ASL environment."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def check_module(name: str, attr: str | None = None) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "installed")
        if attr and not hasattr(mod, attr):
            return False, f"{ver} (missing attribute '{attr}')"
        return True, str(ver)
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    checks = [
        ("numpy", None),
        ("cv2", "imdecode"),
        ("mediapipe", "tasks"),
        ("onnxruntime", "InferenceSession"),
        ("PIL", "Image"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "run"),
    ]
    all_core_ok = True
    for mod_name, attr in checks:
        ok, info = check_module(mod_name, attr)
        tag = "OK" if ok else "MISSING/ISSUE"
        print(f"  [{tag}] {mod_name}: {info}")
        if not ok:
            all_core_ok = False

    # Check MediaPipe Tasks & optional webcam test
    try:
        from asl_live.config import DEFAULT_MEDIAPIPE_DIR, DEFAULT_MODELS_DIR
        from asl_live.landmarks import ensure_mediapipe_assets

        print("\nChecking MediaPipe task bundles...")
        assets = ensure_mediapipe_assets(DEFAULT_MEDIAPIPE_DIR)
        for name, path in assets.items():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {name} ({size_mb:.1f} MB)")

        print("\nChecking exported Colab ONNX models in models/...")
        expected_files = [
            "asl_words_conformer.onnx",
            "word_model_config.json",
            "asl_letters_hybrid.onnx",
            "letter_model_config.json",
        ]
        for fname in expected_files:
            fpath = DEFAULT_MODELS_DIR / fname
            status = "FOUND" if fpath.exists() else "NOT YET EXPORTED (run Colab notebooks)"
            print(f"  [{'OK' if fpath.exists() else 'WAIT'}] {fname}: {status}")
    except Exception as exc:
        print(f"  [WARN] Asset check failed: {exc}")

    # Quick webcam probe
    try:
        import cv2

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ok, frm = cap.read()
            print(f"\nWebcam 0: {'OK (' + str(frm.shape[1]) + 'x' + str(frm.shape[0]) + ')' if ok and frm is not None else 'Opened but frame read failed'}")
        else:
            print("\nWebcam 0: NOT DETECTED (connect webcam or check Windows privacy settings)")
        cap.release()
    except Exception as exc:
        print(f"\nWebcam check skipped: {exc}")

    if not all_core_ok:
        print("\nRun: pip install -r requirements.txt")
        return 1
    print("\nEnvironment looks ready! Launch live web app with: python web_app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
