"""Exercise the actual hosted models, HTTP uploads and WebSocket protocol.

Run against an existing server with --url, or start a local ASGI process with
--spawn-server. No camera, credentials, or external images are required.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import websockets

ROOT = Path(__file__).resolve().parents[1]


def get_json(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=60) as response:
        assert response.status == 200, path
        return json.load(response)


async def check_socket(base: str, jpeg: bytes) -> None:
    url = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    async with websockets.connect(url + "/ws/live", open_timeout=60, close_timeout=5) as ws:
        for mode in ("words", "letters"):
            await ws.send(json.dumps({"type": "config", "mode": mode}))
            ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            assert ack["type"] == "ack" and ack["mode"] == mode
            await ws.send(jpeg)
            prediction = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
            assert prediction["type"] == "prediction"
            assert prediction["mode"] == mode and prediction["model_ready"]


def check_server(base: str) -> None:
    status = get_json(base, "/api/status")
    assert status["words"]["available"] and len(status["words"]["labels"]) == 15
    assert status["letters"]["available"] and len(status["letters"]["labels"]) == 26
    assert get_json(base, "/healthz")["ready"]
    guide = get_json(base, "/api/demo-signs")["signs"]
    assert len(guide) == 15
    for path in ("/", "/static/app.css", "/static/app.js", guide[0]["image_url"]):
        with urllib.request.urlopen(base + path, timeout=60) as response:
            assert response.status == 200 and response.read(), path

    ok, encoded = cv2.imencode(".jpg", np.zeros((180, 240, 3), dtype=np.uint8))
    assert ok
    jpeg = encoded.tobytes()
    boundary = "asl-live-smoke"
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            'filename="blank.jpg"\r\nContent-Type: image/jpeg\r\n\r\n').encode()
    body += jpeg + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(base + "/api/predict/image", data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    })
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
        assert result["hand_present"] is False and result["preview_url"]
    asyncio.run(check_socket(base, jpeg))
    print("PASS: model readiness, UI/assets, image upload, words and letters WebSockets")


def check_model_execution() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from asl_live.predictors import LetterOnnxPredictor, WordOnnxPredictor

    words = WordOnnxPredictor()
    probs, topk, _ = words.predict(np.zeros((48, 688), dtype=np.float32))
    assert len(probs) == 15 and len(topk) == 3 and np.isfinite(probs).all()
    letters = LetterOnnxPredictor()
    probs, topk, _, _ = letters.predict(
        np.zeros((3, 224, 224), dtype=np.float32), np.zeros(98, dtype=np.float32)
    )
    assert len(probs) == 26 and len(topk) == 3 and np.isfinite(probs).all()
    print("PASS: both bundled ONNX models execute on CPU")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--spawn-server", action="store_true")
    args = parser.parse_args()
    process = None
    try:
        if args.spawn_server:
            check_model_execution()
            process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1",
                "--port", "8765", "--log-level", "warning",
            ], cwd=ROOT)
            deadline = time.monotonic() + 90
            while True:
                try:
                    get_json(args.url, "/healthz")
                    break
                except OSError:
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("ASGI server failed to become ready")
                    time.sleep(0.2)
        check_server(args.url.rstrip("/"))
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
