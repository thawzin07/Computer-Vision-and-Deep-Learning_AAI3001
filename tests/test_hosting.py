"""Hosting boundaries: readiness, bounded uploads and per-viewer motion state."""

import asyncio
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import web_app


def test_readiness_and_upload_limit(tmp_path):
    app = web_app.create_app()

    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            response = await client.get("/healthz")
            assert response.status_code == 200 and response.json()["ready"]
            assert response.headers["cache-control"] == "no-store"
            response = await client.post("/api/predict/image", files={
                "file": ("oversized.jpg", b"x" * (web_app.MAX_UPLOAD_BYTES + 1), "image/jpeg"),
            })
            assert response.status_code == 413
        unavailable = web_app.create_app(models_dir=tmp_path)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(unavailable), base_url="http://test") as client:
            response = await client.get("/healthz")
            assert response.status_code == 503 and response.json()["ready"] is False

    asyncio.run(check())


def test_one_viewer_reset_preserves_another_viewers_motion():
    registry = web_app.ModelRegistry()
    first = web_app._LiveWebSession(registry)
    second = web_app._LiveWebSession(registry)
    try:
        hand = np.zeros((21, 3), dtype=np.float32)
        hand[9, :2] = [0.2, 0.2]
        for session in (first, second):
            registry.letter_predictor.predict(
                np.zeros((3, 224, 224), dtype=np.float32), np.zeros(98, dtype=np.float32),
                hand_pts=hand, dynamic_detector=session.jz_detector,
            )
        first.handle_control({"type": "reset"})
        assert len(first.jz_detector.index_hist) == 0
        assert len(second.jz_detector.index_hist) == 1
        assert len(registry.letter_predictor.jz_detector.index_hist) == 0
    finally:
        first.close()
        second.close()
