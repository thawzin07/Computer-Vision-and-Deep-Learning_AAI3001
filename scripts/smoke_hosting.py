"""Check a running ASL Live server with a synthetic frame and no webcam."""

import argparse
import asyncio
import json
import time
import urllib.request

import cv2
import numpy as np
import websockets


def get_json(base: str, path: str) -> dict:
    with urllib.request.urlopen(base + path, timeout=5) as response:
        return json.load(response)


def wait_for_models(base: str) -> None:
    deadline = time.monotonic() + 45
    while True:
        try:
            health = get_json(base, "/health")
            if health["status"] == "ok" and health["words_ready"] and health["letters_ready"]:
                return
        except OSError as exc:
            health = str(exc)
        if time.monotonic() >= deadline:
            raise RuntimeError(f"ASL Live did not become ready at {base}: {health}")
        time.sleep(1)


def check_image_upload(base: str, jpeg: bytes) -> None:
    boundary = "asl-live-smoke"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        'filename="blank.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'
    ).encode() + jpeg + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(base + "/api/predict/image", data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    })
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    assert result["hand_present"] is False
    assert result["preview_url"].startswith("data:image/jpeg;base64,")


async def check_live(base: str, jpeg: bytes) -> None:
    socket_url = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
    async with websockets.connect(socket_url + "/ws/live", open_timeout=30) as connection:
        for mode in ("words", "letters"):
            await connection.send(json.dumps({"type": "config", "mode": mode, "return_overlay": False}))
            ack = json.loads(await asyncio.wait_for(connection.recv(), timeout=30))
            assert ack["type"] == "ack" and ack["mode"] == mode
            await connection.send(jpeg)
            prediction = json.loads(await asyncio.wait_for(connection.recv(), timeout=30))
            assert prediction["type"] == "prediction"
            assert prediction["mode"] == mode and prediction["model_ready"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    base = parser.parse_args().url.rstrip("/")
    wait_for_models(base)

    status = get_json(base, "/api/status")
    assert status["words"]["available"] and len(status["words"]["labels"]) == 15
    assert status["letters"]["available"] and len(status["letters"]["labels"]) == 26
    guide = get_json(base, "/api/demo-signs")["signs"]
    assert len(guide) == 15
    for path in ("/", "/static/app.css", "/static/app.js", guide[0]["image_url"]):
        with urllib.request.urlopen(base + path, timeout=15) as response:
            assert response.status == 200 and response.read(), path

    encoded, image = cv2.imencode(".jpg", np.zeros((180, 240, 3), dtype=np.uint8))
    assert encoded
    jpeg = image.tobytes()
    check_image_upload(base, jpeg)
    asyncio.run(check_live(base, jpeg))
    print("Passed model readiness, homepage and assets, image upload, and both live recognition modes.")


if __name__ == "__main__":
    main()
