"""Native OpenCV desktop HUD renderer for live ASL Words & Letters recognition."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .stabilizer import PredictionState


def _draw_bar(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    ratio: float,
    fill_color: tuple[int, int, int],
    bg_color: tuple[int, int, int] = (45, 52, 66),
) -> None:
    ratio = float(np.clip(ratio, 0.0, 1.0))
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg_color, -1)
    fill_w = int(round(w * ratio))
    if fill_w > 0:
        cv2.rectangle(canvas, (x, y), (x + fill_w, y + h), fill_color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (75, 85, 105), 1)


def render_hud(
    camera_canvas_bgr: np.ndarray,
    mode: str,
    pred: PredictionState,
    model_ready: bool,
    model_status_msg: str,
    fps: float,
    buffer_enabled: bool,
    transcript_text: str,
    auto_word_mode: bool,
    show_skeleton: bool,
    calibration_banner: Optional[str] = None,
) -> np.ndarray:
    """Compose the webcam feed + telemetry sidebar + transcript footer into one crisp frame."""
    target_cam_h = 600
    src_h, src_w = camera_canvas_bgr.shape[:2]
    target_cam_w = max(640, int(round(src_w * (target_cam_h / max(src_h, 1)))))
    cam_resized = cv2.resize(camera_canvas_bgr, (target_cam_w, target_cam_h), interpolation=cv2.INTER_LINEAR)

    # Top overlay pill on camera feed (mode + calibration banner)
    mode_tag = "MODE: ASL WORDS (15)" if mode == "words" else "MODE: ASL LETTERS (A-Z)"
    cv2.rectangle(cam_resized, (14, 12), (310, 48), (20, 24, 34), -1)
    cv2.rectangle(cam_resized, (14, 12), (310, 48), (0, 200, 255) if mode == "words" else (110, 235, 150), 2)
    cv2.putText(
        cam_resized,
        mode_tag,
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (245, 248, 255),
        2,
        cv2.LINE_AA,
    )

    if calibration_banner:
        cv2.rectangle(cam_resized, (14, 58), (target_cam_w - 14, 96), (25, 35, 75), -1)
        cv2.rectangle(cam_resized, (14, 58), (target_cam_w - 14, 96), (80, 175, 255), 2)
        cv2.putText(
            cam_resized,
            calibration_banner,
            (24, 83),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 235, 140),
            2,
            cv2.LINE_AA,
        )

    # Right sidebar
    panel_w = 390
    sidebar = np.full((target_cam_h, panel_w, 3), (22, 26, 35), dtype=np.uint8)
    cv2.line(sidebar, (0, 0), (0, target_cam_h), (60, 70, 90), 2)

    y = 32
    cv2.putText(sidebar, "ASL LIVE TRANSLATOR", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 245, 255), 2, cv2.LINE_AA)
    y += 28
    perf_text = f"FPS: {fps:4.1f}  |  Infer: {pred.latency_ms:4.1f} ms"
    cv2.putText(sidebar, perf_text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (165, 180, 205), 1, cv2.LINE_AA)
    y += 24

    # Model status badge
    status_col = (80, 220, 140) if model_ready else (80, 185, 255)
    cv2.putText(sidebar, model_status_msg[:44], (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_col, 1, cv2.LINE_AA)
    y += 18
    cv2.line(sidebar, (20, y), (panel_w - 20, y), (50, 58, 74), 1)
    y += 28

    # Primary prediction box
    cv2.putText(sidebar, "PREDICTION", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (145, 160, 185), 1, cv2.LINE_AA)
    y += 12
    box_h = 98
    cv2.rectangle(sidebar, (20, y), (panel_w - 20, y + box_h), (32, 38, 52), -1)
    border_col = (80, 230, 145) if pred.confirmed_token else (85, 120, 195)
    cv2.rectangle(sidebar, (20, y), (panel_w - 20, y + box_h), border_col, 2)

    label_disp = pred.predicted_label.upper() if pred.predicted_label else "---"
    font_scale = 1.35 if len(label_disp) <= 6 else (1.05 if len(label_disp) <= 10 else 0.85)
    cv2.putText(sidebar, label_disp, (34, y + 52), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3, cv2.LINE_AA)
    conf_str = f"{pred.confidence * 100:5.1f}%" if pred.confidence > 0 else "--.-%"
    cv2.putText(sidebar, conf_str, (panel_w - 118, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (130, 235, 255), 2, cv2.LINE_AA)
    cv2.putText(sidebar, pred.status[:38], (34, y + 83), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (195, 210, 235), 1, cv2.LINE_AA)
    y += box_h + 26

    # Top-3 Confidence Bars
    cv2.putText(sidebar, "TOP-3 CANDIDATES", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (145, 160, 185), 1, cv2.LINE_AA)
    y += 18
    for idx in range(3):
        if idx < len(pred.topk):
            lbl, prob = pred.topk[idx]
            bar_col = (80, 220, 145) if idx == 0 else (95, 165, 240)
            cv2.putText(sidebar, f"{lbl.upper()[:12]:<12}", (20, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 236, 248), 1, cv2.LINE_AA)
            _draw_bar(sidebar, 145, y + 2, 165, 15, prob, bar_col)
            cv2.putText(sidebar, f"{prob * 100:4.1f}%", (320, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 210, 225), 1, cv2.LINE_AA)
        else:
            cv2.putText(sidebar, "---", (20, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 110, 130), 1, cv2.LINE_AA)
            _draw_bar(sidebar, 145, y + 2, 165, 15, 0.0, (70, 80, 95))
        y += 30

    y += 14
    # Motion / Hold indicator
    if mode == "words":
        cv2.putText(sidebar, "SIGN MOTION ENERGY", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (145, 160, 185), 1, cv2.LINE_AA)
        y += 12
        _draw_bar(sidebar, 20, y, panel_w - 40, 16, min(1.0, pred.motion_energy / 0.06), (70, 200, 255))
        y += 34
        seg_txt = f"Segmenter: {'AUTO MOTION' if auto_word_mode else 'MANUAL [G]'}"
        cv2.putText(sidebar, seg_txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (185, 205, 230), 1, cv2.LINE_AA)
    else:
        cv2.putText(sidebar, "HOLD-TO-CONFIRM PROGRESS", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (145, 160, 185), 1, cv2.LINE_AA)
        y += 12
        _draw_bar(sidebar, 20, y, panel_w - 40, 16, pred.hold_progress, (90, 230, 150))
        y += 34
        cv2.putText(sidebar, "Dynamic J & Z motion active", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (185, 205, 230), 1, cv2.LINE_AA)

    y += 32
    cv2.line(sidebar, (20, y), (panel_w - 20, y), (50, 58, 74), 1)
    y += 24
    # Toggles summary
    flags_line = (
        f"Buffer [B]: {'ON' if buffer_enabled else 'OFF'}   |   "
        f"Skeleton [S]: {'ON' if show_skeleton else 'OFF'}"
    )
    cv2.putText(sidebar, flags_line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (170, 190, 215), 1, cv2.LINE_AA)

    # Combine top row
    top_row = np.hstack([cam_resized, sidebar])
    total_w = top_row.shape[1]

    # Footer: Transcript bar + Controls legend
    footer_h = 108
    footer = np.full((footer_h, total_w, 3), (16, 19, 27), dtype=np.uint8)
    cv2.line(footer, (0, 0), (total_w, 0), (60, 70, 90), 2)

    buf_label = "TRANSCRIPT BUFFER (ON):" if buffer_enabled else "TRANSCRIPT BUFFER (OFF - press [B] to enable):"
    cv2.putText(footer, buf_label, (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (140, 195, 255) if buffer_enabled else (130, 140, 160), 1, cv2.LINE_AA)

    disp_text = transcript_text if transcript_text else "(confirmed words & letters appear here)"
    text_col = (255, 255, 255) if transcript_text else (110, 120, 140)
    cv2.putText(footer, disp_text[-78:], (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.72, text_col, 2, cv2.LINE_AA)

    controls = (
        "[M] Word/Letter  |  [B] Buffer  |  [A] Auto/Manual(Words)  |  [G] Capture Sign  |  "
        "[Space] Space  |  [Bksp] Undo  |  [C] Clear  |  [T] Speak  |  [R] Calibrate  |  [S] Skeleton  |  [Q] Quit"
    )
    cv2.putText(footer, controls, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 175, 195), 1, cv2.LINE_AA)

    return np.vstack([top_row, footer])
