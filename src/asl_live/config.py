"""Shared configuration, vocabularies, MediaPipe asset URLs, and inference defaults."""

from pathlib import Path

# 15 curated, visually distinct everyday ASL words (alphabetical order)
WORDS_15 = [
    "eat",
    "family",
    "finish",
    "friend",
    "hello",
    "help",
    "love",
    "more",
    "no",
    "please",
    "stop",
    "thank you",
    "want",
    "water",
    "yes",
]

# 26 ASL fingerspelling letters (A-Z)
LETTERS_26 = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

# Feature dimensions (must stay identical between Colab notebooks and local runtime)
POSE_INDICES = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24]
WORD_SEQ_LEN = 48
WORD_STATIC_DIM = 346
WORD_FEATURE_DIM = 688  # 346 static + 171 velocity + 171 acceleration

LETTER_CROP_SIZE = 224
LETTER_GEO_DIM = 98  # 63 coords + 15 cosines + 5 tip-wrist + 10 tip-tip + 5 tip-mcp
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Finger joint triples for angle cosines (15 angles per hand)
FINGER_ANGLE_TRIPLES = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),        # Thumb
    (0, 5, 6), (5, 6, 7), (6, 7, 8),        # Index
    (0, 9, 10), (9, 10, 11), (10, 11, 12),  # Middle
    (0, 13, 14), (13, 14, 15), (14, 15, 16),# Ring
    (0, 17, 18), (17, 18, 19), (18, 19, 20),# Pinky
]

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
FINGER_MCP_REF = [(4, 5), (8, 5), (12, 9), (16, 13), (20, 17)]

# Official MediaPipe Tasks vision bundle URLs
MEDIAPIPE_MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}

# Default directory paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MEDIAPIPE_DIR = DEFAULT_MODELS_DIR / "mediapipe"
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "calibration_data"

# Live inference stabilization & UI thresholds
WORD_CONFIDENCE_THRESHOLD = 0.68
WORD_MARGIN_THRESHOLD = 0.18
WORD_MOTION_ENERGY_THRESHOLD = 0.014
WORD_MIN_ACTIVE_FRAMES = 14
WORD_COOLDOWN_SEC = 1.25

LETTER_CONFIDENCE_THRESHOLD = 0.72
LETTER_HOLD_CONFIRM_SEC = 0.65
LETTER_EMA_ALPHA = 0.45
JZ_TRAJECTORY_WINDOW = 18
