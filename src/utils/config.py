"""
Configuration settings for Indian Sign Language (ISL) Recognition & Translation System.
"""
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path

# Project Roots
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model and Label Paths
# Checks models/ and root for backwards compatibility with original PBL4 repo
DEFAULT_MODEL_PATH = MODELS_DIR / "sign_language_model.h5"
LEGACY_MODEL_PATH = BASE_DIR / "sign_language_model.h5"
MODEL_PATH = str(DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else LEGACY_MODEL_PATH)

DEFAULT_LABELS_PATH = DATA_DIR / "gesture_labels.json"
LEGACY_LABELS_PATH = BASE_DIR / "gesture_labels.json"
LABELS_PATH = str(DEFAULT_LABELS_PATH if DEFAULT_LABELS_PATH.exists() else LEGACY_LABELS_PATH)

# Camera & Vision Configuration
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 30

# MediaPipe Hands Settings
MIN_DETECTION_CONFIDENCE = 0.70
MIN_TRACKING_CONFIDENCE = 0.70
MAX_NUM_HANDS = 2

# Feature Extraction Settings
FEATURE_DIM_SINGLE_HAND = 63   # 21 landmarks * 3 coords
FEATURE_DIM_DUAL_HAND = 126    # 42 landmarks * 3 coords
NORMALIZE_LANDMARKS = False    # Set False for legacy model compatibility; True for new models

# Sentence Formation & Smoothing Settings
PREDICTION_CONFIDENCE_THRESHOLD = 0.70  # Min confidence to accept a gesture
SMOOTHING_WINDOW_SIZE = 7              # Rolling window for majority voting
GESTURE_HOLD_FRAMES = 5                # Consecutive stable frames required
COOLDOWN_SECONDS = 1.2                 # Minimum cooldown between identical signs

# Text-To-Speech (TTS) Settings
TTS_BACKEND = "pyttsx3"                # Options: "pyttsx3", "speechbrain", "console"
TTS_RATE = 160                         # Speaking speed (words per minute)
TTS_VOLUME = 1.0                       # Audio volume (0.0 to 1.0)
SPEECHBRAIN_TTS_DIR = BASE_DIR / "tts_model"
SPEECHBRAIN_VOCODER_DIR = BASE_DIR / "vocoder_model"
