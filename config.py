"""Centralized configuration for the service's paths and limits.

Everything lives relative to the project folder, regardless of where it's deployed.
Environment variables are read from the .env file next to this one.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL_DIR = Path(os.getenv("VOICE_CLONE_MODEL_DIR", str(BASE_DIR / "models")))
SPEAKERS_DIR = Path(os.getenv("VOICE_CLONE_SPEAKERS_DIR", str(BASE_DIR / "speakers")))
LOG_DIR = Path(os.getenv("VOICE_CLONE_LOG_DIR", str(BASE_DIR / "logs")))

MAX_TRAIN_AUDIO_SECONDS = 45
MAX_TRAIN_AUDIO_BYTES = 5 * 1024 * 1024

API_TOKEN = os.getenv("VOICE_CLONE_API_TOKEN", "")

MAX_GPU_QUEUE_SIZE = int(os.getenv("VOICE_CLONE_MAX_QUEUE", "10"))
GPU_MEMORY_FRACTION = float(os.getenv("VOICE_CLONE_GPU_MEMORY_FRACTION", "0.9"))
MAX_SYNTHESIZE_TEXT_LENGTH = int(os.getenv("VOICE_CLONE_MAX_TEXT_LENGTH", "2000"))

AUDIO_OUTPUT_DEVICE = os.getenv("VOICE_CLONE_AUDIO_DEVICE", "default")

CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("VOICE_CLONE_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
