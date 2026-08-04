"""Configuración centralizada de rutas y límites del servicio.

Todo vive relativo a la carpeta del proyecto, sin importar dónde se despliegue.
Las variables de entorno se leen desde el .env ubicado junto a este archivo.
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
