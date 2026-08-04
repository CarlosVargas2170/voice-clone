"""Motor singleton de clonación de voz sobre XTTS v2."""
import io
import os
import re
import threading

import numpy as np
import soundfile as sf
import torch
from TTS.api import TTS

from config import GPU_MEMORY_FRACTION, MODEL_DIR, SPEAKERS_DIR

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SAMPLE_RATE = 24000
SENTENCE_END = re.compile(r".*?[.!?]\s")


class VoiceEngine:
    """Único punto de acceso al modelo. Se carga una vez y se reutiliza en todo el servicio."""

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
        os.environ["TTS_HOME"] = str(MODEL_DIR)

        gpu = torch.cuda.is_available()
        if gpu:
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)

        api = TTS(MODEL_NAME, gpu=gpu)
        self.model = api.synthesizer.tts_model
        self.speakers = {}
        self.speakers_lock = threading.Lock()
        self._load_saved_speakers()

    def _speaker_path(self, speaker_id: str):
        return SPEAKERS_DIR / f"{speaker_id}.pt"

    def _load_saved_speakers(self):
        """Carga en memoria los latentes de voces ya entrenadas en corridas anteriores."""
        for path in SPEAKERS_DIR.glob("*.pt"):
            gpt_latent, speaker_embedding = torch.load(path, map_location="cpu")
            self.speakers[path.stem] = (gpt_latent, speaker_embedding)

    def train(self, speaker_id: str, speaker_wav_path: str):
        """Calcula los latentes de una voz a partir de un audio de referencia y los persiste en disco."""
        gpt_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=speaker_wav_path)
        with self.speakers_lock:
            self.speakers[speaker_id] = (gpt_latent, speaker_embedding)
            torch.save((gpt_latent.cpu(), speaker_embedding.cpu()), self._speaker_path(speaker_id))

    def delete_speaker(self, speaker_id: str):
        """Elimina una voz entrenada de memoria y de disco."""
        if not self.has_speaker(speaker_id):
            raise KeyError(f"Voz '{speaker_id}' no existe.")
        with self.speakers_lock:
            del self.speakers[speaker_id]
            self._speaker_path(speaker_id).unlink(missing_ok=True)

    def list_speakers(self) -> list[str]:
        return sorted(self.speakers.keys())

    def has_speaker(self, speaker_id: str) -> bool:
        return speaker_id in self.speakers

    def _latents(self, speaker_id: str):
        if not self.has_speaker(speaker_id):
            raise KeyError(f"Voz '{speaker_id}' no entrenada. Llama antes a /train.")
        return self.speakers[speaker_id]

    def synthesize(self, text: str, speaker_id: str, language: str = "es") -> bytes:
        """Genera el audio completo de un texto y lo devuelve como WAV."""
        gpt_latent, speaker_embedding = self._latents(speaker_id)
        output = self.model.inference(
            text, language, gpt_latent, speaker_embedding, enable_text_splitting=True
        )
        return wav_bytes(output["wav"])

    def synthesize_stream(self, text: str, speaker_id: str, language: str = "es"):
        """Genera audio en chunks a medida que el modelo los produce."""
        gpt_latent, speaker_embedding = self._latents(speaker_id)
        for chunk in self.model.inference_stream(
            text, language, gpt_latent, speaker_embedding, enable_text_splitting=True
        ):
            yield chunk.detach().cpu().numpy().astype(np.float32).tobytes()


class TextStreamBuffer:
    """Acumula texto que llega en fragmentos y libera oraciones completas para sintetizar."""

    def __init__(self):
        self.buffer = ""

    def feed(self, chunk: str) -> list[str]:
        """Agrega texto al buffer y devuelve las oraciones completas listas para sintetizar."""
        self.buffer += chunk
        sentences = []
        while True:
            match = SENTENCE_END.match(self.buffer)
            if not match:
                break
            sentences.append(match.group().strip())
            self.buffer = self.buffer[match.end():]
        return sentences

    def flush(self) -> str | None:
        """Devuelve el texto restante en el buffer, si queda alguno."""
        remainder = self.buffer.strip()
        self.buffer = ""
        return remainder or None


def wav_bytes(wav: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, wav, SAMPLE_RATE, format="WAV")
    return buffer.getvalue()
