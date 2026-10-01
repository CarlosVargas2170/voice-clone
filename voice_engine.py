"""Singleton voice-clone engine built on XTTS v2."""
import io
import logging
import os
import re
import threading
import time

import numpy as np
import soundfile as sf
import torch
from TTS.api import TTS

from config import GPU_MEMORY_FRACTION, MODEL_DIR, SPEAKERS_DIR

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SAMPLE_RATE = 24000
SENTENCE_END = re.compile(r".*?[.!?]\s")

logger = logging.getLogger("voice-clone")


class VoiceEngine:
    """Single access point to the model. Loaded once and reused across the whole service."""

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
        """Loads into memory the latents of voices trained in previous runs."""
        for path in SPEAKERS_DIR.glob("*.pt"):
            gpt_latent, speaker_embedding = torch.load(path, map_location="cpu")
            self.speakers[path.stem] = (gpt_latent, speaker_embedding)

    def train(self, speaker_id: str, speaker_wav_path: str):
        """Computes the latents of a voice from a reference audio and persists them to disk."""
        gpt_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=speaker_wav_path)
        with self.speakers_lock:
            self.speakers[speaker_id] = (gpt_latent, speaker_embedding)
            torch.save((gpt_latent.cpu(), speaker_embedding.cpu()), self._speaker_path(speaker_id))

    def delete_speaker(self, speaker_id: str):
        """Deletes a trained voice from memory and disk."""
        if not self.has_speaker(speaker_id):
            raise KeyError(f"Voice '{speaker_id}' does not exist.")
        with self.speakers_lock:
            del self.speakers[speaker_id]
            self._speaker_path(speaker_id).unlink(missing_ok=True)

    def list_speakers(self) -> list[str]:
        return sorted(self.speakers.keys())

    def has_speaker(self, speaker_id: str) -> bool:
        return speaker_id in self.speakers

    def _latents(self, speaker_id: str):
        if not self.has_speaker(speaker_id):
            raise KeyError(f"Voice '{speaker_id}' not trained. Call /train first.")
        return self.speakers[speaker_id]

    def synthesize(self, text: str, speaker_id: str, language: str = "es") -> bytes:
        """Generates the full audio for a text and returns it as WAV."""
        return wav_bytes(self._synthesize_array(text, speaker_id, language))

    def synthesize_raw(self, text: str, speaker_id: str, language: str = "es") -> bytes:
        """Generates the full audio for a text and returns it as raw float32 PCM (no header)."""
        audio = self._synthesize_array(text, speaker_id, language)
        return audio.astype(np.float32).tobytes()

    def _synthesize_array(self, text: str, speaker_id: str, language: str) -> np.ndarray:
        gpt_latent, speaker_embedding = self._latents(speaker_id)
        output = self.model.inference(
            text, language, gpt_latent, speaker_embedding, enable_text_splitting=True
        )
        return output["wav"]

    def synthesize_stream(self, text: str, speaker_id: str, language: str = "es"):
        """Generates audio in chunks as the model produces them.

        Logs generation time vs. audio duration per chunk (temporary diagnostic for
        choppy playback) — a ratio above 1.0 means the GPU can't keep up with real time.
        """
        gpt_latent, speaker_embedding = self._latents(speaker_id)
        last_yield = time.monotonic()
        for chunk in self.model.inference_stream(
            text, language, gpt_latent, speaker_embedding, enable_text_splitting=True
        ):
            audio = chunk.detach().cpu().numpy().astype(np.float32)
            now = time.monotonic()
            generation_time = now - last_yield
            audio_duration = len(audio) / SAMPLE_RATE
            logger.info(
                "chunk: %d samples, %.3fs audio, %.3fs to generate, realtime ratio %.2f",
                len(audio), audio_duration, generation_time, generation_time / audio_duration,
            )
            last_yield = now
            yield audio.tobytes()


class TextStreamBuffer:
    """Accumulates text arriving in fragments and releases complete sentences ready for synthesis."""

    def __init__(self):
        self.buffer = ""

    def feed(self, chunk: str) -> list[str]:
        """Adds text to the buffer and returns the complete sentences ready for synthesis."""
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
        """Returns the remaining text in the buffer, if any is left."""
        remainder = self.buffer.strip()
        self.buffer = ""
        return remainder or None


def wav_bytes(wav: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, wav, SAMPLE_RATE, format="WAV")
    return buffer.getvalue()
