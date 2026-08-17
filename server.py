"""HTTP voice-clone service: training, full synthesis, and streaming synthesis."""
import asyncio
import io
import logging
import secrets
import subprocess

import soundfile as sf
from fastapi import Depends, FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from auth import verify_token
from config import (
    API_TOKEN,
    AUDIO_OUTPUT_DEVICE,
    CORS_ALLOWED_ORIGINS,
    LOG_DIR,
    MAX_SYNTHESIZE_TEXT_LENGTH,
    MAX_TRAIN_AUDIO_BYTES,
    MAX_TRAIN_AUDIO_SECONDS,
    SPEAKERS_DIR,
)
from gpu_queue import GpuOutOfMemory, GpuQueueFull, gpu_worker
from voice_engine import SAMPLE_RATE, TextStreamBuffer, VoiceEngine

DEFAULT_SPEAKER = "default"
DEFAULT_LANGUAGE = "es"

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "service.log"), logging.StreamHandler()],
)
logger = logging.getLogger("voice-clone")

app = FastAPI(title="Voice Clone Service")
protected = Depends(verify_token)

if CORS_ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.middleware("http")
async def allow_private_network(request, call_next):
    """Answers Chrome's Private Network Access preflight so public HTTPS pages can reach this LAN/Tailscale IP."""
    response = await call_next(request)
    if request.headers.get("access-control-request-private-network") == "true":
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


@app.exception_handler(GpuQueueFull)
def handle_gpu_queue_full(request, error):
    """Returns 503 when the GPU queue is full, instead of crashing the request."""
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.exception_handler(GpuOutOfMemory)
def handle_gpu_out_of_memory(request, error):
    """Returns 503 when a synthesis runs out of GPU memory, instead of a generic 500."""
    return JSONResponse(status_code=503, content={"detail": str(error)})


class SynthesizeRequest(BaseModel):
    text: str = Field(max_length=MAX_SYNTHESIZE_TEXT_LENGTH)
    speaker_id: str = DEFAULT_SPEAKER
    language: str = DEFAULT_LANGUAGE


@app.post("/train", dependencies=[protected])
async def train(file: UploadFile, speaker_id: str = DEFAULT_SPEAKER):
    """Uploads a reference audio and trains the latents for a voice."""
    audio_bytes = await file.read()
    _validate_train_audio(audio_bytes)

    speaker_wav = SPEAKERS_DIR / f"{speaker_id}_source.wav"
    speaker_wav.write_bytes(audio_bytes)

    engine = VoiceEngine()
    await gpu_worker.run(engine.train, speaker_id, str(speaker_wav))
    logger.info("Voice trained: %s", speaker_id)
    return {"speaker_id": speaker_id, "status": "trained"}


@app.delete("/voices/{speaker_id}", dependencies=[protected])
def delete_voice(speaker_id: str):
    """Deletes a trained voice, both from memory and disk."""
    engine = VoiceEngine()
    try:
        engine.delete_speaker(speaker_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error))

    source_wav = SPEAKERS_DIR / f"{speaker_id}_source.wav"
    source_wav.unlink(missing_ok=True)
    logger.info("Voice deleted: %s", speaker_id)
    return {"speaker_id": speaker_id, "status": "deleted"}


@app.get("/voices", dependencies=[protected])
def list_voices():
    """Lists the currently trained voices."""
    return {"speakers": VoiceEngine().list_speakers()}


@app.post("/synthesize", dependencies=[protected])
async def synthesize(request: SynthesizeRequest):
    """Returns the full audio generated from a text."""
    engine = VoiceEngine()
    if not engine.has_speaker(request.speaker_id):
        raise HTTPException(status_code=404, detail=f"Voice '{request.speaker_id}' not trained.")

    audio = await gpu_worker.run(engine.synthesize, request.text, request.speaker_id, request.language)
    return Response(content=audio, media_type="audio/wav")


PLAYBACK_CHUNK_BYTES = 24_000 * 4 // 5  # ~200ms of float32 mono audio at SAMPLE_RATE


@app.post("/synthesize/play", dependencies=[protected])
async def synthesize_play(request: SynthesizeRequest):
    """Synthesizes the full audio first (the Jetson GPU can't keep up with real-time
    generation, so streaming chunk-by-chunk produces choppy playback), then plays it
    on the Jetson's speakers and streams it back to the client at the same time.
    """
    engine = VoiceEngine()
    if not engine.has_speaker(request.speaker_id):
        raise HTTPException(status_code=404, detail=f"Voice '{request.speaker_id}' not trained.")

    audio = await gpu_worker.run(engine.synthesize_raw, request.text, request.speaker_id, request.language)

    try:
        process = await asyncio.create_subprocess_exec(
            "aplay", "-q", "-D", AUDIO_OUTPUT_DEVICE,
            "-f", "FLOAT_LE", "-c", "1", "-r", str(SAMPLE_RATE), "-t", "raw", "-",
            stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="aplay is not installed on the system.")

    return StreamingResponse(_play_and_forward(process, audio), media_type="application/octet-stream")


async def _play_and_forward(process: asyncio.subprocess.Process, audio: bytes):
    """Sends the already-generated audio to aplay and to the client in lockstep chunks."""
    try:
        for offset in range(0, len(audio), PLAYBACK_CHUNK_BYTES):
            chunk = audio[offset:offset + PLAYBACK_CHUNK_BYTES]
            process.stdin.write(chunk)
            await process.stdin.drain()
            yield chunk
    finally:
        process.stdin.close()
        stderr = await process.stderr.read()
        await process.wait()
        if process.returncode != 0:
            logger.warning(
                "aplay failed (device=%s, code=%d): %s",
                AUDIO_OUTPUT_DEVICE, process.returncode, stderr.decode(errors="replace").strip(),
            )


@app.post("/synthesize/stream", dependencies=[protected])
def synthesize_stream(request: SynthesizeRequest):
    """Returns the audio in chunks as it gets generated."""
    engine = VoiceEngine()
    if not engine.has_speaker(request.speaker_id):
        raise HTTPException(status_code=404, detail=f"Voice '{request.speaker_id}' not trained.")

    chunks = gpu_worker.open_stream(engine.synthesize_stream, request.text, request.speaker_id, request.language)
    return StreamingResponse(chunks, media_type="application/octet-stream")


@app.websocket("/synthesize/stream-text")
async def synthesize_stream_text(websocket: WebSocket):
    """Receives text in fragments and returns audio in chunks as complete sentences become available."""
    query = websocket.query_params
    if not _websocket_token_is_valid(query.get("token")):
        await websocket.close(code=4401, reason="Invalid or missing token.")
        return

    await websocket.accept()
    engine = VoiceEngine()
    text_buffer = TextStreamBuffer()
    total_received = 0

    speaker_id = query.get("speaker_id", DEFAULT_SPEAKER)
    language = query.get("language", DEFAULT_LANGUAGE)

    if not engine.has_speaker(speaker_id):
        await websocket.close(code=4404, reason=f"Voice '{speaker_id}' not trained.")
        return

    try:
        while True:
            text_chunk = await websocket.receive_text()
            total_received += len(text_chunk)
            if total_received > MAX_SYNTHESIZE_TEXT_LENGTH:
                await websocket.close(code=1009, reason="Total text exceeds the allowed limit.")
                return

            if text_chunk == "<END>":
                remainder = text_buffer.flush()
                if remainder:
                    await _stream_sentence(websocket, engine, remainder, speaker_id, language)
                break

            for sentence in text_buffer.feed(text_chunk):
                await _stream_sentence(websocket, engine, sentence, speaker_id, language)
    except WebSocketDisconnect:
        return
    except (GpuQueueFull, GpuOutOfMemory) as error:
        await websocket.close(code=1013, reason=str(error))
        return

    await websocket.close()


async def _stream_sentence(websocket: WebSocket, engine: VoiceEngine, sentence: str, speaker_id: str, language: str):
    """Synthesizes one sentence and sends its audio chunks over the websocket."""
    chunks = gpu_worker.open_stream(engine.synthesize_stream, sentence, speaker_id, language)
    async for chunk in chunks:
        await websocket.send_bytes(chunk)


def _websocket_token_is_valid(token: str | None) -> bool:
    """Validates the token received as a query param, the only mechanism available in a websocket handshake."""
    if not token or not API_TOKEN:
        return False
    return secrets.compare_digest(token, API_TOKEN)


def _validate_train_audio(audio_bytes: bytes):
    """Rejects training audio that exceeds the allowed size or duration."""
    if len(audio_bytes) > MAX_TRAIN_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail=f"Audio exceeds the maximum of {MAX_TRAIN_AUDIO_BYTES} bytes.")

    try:
        info = sf.info(io.BytesIO(audio_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio file, expected a WAV.")

    if info.duration > MAX_TRAIN_AUDIO_SECONDS:
        raise HTTPException(
            status_code=413, detail=f"Audio exceeds the maximum of {MAX_TRAIN_AUDIO_SECONDS} seconds."
        )


@app.get("/health")
def health():
    """Reports whether the model is loaded and ready to serve requests."""
    engine = VoiceEngine()
    return {
        "status": "ok",
        "gpu": engine.model.device.type == "cuda",
        "speakers_loaded": len(engine.list_speakers()),
    }


@app.on_event("startup")
def load_model():
    VoiceEngine()
