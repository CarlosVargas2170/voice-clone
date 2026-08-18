"""Servicio HTTP de clonación de voz: entrenamiento, síntesis completa y síntesis en streaming."""
import io
import logging
import secrets

import soundfile as sf
from fastapi import Depends, FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from auth import verify_token
from config import (
    API_TOKEN,
    LOG_DIR,
    MAX_SYNTHESIZE_TEXT_LENGTH,
    MAX_TRAIN_AUDIO_BYTES,
    MAX_TRAIN_AUDIO_SECONDS,
    SPEAKERS_DIR,
)
from gpu_queue import GpuOutOfMemory, GpuQueueFull, gpu_worker
from voice_engine import TextStreamBuffer, VoiceEngine

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


@app.exception_handler(GpuQueueFull)
def handle_gpu_queue_full(request, error):
    """Devuelve 503 cuando la cola de GPU está saturada, en vez de tumbar la request."""
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.exception_handler(GpuOutOfMemory)
def handle_gpu_out_of_memory(request, error):
    """Devuelve 503 cuando una síntesis agota la memoria de la GPU, en vez de un 500 genérico."""
    return JSONResponse(status_code=503, content={"detail": str(error)})


class SynthesizeRequest(BaseModel):
    text: str = Field(max_length=MAX_SYNTHESIZE_TEXT_LENGTH)
    speaker_id: str = DEFAULT_SPEAKER
    language: str = DEFAULT_LANGUAGE


@app.post("/train", dependencies=[protected])
async def train(file: UploadFile, speaker_id: str = DEFAULT_SPEAKER):
    """Sube un audio de referencia y entrena los latentes de una voz."""
    audio_bytes = await file.read()
    _validate_train_audio(audio_bytes)

    speaker_wav = SPEAKERS_DIR / f"{speaker_id}_source.wav"
    speaker_wav.write_bytes(audio_bytes)

    engine = VoiceEngine()
    await gpu_worker.run(engine.train, speaker_id, str(speaker_wav))
    logger.info("Voz entrenada: %s", speaker_id)
    return {"speaker_id": speaker_id, "status": "trained"}


@app.delete("/voices/{speaker_id}", dependencies=[protected])
def delete_voice(speaker_id: str):
    """Elimina una voz entrenada, tanto de memoria como de disco."""
    engine = VoiceEngine()
    try:
        engine.delete_speaker(speaker_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error))

    source_wav = SPEAKERS_DIR / f"{speaker_id}_source.wav"
    source_wav.unlink(missing_ok=True)
    logger.info("Voz eliminada: %s", speaker_id)
    return {"speaker_id": speaker_id, "status": "deleted"}


@app.get("/voices", dependencies=[protected])
def list_voices():
    """Lista las voces actualmente entrenadas."""
    return {"speakers": VoiceEngine().list_speakers()}


@app.post("/synthesize", dependencies=[protected])
async def synthesize(request: SynthesizeRequest):
    """Devuelve el audio completo generado a partir de un texto."""
    engine = VoiceEngine()
    if not engine.has_speaker(request.speaker_id):
        raise HTTPException(status_code=404, detail=f"Voz '{request.speaker_id}' no entrenada.")

    audio = await gpu_worker.run(engine.synthesize, request.text, request.speaker_id, request.language)
    return Response(content=audio, media_type="audio/wav")


@app.post("/synthesize/stream", dependencies=[protected])
def synthesize_stream(request: SynthesizeRequest):
    """Devuelve el audio en chunks a medida que se va generando."""
    engine = VoiceEngine()
    if not engine.has_speaker(request.speaker_id):
        raise HTTPException(status_code=404, detail=f"Voz '{request.speaker_id}' no entrenada.")

    chunks = gpu_worker.open_stream(engine.synthesize_stream, request.text, request.speaker_id, request.language)
    return StreamingResponse(chunks, media_type="application/octet-stream")


@app.websocket("/synthesize/stream-text")
async def synthesize_stream_text(websocket: WebSocket):
    """Recibe texto en fragmentos y devuelve audio en fragmentos a medida que hay oraciones completas."""
    query = websocket.query_params
    if not _websocket_token_is_valid(query.get("token")):
        await websocket.close(code=4401, reason="Token inválido o ausente.")
        return

    await websocket.accept()
    engine = VoiceEngine()
    text_buffer = TextStreamBuffer()
    total_received = 0

    speaker_id = query.get("speaker_id", DEFAULT_SPEAKER)
    language = query.get("language", DEFAULT_LANGUAGE)
    logger.info("WS conectado speaker_id=%s language=%s", speaker_id, language)

    if not engine.has_speaker(speaker_id):
        logger.warning("WS rechazado: voz '%s' no entrenada", speaker_id)
        await websocket.close(code=4404, reason=f"Voz '{speaker_id}' no entrenada.")
        return

    try:
        while True:
            text_chunk = await websocket.receive_text()
            logger.info("WS recibido (%d chars): %r", len(text_chunk), text_chunk)
            total_received += len(text_chunk)
            if total_received > MAX_SYNTHESIZE_TEXT_LENGTH:
                logger.warning("WS texto total excede el límite (%d)", total_received)
                await websocket.close(code=1009, reason="Texto total excede el límite permitido.")
                return

            if text_chunk == "<END>":
                remainder = text_buffer.flush()
                if remainder:
                    logger.info("WS sintetizando oración final: %r", remainder)
                    await _stream_sentence(websocket, engine, remainder, speaker_id, language)
                break

            for sentence in text_buffer.feed(text_chunk):
                logger.info("WS sintetizando oración: %r", sentence)
                await _stream_sentence(websocket, engine, sentence, speaker_id, language)
    except WebSocketDisconnect:
        logger.info("WS desconectado")
        return
    except (GpuQueueFull, GpuOutOfMemory) as error:
        logger.warning("WS error de GPU: %s", error)
        await websocket.close(code=1013, reason=str(error))
        return

    logger.info("WS turno completo, cerrando")
    await websocket.close()


async def _stream_sentence(websocket: WebSocket, engine: VoiceEngine, sentence: str, speaker_id: str, language: str):
    """Sintetiza una oración y envía sus chunks de audio por el websocket."""
    chunks = gpu_worker.open_stream(engine.synthesize_stream, sentence, speaker_id, language)
    async for chunk in chunks:
        await websocket.send_bytes(chunk)


def _websocket_token_is_valid(token: str | None) -> bool:
    """Valida el token recibido por query param, único mecanismo disponible en un handshake de websocket."""
    if not token or not API_TOKEN:
        return False
    return secrets.compare_digest(token, API_TOKEN)


def _validate_train_audio(audio_bytes: bytes):
    """Rechaza audios de entrenamiento que excedan el tamaño o duración permitidos."""
    if len(audio_bytes) > MAX_TRAIN_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail=f"El audio supera el máximo de {MAX_TRAIN_AUDIO_BYTES} bytes.")

    try:
        info = sf.info(io.BytesIO(audio_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Archivo de audio inválido, se espera un WAV.")

    if info.duration > MAX_TRAIN_AUDIO_SECONDS:
        raise HTTPException(
            status_code=413, detail=f"El audio supera el máximo de {MAX_TRAIN_AUDIO_SECONDS} segundos."
        )


@app.get("/health")
def health():
    """Reporta si el modelo está cargado y listo para atender solicitudes."""
    engine = VoiceEngine()
    return {
        "status": "ok",
        "gpu": engine.model.device.type == "cuda",
        "speakers_loaded": len(engine.list_speakers()),
    }


@app.on_event("startup")
def load_model():
    VoiceEngine()
