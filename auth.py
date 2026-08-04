"""Verificación del token fijo que usa la única aplicación cliente autorizada."""
import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import API_TOKEN

bearer_scheme = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """Compara el token recibido contra el token hardcodeado del servidor."""
    if not API_TOKEN:
        raise HTTPException(status_code=500, detail="VOICE_CLONE_API_TOKEN no configurado en el servidor.")

    if not secrets.compare_digest(credentials.credentials, API_TOKEN):
        raise HTTPException(status_code=401, detail="Token inválido.")
