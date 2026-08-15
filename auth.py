"""Verification of the fixed token used by the single authorized client application."""
import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import API_TOKEN

bearer_scheme = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    """Compares the received token against the server's configured token."""
    if not API_TOKEN:
        raise HTTPException(status_code=500, detail="VOICE_CLONE_API_TOKEN not configured on the server.")

    if not secrets.compare_digest(credentials.credentials, API_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid token.")
