#!/usr/bin/env bash
# Instala el servicio de voice-clone de forma nativa, en el mismo directorio donde vive el proyecto.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_FILE="/etc/systemd/system/voice-clone.service"

if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi no encontrado. Instala el driver NVIDIA (ej. AMI 'Deep Learning Base') antes de continuar."
    exit 1
fi

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip ffmpeg

python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

mkdir -p "$APP_DIR"/{models,speakers,logs}

if [ ! -f "$APP_DIR/.env" ]; then
    echo "VOICE_CLONE_API_TOKEN=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" > "$APP_DIR/.env"
    echo "Token generado en $APP_DIR/.env"
fi

sed \
    -e "s|__APP_DIR__|$APP_DIR|g" \
    -e "s|__USER__|$(whoami)|g" \
    "$APP_DIR/deploy/voice-clone.service.template" | sudo tee "$SERVICE_FILE" > /dev/null

sudo systemctl daemon-reload

echo "Instalación lista. Corre:"
echo "  sudo systemctl enable --now voice-clone"
