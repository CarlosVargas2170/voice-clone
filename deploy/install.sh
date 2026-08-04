#!/usr/bin/env bash
# Instala el servicio de voice-clone sin permisos de root, en el mismo directorio donde vive el proyecto.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_SYSTEMD_DIR="$HOME/.config/systemd/user"

if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi no encontrado. Instala el driver NVIDIA (ej. AMI 'Deep Learning Base') antes de continuar."
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg no encontrado. Instálalo sin sudo con: conda install -y -c conda-forge ffmpeg"
    exit 1
fi

python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

mkdir -p "$APP_DIR"/{models,speakers,logs}

if [ ! -f "$APP_DIR/.env" ]; then
    echo "VOICE_CLONE_API_TOKEN=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" > "$APP_DIR/.env"
    echo "Token generado en $APP_DIR/.env"
fi

mkdir -p "$USER_SYSTEMD_DIR"
sed \
    -e "s|__APP_DIR__|$APP_DIR|g" \
    "$APP_DIR/deploy/voice-clone.service.template" > "$USER_SYSTEMD_DIR/voice-clone.service"

systemctl --user daemon-reload

echo "Instalación lista. Corre:"
echo "  systemctl --user enable --now voice-clone"
echo ""
echo "Para que el servicio siga corriendo después de cerrar la sesión SSH, alguien con sudo debe correr una vez:"
echo "  sudo loginctl enable-linger $(whoami)"
