#!/usr/bin/env bash
# Instala el servicio de voice-clone en una Jetson (JetPack 6.2 / L4T R36.5), sin conda.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_SYSTEMD_DIR="$HOME/.config/systemd/user"
CUDA_VERSION="12.6"
CUDA_LIB_PATH="/usr/local/cuda-$CUDA_VERSION/lib64"
TORCH_INDEX_URL="https://pypi.jetson-ai-lab.io/jp6/cu126"

if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi no encontrado. Este script asume una Jetson con JetPack ya instalado."
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "== Instalando ffmpeg vía apt =="
    sudo apt-get install -y ffmpeg
fi

echo "== Instalando librerías runtime de CUDA $CUDA_VERSION y cuDNN =="
sudo apt-get install -y \
    cuda-cudart-$CUDA_VERSION cuda-nvrtc-$CUDA_VERSION cuda-libraries-$CUDA_VERSION cuda-cupti-$CUDA_VERSION \
    libcublas-$CUDA_VERSION libcufft-$CUDA_VERSION libcurand-$CUDA_VERSION libcusolver-$CUDA_VERSION \
    libcusparse-$CUDA_VERSION libnvjitlink-$CUDA_VERSION libnvfatbin-$CUDA_VERSION libcufile-$CUDA_VERSION \
    libcudnn9-cuda-12

if [ ! -d "$APP_DIR/.venv" ]; then
    echo "== Creando entorno virtual (Python $(python3 --version 2>&1 | cut -d' ' -f2)) =="
    python3 -m venv "$APP_DIR/.venv"
fi

# El LD_LIBRARY_PATH de CUDA se necesita tanto en el venv (uso manual) como en el
# servicio systemd (Environment= en voice-clone.service.template).
if ! grep -q "LD_LIBRARY_PATH" "$APP_DIR/.venv/bin/activate"; then
    echo "export LD_LIBRARY_PATH=$CUDA_LIB_PATH:\$LD_LIBRARY_PATH" >> "$APP_DIR/.venv/bin/activate"
fi

source "$APP_DIR/.venv/bin/activate"
export LD_LIBRARY_PATH="$CUDA_LIB_PATH:${LD_LIBRARY_PATH:-}"

pip install --upgrade pip

# PyTorch para Jetson no está en PyPI estándar (los wheels de download.pytorch.org son x86_64).
# Se instala aparte, desde el índice de la comunidad Jetson AI Lab, antes de requirements.txt
# para que TTS no lo pise con una build sin CUDA.
if ! python3 -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
    echo "== Instalando PyTorch con soporte CUDA para Jetson (JetPack 6.2) =="
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url "$TORCH_INDEX_URL"
fi

echo "== Instalando dependencias del proyecto =="
pip install -r "$APP_DIR/requirements.txt"

# TTS==0.22.0 puede pisar el torch con GPU al resolver sus propias dependencias; se
# reinstala sin deps para recuperar la build de Jetson, y se fija numpy<2 (compatible
# con torch 2.8; TTS pide numpy==1.22.0 pero es solo un pin de instalación, no algo que
# verifique en tiempo de ejecución).
pip install --no-deps torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install "numpy<2"

# torch>=2.6 cambió el default de weights_only en torch.load a True, lo que rompe la
# carga del checkpoint de XTTS (trae objetos de configuración, no solo tensores).
# TTS==0.22.0 no lo pasa explícito, así que se parchea para que use weights_only=False.
TTS_IO_FILE="$APP_DIR/.venv/lib/python3.10/site-packages/TTS/utils/io.py"
if ! grep -q "weights_only" "$TTS_IO_FILE"; then
    sed -i '/is_local = os.path.isdir(path) or os.path.isfile(path)/i\    kwargs.setdefault("weights_only", False)' "$TTS_IO_FILE"
fi

mkdir -p "$APP_DIR"/{models,speakers,logs}

if [ ! -f "$APP_DIR/.env" ]; then
    echo "VOICE_CLONE_API_TOKEN=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" > "$APP_DIR/.env"
    echo "Token generado en $APP_DIR/.env"
fi

if ! grep -q "VOICE_CLONE_AUDIO_DEVICE" "$APP_DIR/.env"; then
    echo "VOICE_CLONE_AUDIO_DEVICE=plughw:2,0" >> "$APP_DIR/.env"
    echo "Dispositivo de audio por defecto: plughw:2,0 (USB Audio Device). Ajusta en .env si 'aplay -l' muestra otra tarjeta."
fi

mkdir -p "$USER_SYSTEMD_DIR"
sed \
    -e "s|__APP_DIR__|$APP_DIR|g" \
    "$APP_DIR/deploy/voice-clone.service.template" > "$USER_SYSTEMD_DIR/voice-clone.service"
cp "$APP_DIR/deploy/voice-clone.timer" "$USER_SYSTEMD_DIR/voice-clone.timer"

systemctl --user daemon-reload

echo "Instalación lista. Corre:"
echo "  systemctl --user enable --now voice-clone.timer"
echo ""
echo "Esto arrancará voice-clone.service automáticamente 1 minuto después de cada boot."
echo "(No uses 'enable --now voice-clone' directamente, o arrancará sin el retraso)."
echo ""
echo "Para que el servicio siga corriendo después de cerrar la sesión SSH, alguien con sudo debe correr una vez:"
echo "  sudo loginctl enable-linger $(whoami)"
