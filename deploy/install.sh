#!/usr/bin/env bash
# Installs the voice-clone service on a Jetson (JetPack 6.2 / L4T R36.5), no conda.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_SYSTEMD_DIR="$HOME/.config/systemd/user"
CUDA_VERSION="12.6"
CUDA_LIB_PATH="/usr/local/cuda-$CUDA_VERSION/lib64"
TORCH_INDEX_URL="https://pypi.jetson-ai-lab.io/jp6/cu126"

if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found. This script assumes a Jetson with JetPack already installed."
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "== Installing ffmpeg via apt =="
    sudo apt-get install -y ffmpeg
fi

echo "== Installing CUDA $CUDA_VERSION and cuDNN runtime libraries =="
sudo apt-get install -y \
    cuda-cudart-$CUDA_VERSION cuda-nvrtc-$CUDA_VERSION cuda-libraries-$CUDA_VERSION cuda-cupti-$CUDA_VERSION \
    libcublas-$CUDA_VERSION libcufft-$CUDA_VERSION libcurand-$CUDA_VERSION libcusolver-$CUDA_VERSION \
    libcusparse-$CUDA_VERSION libnvjitlink-$CUDA_VERSION libnvfatbin-$CUDA_VERSION libcufile-$CUDA_VERSION \
    libcudnn9-cuda-12

if [ ! -d "$APP_DIR/.venv" ]; then
    echo "== Creating virtual environment (Python $(python3 --version 2>&1 | cut -d' ' -f2)) =="
    python3 -m venv "$APP_DIR/.venv"
fi

# The CUDA LD_LIBRARY_PATH is needed both in the venv (manual use) and in the
# systemd service (Environment= in voice-clone.service.template).
if ! grep -q "LD_LIBRARY_PATH" "$APP_DIR/.venv/bin/activate"; then
    echo "export LD_LIBRARY_PATH=$CUDA_LIB_PATH:\$LD_LIBRARY_PATH" >> "$APP_DIR/.venv/bin/activate"
fi

source "$APP_DIR/.venv/bin/activate"
export LD_LIBRARY_PATH="$CUDA_LIB_PATH:${LD_LIBRARY_PATH:-}"

pip install --upgrade pip

# PyTorch for Jetson isn't on standard PyPI (the download.pytorch.org wheels are
# x86_64). It's installed separately, from the Jetson AI Lab community index,
# before requirements.txt so TTS doesn't overwrite it with a non-CUDA build.
if ! python3 -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
    echo "== Installing PyTorch with CUDA support for Jetson (JetPack 6.2) =="
    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url "$TORCH_INDEX_URL"
fi

echo "== Installing project dependencies =="
pip install -r "$APP_DIR/requirements.txt"

# TTS==0.22.0 may overwrite the GPU-enabled torch while resolving its own
# dependencies; it's reinstalled with --no-deps to restore the Jetson build,
# and numpy<2 is pinned (compatible with torch 2.8; TTS asks for
# numpy==1.22.0 but that's just an install-time pin, not something it checks
# at runtime).
pip install --no-deps torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install "numpy<2"

# torch>=2.6 changed the default of weights_only in torch.load to True, which
# breaks loading the XTTS checkpoint (it carries config objects, not just
# tensors). TTS==0.22.0 doesn't pass it explicitly, so it's patched to use
# weights_only=False.
TTS_IO_FILE="$APP_DIR/.venv/lib/python3.10/site-packages/TTS/utils/io.py"
if ! grep -q "weights_only" "$TTS_IO_FILE"; then
    sed -i '/is_local = os.path.isdir(path) or os.path.isfile(path)/i\    kwargs.setdefault("weights_only", False)' "$TTS_IO_FILE"
fi

mkdir -p "$APP_DIR"/{models,speakers,logs}

if [ ! -f "$APP_DIR/.env" ]; then
    echo "VOICE_CLONE_API_TOKEN=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" > "$APP_DIR/.env"
    echo "Token generated at $APP_DIR/.env"
fi

if ! grep -q "VOICE_CLONE_AUDIO_DEVICE" "$APP_DIR/.env"; then
    echo "VOICE_CLONE_AUDIO_DEVICE=plughw:2,0" >> "$APP_DIR/.env"
    echo "Default audio device: plughw:2,0 (USB Audio Device). Adjust in .env if 'aplay -l' shows a different card."
fi

mkdir -p "$USER_SYSTEMD_DIR"
sed \
    -e "s|__APP_DIR__|$APP_DIR|g" \
    "$APP_DIR/deploy/voice-clone.service.template" > "$USER_SYSTEMD_DIR/voice-clone.service"
cp "$APP_DIR/deploy/voice-clone.timer" "$USER_SYSTEMD_DIR/voice-clone.timer"

systemctl --user daemon-reload

echo "Installation ready. Run:"
echo "  systemctl --user enable --now voice-clone.timer"
echo ""
echo "This will start voice-clone.service automatically 1 minute after every boot."
echo "(Don't use 'enable --now voice-clone' directly, it would skip the delay)."
echo ""
echo "For the service to keep running after closing the SSH session, someone with sudo must run once:"
echo "  sudo loginctl enable-linger $(whoami)"
