#!/usr/bin/env bash
# Installs the voice-clone service on a Jetson (JetPack 6+ / L4T R36+), no conda.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_SYSTEMD_DIR="$HOME/.config/systemd/user"
PY_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found. This script assumes a Jetson with JetPack already installed."
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "== Installing ffmpeg via apt =="
    sudo apt-get install -y ffmpeg
fi

# CUDA apt package versions are suffixed either with a dot (older repos, e.g.
# cuda-libraries-12.6) or a dash (newer repos, e.g. cuda-libraries-13-2).
# Detect whichever the configured apt repo actually has instead of hardcoding one.
CUDA_PKG_VERSION="$(apt-cache search '^cuda-libraries-[0-9]' | grep -oP 'cuda-libraries-\K[0-9]+[.-][0-9]+' | sort -V | tail -1)"
if [ -z "$CUDA_PKG_VERSION" ]; then
    echo "No cuda-libraries-* package found via apt-cache search."
    echo "Is the NVIDIA CUDA apt repo configured on this Jetson (check /etc/apt/sources.list.d/)?"
    exit 1
fi
CUDA_VERSION_DOT="${CUDA_PKG_VERSION/-/.}"
CUDA_MAJOR="${CUDA_VERSION_DOT%%.*}"
CUDA_LIB_PATH="/usr/local/cuda-$CUDA_VERSION_DOT/lib64"

echo "== Detected CUDA $CUDA_VERSION_DOT (apt suffix: $CUDA_PKG_VERSION, Python $PY_VERSION) =="

echo "== Installing CUDA $CUDA_VERSION_DOT and cuDNN runtime libraries =="
sudo apt-get install -y \
    cuda-cudart-$CUDA_PKG_VERSION cuda-nvrtc-$CUDA_PKG_VERSION cuda-libraries-$CUDA_PKG_VERSION cuda-cupti-$CUDA_PKG_VERSION \
    libcublas-$CUDA_PKG_VERSION libcufft-$CUDA_PKG_VERSION libcurand-$CUDA_PKG_VERSION libcusolver-$CUDA_PKG_VERSION \
    libcusparse-$CUDA_PKG_VERSION libnvjitlink-$CUDA_PKG_VERSION libnvfatbin-$CUDA_PKG_VERSION libcufile-$CUDA_PKG_VERSION \
    libcudnn9-cuda-$CUDA_MAJOR

if [ "$PY_VERSION" != "3.10" ] && [ "$PY_VERSION" != "3.11" ]; then
    echo "WARNING: TTS==0.22.0 (pinned in requirements.txt) is only known to support Python 3.9-3.11."
    echo "This system has Python $PY_VERSION. 'pip install -r requirements.txt' below may fail to resolve TTS."
    echo "If it does, install Python 3.11 (e.g. via deadsnakes PPA) and recreate .venv with it."
fi

if [ ! -d "$APP_DIR/.venv" ]; then
    echo "== Creating virtual environment (Python $(python3 --version 2>&1 | cut -d' ' -f2)) =="
    python3 -m venv "$APP_DIR/.venv"
fi

# The CUDA LD_LIBRARY_PATH is needed both in the venv (manual use) and in the
# systemd service (Environment= in voice-clone.service.template).
if ! grep -q "LD_LIBRARY_PATH" "$APP_DIR/.venv/bin/activate"; then
    echo "export LD_LIBRARY_PATH=$CUDA_LIB_PATH:\${LD_LIBRARY_PATH:-}" >> "$APP_DIR/.venv/bin/activate"
fi

source "$APP_DIR/.venv/bin/activate"
export LD_LIBRARY_PATH="$CUDA_LIB_PATH:${LD_LIBRARY_PATH:-}"

pip install --upgrade pip

# PyTorch for Jetson isn't on standard PyPI (the download.pytorch.org wheels are
# x86_64). It's installed separately, before requirements.txt so TTS doesn't
# overwrite it with a non-CUDA build. The Jetson AI Lab community index only
# publishes JetPack 6 (CUDA 12.x) builds; for anything newer set
# TORCH_INDEX_URL yourself to a verified index/wheel source before running
# this script (e.g. a community JetPack 7/CUDA 13 wheel repo) — none is
# assumed by default since it hasn't been verified against this project's
# pinned torch/TTS versions.
if [ "$CUDA_MAJOR" = "12" ]; then
    TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://pypi.jetson-ai-lab.io/jp6/cu$(echo "$CUDA_VERSION_DOT" | tr -d '.')}"
elif [ -z "${TORCH_INDEX_URL:-}" ]; then
    echo "No known PyTorch index for CUDA $CUDA_VERSION_DOT (JetPack 7+)."
    echo "Set TORCH_INDEX_URL to a verified wheel index for cu$(echo "$CUDA_VERSION_DOT" | tr -d '.')/py$(echo "$PY_VERSION" | tr -d '.') and re-run."
    exit 1
fi

if ! python3 -c "import torch; assert torch.cuda.is_available()" &> /dev/null; then
    echo "== Installing PyTorch with CUDA support for Jetson (CUDA $CUDA_VERSION_DOT) =="
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
TTS_IO_FILE="$APP_DIR/.venv/lib/python$PY_VERSION/site-packages/TTS/utils/io.py"
if ! grep -q "weights_only" "$TTS_IO_FILE"; then
    sed -i '/is_local = os.path.isdir(path) or os.path.isfile(path)/i\    kwargs.setdefault("weights_only", False)' "$TTS_IO_FILE"
fi

mkdir -p "$APP_DIR"/{models,speakers,logs}

if [ ! -f "$APP_DIR/.env" ]; then
    echo "VOICE_CLONE_API_TOKEN=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" > "$APP_DIR/.env"
    echo "Token generated at $APP_DIR/.env"
fi

# Without this, the first model download prompts interactively for the Coqui
# license (input()), which hangs/fails under systemd (no tty attached).
if ! grep -q "COQUI_TOS_AGREED" "$APP_DIR/.env"; then
    echo "COQUI_TOS_AGREED=1" >> "$APP_DIR/.env"
fi

if ! grep -q "VOICE_CLONE_AUDIO_DEVICE" "$APP_DIR/.env"; then
    echo "VOICE_CLONE_AUDIO_DEVICE=plughw:2,0" >> "$APP_DIR/.env"
    echo "Default audio device: plughw:2,0 (USB Audio Device). Adjust in .env if 'aplay -l' shows a different card."
fi

mkdir -p "$USER_SYSTEMD_DIR"
sed \
    -e "s|__APP_DIR__|$APP_DIR|g" \
    -e "s|__CUDA_LIB_PATH__|$CUDA_LIB_PATH|g" \
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
