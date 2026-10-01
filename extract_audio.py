"""Extracts and cleans up the audio from main.mp4 to use as a voice sample for XTTS v2."""
import subprocess
import sys
from pathlib import Path

SRC = Path("main.mp4")
OUT_DIR = Path("samples")
OUT_WAV = OUT_DIR / "voice_sample.wav"


def main():
    if not SRC.exists():
        sys.exit(f"{SRC.resolve()} not found. Copy your main.mp4 video into this folder.")

    OUT_DIR.mkdir(exist_ok=True)

    # XTTS v2 expects mono, 22050 Hz, 16-bit PCM.
    cmd = [
        "ffmpeg", "-y",
        "-i", str(SRC),
        "-ac", "1",
        "-ar", "22050",
        "-vn",
        str(OUT_WAV),
    ]
    subprocess.run(cmd, check=True)
    print(f"Audio extracted to: {OUT_WAV.resolve()}")


if __name__ == "__main__":
    main()
