"""Extrae y limpia el audio de main.mp4 para usarlo como muestra de voz para XTTS v2."""
import subprocess
import sys
from pathlib import Path

SRC = Path("main.mp4")
OUT_DIR = Path("samples")
OUT_WAV = OUT_DIR / "voice_sample.wav"


def main():
    if not SRC.exists():
        sys.exit(f"No se encontró {SRC.resolve()}. Copia tu video main.mp4 en esta carpeta.")

    OUT_DIR.mkdir(exist_ok=True)

    # XTTS v2 espera mono, 22050 Hz, 16-bit PCM.
    cmd = [
        "ffmpeg", "-y",
        "-i", str(SRC),
        "-ac", "1",
        "-ar", "22050",
        "-vn",
        str(OUT_WAV),
    ]
    subprocess.run(cmd, check=True)
    print(f"Audio extraído en: {OUT_WAV.resolve()}")


if __name__ == "__main__":
    main()
