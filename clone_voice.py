"""Generates cloned voice with XTTS v2 from samples/voice_sample.wav."""
import argparse
from pathlib import Path

from TTS.api import TTS

SPEAKER_WAV = Path("samples/voice_sample.wav")
OUT_DIR = Path("output")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Text to synthesize with the cloned voice")
    parser.add_argument("--lang", default="es", help="Language code (es, en, etc.)")
    parser.add_argument("--out", default="output.wav", help="Output file name")
    args = parser.parse_args()

    if not SPEAKER_WAV.exists():
        raise SystemExit(
            f"{SPEAKER_WAV.resolve()} not found. Run extract_audio.py first."
        )

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / args.out

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    tts.tts_to_file(
        text=args.text,
        speaker_wav=str(SPEAKER_WAV),
        language=args.lang,
        file_path=str(out_path),
    )
    print(f"Audio generated at: {out_path.resolve()}")


if __name__ == "__main__":
    main()
