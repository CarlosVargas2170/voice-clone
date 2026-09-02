#!/usr/bin/env python3
r"""Prepara un lote de output/batch/ para subir al tótem (avatar-registration-backen).

`batch_synthesize.py` ya deja los WAV + manifest.json en el formato del Runtime.
Esto sólo:
  - descarta los pasos obsoletos (photo_left / photo_right, quitados en P4),
  - avisa qué pasos canónicos faltan o tienen pocas variantes,
  - imprime el `scp` exacto para subir output/batch/<lote>/assets/audio a la Jetson.

Uso (desde la carpeta voice-clone):
    python pack_totem_audio.py                 # lote 'franco-expoteleinfo'
    python pack_totem_audio.py camila-expoteleinfo
"""

from __future__ import annotations

import json
import shutil
import sys
import wave
from pathlib import Path

# --- ajustá si cambia el destino -------------------------------------------
JETSON_USER_HOST = "nexus2026@192.168.19.30"
JETSON_REPO = "/home/nexus2026/projects/avatar-registration-backen"
# -------------------------------------------------------------------------

OUT_DIR = Path(__file__).resolve().parent / "output" / "batch"
SAMPLE_RATE = 24_000
MIN_VARIANTS = 10
OBSOLETE = ("photo_left", "photo_right")

# 29 pasos canónicos del guion (ver fallback_emergency/docs/catalogo-audios.md).
CANONICAL = [
    "greeting", "resume", "ask_name", "nudge_name", "ask_company_role",
    "nudge_company_role", "ask_contact", "nudge_contact", "ask_consent",
    "confirm_data", "final_confirm", "farewell", "farewell_cancel",
    "photo_intro", "photo_front", "photo_hold", "photo_multiface",
    "tablet_hint", "recheck", "didnt_understand", "manual_fallback",
    "welcome_back", "update_intro", "offer_ticket", "already_registered",
    "invite_register", "ticket_printing", "ticket_ready", "ticket_error",
]


def _wav_bad(path: Path) -> str | None:
    try:
        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2:
                return "no es PCM16"
            if w.getnchannels() != 1:
                return f"{w.getnchannels()} canales (se espera mono)"
            if w.getframerate() != SAMPLE_RATE:
                return f"{w.getframerate()} Hz (se espera {SAMPLE_RATE})"
    except wave.Error as exc:
        return f"WAV inválido ({exc})"
    return None


def main() -> int:
    batch = sys.argv[1] if len(sys.argv) > 1 else "franco-expoteleinfo"
    audio = OUT_DIR / batch / "assets" / "audio"
    if not audio.is_dir():
        print(f"No existe el lote: {audio}", file=sys.stderr)
        if OUT_DIR.is_dir():
            print("Lotes disponibles:", file=sys.stderr)
            for d in sorted(p.name for p in OUT_DIR.iterdir() if p.is_dir()):
                print(f"  {d}", file=sys.stderr)
        return 2

    print(f"=== lote: {batch} ===")
    print(f"carpeta: {audio}\n")

    manifest_path = audio / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {"sample_rate": SAMPLE_RATE, "steps": {}}
    )
    steps = manifest.setdefault("steps", {})

    # 1) borrar pasos obsoletos (carpeta + entrada del manifest)
    for step in OBSOLETE:
        folder = audio / step
        if folder.is_dir():
            shutil.rmtree(folder)
            print(f"  borrado paso obsoleto: {step}")
        steps.pop(step, None)

    # 2) reordenar el manifest a los pasos canónicos y verificar cobertura
    ordered: dict[str, list] = {}
    missing, few, bad = [], [], []
    for step in CANONICAL:
        folder = audio / step
        wavs = sorted(p.name for p in folder.glob("*.wav")) if folder.is_dir() else []
        if not wavs:
            missing.append(step)
            continue
        if len(wavs) < MIN_VARIANTS:
            few.append(f"{step} ({len(wavs)})")
        by_name = {e["file"]: e.get("text", "") for e in steps.get(step, [])}
        entries = []
        for name in wavs:
            problem = _wav_bad(folder / name)
            if problem:
                bad.append(f"{step}/{name}: {problem}")
            entries.append({"file": name, "text": by_name.get(name, "")})
            if not by_name.get(name):
                bad.append(f"{step}/{name}: sin transcripción en el manifest")
        ordered[step] = entries

    manifest["steps"] = ordered
    manifest["sample_rate"] = SAMPLE_RATE
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  manifest reescrito: {manifest_path}\n")

    total = sum(len(v) for v in ordered.values())
    print(f"pasos con voz: {len(ordered)}/{len(CANONICAL)}   ·   wav: {total}")
    if few:
        print(f"  pocas variantes: {', '.join(few)}")
    if missing:
        print(f"  SIN audio (mudos, no bloquea): {', '.join(missing)}")
    if bad:
        print("\n!! problemas que hay que corregir antes de subir:")
        for b in bad:
            print(f"   - {b}")
        return 1

    print("\n=== subir a la Jetson: ===")
    print(f"  ssh {JETSON_USER_HOST} 'rm -rf {JETSON_REPO}/assets/audio'")
    print(f"  scp -r -C \"{audio}\" {JETSON_USER_HOST}:{JETSON_REPO}/assets/")
    print(f"  ssh {JETSON_USER_HOST} 'sudo systemctl restart fallback-emergency'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
