"""
Unduh subset MLAAD lintas-generasi TTS.

Tujuan: mengukur apakah detektor yang dilatih pada FoR (TTS era 2019) masih
mengenali TTS modern 2025-2026. Dipilih model yang merentang empat generasi
supaya kurva kebutaannya terlihat, bukan hanya satu titik.
"""
import os
import sys

from huggingface_hub import snapshot_download

REPO = "mueller91/MLAAD"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mlaad")

# nama folder harus persis seperti di repo
GENERASI = {
    "2019-era (setara FoR)": [
        "griffin_lim",
        "tts_models_en_ljspeech_tacotron2-DDC",
        "tts_models_en_ljspeech_speedy-speech",
    ],
    "2021-2022": [
        "tts_models_en_ljspeech_vits",
        "tts_models_multilingual_multi-dataset_xtts_v2",
        "suno_bark",
    ],
    "2024-2025 open": [
        "f5-tts",
        "kokoro",
        "sesame_csm",
        "orpheus-tts-0.1-finetune",
    ],
    "2025-2026 komersial/SOTA": [
        "ElevenLabs-v3",
        "Chatterbox",
        "OpenAI TTS-1 HD",
        "Higgs-Audio-V2",
    ],
}

pats = []
for era, models in GENERASI.items():
    for m in models:
        pats.append(f"fake/en/{m}/*")

print(f"mengunduh {len(pats)} folder model TTS dari {REPO} ...")
for era, models in GENERASI.items():
    print(f"  {era}: {', '.join(models)}")

p = snapshot_download(repo_id=REPO, repo_type="dataset", allow_patterns=pats,
                      local_dir=OUT, max_workers=8)
print("\nselesai ->", p)

# ringkas apa yang benar-benar terunduh
import glob
tot = 0
for d in sorted(glob.glob(os.path.join(OUT, "fake", "en", "*"))):
    n = len(glob.glob(os.path.join(d, "*.wav")))
    tot += n
    print(f"  {os.path.basename(d):48s} {n:5d} wav")
print(f"total {tot} berkas")
