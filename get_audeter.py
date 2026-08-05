"""
Unduh subset AUDETER: TTS modern (2024-2025) + bona-fide berpasangan.

AUDETER (arXiv 2509.04345) berisi 4.500 jam dari 11 TTS + 10 vocoder modern.
Berbeda dari MLAAD, repo ini TIDAK gated.

Yang diambil: split `test` untuk TTS modern, plus `bona-fide` sebagai kelas asli.
Memakai bona-fide dari korpus yang SAMA (celebrity) dengan spoof-nya menghindari
confound provenance — pelajaran dari artefak MP3 di FoR.
"""
import os
import sys
import tarfile
import glob

from huggingface_hub import hf_hub_download

REPO = "wqz995/AUDETER"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "audeter")

# TTS modern, diurut dari yang paling relevan untuk "AI yang sulit dibedakan"
WANT = [
    "celebrity/tts/f5_tts/bona-fide/v0.tar",
    "celebrity/tts/cosyvoice/bona-fide/v0.tar",
    "celebrity/tts/cosyvoice/test/v0.tar",
    "audiobook/tts/f5_tts/test/v0.tar",
    "audiobook/tts/xtts/test/v0.tar",
    "audiobook/tts/cosyvoice/test/v0.tar",
    "audiobook/tts/sparktts/test/v0.tar",
    "audiobook/tts/zonos/test/v0.tar",
    "audiobook/tts/fish_speech/test/v0.tar",
    "audiobook/tts/bark/test/v0.tar",
    "audiobook/tts/vits/test/v0.tar",
    "audiobook/tts/chattts/test/v0.tar",
]

os.makedirs(OUT, exist_ok=True)
got = []
for rel in WANT:
    try:
        p = hf_hub_download(REPO, rel, repo_type="dataset", local_dir=OUT)
        mb = os.path.getsize(p) / 1024 ** 2
        print(f"  OK   {rel:52s} {mb:8.1f} MB")
        got.append((rel, p))
    except Exception as e:
        print(f"  GAGAL {rel:52s} {type(e).__name__}: {str(e)[:90]}")

print(f"\n{len(got)} arsip terunduh. Mengekstrak ...")
for rel, p in got:
    d = os.path.join(OUT, "extracted", rel.replace("/", "__").replace(".tar", ""))
    if os.path.isdir(d) and glob.glob(os.path.join(d, "**", "*.wav"), recursive=True):
        continue
    os.makedirs(d, exist_ok=True)
    try:
        with tarfile.open(p) as t:
            t.extractall(d, filter="data")
    except Exception as e:
        print(f"  ekstrak gagal {rel}: {e}")

print("\nIsi:")
tot = 0
for d in sorted(glob.glob(os.path.join(OUT, "extracted", "*"))):
    n = len(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True)) + \
        len(glob.glob(os.path.join(d, "**", "*.flac"), recursive=True))
    tot += n
    print(f"  {os.path.basename(d):58s} {n:6d}")
print(f"total {tot} berkas audio")
