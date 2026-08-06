"""
Eksperimen lintas-generasi TTS: apakah detektor masih relevan pada suara AI 2026?

RANCANGAN YANG MENGHINDARI JEBAKAN SPESIFISITAS
-----------------------------------------------
Pengukuran sebelumnya menunjukkan model SOTA menandai 100% audio asli sebagai
palsu, sehingga "detection rate 99%" pada TTS modern tidak bermakna. Karena itu
di sini ambang TIDAK ditetapkan sembarang, melainkan dikalibrasi pada himpunan
audio ASLI yang terpisah:

    ambang dipilih agar SPESIFISITAS = 95% pada bona-fide In-the-Wild

Baru setelah itu recall diukur pada tiap sistem TTS. Dengan cara ini seluruh
model dibandingkan pada TITIK OPERASI YANG SAMA (FPR 5%), sehingga angka recall
benar-benar berarti dan tidak dapat dinaikkan dengan menjawab "palsu" terus.

Generasi TTS yang diuji (MLAAD, 1.000 berkas per sistem):
  2019-era  : griffin_lim, tacotron2-DDC, speedy-speech
  2021-2022 : vits, xtts_v2, suno_bark
  2024-2025 : f5-tts, kokoro, sesame_csm, orpheus-tts
  2025-2026 : ElevenLabs-v3, Chatterbox, OpenAI TTS-1 HD, Higgs-Audio-V2
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.models import build_model
from eval_sota_modern import score

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GENERASI = {
    "2019 (era FoR)": ["griffin_lim", "tts_models_en_ljspeech_tacotron2-DDC",
                       "tts_models_en_ljspeech_speedy-speech"],
    "2021-2022": ["tts_models_en_ljspeech_vits",
                  "tts_models_multilingual_multi-dataset_xtts_v2", "suno_bark"],
    "2024-2025 open": ["f5-tts", "kokoro", "sesame_csm",
                       "orpheus-tts-0.1-finetune"],
    "2025-2026 komersial": ["ElevenLabs-v3", "Chatterbox", "OpenAI TTS-1 HD",
                            "Higgs-Audio-V2"],
}
N_PER_TTS = 300
N_REAL = 1500
TARGET_SPEC = 0.95


def real_paths():
    """Audio ASLI dari In-the-Wild — dipakai untuk mengalibrasi ambang."""
    root = os.path.join(HERE, "data", "release_in_the_wild")
    out = []
    with open(os.path.join(root, "meta.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["label"].strip() == "bona-fide":
                out.append(os.path.join(root, r["file"]))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(out))[:N_REAL]
    return [out[i] for i in sorted(idx)]


def tts_paths(name):
    d = os.path.join(HERE, "data", "mlaad", "fake", "en", name)
    ws = sorted(glob.glob(os.path.join(d, "*.wav")))
    return ws[:N_PER_TTS]


def load(tag):
    d = os.path.join(HERE, "runs", tag)
    ck = os.path.join(d, "best.pt")
    m = re.match(r"^(.+?)_official_", tag)
    arch = m.group(1)
    kw = {"freeze": True, "layer_weighting": True} if arch in (
        "wav2vec2", "hubert", "wavlm", "ast") else {}
    mod = build_model(arch, **kw).to(DEV)
    mod.load_state_dict(torch.load(ck, map_location=DEV))
    return mod.eval()


def main():
    reals = real_paths()
    print(f"audio asli untuk kalibrasi ambang: {len(reals)} berkas (In-the-Wild bona-fide)")

    cands = []
    for tag in ["wavlm_official_fullbg_b16e10_s42",
                "wavlm_official_fullbg_b16e10_s1337",
                "wavlm_official_fullbg_b16e10_s2024",
                "wavlm_official_full_b16e10_s1337",
                "wavlm_official_full_b16e10_s2024",
                "nes2net_official_fullbgrb_b16e10_s42",
                "nes2net_official_fullbgrb_b16e10_s1337",
                "nes2net_official_fullbgrb_b16e10_s2024",
                "nes2net_official_fullbg_b16e10_s42",
                "nes2net_official_fullbg_b16e10_s1337",
                "nes2net_official_fullbg_b16e10_s2024",
                "nes2net_official_soft_b16e10_s42",
                "nes2net_official_soft_b16e10_s1337",
                "nes2net_official_soft_b16e10_s2024",
                "nes2net_official_full_b16e10_s42",
                "nes2net_official_full_b16e10_s1337",
                "nes2net_official_full_b16e10_s2024",
                "nes2net_official_fullrb_b16e10_s42",
                "nes2net_official_fullrb_b16e10_s1337",
                "nes2net_official_fullrb_b16e10_s2024",
                "wavlm_official_full_b16e10_s42",
                "hubert_official_full_b16e10_s42"]:
        if os.path.exists(os.path.join(HERE, "runs", tag, "best.pt")):
            cands.append(tag)
    print(f"model diuji: {len(cands)}\n")

    res = {}
    for tag in cands:
        try:
            model = load(tag)
        except Exception as e:
            print(f"  {tag}: GAGAL muat — {type(e).__name__}: {e}")
            continue
        p_real = score(model, reals)
        thr = float(np.quantile(p_real, TARGET_SPEC))     # 95% real di bawah ambang
        spec = float((p_real < thr).mean())
        row = {"threshold": thr, "specificity": spec, "tts": {}}
        print(f"{tag}")
        print(f"  ambang @spesifisitas {spec*100:.1f}% = {thr:.4f}")
        for era, names in GENERASI.items():
            for nm in names:
                ws = tts_paths(nm)
                if len(ws) < 50:
                    continue
                p = score(model, ws)
                rec = float((p >= thr).mean())
                row["tts"][nm] = {"era": era, "n": len(ws), "recall": rec,
                                  "mean_score": float(p.mean())}
                print(f"    {era:22s} {nm:44s} recall={rec*100:6.2f}%")
        res[tag] = row
        del model
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
        print()

    json.dump(res, open(os.path.join(HERE, "generations_results.json"), "w"),
              indent=1)
    print("-> generations_results.json")


if __name__ == "__main__":
    main()
