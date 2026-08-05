"""Laporan + grafik lintas-generasi TTS pada titik operasi tersamakan."""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, "generations_results.json"), encoding="utf-8"))

ERAS = ["2019 (era FoR)", "2021-2022", "2024-2025 open", "2025-2026 komersial"]
SHORT = {
    "griffin_lim": "Griffin-Lim",
    "tts_models_en_ljspeech_tacotron2-DDC": "Tacotron2-DDC",
    "tts_models_en_ljspeech_speedy-speech": "SpeedySpeech",
    "tts_models_en_ljspeech_vits": "VITS",
    "tts_models_multilingual_multi-dataset_xtts_v2": "XTTS-v2",
    "suno_bark": "Bark",
    "f5-tts": "F5-TTS", "kokoro": "Kokoro", "sesame_csm": "Sesame-CSM",
    "orpheus-tts-0.1-finetune": "Orpheus",
    "ElevenLabs-v3": "ElevenLabs-v3", "Chatterbox": "Chatterbox",
    "OpenAI TTS-1 HD": "OpenAI TTS-1 HD", "Higgs-Audio-V2": "Higgs-Audio-V2",
}
MLAB = {
    "wavlm_official_full_b16e10_s42": "WavLM + aug penuh",
    "hubert_official_full_b16e10_s42": "HuBERT + aug penuh",
    "nes2net_official_full_b16e10_s42": "Nes2Net-X + aug penuh",
    "wavlm_official_codec_b16e10_s42": "WavLM + codec saja",
    "ast_official_codec_b32e10_s42": "AST + codec saja",
}
COL = {"wavlm_official_full_b16e10_s42": "#1864AB",
       "hubert_official_full_b16e10_s42": "#2F9E44",
       "nes2net_official_full_b16e10_s42": "#0B7285",
       "wavlm_official_codec_b16e10_s42": "#74C0FC",
       "ast_official_codec_b32e10_s42": "#C2255C"}

L = []
def out(s=""):
    print(s); L.append(s)

out("# Deteksi Lintas-Generasi TTS pada Titik Operasi Tersamakan\n")
out("Setiap model dikalibrasi agar **spesifisitas = 95%** pada 1.500 berkas asli "
    "In-the-Wild, baru kemudian recall diukur pada tiap sistem TTS (MLAAD, "
    "300 berkas per sistem).\n")
out("Kalibrasi ini penting: tanpanya, detektor yang selalu menjawab 'palsu' akan "
    "mencatat recall 100% — persis yang terjadi pada checkpoint SOTA "
    "(spesifisitas 0%). Di sini seluruh model dibandingkan pada FPR 5% yang sama.\n")

# ---- tabel ----
models = list(res)
out("| sistem TTS | generasi | " + " | ".join(MLAB.get(m, m) for m in models) + " |")
out("|" + "---|" * (len(models) + 2))
by_era = defaultdict(lambda: defaultdict(list))
for era in ERAS:
    names = [n for n in SHORT if any(
        res[m]["tts"].get(n, {}).get("era") == era for m in models)]
    for n in names:
        cells = []
        for m in models:
            r = res[m]["tts"].get(n)
            if r:
                cells.append(f"{r['recall']*100:.1f}%")
                by_era[m][era].append(r["recall"])
            else:
                cells.append("—")
        out(f"| {SHORT[n]} | {era} | " + " | ".join(cells) + " |")
out("")

out("## Rerata per generasi\n")
out("| model | " + " | ".join(ERAS) + " | **turun 2019→2026** |")
out("|" + "---|" * (len(ERAS) + 2))
for m in models:
    v = [np.mean(by_era[m][e]) * 100 if by_era[m][e] else np.nan for e in ERAS]
    drop = v[0] - v[-1]
    out(f"| {MLAB.get(m, m)} | " + " | ".join(f"{x:.1f}%" for x in v) +
        f" | **{drop:+.1f} pp** |")
out("")

# ---- grafik ----
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "legend.frameon": False, "figure.facecolor": "white"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                               gridspec_kw={"width_ratios": [1.55, 1]})

order = [n for e in ERAS for n in SHORT if any(
    res[m]["tts"].get(n, {}).get("era") == e for m in models)]
x = np.arange(len(order))
for m in models:
    ys = [res[m]["tts"].get(n, {}).get("recall", np.nan) * 100 for n in order]
    ax1.plot(x, ys, "-o", color=COL.get(m, "#495057"), lw=2, ms=6,
             label=MLAB.get(m, m))
ax1.set_xticks(x)
ax1.set_xticklabels([SHORT[n] for n in order], rotation=40, ha="right", fontsize=8.5)
ax1.set_ylabel("Recall pada spesifisitas 95% (%)")
ax1.set_ylim(0, 103)
b = 0
for e in ERAS:
    k = sum(1 for n in order if any(res[m]["tts"].get(n, {}).get("era") == e
                                    for m in models))
    if k:
        ax1.axvspan(b - 0.5, b + k - 0.5, color="#000", alpha=0.03)
        ax1.text(b + k / 2 - 0.5, 101, e, ha="center", fontsize=8, color="#555")
        b += k
ax1.legend(fontsize=8, loc="lower left")
ax1.set_title("Per sistem TTS", loc="left", fontsize=11)

for m in models:
    v = [np.mean(by_era[m][e]) * 100 if by_era[m][e] else np.nan for e in ERAS]
    ax2.plot(range(len(ERAS)), v, "-o", color=COL.get(m, "#495057"), lw=2.6, ms=8)
ax2.set_xticks(range(len(ERAS)))
ax2.set_xticklabels(["2019", "2021-22", "2024-25", "2025-26"], fontsize=9)
ax2.set_ylabel("Recall rata-rata (%)")
ax2.set_ylim(0, 103)
ax2.set_title("Rerata per generasi", loc="left", fontsize=11)
fig.suptitle("Apakah detektor masih mengenali suara AI generasi terbaru?  "
             "Semua diukur pada spesifisitas 95% yang sama",
             x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "charts", "05_generasi_tts.png"), bbox_inches="tight")
plt.close(fig)

open(os.path.join(HERE, "HASIL_GENERASI.md"), "w", encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_GENERASI.md, charts/05_generasi_tts.png")
