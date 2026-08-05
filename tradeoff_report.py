"""
Uji apakah band-gain memecah trade-off FoR-vs-TTS modern.

Membandingkan strategi augmentasi pada arsitektur yang SAMA (Nes2Net-X),
masing-masing 3 seed, pada dua sumbu sekaligus:
  sumbu 1 = akurasi FoR-2sec (in-domain)
  sumbu 2 = recall TTS 2025-2026 pada spesifisitas 95% (lintas generasi)
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.metrics import full_metrics, prior_matched_threshold

HERE = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(os.path.join(HERE, "generations_results.json"), encoding="utf-8"))

ERAS = ["2019 (era FoR)", "2021-2022", "2024-2025 open", "2025-2026 komersial"]
OLD_NONMP3 = ["tts_models_en_ljspeech_tacotron2-DDC",
              "tts_models_en_ljspeech_speedy-speech",
              "tts_models_en_ljspeech_vits"]


def parse(tag):
    m = re.match(r"^(.+?)_official_([a-z]+?)(?:_b\d+e\d+)?_s(\d+)$", tag)
    return None if not m else (m.group(1), m.group(2), m.group(3))


foracc = {}
for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
    f = os.path.join(d, "test_scores.npy")
    p = parse(os.path.basename(d))
    if not (os.path.exists(f) and p):
        continue
    y, s, _ = np.load(f)
    if len(y) != 1088:
        continue
    m = full_metrics(y.astype(int), s, prior_matched_threshold(s, 0.5))
    foracc[p] = m["accuracy"]

rows = defaultdict(list)
for tag, r in G.items():
    p = parse(tag)
    if not p or p[0] != "nes2net":
        continue
    modern = [v["recall"] for v in r["tts"].values()
              if v["era"] == "2025-2026 komersial"]
    old = [r["tts"][k]["recall"] for k in OLD_NONMP3 if k in r["tts"]]
    fa = foracc.get(p)
    if modern and fa is not None:
        rows[p[1]].append({"seed": p[2], "for": fa, "modern": np.mean(modern),
                           "old_nonmp3": np.mean(old) if old else np.nan})

L = []
def out(s=""):
    print(s); L.append(s)


NAMA = {"full": "aug penuh (basis)",
        "fullrb": "aug penuh + RawBoost",
        "fullbg": "**aug penuh + band-gain (usulan)**",
        "soft": "preset 'soft' (cacat: 4 variabel berubah)"}
COL = {"full": "#0B7285", "fullrb": "#C2255C",
       "fullbg": "#2F9E44", "soft": "#ADB5BD"}

out("# Apakah Band-Gain Memecah Trade-off FoR vs TTS Modern?\n")
out("Arsitektur identik (Nes2Net-X), data identik, 3 seed per strategi. "
    "Yang berbeda hanya augmentasi.\n")
out("| strategi augmentasi | n | akurasi FoR | recall TTS 2025-26 | "
    "recall TTS-2019 non-MP3 |")
out("|---|---|---|---|---|")

summ = {}
for k in ["full", "fullrb", "fullbg", "soft"]:
    v = rows.get(k, [])
    if not v:
        continue
    fa = np.array([x["for"] for x in v]) * 100
    md = np.array([x["modern"] for x in v]) * 100
    ol = np.array([x["old_nonmp3"] for x in v]) * 100
    summ[k] = (fa, md, ol)
    sd = lambda a: ("+/-%.2f" % a.std(ddof=1)) if len(a) > 1 else ""
    out("| %s | %d | %.2f%% %s | **%.2f%%** %s | %.1f%% %s |"
        % (NAMA.get(k, k), len(v), fa.mean(), sd(fa), md.mean(), sd(md),
           ol.mean(), sd(ol)))
out("")

if all(k in summ for k in ("full", "fullrb", "fullbg")):
    f0, m0, o0 = summ["full"]
    fr, mr, orb = summ["fullrb"]
    fb, mb, ob = summ["fullbg"]
    out("## Ablasi variabel-tunggal terhadap basis `aug penuh`\n")
    out("| perbandingan | d-akurasi FoR | d-TTS modern | d-TTS-2019 non-MP3 |")
    out("|---|---|---|---|")
    out("| + RawBoost | **%+.2f pp** | **%+.2f pp** | **%+.1f pp** |"
        % (fr.mean() - f0.mean(), mr.mean() - m0.mean(), orb.mean() - o0.mean()))
    out("| + band-gain | **%+.2f pp** | **%+.2f pp** | **%+.1f pp** |"
        % (fb.mean() - f0.mean(), mb.mean() - m0.mean(), ob.mean() - o0.mean()))
    out("")
    out("Simpangan baku TTS-2019 non-MP3: basis +/-%.1f, RawBoost +/-%.1f, "
        "**band-gain +/-%.1f**\n" % (o0.std(ddof=1), orb.std(ddof=1), ob.std(ddof=1)))

    if fb.mean() > f0.mean() and mb.mean() >= m0.mean() - m0.std(ddof=1):
        out("**Band-gain sebagian besar memecah trade-off.** Akurasi FoR naik "
            "%+.2f pp sementara deteksi TTS modern hanya bergeser %+.2f pp - di "
            "dalam derau antar-seed (+/-%.2f). RawBoost, pada kenaikan FoR yang "
            "serupa, kehilangan %.2f pp.\n"
            % (fb.mean() - f0.mean(), mb.mean() - m0.mean(), m0.std(ddof=1),
               abs(mr.mean() - m0.mean())))
        out("Sinyal terkuatnya ada di TTS-2019 non-MP3, proksi ketergantungan "
            "pintasan codec: band-gain %+.1f pp dengan simpangan baku runtuh dari "
            "+/-%.1f ke +/-%.1f, sedangkan RawBoost %+.1f pp. Ini persis yang "
            "diprediksi diagnosis mekanistiknya - menetralkan LEVEL energi HF "
            "menghapus pintasan, sementara menghancurkan STRUKTUR HALUS HF "
            "(low-pass, RawBoost) ikut menghapus buktinya.\n"
            % (ob.mean() - o0.mean(), o0.std(ddof=1), ob.std(ddof=1),
               orb.mean() - o0.mean()))
        out("> Preset `soft` sebelumnya tampak gagal karena mengubah empat "
            "variabel sekaligus. Hasil di sini berasal dari ablasi "
            "variabel-tunggal (`fullbg`) dan menyimpulkan sebaliknya.\n")
    else:
        out("Band-gain belum memecah trade-off pada konfigurasi ini.\n")

# ---- grafik Pareto ----
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "legend.frameon": False, "figure.facecolor": "white"})
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6))

for k, (fa, md, ol) in summ.items():
    c = COL.get(k, "#495057")
    nm = NAMA.get(k, k).replace("**", "")
    ax.scatter(fa, md, s=80, color=c, alpha=0.5, edgecolor="white", zorder=3)
    ax.scatter([fa.mean()], [md.mean()], s=330, marker="D", color=c,
               edgecolor="white", lw=2, zorder=4, label=nm)
    ax2.scatter(fa, ol, s=80, color=c, alpha=0.5, edgecolor="white", zorder=3)
    ax2.scatter([fa.mean()], [ol.mean()], s=330, marker="D", color=c,
                edgecolor="white", lw=2, zorder=4)

ax.set_xlabel("Akurasi FoR-2sec (%) - metrik in-domain")
ax.set_ylabel("Recall TTS 2025-2026 @ spesifisitas 95% (%)")
ax.set_title("vs TTS komersial terbaru", loc="left", fontsize=11)
ax.legend(fontsize=8.5, loc="lower left")
ax2.set_xlabel("Akurasi FoR-2sec (%) - metrik in-domain")
ax2.set_ylabel("Recall TTS-2019 non-MP3 (%)")
ax2.set_title("vs TTS lama tanpa MP3 - proksi ketergantungan pintasan",
              loc="left", fontsize=11)
for a in (ax, ax2):
    a.annotate("arah yang diinginkan", xy=(0.97, 0.06), xycoords="axes fraction",
               ha="right", fontsize=8.5, color="#2F9E44")
fig.suptitle("Trade-off in-domain vs generalisasi - Nes2Net-X, hanya augmentasi "
             "yang berbeda (titik = seed, berlian = rerata)",
             x=0.01, ha="left", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "charts", "06_tradeoff.png"), bbox_inches="tight")
plt.close(fig)

open(os.path.join(HERE, "HASIL_TRADEOFF.md"), "w", encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_TRADEOFF.md, charts/06_tradeoff.png")
