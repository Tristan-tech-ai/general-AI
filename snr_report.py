"""Laporan grid SNR: kurva degradasi + dekomposisi diskriminasi vs kalibrasi."""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, "snr_results.json"), encoding="utf-8"))
SNRS = [None, 30, 25, 20, 15, 10, 5, 0, -5]


def key(s):
    return "bersih" if s is None else f"{s} dB"


g = defaultdict(list)
for r in res:
    g[(r["arch"], r["snr"])].append(r)

archs = sorted({r["arch"] for r in res})
L = []
def out(s=""):
    print(s); L.append(s)

out("# Grid SNR — Ketahanan terhadap Noise Lingkungan yang Belum Pernah Dilihat\n")
out("Model dilatih dengan augmentasi **colored noise sintetis**; diuji dengan "
    "**DEMAND** (6 lingkungan nyata: dapur, taman, kantor, kafetaria, lalu lintas, bus). "
    "Kedua korpus sepenuhnya terpisah.\n")
out(f"Test set 1.088 berkas, {len(archs)} arsitektur × 3 seed. "
    "Nilai = rerata ± simpangan baku atas seed.\n")

# ---------- akurasi (ambang prior-matched) ----------
out("## 1. Akurasi vs SNR (ambang prior-matched)\n")
out("| arsitektur | " + " | ".join(key(s) for s in SNRS) + " |")
out("|" + "---|" * (len(SNRS) + 1))
curves = {}
for a in archs:
    cells, vals = [], []
    for s in SNRS:
        rs = g.get((a, s), [])
        if not rs:
            cells.append("—"); vals.append(np.nan); continue
        v = np.array([r["acc_pm"] for r in rs]) * 100
        vals.append(v.mean())
        cells.append(f"{v.mean():.1f}")
    curves[a] = vals
    out(f"| `{a}` | " + " | ".join(cells) + " |")
out("")

# ---------- EER ----------
out("## 2. EER vs SNR\n")
out("| arsitektur | " + " | ".join(key(s) for s in SNRS) + " |")
out("|" + "---|" * (len(SNRS) + 1))
for a in archs:
    cells = []
    for s in SNRS:
        rs = g.get((a, s), [])
        cells.append(f"{np.mean([r['eer'] for r in rs])*100:.1f}" if rs else "—")
    out(f"| `{a}` | " + " | ".join(cells) + " |")
out("")

# ---------- degradasi ----------
out("## 3. Degradasi dari kondisi bersih (poin persentase)\n")
out("| arsitektur | bersih | @10 dB | @0 dB | @−5 dB | **turun bersih→0 dB** |")
out("|---|---|---|---|---|---|")
deg = {}
for a in archs:
    v = curves[a]
    i = {s: k for k, s in enumerate(SNRS)}
    c, d10, d0, dm5 = v[i[None]], v[i[10]], v[i[0]], v[i[-5]]
    deg[a] = c - d0
    out(f"| `{a}` | {c:.1f}% | {d10:.1f}% | {d0:.1f}% | {dm5:.1f}% | **−{c-d0:.1f} pp** |")
out("")
rank = sorted(deg, key=lambda a: deg[a])
out(f"**Paling tahan (degradasi terkecil bersih→0 dB):** `{rank[0]}` "
    f"(−{deg[rank[0]]:.1f} pp) · **paling rapuh:** `{rank[-1]}` (−{deg[rank[-1]]:.1f} pp)\n")

# ---------- dekomposisi ----------
out("## 4. Dekomposisi: diskriminasi vs kalibrasi\n")
out("`acc_fix` memakai ambang yang dibekukan dari kondisi bersih; `acc_pm` "
    "memakai ambang prior-matched per kondisi. Selisihnya adalah akurasi yang "
    "hilang **semata karena ambang meleset**, bukan karena model kehilangan daya pisah.\n")
out("| arsitektur | SNR | acc (ambang beku) | acc (prior-matched) | **hilang krn ambang** | AUC |")
out("|---|---|---|---|---|---|")
tot_cal = []
for a in archs:
    for s in [20, 10, 0]:
        rs = g.get((a, s), [])
        if not rs:
            continue
        fx = np.mean([r["acc_fx"] for r in rs]) * 100
        pm = np.mean([r["acc_pm"] for r in rs]) * 100
        au = np.mean([r["auc"] for r in rs])
        tot_cal.append(pm - fx)
        out(f"| `{a}` | {s} dB | {fx:.1f}% | {pm:.1f}% | **{pm-fx:+.1f} pp** | {au:.3f} |")
out("")
out(f"Rerata akurasi yang dapat dipulihkan hanya dengan mengoreksi ambang: "
    f"**{np.mean(tot_cal):+.1f} pp** (maks {np.max(tot_cal):+.1f} pp).\n")

# ---------- titik runtuh ----------
out("## 5. Titik runtuh (SNR saat akurasi turun di bawah 80%)\n")
out("| arsitektur | SNR runtuh |")
out("|---|---|")
for a in archs:
    v = curves[a]
    bp = "tidak runtuh"
    for k, s in enumerate(SNRS):
        if s is not None and v[k] < 80:
            bp = f"{s} dB"; break
    out(f"| `{a}` | {bp} |")
out("")

open(os.path.join(HERE, "HASIL_SNR.md"), "w", encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_SNR.md")
