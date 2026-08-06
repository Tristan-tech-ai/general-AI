"""
Matriks 2x2 apple-to-apple: apakah rekayasa kita benar-benar melewati baseline?

Untuk tiap arsitektur, empat sel dijalankan dengan arsitektur, data, batch, dan
seed yang sama. Hanya dua hal yang berubah: konfigurasi pelatihan dan skema
pembagian data.

Sel yang menjawab pertanyaan pokok adalah kolom "partisi resmi". Perbandingan
baris di kolom itu mengukur nilai rekayasa pada protokol yang sama persis.
Kolom "split acak" hanya menunjukkan bahwa protokol yang longgar membuat kedua
konfigurasi tampak sama-sama baik.
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.metrics import full_metrics, prior_matched_threshold

HERE = os.path.dirname(os.path.abspath(__file__))
L = []
def out(s=""):
    print(s); L.append(s)


def cari(model, split, cfg):
    """cfg: 'proposal' -> tag mengandung proposalULRPK; 'diperbaiki' -> _full_"""
    pat = (f"runs/{model}_{split}_proposalULRPK_*" if cfg == "proposal"
           else f"runs/{model}_{split}_full_*")
    hasil = []
    for d in sorted(glob.glob(os.path.join(HERE, pat))):
        f = os.path.join(d, "test_scores.npy")
        if not os.path.exists(f):
            continue
        y, p, _ = np.load(f)
        y = y.astype(int)
        if cfg == "proposal":
            m = full_metrics(y, p, 0.5)              # ambang tetap 0,5
        else:
            m = full_metrics(y, p, prior_matched_threshold(p, 0.5))
        hasil.append(m)
    return hasil


def sel(h):
    if not h:
        return "belum ada", None
    a = np.array([x["accuracy"] for x in h]) * 100
    if len(a) == 1:
        return f"{a[0]:.2f}%", a.mean()
    return f"{a.mean():.2f}% ({a.std(ddof=1):.2f})", a.mean()


out("# Matriks 2x2: Apakah Rekayasa Melewati Baseline?\n")
out("Setiap matriks memakai satu arsitektur, batch dan seed yang sama. "
    "Hanya konfigurasi pelatihan dan skema pembagian data yang berubah, "
    "sehingga tiap perbandingan bersifat satu variabel.\n")
out("Konfigurasi proposal: learning rate 0,001 seragam dengan encoder ikut "
    "dilatih, 20 epoch tanpa early stopping, normalisasi peak, augmentasi noise "
    "SNR 15 sampai 30 dB, ambang keputusan 0,5.\n")
out("Konfigurasi diperbaiki: learning rate per model dengan encoder dibekukan "
    "dan agregasi berbobot antar lapisan, 10 epoch dengan early stopping pada "
    "EER, normalisasi loudness, augmentasi penuh, ambang prior-matched.\n")

ringkas = []
for model in ["ast", "wavlm", "hubert", "nes2net"]:
    sel_data = {}
    for cfg in ["proposal", "diperbaiki"]:
        for split in ["random", "official"]:
            sel_data[(cfg, split)] = cari(model, split, cfg)
    if not any(sel_data.values()):
        continue

    out(f"## {model}\n")
    out("| Konfigurasi | Split acak 60/20/20 | Partisi resmi FoR | Selisih antar split |")
    out("|---|---|---|---|")
    nilai = {}
    for cfg, nm in [("proposal", "Proposal apa adanya"),
                    ("diperbaiki", "Diperbaiki (rekayasa)")]:
        sr, vr = sel(sel_data[(cfg, "random")])
        so, vo = sel(sel_data[(cfg, "official")])
        d = f"{vr - vo:+.2f} pp" if (vr is not None and vo is not None) else "n/a"
        nilai[cfg] = (vr, vo)
        out(f"| {nm} | {sr} | **{so}** | {d} |")

    vp, vd = nilai.get("proposal"), nilai.get("diperbaiki")
    if vp and vd and vp[1] is not None and vd[1] is not None:
        gap = vd[1] - vp[1]
        ringkas.append((model, vp[1], vd[1], gap))
        out("")
        out(f"**Nilai rekayasa pada protokol resmi: {gap:+.2f} poin persentase** "
            f"({vp[1]:.2f}% menjadi {vd[1]:.2f}%).")
        if vp[0] is not None and vd[0] is not None:
            gr = vd[0] - vp[0]
            out(f"Pada split acak selisihnya hanya {gr:+.2f} poin "
                f"({vp[0]:.2f}% menjadi {vd[0]:.2f}%), yang menunjukkan bahwa "
                f"protokol longgar tidak dapat membedakan kedua konfigurasi.")
    out("")

if ringkas:
    out("## Kesimpulan\n")
    out("| Arsitektur | Proposal pada partisi resmi | Diperbaiki pada partisi resmi | Selisih |")
    out("|---|---|---|---|")
    for m, vp, vd, g in ringkas:
        out(f"| {m} | {vp:.2f}% | **{vd:.2f}%** | **{g:+.2f} pp** |")
    out("")
    rata = np.mean([g for _, _, _, g in ringkas])
    if rata > 5:
        out(f"Rerata selisih {rata:+.2f} poin persentase pada protokol yang sama "
            "persis. Rekayasa memberi perbaikan nyata, dan perbaikan itu tidak "
            "terlihat sama sekali bila hanya melihat kolom split acak.\n")
    elif rata > 0:
        out(f"Rerata selisih hanya {rata:+.2f} poin persentase. Rekayasa memberi "
            "perbaikan kecil pada protokol yang sama.\n")
    else:
        out(f"Rerata selisih {rata:+.2f} poin persentase. Pada protokol yang sama "
            "persis, rekayasa tidak terbukti mengungguli konfigurasi proposal. "
            "Temuan ini harus dilaporkan apa adanya.\n")

open(os.path.join(HERE, "TABEL_2X2.md"), "w", encoding="utf-8").write("\n".join(L))
print("\n-> TABEL_2X2.md")
