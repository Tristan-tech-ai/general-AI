"""
Tangga ablasi: memecah nilai rekayasa menjadi sumbangan tiap perbaikan.

Bertolak dari konfigurasi proposal pada partisi resmi, lalu menambahkan satu
perbaikan pada satu waktu. Seluruh langkah memakai AST, partisi resmi, batch 32,
seed 42, sehingga tiap selisih hanya mencerminkan satu variabel yang berubah.

Akurasi dilaporkan pada ambang prior-matched untuk semua langkah, supaya sumbu
ambang tidak ikut bercampur ke dalam tangga. Sumbangan ambang sendiri sudah
dipisahkan tersendiri di dekomposisi.py. AUC dan EER disertakan karena keduanya
tidak bergantung pada ambang sama sekali.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.metrics import full_metrics, prior_matched_threshold

HERE = os.path.dirname(os.path.abspath(__file__))
L = []
def out(s=""):
    print(s); L.append(s)

TANGGA = [
    ("L1", "Konfigurasi proposal apa adanya",
     "runs/ast_official_proposalULRPK_b32e20_s42",
     "LR 0,001 seragam dengan encoder ikut dilatih, normalisasi peak, "
     "20 epoch tanpa early stopping, augmentasi noise saja"),
    ("L2", "Normalisasi loudness",
     "runs/ast_official_proposalULR_b32e20_s42",
     "normalisasi peak diganti loudness, selebihnya sama"),
    ("L3", "LR per model dan encoder dibekukan",
     "runs/ast_official_proposal_b32e20_s42",
     "encoder tidak lagi dilatih, head 0,001 dan encoder 2e-5, "
     "ditambah agregasi berbobot antar lapisan"),
    ("L4", "Early stopping pada EER",
     "runs/ast_official_proposal_b32e10_s42",
     "10 epoch dengan pemilihan bobot terbaik menurut EER validasi"),
    ("L5", "Augmentasi penuh",
     "runs/ast_official_full_b32e10_s42",
     "augmentasi noise saja diganti augmentasi penuh, yaitu codec, noise, "
     "reverb, dan band-gain"),
]


def baca(d):
    f = os.path.join(HERE, d, "test_scores.npy")
    if not os.path.exists(f):
        return None
    y, p, _ = np.load(f)
    y = y.astype(int)
    m0 = full_metrics(y, p, 0.5)
    mp = full_metrics(y, p, prior_matched_threshold(p, 0.5))
    return {"a05": m0["accuracy"] * 100, "apm": mp["accuracy"] * 100,
            "auc": m0["auc"], "eer": m0["eer"] * 100}


out("# Tangga Ablasi: Perbaikan Mana yang Membeli Berapa\n")
out("Semua langkah memakai AST pada partisi resmi Fake-or-Real, batch 32, seed "
    "42. Tiap baris menambahkan satu perbaikan di atas baris sebelumnya, "
    "sehingga selisih antar baris hanya mencerminkan satu variabel.\n")
out("Akurasi dilaporkan pada ambang prior-matched untuk seluruh langkah agar "
    "sumbu ambang tidak bercampur ke dalam tangga. Sumbangan ambang itu sendiri "
    "dipisahkan tersendiri di HASIL_DEKOMPOSISI.md.\n")

data, prev = [], None
out("| Langkah | Perbaikan yang ditambahkan | Akurasi | Selisih | AUC | EER |")
out("|---|---|---|---|---|---|")
for kode, nama, d, _ in TANGGA:
    m = baca(d)
    if m is None:
        out(f"| {kode} | {nama} | belum ada | | | |")
        continue
    sel = "" if prev is None else f"**{m['apm'] - prev:+.2f}**"
    out(f"| {kode} | {nama} | {m['apm']:.2f} | {sel} | {m['auc']:.4f} | "
        f"{m['eer']:.2f} |")
    data.append((kode, nama, m, None if prev is None else m["apm"] - prev))
    prev = m["apm"]
out("")

if len(data) >= 2:
    tot = data[-1][2]["apm"] - data[0][2]["apm"]
    out(f"Total kenaikan sepanjang tangga adalah {tot:+.2f} poin persentase, "
        f"dari {data[0][2]['apm']:.2f} persen menjadi {data[-1][2]['apm']:.2f} "
        "persen.\n")

    out("## Rincian tiap langkah\n")
    for kode, nama, m, sel in data:
        _, _, _, ket = next(t for t in TANGGA if t[0] == kode)
        if sel is None:
            out(f"**{kode}, {nama}.** Titik tolak, yaitu {ket}. Akurasi "
                f"{m['apm']:.2f} persen dengan AUC {m['auc']:.4f}.\n")
        else:
            arah = ("menaikkan" if sel > 0 else
                    "menurunkan" if sel < 0 else "tidak mengubah")
            out(f"**{kode}, {nama}.** Perubahan yang dilakukan adalah {ket}. "
                f"Langkah ini {arah} akurasi sebesar {abs(sel):.2f} poin "
                f"persentase menjadi {m['apm']:.2f} persen, dengan AUC "
                f"{m['auc']:.4f} dan EER {m['eer']:.2f} persen.\n")

    naik = [(k, n, s) for k, n, m, s in data if s is not None and s > 0]
    turun = [(k, n, s) for k, n, m, s in data if s is not None and s < 0]
    out("## Bacaan\n")
    if naik:
        b = max(naik, key=lambda t: t[2])
        out(f"Perbaikan yang paling banyak menyumbang adalah {b[1].lower()} "
            f"pada langkah {b[0]}, sebesar {b[2]:+.2f} poin persentase.\n")
    if turun:
        w = min(turun, key=lambda t: t[2])
        out(f"Tidak semua perbaikan berguna sendirian. Langkah {w[0]}, yaitu "
            f"{w[1].lower()}, justru {w[2]:.2f} poin persentase ketika "
            "diterapkan tanpa perbaikan lain. Langkah itu tetap dipertahankan "
            "dalam konfigurasi akhir karena bermanfaat dalam kombinasi, namun "
            "temuan negatifnya dilaporkan apa adanya di sini.\n")

open(os.path.join(HERE, "HASIL_ABLASI.md"), "w",
     encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_ABLASI.md")
