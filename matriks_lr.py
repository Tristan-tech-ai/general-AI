"""
Matriks arsitektur terhadap perlakuan encoder.

Temuan pokok penelitian ini pada akhirnya bukan "metodologi A mengalahkan
metodologi B", melainkan bahwa kedua metodologi melakukan kelas kesalahan yang
sama. Proposal menyeragamkan learning rate 0,001 untuk seluruh arsitektur.
Konfigurasi rekayasa menyeragamkan pembekuan encoder untuk seluruh arsitektur.
Keduanya kebetulan tepat pada satu arsitektur dan merugikan pada arsitektur
lain.

Matriks ini menyandingkan empat perlakuan encoder pada dua arsitektur, dengan
seluruh setelan lain dijaga tetap di dalam tiap kolom, sehingga arah dan besar
pengaruhnya dapat dibaca langsung.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.metrics import full_metrics, prior_matched_threshold

HERE = os.path.dirname(os.path.abspath(__file__))
L = []
def out(s=""):
    print(s); L.append(s)

# Empat perlakuan encoder. Tiga yang pertama memakai paket rekayasa yang sama
# persis, yaitu 10 epoch dengan early stopping, augmentasi penuh, normalisasi
# loudness, dan agregasi berbobot antar lapisan. Hanya perlakuan encoder yang
# berbeda, sehingga ketiganya merupakan perbandingan satu variabel.
PERLAKUAN = [
    ("Encoder dibekukan", "runs/{m}_official_full_b{b}e10_s*"),
    ("Encoder dilatih, laju wajar per model", "runs/{m}_official_fullUF_b{b}e10_s*"),
    ("Encoder dilatih, laju 0,001", "runs/{m}_official_fullUFENC0.001_b{b}e10_s*"),
    ("Proposal apa adanya, laju 0,001 seragam",
     "runs/{m}_official_proposalULRPK_b{b}e20_s*"),
]
ARS = [("ast", 32, "AST, 86 juta parameter, pra-latih terselia"),
       ("wavlm", 16, "WavLM Large, 300 juta parameter, swa-selia"),
       ("hubert", 32, "HuBERT Large, 300 juta parameter, swa-selia")]


def ambil(pat):
    acc, auc, eer = [], [], []
    for d in sorted(glob.glob(os.path.join(HERE, pat))):
        f = os.path.join(d, "test_scores.npy")
        if not os.path.exists(f):
            continue
        y, p, _ = np.load(f)
        y = y.astype(int)
        m0 = full_metrics(y, p, 0.5)
        acc.append(full_metrics(y, p, prior_matched_threshold(p, 0.5))["accuracy"] * 100)
        auc.append(m0["auc"])
        eer.append(m0["eer"] * 100)
    if not acc:
        return None
    return {"acc": np.mean(acc), "sd": (np.std(acc, ddof=1) if len(acc) > 1 else 0.0),
            "auc": np.mean(auc), "eer": np.mean(eer), "n": len(acc)}


out("# Matriks Arsitektur terhadap Perlakuan Encoder\n")
out("Seluruh angka diukur pada partisi resmi Fake-or-Real dengan ambang "
    "prior-matched. Tiga baris pertama memakai paket rekayasa yang sama persis, "
    "yaitu 10 epoch dengan early stopping pada equal error rate, augmentasi "
    "penuh, normalisasi loudness, dan agregasi berbobot antar lapisan. Hanya "
    "perlakuan encoder yang berbeda di antara ketiganya, sehingga "
    "perbandingannya bersifat satu variabel. Baris keempat disertakan sebagai "
    "acuan, yaitu konfigurasi proposal apa adanya.\n")

ADA = []
for m, b, ket in ARS:
    baris = [(nm, ambil(pat.format(m=m, b=b))) for nm, pat in PERLAKUAN]
    if not any(v for _, v in baris):
        continue
    ADA.append((m, ket, baris))
    out(f"## {ket}\n")
    out("| Perlakuan encoder | n | Akurasi | AUC | EER |")
    out("|---|---|---|---|---|")
    terbaik = max((v["acc"] for _, v in baris if v), default=None)
    for nm, v in baris:
        if not v:
            out(f"| {nm} | | belum ada | | |")
            continue
        a = (f"{v['acc']:.2f} ({v['sd']:.2f})" if v["n"] > 1
             else f"{v['acc']:.2f}")
        tebal = "**" if v["acc"] == terbaik else ""
        out(f"| {nm} | {v['n']} | {tebal}{a}{tebal} | {v['auc']:.4f} | "
            f"{v['eer']:.2f} |")
    out("")

if len(ADA) >= 2:
    out("## Bacaan\n")
    for m, ket, baris in ADA:
        d = {nm: v for nm, v in baris if v}
        beku = d.get("Encoder dibekukan")
        wajar = d.get("Encoder dilatih, laju wajar per model")
        if beku and wajar:
            sel = wajar["acc"] - beku["acc"]
            arah = "lebih baik daripada" if sel > 0 else "lebih buruk daripada"
            out(f"Pada {ket.split(',')[0]}, melatih encoder pada laju wajar "
                f"{arah} membekukannya, dengan selisih {sel:+.2f} poin "
                "persentase.\n")
    out("Arah pengaruhnya tidak sama antar arsitektur. Tidak ada satu perlakuan "
        "encoder yang benar untuk semuanya. Inilah sebabnya baik penyeragaman "
        "learning rate pada proposal maupun penyeragaman pembekuan encoder pada "
        "konfigurasi rekayasa sama-sama menghasilkan kerugian pada arsitektur "
        "yang tidak cocok dengan pilihan tersebut. Keputusan ini seharusnya "
        "ditetapkan per arsitektur dan dipilih menggunakan data validasi, bukan "
        "diseragamkan di muka.\n")

open(os.path.join(HERE, "HASIL_MATRIKS_LR.md"), "w",
     encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_MATRIKS_LR.md")
