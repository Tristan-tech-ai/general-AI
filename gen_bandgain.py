"""
Sapuan parameter augmentasi band-gain.

Band-gain adalah usulan orisinal penelitian ini, namun ketiga parameternya
ditetapkan sekali di awal berdasarkan penalaran mekanistik saja dan tidak pernah
diuji. Skrip ini melaporkan hasil sapuan satu variabel pada satu waktu, dengan
dua parameter lain dipertahankan pada nilai bawaannya.

Ketiga parameter mengendalikan keseimbangan yang menjadi dasar gagasannya, yaitu
menetralkan LEVEL energi pita tinggi tanpa merusak STRUKTUR HALUS di dalamnya:

  f_lo    dari frekuensi berapa penetralan dimulai
  n_bands sehalus apa penetralannya; semakin banyak pita semakin menyerupai
          penyaringan yang justru ikut merusak struktur
  db      sekuat apa level diacak

Sapuan dijalankan pada WavLM Large berencoder beku karena konfigurasi itu
memiliki simpangan baku terkecil dalam penelitian ini, yaitu 0,63 poin
persentase. Menyapu parameter pada arsitektur dengan ragam besar akan
menenggelamkan selisih antar parameter di dalam derau.
"""
import glob
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

DASAR = "wavlm_official_fullbg_b16e10_s*"
# Penanda pada tag: F untuk f_lo, N untuk jumlah pita, D untuk redaman maksimum.
POLA = re.compile(r"^wavlm_official_fullbg"
                  r"(?:F(?P<f>\d+))?(?:N(?P<n>\d+))?(?:D(?P<d>\d+))?"
                  r"_b16e10_s(?P<s>\d+)$")


def akurasi(d):
    f = os.path.join(d, "test_scores.npy")
    if not os.path.exists(f):
        return None
    y, p, _ = np.load(f)
    y = y.astype(int)
    m0 = full_metrics(y, p, 0.5)
    return {"acc": full_metrics(y, p, prior_matched_threshold(p, 0.5))["accuracy"] * 100,
            "auc": m0["auc"], "eer": m0["eer"] * 100}


kel, lewat = {}, []

# JANGKAR NOL. Sumbu redaman tidak dapat ditafsirkan tanpa titik nol, yaitu
# konfigurasi yang sama persis tetapi tanpa band-gain sama sekali. Tanpa jangkar
# ini, tren "semakin lembut semakin baik" akan terbaca seolah band-gain hanya
# merugikan dan sebaiknya dimatikan, padahal titik nolnya justru lebih rendah
# daripada beberapa titik bukan nol. Preset `full` adalah titik nol tersebut
# karena identik dengan `fullbg` kecuali band-gain-nya dimatikan.
for d in sorted(glob.glob(os.path.join(HERE, "runs",
                                       "wavlm_official_full_b16e10_s*"))):
    v = akurasi(d)
    if v is not None:
        kel.setdefault(("3000", "6", "0"), []).append(v)

for d in sorted(glob.glob(os.path.join(HERE, "runs", "wavlm_official_fullbg*"))):
    nama = os.path.basename(d)
    m = POLA.match(nama)
    if not m:
        if os.path.exists(os.path.join(d, "test_scores.npy")):
            lewat.append(nama)
        continue
    v = akurasi(d)
    if v is None:
        continue
    kunci = (m.group("f") or "3000", m.group("n") or "6", m.group("d") or "12")
    kel.setdefault(kunci, []).append(v)


def gab(v):
    a = np.array([x["acc"] for x in v])
    return {"acc": a.mean(), "sd": (a.std(ddof=1) if len(a) > 1 else 0.0),
            "n": len(a), "auc": np.mean([x["auc"] for x in v]),
            "eer": np.mean([x["eer"] for x in v])}


out("# Sapuan Parameter Augmentasi Band-Gain\n")
out("Seluruh sapuan dijalankan pada WavLM Large berencoder beku dengan preset "
    "augmentasi penuh ditambah band-gain, pada partisi resmi dan ambang "
    "prior-matched. Satu parameter diubah pada satu waktu, dua yang lain "
    "dipertahankan pada nilai bawaannya, yaitu f_lo 3000 Hz, enam pita, dan "
    "redaman sampai 12 dB.\n")
out("Konfigurasi ini dipilih karena simpangan bakunya terkecil di antara "
    "seluruh konfigurasi dalam penelitian ini. Menyapu parameter pada "
    "arsitektur dengan ragam besar akan menenggelamkan selisih antar parameter "
    "di dalam derau antar inisialisasi.\n")

dasar = kel.get(("3000", "6", "12"))
if not kel:
    out("Belum ada hasil.")
else:
    out("| f_lo (Hz) | jumlah pita | redaman maks (dB) | n | Akurasi | "
        "Selisih dari bawaan | AUC | EER |")
    out("|---|---|---|---|---|---|---|---|")
    b = gab(dasar) if dasar else None
    for kunci in sorted(kel, key=lambda k: (int(k[0]), int(k[1]), int(k[2]))):
        g = gab(kel[kunci])
        tanda = (" (bawaan)" if kunci == ("3000", "6", "12")
                 else " (tanpa band-gain)" if kunci[2] == "0" else "")
        sel = "" if b is None else f"{g['acc'] - b['acc']:+.2f}"
        akur = (f"{g['acc']:.2f} ({g['sd']:.2f})" if g["n"] > 1
                else f"{g['acc']:.2f}")
        out(f"| {kunci[0]}{tanda} | {kunci[1]} | {kunci[2]} | {g['n']} | "
            f"{akur} | {sel} | {g['auc']:.4f} | {g['eer']:.2f} |")
    out("")

    if b is not None and len(kel) > 1:
        lain = [(k, gab(v)) for k, v in kel.items() if k != ("3000", "6", "12")]
        terbaik = max(lain, key=lambda t: t[1]["acc"])
        out(f"Nilai bawaan mencapai {b['acc']:.2f} persen. Kombinasi terbaik "
            f"dalam sapuan ini adalah f_lo {terbaik[0][0]} Hz dengan "
            f"{terbaik[0][1]} pita dan redaman {terbaik[0][2]} dB, yaitu "
            f"{terbaik[1]['acc']:.2f} persen atau "
            f"{terbaik[1]['acc'] - b['acc']:+.2f} poin persentase.\n")
        out("Selisih itu belum dapat dinyatakan bermakna. Sapuan ini memakai "
            "satu inisialisasi acak per titik, sedangkan simpangan baku "
            f"konfigurasi bawaannya sendiri {b['sd']:.2f} poin persentase pada "
            f"{b['n']} inisialisasi. Titik terbaik perlu diulang dengan "
            "beberapa inisialisasi sebelum dapat dibandingkan secara sah, dan "
            "itu berlaku juga bila selisihnya tampak besar.\n")

if lewat:
    out("## Cakupan\n")
    out(f"Sebanyak {len(lewat)} run bertag fullbg tidak masuk sapuan ini karena "
        "nama tag-nya di luar pola yang ditangani. Jumlahnya dicatat supaya "
        "pelewatan tidak berlangsung tanpa diketahui.\n")
    out("Tag yang dilewati: " + ", ".join(f"`{t}`" for t in lewat[:8])
        + (f", dan {len(lewat) - 8} lainnya." if len(lewat) > 8 else "."))
    out("")

open(os.path.join(HERE, "HASIL_BANDGAIN.md"), "w",
     encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_BANDGAIN.md")
if lewat:
    print(f"   catatan: {len(lewat)} run fullbg dilewati pola tag")
