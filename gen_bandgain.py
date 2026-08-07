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


def welch_p(a, b):
    """Nilai p dua sisi uji t Welch, implementasi sama dengan signifikansi.py
    yang sudah dicocokkan terhadap SciPy sampai selisih 1e-10."""
    from math import lgamma
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb == 0:
        return None
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    x = df / (df + t * t)
    aa, bb = df / 2.0, 0.5
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = lgamma(aa) + lgamma(bb) - lgamma(aa + bb)
    front = np.exp(np.log(x) * aa + np.log(1 - x) * bb - lbeta) / aa
    f, c, d = 1.0, 1.0, 0.0
    for i in range(300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (bb - m) * x) / ((aa + 2 * m - 1) * (aa + 2 * m))
        else:
            num = -((aa + m) * (aa + bb + m) * x) / ((aa + 2 * m) * (aa + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-10:
            break
    return float(front * (f - 1.0))


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

    # Pengujian terhadap dua acuan sekaligus: konfigurasi bawaan dan titik nol.
    # Sebuah titik hanya berguna bila mengungguli keduanya. Mengungguli bawaan
    # saja tidak cukup, karena bisa jadi band-gain memang sebaiknya dilemahkan
    # sampai hampir tidak ada.
    nol = kel.get(("3000", "6", "0"))
    uji = []
    for k, v in kel.items():
        if k in (("3000", "6", "12"), ("3000", "6", "0")) or len(v) < 2:
            continue
        a = np.array([x["acc"] for x in v])
        for nama_acuan, acuan in [("bawaan 12 dB", dasar), ("tanpa band-gain", nol)]:
            if not acuan or len(acuan) < 2:
                continue
            c = np.array([x["acc"] for x in acuan])
            uji.append({"titik": f"f_lo {k[0]}, {k[1]} pita, {k[2]} dB",
                        "acuan": nama_acuan, "n": f"{len(a)}/{len(c)}",
                        "sel": a.mean() - c.mean(), "p": welch_p(a, c)})
    uji = [u for u in uji if u["p"] is not None]
    if uji:
        for rank, u in enumerate(sorted(uji, key=lambda x: x["p"])):
            u["ph"] = min(1.0, u["p"] * (len(uji) - rank))
        jl = 0.0
        for u in sorted(uji, key=lambda x: x["p"]):
            jl = max(jl, u["ph"])
            u["ph"] = jl
        out("## Pengujian terhadap dua acuan\n")
        out("Sebuah titik hanya berguna bila mengungguli konfigurasi bawaan dan "
            "titik tanpa band-gain sekaligus. Mengungguli bawaan saja tidak "
            "cukup, karena hal itu juga akan terjadi bila band-gain sebaiknya "
            "dilemahkan sampai hampir tidak ada.\n")
        out("| Titik | Acuan | n | Selisih | p mentah | p Holm | Bacaan |")
        out("|---|---|---|---|---|---|---|")
        for u in uji:
            bacaan = ("melampaui ragam" if u["ph"] < 0.05 else
                      "di garis batas" if u["ph"] < 0.15 else
                      "belum terbukti berbeda")
            out(f"| {u['titik']} | {u['acuan']} | {u['n']} | {u['sel']:+.2f} | "
                f"{u['p']:.4f} | {u['ph']:.4f} | **{bacaan}** |")
        out("")
    else:
        out("Belum ada titik selain bawaan yang memiliki lebih dari satu "
            "inisialisasi acak, sehingga belum ada yang dapat diuji. Sapuan "
            "dengan satu inisialisasi per titik hanya menunjukkan bentuk kurva, "
            "bukan besaran yang dapat dipertanggungjawabkan, dan itu berlaku "
            "juga bila selisihnya tampak besar.\n")

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
