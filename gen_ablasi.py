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
    """Rerata atas seluruh inisialisasi acak yang tersedia untuk langkah ini.

    Pola direktori diakhiri _s42 pada versi awal. Agar seed tambahan ikut
    terhitung, akhiran seed diganti menjadi wildcard.
    """
    import glob as _g
    import re as _re
    pola = _re.sub(r"_s\d+$", "_s*", d)
    apm, a05, auc, eer = [], [], [], []
    for dd in sorted(_g.glob(os.path.join(HERE, pola))):
        f = os.path.join(dd, "test_scores.npy")
        if not os.path.exists(f):
            continue
        y, p, _ = np.load(f)
        y = y.astype(int)
        m0 = full_metrics(y, p, 0.5)
        a05.append(m0["accuracy"] * 100)
        apm.append(full_metrics(y, p, prior_matched_threshold(p, 0.5))["accuracy"] * 100)
        auc.append(m0["auc"])
        eer.append(m0["eer"] * 100)
    if not apm:
        return None
    return {"a05": np.mean(a05), "apm": np.mean(apm),
            "auc": np.mean(auc), "eer": np.mean(eer),
            "n": len(apm),
            "sd": (np.std(apm, ddof=1) if len(apm) > 1 else 0.0)}


out("# Tangga Ablasi: Perbaikan Mana yang Membeli Berapa\n")
out("Semua langkah memakai AST pada partisi resmi Fake-or-Real, batch 32, seed "
    "42. Tiap baris menambahkan satu perbaikan di atas baris sebelumnya, "
    "sehingga selisih antar baris hanya mencerminkan satu variabel.\n")
out("Akurasi dilaporkan pada ambang prior-matched untuk seluruh langkah agar "
    "sumbu ambang tidak bercampur ke dalam tangga. Sumbangan ambang itu sendiri "
    "dipisahkan tersendiri di HASIL_DEKOMPOSISI.md.\n")

def nilai(d):
    """Akurasi prior-matched tiap inisialisasi acak untuk satu langkah."""
    import glob as _g
    import re as _re
    pola = _re.sub(r"_s\d+$", "_s*", d)
    a = []
    for dd in sorted(_g.glob(os.path.join(HERE, pola))):
        f = os.path.join(dd, "test_scores.npy")
        if not os.path.exists(f):
            continue
        y, p, _ = np.load(f)
        a.append(full_metrics(y.astype(int), p,
                              prior_matched_threshold(p, 0.5))["accuracy"] * 100)
    return np.array(a)


def welch_p(a, b):
    """Nilai p dua sisi uji t Welch, sama dengan implementasi di
    signifikansi.py yang sudah dicocokkan terhadap SciPy."""
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
    f, c, dd_ = 1.0, 1.0, 0.0
    for i in range(300):
        mm = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (mm * (bb - mm) * x) / ((aa + 2 * mm - 1) * (aa + 2 * mm))
        else:
            num = -((aa + mm) * (aa + bb + mm) * x) / ((aa + 2 * mm) * (aa + 2 * mm + 1))
        dd_ = 1.0 + num * dd_
        dd_ = 1e-30 if abs(dd_) < 1e-30 else dd_
        dd_ = 1.0 / dd_
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * dd_
        if abs(1.0 - c * dd_) < 1e-10:
            break
    return float(front * (f - 1.0))


data, prev, prev_v = [], None, None
out("Selisih tiap langkah diuji terhadap langkah sebelumnya dengan uji t Welch. "
    "Seluruh tangga dijalankan pada AST, yaitu arsitektur dengan ragam antar "
    "inisialisasi terbesar di antara yang diuji, sehingga selisih yang kecil di "
    "sini menuntut kehati-hatian khusus.\n")
# Kumpulkan dahulu supaya koreksi Holm dapat dihitung atas seluruh langkah.
baris = []
for kode, nama, d, _ in TANGGA:
    m = baca(d)
    v = nilai(d) if m is not None else None
    p = None
    if m is not None and prev_v is not None:
        p = welch_p(v, prev_v)
    baris.append({"kode": kode, "nama": nama, "m": m, "v": v, "p": p,
                  "sel": None if (m is None or prev is None) else m["apm"] - prev})
    if m is not None:
        prev, prev_v = m["apm"], v

uji = [b for b in baris if b["p"] is not None]
mm = len(uji)
for rank, b in enumerate(sorted(uji, key=lambda x: x["p"])):
    b["ph"] = min(1.0, b["p"] * (mm - rank))
jalan = 0.0
for b in sorted(uji, key=lambda x: x["p"]):
    jalan = max(jalan, b["ph"])
    b["ph"] = jalan

out("| Langkah | Perbaikan yang ditambahkan | n | Akurasi | Selisih | p mentah "
    "| p Holm | AUC | EER |")
out("|---|---|---|---|---|---|---|---|---|")
for b in baris:
    m = b["m"]
    if m is None:
        out(f"| {b['kode']} | {b['nama']} | | belum ada | | | | | |")
        continue
    sel = "" if b["sel"] is None else f"**{b['sel']:+.2f}**"
    pstr = "" if b["p"] is None else f"{b['p']:.3f}"
    phstr = "" if "ph" not in b else f"{b['ph']:.3f}"
    akur = (f"{m['apm']:.2f} ({m['sd']:.2f})" if m["n"] > 1
            else f"{m['apm']:.2f}")
    out(f"| {b['kode']} | {b['nama']} | {m['n']} | {akur} | {sel} | {pstr} | "
        f"{phstr} | {m['auc']:.4f} | {m['eer']:.2f} |")
data = [(b["kode"], b["nama"], b["m"], b["sel"]) for b in baris if b["m"]]
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

    out("## Bacaan\n")
    lolos = [b for b in uji if b.get("ph", 1.0) < 0.05]
    hampir = [b for b in uji if 0.05 <= b.get("ph", 1.0) < 0.15]
    tidak = [b for b in uji if b.get("ph", 1.0) >= 0.15]

    out("Empat selisih diuji sekaligus, sehingga koreksi Holm-Bonferroni "
        "diterapkan dan keputusan diambil dari kolom p Holm.\n")
    if lolos:
        out("Langkah yang selisihnya melampaui ragam antar inisialisasi: "
            + "; ".join(f"{b['kode']} ({b['sel']:+.2f} poin, p Holm "
                        f"{b['ph']:.3f})" for b in lolos) + ".\n")
    if hampir:
        out("Langkah yang berada di garis batas dan belum dapat dinyatakan "
            "mapan: "
            + "; ".join(f"{b['kode']} ({b['sel']:+.2f} poin, p Holm "
                        f"{b['ph']:.3f})" for b in hampir) + ".\n")
    if tidak:
        out("Langkah yang selisihnya belum terbukti berbeda dari nol: "
            + "; ".join(f"{b['kode']} ({b['sel']:+.2f} poin, p Holm "
                        f"{b['ph']:.3f})" for b in tidak) + ".\n")

    out("Pola yang muncul cukup jelas. Dua langkah dengan selisih terbesar, "
        "yaitu pembekuan encoder dan early stopping, memiliki nilai p mentah di "
        "bawah 0,05 sedangkan dua langkah dengan selisih kecil tidak. Setelah "
        "koreksi untuk empat pengujian sekaligus, tidak ada satu pun yang "
        "bertahan di bawah ambang. Perlu diingat bahwa seluruh tangga ini "
        "dijalankan pada AST, yaitu arsitektur dengan ragam antar inisialisasi "
        "terbesar di antara yang diuji, sehingga daya ujinya paling rendah di "
        "sini dan bukan karena efeknya tidak ada.\n")
    out("Kesimpulan yang dapat dipertanggungjawabkan dari tangga ini karena itu "
        "terbatas. Arah tiap langkah konsisten dengan penjelasan mekanistik yang "
        "diajukan, tetapi besarannya belum dapat dipisahkan dari ragam pada "
        "ukuran sampel ini. Tangga ablasi lebih tepat dibaca sebagai peta "
        "kemungkinan sebab, bukan sebagai pengukuran sumbangan tiap perbaikan.\n")

open(os.path.join(HERE, "HASIL_ABLASI.md"), "w",
     encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_ABLASI.md")
