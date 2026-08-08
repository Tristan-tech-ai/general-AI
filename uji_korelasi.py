"""
Menguji klaim korelasi dalam penelitian ini terhadap ukuran sampelnya.

Dua korelasi dilaporkan sebagai temuan:

  r = -0,542 (n = 7)   antara akurasi pada Fake-or-Real dan recall terhadap
                       sistem text-to-speech generasi terbaru
  r = -0,980 (n = 3)   antara ruang perbaikan yang tersisa dan besarnya manfaat
                       band-gain, yaitu hipotesis ceiling

Keduanya dilaporkan tanpa nilai p maupun selang kepercayaan. Pada ukuran sampel
sekecil itu, koefisien korelasi memiliki sebaran yang sangat lebar, sehingga
nilai yang tampak besar sekalipun dapat muncul dari data tanpa hubungan apa pun.

Skrip ini menghitung nilai p dua sisi lewat uji permutasi, yang tidak
mengandaikan kenormalan, beserta selang kepercayaan bootstrap. Uji permutasi
dipilih karena pada n kecil pendekatan berbasis distribusi t untuk koefisien
korelasi tidak dapat diandalkan.
"""
import itertools
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.metrics import full_metrics, prior_matched_threshold

HERE = os.path.dirname(os.path.abspath(__file__))
L = []
def out(s=""):
    print(s); L.append(s)


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    d = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    return float((xc * yc).sum() / d) if d > 0 else 0.0


def p_permutasi(x, y):
    """Nilai p dua sisi lewat permutasi lengkap bila n kecil.

    Untuk n sampai 8, seluruh n! permutasi dapat dicacah sehingga nilai p yang
    dihasilkan eksak dan tidak bergantung pada penarikan acak.
    """
    n = len(x)
    r0 = abs(pearson(x, y))
    if n <= 8:
        semua = [abs(pearson(x, list(p))) for p in itertools.permutations(y)]
        return float(np.mean([a >= r0 - 1e-12 for a in semua])), len(semua), True
    rng = np.random.default_rng(0)
    tot = 20000
    c = sum(abs(pearson(x, rng.permutation(y))) >= r0 - 1e-12
            for _ in range(tot))
    return float((c + 1) / (tot + 1)), tot, False


def ci_bootstrap(x, y, n_boot=20000):
    rng = np.random.default_rng(0)
    n = len(x)
    x, y = np.asarray(x, float), np.asarray(y, float)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(idx.tolist())) < 3:
            continue
        rs.append(pearson(x[idx], y[idx]))
    if len(rs) < 100:
        return None
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def parse(tag):
    m = re.match(r"^(.+?)_official_([a-zA-Z0-9.]+?)(_b\d+e\d+)?_s(\d+)$", tag)
    # Konfigurasi ikut menjadi bagian kunci supaya titik data pada korelasi
    # tidak mencampur run dengan jumlah epoch yang berbeda.
    return (None if not m else
            (m.group(1), m.group(2) + "@" + (m.group(3) or "_lama").lstrip("_"),
             m.group(4)))


out("# Apakah Korelasi yang Dilaporkan Bertahan pada Ukuran Sampelnya?\n")
out("Dua korelasi dilaporkan dalam penelitian ini tanpa nilai p maupun selang "
    "kepercayaan. Pada ukuran sampel kecil, koefisien korelasi memiliki sebaran "
    "yang sangat lebar sehingga nilai besar sekalipun dapat muncul dari data "
    "yang sebenarnya tidak berhubungan. Nilai p di bawah dihitung dengan uji "
    "permutasi lengkap, yang eksak untuk ukuran sampel sekecil ini dan tidak "
    "mengandaikan kenormalan. Selang kepercayaan dihitung dengan bootstrap "
    "persentil.\n")

gp = os.path.join(HERE, "generations_results.json")
if not os.path.exists(gp):
    out("generations_results.json belum ada.")
else:
    G = json.load(open(gp, encoding="utf-8"))
    # Akurasi FoR partisi resmi per run, dipasangkan dengan recall TTS modern.
    acc = {}
    for d in sorted(os.listdir(os.path.join(HERE, "runs"))):
        f = os.path.join(HERE, "runs", d, "test_scores.npy")
        if not os.path.exists(f):
            continue
        y, p, _ = np.load(f)
        if len(y) != 1088:
            continue
        acc[d] = full_metrics(y.astype(int), p,
                              prior_matched_threshold(p, 0.5))["accuracy"] * 100

    # Dikelompokkan per (arsitektur, augmentasi) lalu dirata-ratakan antar seed,
    # supaya tiap titik data mewakili satu konfigurasi dan bukan satu run.
    ga, gm = defaultdict(list), defaultdict(list)
    for tag, r in G.items():
        pr = parse(tag)
        if not pr or tag not in acc:
            continue
        mod = [v["recall"] for v in r["tts"].values()
               if v["era"] == "2025-2026 komersial"]
        if not mod:
            continue
        ga[(pr[0], pr[1])].append(acc[tag])
        gm[(pr[0], pr[1])].append(np.mean(mod) * 100)

    kunci = sorted(k for k in ga if len(ga[k]) >= 1)
    if len(kunci) < 4:
        out("Titik data belum cukup untuk menguji korelasi.")
    else:
        xs = [float(np.mean(ga[k])) for k in kunci]
        ys = [float(np.mean(gm[k])) for k in kunci]
        r = pearson(xs, ys)
        p, n_perm, eksak = p_permutasi(xs, ys)
        ci = ci_bootstrap(xs, ys)
        out("## Akurasi Fake-or-Real terhadap recall TTS generasi terbaru\n")
        out(f"Titik data: {len(xs)} konfigurasi, yaitu pasangan arsitektur dan "
            "strategi augmentasi, masing-masing dirata-ratakan atas "
            "inisialisasi acak yang tersedia.\n")
        out("| Konfigurasi | Akurasi FoR | Recall TTS 2025-2026 |")
        out("|---|---|---|")
        for k, a, m in sorted(zip(kunci, xs, ys), key=lambda t: -t[1]):
            out(f"| {k[0]} + {k[1]} | {a:.2f} | {m:.2f} |")
        out("")
        out(f"Koefisien korelasi Pearson r = {r:.3f} dengan n = {len(xs)}. "
            f"Nilai p dua sisi dari uji permutasi "
            f"{'lengkap atas ' + format(n_perm, ',').replace(',', '.') + ' permutasi' if eksak else 'acak'} "
            f"adalah {p:.4f}.")
        if ci:
            out(f"Selang kepercayaan bootstrap 95 persen membentang dari "
                f"{ci[0]:.3f} sampai {ci[1]:.3f}.")
        out("")
        if ci and ci[0] < 0 < ci[1]:
            out("Selang kepercayaannya memuat nol. Arah hubungan karena itu "
                "belum dapat ditetapkan dari data ini saja, sekalipun koefisien "
                "titiknya bertanda negatif.\n")
        elif p < 0.05:
            out("Korelasi ini bertahan pada tingkat lima persen.\n")
        else:
            out("Korelasi ini belum terbukti berbeda dari nol.\n")

        out("Perlu ditegaskan bahwa yang menopang temuan mengenai hubungan "
            "terbalik ini bukan koefisien korelasinya, melainkan mekanismenya "
            "yang terdokumentasi secara terpisah, yaitu audit kebocoran codec "
            "pada dataset, eksperimen augmentasi terkontrol, dan pola kebutaan "
            "model terhadap sistem yang tidak dikompresi. Koefisien korelasi "
            "pada ukuran sampel sekecil ini sebaiknya dibaca sebagai ringkasan "
            "deskriptif, bukan sebagai bukti.\n")

open(os.path.join(HERE, "HASIL_UJI_KORELASI.md"), "w",
     encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_UJI_KORELASI.md")
