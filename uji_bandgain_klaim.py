"""
Menguji klaim generalisasi band-gain terhadap ragam antar inisialisasi.

Klaim mengenai band-gain dalam penelitian ini dinyatakan pada dua sumbu, yaitu
recall terhadap sistem text-to-speech generasi 2025 sampai 2026 dan recall
terhadap sistem lama yang tidak dikompresi MP3. Angka yang dilaporkan berupa
rerata atas tiga inisialisasi acak, namun selisihnya tidak pernah diuji terhadap
sebaran itu sendiri.

Skrip ini melakukan pengujian tersebut. Perlu dicatat di muka bahwa recall pada
sumbu-sumbu ini memiliki ragam yang jauh lebih besar daripada akurasi Fake-or-
Real, dalam beberapa kasus berpuluh poin persentase antar inisialisasi, sehingga
selisih belasan poin sekalipun belum tentu dapat dibedakan dari derau.
"""
import json
import os
import sys
from math import lgamma

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L = []
def out(s=""):
    print(s); L.append(s)

OLD_NONMP3 = ["tts_models_en_ljspeech_tacotron2-DDC",
              "tts_models_en_ljspeech_speedy-speech",
              "tts_models_en_ljspeech_vits"]
SEED = ["s42", "s1337", "s2024"]


def welch_p(a, b):
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
    lb = lgamma(aa) + lgamma(bb) - lgamma(aa + bb)
    front = np.exp(np.log(x) * aa + np.log(1 - x) * bb - lb) / aa
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


gp = os.path.join(HERE, "generations_results.json")
if not os.path.exists(gp):
    print("generations_results.json belum ada")
    sys.exit(0)
G = json.load(open(gp, encoding="utf-8"))


def ambil(pref):
    mod, old = [], []
    for s in SEED:
        k = f"{pref}_b16e10_{s}"
        if k not in G:
            continue
        r = G[k]["tts"]
        m = [v["recall"] for v in r.values()
             if v["era"] == "2025-2026 komersial"]
        o = [r[x]["recall"] for x in OLD_NONMP3 if x in r]
        if m:
            mod.append(np.mean(m) * 100)
        if o:
            old.append(np.mean(o) * 100)
    return np.array(mod), np.array(old)


out("# Apakah Klaim Generalisasi Band-Gain Bertahan?\n")
out("Klaim mengenai band-gain dinyatakan pada dua sumbu, yaitu recall terhadap "
    "sistem text-to-speech generasi 2025 sampai 2026 dan recall terhadap sistem "
    "lama yang tidak dikompresi MP3. Angka yang dilaporkan berupa rerata atas "
    "tiga inisialisasi acak, namun selisihnya belum pernah diuji terhadap "
    "sebaran itu sendiri. Tabel berikut melakukan pengujian tersebut dengan uji "
    "t Welch dan koreksi Holm-Bonferroni.\n")

uji = []
for arch in ["nes2net", "wavlm", "hubert"]:
    mb, ob = ambil(f"{arch}_official_fullbg")
    mn, on = ambil(f"{arch}_official_full")
    for nama, a, b in [("recall TTS 2025-2026", mb, mn),
                       ("recall TTS 2019 non-MP3", ob, on)]:
        if len(a) < 2 or len(b) < 2:
            continue
        uji.append({"arch": arch, "sumbu": nama, "n": f"{len(a)}/{len(b)}",
                    "dengan": a, "tanpa": b, "sel": a.mean() - b.mean(),
                    "p": welch_p(a, b)})
uji = [u for u in uji if u["p"] is not None]

if not uji:
    out("Belum ada pasangan yang dapat diuji.")
else:
    for rank, u in enumerate(sorted(uji, key=lambda x: x["p"])):
        u["ph"] = min(1.0, u["p"] * (len(uji) - rank))
    jl = 0.0
    for u in sorted(uji, key=lambda x: x["p"]):
        jl = max(jl, u["ph"])
        u["ph"] = jl

    out("| Arsitektur | Sumbu | n | Tanpa band-gain | Dengan band-gain | "
        "Selisih | p mentah | p Holm | Bacaan |")
    out("|---|---|---|---|---|---|---|---|---|")
    for u in uji:
        bacaan = ("melampaui ragam" if u["ph"] < 0.05 else
                  "di garis batas" if u["ph"] < 0.15 else
                  "belum terbukti berbeda")
        out(f"| {u['arch']} | {u['sumbu']} | {u['n']} | "
            f"{u['tanpa'].mean():.2f} ({u['tanpa'].std(ddof=1):.2f}) | "
            f"{u['dengan'].mean():.2f} ({u['dengan'].std(ddof=1):.2f}) | "
            f"{u['sel']:+.2f} | {u['p']:.4f} | {u['ph']:.4f} | **{bacaan}** |")
    out("")

    lolos = [u for u in uji if u["ph"] < 0.05]
    out("## Bacaan\n")
    if not lolos:
        out("Tidak satu pun klaim bertahan. Seluruh selisih, termasuk yang "
            "besarannya belasan sampai puluhan poin persentase, berada di dalam "
            "ragam antar inisialisasi acak.\n")
    out("Penyebabnya terlihat langsung pada kolom simpangan baku. Recall pada "
        "sumbu-sumbu ini jauh lebih tidak stabil daripada akurasi Fake-or-Real. "
        "Sebagai contoh, Nes2Net tanpa band-gain menghasilkan recall 93,8 dan "
        "96,9 dan 55,7 persen pada sistem lama non-MP3, sehingga selisih 10 poin "
        "persentase yang sempat dilaporkan sebagai keunggulan band-gain "
        "sebenarnya ditentukan hampir seluruhnya oleh satu inisialisasi yang "
        "buruk.\n")
    out("Konsekuensinya, klaim mengenai keunggulan generalisasi band-gain harus "
        "ditarik sebagai temuan dan dinyatakan ulang sebagai pengamatan yang "
        "belum diuji. Yang masih dapat dinyatakan adalah bahwa band-gain tidak "
        "merusak, sedangkan RawBoost menurunkan recall pada kedua sumbu dengan "
        "besaran yang lebih konsisten. Perbandingan itu pun perlu diuji dengan "
        "cara yang sama sebelum dipakai.\n")
    out("Sumbu recall menuntut jumlah inisialisasi yang jauh lebih banyak "
        "daripada tiga. Dengan simpangan baku belasan poin persentase, "
        "mendeteksi selisih 10 poin secara meyakinkan membutuhkan puluhan "
        "inisialisasi, dan itu di luar anggaran komputasi penelitian ini. "
        "Keterbatasan tersebut dilaporkan apa adanya.\n")

open(os.path.join(HERE, "HASIL_UJI_KLAIM_BANDGAIN.md"), "w",
     encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_UJI_KLAIM_BANDGAIN.md")
