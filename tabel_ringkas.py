"""
Tabel ringkas siap-kirim: seluruh angka kunci dalam satu tempat.

Dibuat untuk menjawab "kirim tabelnya" dengan angka yang dapat dipertanggungjawabkan,
bukan angka perkiraan.
"""
import glob
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


def parse(tag):
    m = re.match(r"^(.+?)_(official|random|clean_val|wavval)_([a-zA-Z]+?)"
                 r"(_b\d+e\d+)?_s(\d+)$", tag)
    # Konfigurasi ikut menjadi bagian kunci. Tanpa ini, run dengan jumlah epoch
    # atau batch yang berbeda tergabung menjadi satu kelompok dan sebarannya
    # terlaporkan sebagai ragam antar inisialisasi acak.
    return (None if not m else
            (m.group(1), m.group(2),
             m.group(3) + "@" + (m.group(4) or "_lama").lstrip("_"),
             m.group(5)))


# ---------- FoR ----------
FOR = defaultdict(list)
for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
    f = os.path.join(d, "test_scores.npy")
    p = parse(os.path.basename(d))
    if not (os.path.exists(f) and p):
        continue
    y, s, _ = np.load(f)
    y = y.astype(int)
    pm = full_metrics(y, s, prior_matched_threshold(s, 0.5))
    FOR[(p[0], p[1], p[2])].append(pm)


def agg(v, k):
    a = np.array([x[k] for x in v], dtype=float) * 100
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0), len(a)


# ---------- generasi TTS ----------
GEN = defaultdict(list)
gp = os.path.join(HERE, "generations_results.json")
if os.path.exists(gp):
    G = json.load(open(gp, encoding="utf-8"))
    for tag, r in G.items():
        p = parse(tag)
        if not p:
            continue
        md = [v["recall"] for v in r["tts"].values()
              if v["era"] == "2025-2026 komersial"]
        if md:
            GEN[(p[0], p[2])].append(np.mean(md) * 100)

# ---------- SNR ----------
SNR = defaultdict(dict)
sp = os.path.join(HERE, "snr_results.json")
if os.path.exists(sp):
    tmp = defaultdict(lambda: defaultdict(list))
    for r in json.load(open(sp, encoding="utf-8")):
        tmp[r["arch"]][r["snr"]].append(r["acc_pm"] * 100)
    for a, d in tmp.items():
        SNR[a] = {k: float(np.mean(v)) for k, v in d.items()}

out("# Tabel Ringkas Hasil (angka terukur, bukan perkiraan)\n")
out("Seluruh angka berasal dari berkas hasil di repositori dan dapat "
    "direproduksi. Format: rerata (simpangan baku) dalam persen, n = jumlah "
    "inisialisasi acak.\n")

# ---------- 1 ----------
out("## 1. Efek protokol pembagian data\n")
out("Dua baris pertama memakai model, data, dan hyperparameter yang identik, "
    "yaitu sepuluh epoch dengan batch tiga puluh dua, sehingga hanya cara "
    "pembagian datanya yang berbeda. Angka dilaporkan pada ambang prior-matched. "
    "Pada ambang tetap 0,5, keduanya menjadi 99,93 dan 50,03 persen, dan selisih "
    "yang jauh lebih besar itu berasal dari kalibrasi ambang, bukan dari "
    "protokol. Pemecahannya ada di HASIL_TEMUAN1.md.\n")
out("| Konfigurasi | Split | n | Akurasi | EER |")
out("|---|---|---|---|---|")
for key, nm in [(("cnn_asp", "random", "none@b32e10"), "CNN+ASP tanpa augmentasi"),
                (("cnn_asp", "official", "none@b32e10"), "CNN+ASP tanpa augmentasi")]:
    if key in FOR:
        m, s, n = agg(FOR[key], "accuracy")
        e, _, _ = agg(FOR[key], "eer")
        sp_ = "acak 60/20/20" if key[1] == "random" else "resmi FoR"
        out(f"| {nm} | {sp_} | {n} | **{m:.2f}** | {e:.2f} |")
# Replikasi proposal, dikelompokkan menurut model DAN skema pembagian data.
#
# BUG YANG DIPERBAIKI: pola glob lama menarik seluruh run bertag ULRPK lalu
# melabeli semuanya "acak 60/20/20", padahal sejak arm proposal juga dijalankan
# pada partisi resmi pola itu ikut menangkap keduanya. Akibatnya baris partisi
# resmi muncul di tabel sebagai hasil split acak, yang justru membalik makna
# tabel ini.
rep = defaultdict(lambda: {"acc": [], "eer": []})
for d in sorted(glob.glob(os.path.join(HERE, "runs", "*ULRPK*"))):
    fj, fs = os.path.join(d, "results.json"), os.path.join(d, "test_scores.npy")
    if not (os.path.exists(fj) and os.path.exists(fs)):
        continue
    r = json.load(open(fj, encoding="utf-8"))
    y, p, _ = np.load(fs)
    m05 = full_metrics(y.astype(int), p, 0.5)
    k = (r["args"]["model"], r["args"]["split"])
    rep[k]["acc"].append(m05["accuracy"] * 100)
    rep[k]["eer"].append(m05["eer"] * 100)

for (mdl, sp_), g in sorted(rep.items(), key=lambda t: (t[0][1], -np.mean(t[1]["acc"]))):
    n = len(g["acc"])
    sd = np.std(g["acc"], ddof=1) if n > 1 else 0.0
    lbl = "acak 60/20/20" if sp_ == "random" else "resmi FoR"
    nilai = (f"**{np.mean(g['acc']):.2f}** ({sd:.2f})" if n > 1
             else f"**{np.mean(g['acc']):.2f}**")
    out(f"| Replikasi proposal: {mdl} | {lbl} | {n} | {nilai} | "
        f"{np.mean(g['eer']):.2f} |")
out("")
out("Metrik replikasi proposal dilaporkan pada ambang 0,5 seperti yang tersirat "
    "di proposal. Angka dalam kurung adalah simpangan baku antar inisialisasi "
    "acak.\n")

# ---------- 2 ----------
out("## 2. Kinerja pada partisi resmi FoR\n")
out("| Arsitektur | Augmentasi | n | Akurasi | EER |")
out("|---|---|---|---|---|")
rows = []
for (a, sp_, au), v in FOR.items():
    if sp_ != "official" or len(v) < 3:
        continue
    m, s, n = agg(v, "accuracy")
    e, _, _ = agg(v, "eer")
    rows.append((m, a, au, n, s, e))
for m, a, au, n, s, e in sorted(rows, reverse=True)[:10]:
    out(f"| {a} | {au} | {n} | **{m:.2f}** ({s:.2f}) | {e:.2f} |")
out("")

# ---------- 3 ----------
out("## 3. Deteksi TTS komersial 2025-2026\n")
out("Diukur pada spesifisitas 95 persen yang disamakan untuk semua model, "
    "memakai 1.500 berkas asli In-the-Wild sebagai acuan ambang.\n")
out("| Arsitektur | Augmentasi | n | Recall TTS 2025-2026 |")
out("|---|---|---|---|")
for (a, au), v in sorted(GEN.items(), key=lambda kv: -np.mean(kv[1])):
    if len(v) < 3:
        continue
    arr = np.array(v)
    out(f"| {a} | {au} | {len(arr)} | **{arr.mean():.2f}** ({arr.std(ddof=1):.2f}) |")
out("")

# ---------- 4 ----------
out("## 4. Ketahanan terhadap noise\n")
out("Noise DEMAND, korpus yang tidak dipakai saat pelatihan.\n")
snrs = [None, 20, 10, 5, 0]
out("| Arsitektur dan augmentasi | " +
    " | ".join("bersih" if s is None else f"{s} dB" for s in snrs) + " |")
out("|" + "---|" * (len(snrs) + 1))
for a in sorted(SNR, key=lambda a: -SNR[a].get(0, 0)):
    cells = [f"{SNR[a][s]:.1f}" if s in SNR[a] else "n/a" for s in snrs]
    out(f"| {a} | " + " | ".join(cells) + " |")
out("")

# ---------- 5 ----------
out("## 5. Angka kunci untuk dikutip\n")
out("Hanya angka yang sudah lolos verifikasi yang dicantumkan di sini. Angka "
    "yang sempat dikutip namun kemudian ditarik didaftar terpisah di bawahnya, "
    "supaya tidak terpakai lagi tanpa sengaja.\n")
out("| Klaim | Angka | Sumber |")
out("|---|---|---|")
out("| Sampel palsu di data latih FoR yang berasal MP3 | 90,7 persen | probe_codec_report.md |")
out("| Sampel palsu di data uji FoR yang berasal MP3 | 0 persen | probe_codec_report.md |")
out("| Spesifisitas model SOTA publik di luar domain | 0,00 persen | HASIL_SOTA_COLLAPSE.md |")
out("| Selisih akibat ambang keputusan pada partisi resmi | 42,52 poin | HASIL_TEMUAN1.md |")
out("| Selisih akibat protokol split saja | 6,92 poin, p = 0,0822 | HASIL_TEMUAN1.md |")
out("| WavLM rekayasa lawan proposal, partisi resmi | +35,07 poin, p Holm 0,0002 | HASIL_SIGNIFIKANSI.md |")
out("| HuBERT rekayasa lawan proposal, partisi resmi | +44,56 poin, p Holm 0,0002 | HASIL_SIGNIFIKANSI.md |")
out("")
out("### Angka yang sudah ditarik, jangan dipakai lagi\n")
out("| Klaim yang ditarik | Alasan |")
out("|---|---|")
out("| Selisih protokol sekitar 50 poin | menggabungkan tiga sebab, protokol hanya 6,92 poin |")
out("| r = -0,542 antara akurasi FoR dan recall TTS modern | dihitung ulang menjadi -0,048 dengan p = 0,895 |")
out("| r = -0,980 hipotesis ceiling | n = 3 hanya punya enam permutasi sehingga p minimum 0,33 |")
out("| Band-gain memperbaiki recall TTS-2019 sebesar 10 poin | dua belas perbandingan, seluruhnya p Holm 1,0000 |")
out("")

open(os.path.join(HERE, "TABEL_RINGKAS.md"), "w", encoding="utf-8").write("\n".join(L))
print("\n-> TABEL_RINGKAS.md")
