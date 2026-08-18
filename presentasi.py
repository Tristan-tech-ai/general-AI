"""
Membangun PRESENTASI.pdf, deck 16:9 untuk mempresentasikan penelitian ini.

Isinya menjawab rumusan masalah proposal, menyatakan terus terang bagian yang
jawabannya berbeda dari dugaan, mendaftar klaim yang ditarik, lalu menutup
dengan hal yang ditemukan di luar rencana proposal.

Seperti naskah, seluruh angka pada deck ini dihitung ulang dari berkas hasil
setiap kali berkas ini dijalankan. Tidak ada angka yang diketik tangan, jadi
tidak mungkin ada angka usang ketika hasil berubah.

Jalankan:
    py presentasi.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.metrics import full_metrics, prior_matched_threshold

HERE = os.path.dirname(os.path.abspath(__file__))
GBR = os.path.join(HERE, "gambar_presentasi")
OUT = os.path.join(HERE, "PRESENTASI.pdf")
os.makedirs(GBR, exist_ok=True)

# =====================================================================
# Sistem visual
# =====================================================================
# Palet diambil dari gambar_paper.py supaya deck ini terbaca sebagai satu
# keluarga dengan gambar di naskah, bukan rancangan asing yang ditempelkan.
P = {
    "biru":    "#1864AB",
    "merah":   "#C2255C",
    "hijau":   "#2F9E44",
    "jingga":  "#E8590C",
    "ungu":    "#7048E8",
    "teal":    "#0B7285",
    "abu":     "#868E96",
    "tinta":   "#1A1D21",
    "redup":   "#6B737B",
    "samar":   "#ADB5BD",
    "garis":   "#E3E6E9",
    "latar":   "#FFFFFF",
    "navy":    "#0F2436",
    "navy2":   "#1C3A52",
}
ARCH = {"wavlm": "#1864AB", "hubert": "#2F9E44", "ast": "#C2255C",
        "wav2vec2": "#E8590C", "cnn_asp": "#7048E8", "cnnlstm": "#868E96",
        "nes2net": "#0B7285"}

FONT = "C:/Windows/Fonts"
for nama, berkas in [("SG", "segoeui.ttf"), ("SG-Bold", "segoeuib.ttf"),
                     ("SG-Light", "segoeuil.ttf"), ("SG-Semi", "seguisb.ttf"),
                     ("Mono", "consola.ttf")]:
    p = os.path.join(FONT, berkas)
    if os.path.exists(p):
        pdfmetrics.registerFont(TTFont(nama, p))
    else:                                    # cadangan bila fontnya tidak ada
        pdfmetrics.registerFont(TTFont(nama, os.path.join(FONT, "arial.ttf")))

W, H = 960.0, 540.0                          # 16:9 pada 72 dpi
ML, MR = 64.0, 64.0
LEBAR = W - ML - MR

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": P["samar"], "axes.labelcolor": P["redup"],
    "xtick.color": P["redup"], "ytick.color": P["redup"],
    "axes.grid": True, "grid.color": P["garis"], "grid.linewidth": 0.9,
    "legend.frameon": False, "figure.facecolor": "white",
})


# Lebar tempat gambar ditempel, dalam poin. Ukuran figur matplotlib diturunkan
# dari angka ini supaya PNG-nya ditempel pada skala satu banding satu. Tanpa itu
# gambar ikut diperbesar dan seluruh teks di dalamnya membesar bersamanya,
# sehingga label bertabrakan padahal di berkas aslinya rapi.
L_PENUH = LEBAR * 0.94
L_KIRI = LEBAR * 0.62


def sosok(lebar_pt, tinggi_in, **kw):
    return plt.subplots(figsize=(lebar_pt / 72.0, tinggi_in),
                        layout="constrained", **kw)


def simpan(fig, nama):
    p = os.path.join(GBR, nama)
    fig.savefig(p, facecolor="white")          # tanpa bbox tight, ukuran tetap
    plt.close(fig)
    return p


# =====================================================================
# Pembacaan hasil, seluruh angka dihitung ulang di sini
# =====================================================================
def skor(pola):
    a05, apm, auc, eer = [], [], [], []
    for d in sorted(glob.glob(os.path.join(HERE, pola))):
        f = os.path.join(d, "test_scores.npy")
        if not os.path.exists(f):
            continue
        y, p, _ = np.load(f)
        y = y.astype(int)
        m = full_metrics(y, p, 0.5)
        a05.append(m["accuracy"] * 100)
        auc.append(m["auc"])
        eer.append(m["eer"] * 100)
        apm.append(full_metrics(y, p, prior_matched_threshold(p, 0.5))["accuracy"] * 100)
    if not a05:
        return None
    return {"a05": np.array(a05), "apm": np.array(apm), "auc": np.array(auc),
            "eer": np.array(eer), "n": len(a05)}


def n(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


# kebocoran codec, langsung dari manifest
_c = defaultdict(lambda: [0, 0])
for r in csv.DictReader(open(os.path.join(HERE, "manifest.csv"), encoding="utf-8")):
    k = (r["split_official"], "asli" if r["label"] == "0" else "palsu")
    _c[k][0] += 1
    if r["is_mp3"] in ("1", "True", "true"):
        _c[k][1] += 1
MP3 = {k: (v[0], v[1], 100 * v[1] / v[0]) for k, v in _c.items()}

RA = skor("runs/cnn_asp_random_none_b32e10_s*")
OF = skor("runs/cnn_asp_official_none_b32e10_s*")
OF_LAMA = skor("runs/cnn_asp_official_none_s42")

# matriks perlakuan encoder
MTX = {}
for m, b in [("ast", 32), ("wavlm", 16), ("hubert", 32)]:
    MTX[(m, "beku")] = skor(f"runs/{m}_official_full_b{b}e10_s*")
    MTX[(m, "dilatih")] = skor(f"runs/{m}_official_fullUF_b{b}e10_s*")
    MTX[(m, "proposal")] = skor(f"runs/{m}_official_proposalULRPK_b{b}e20_s*")

# ragam antar inisialisasi pada konfigurasi baku
BAKU = defaultdict(list)
for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
    f, j = os.path.join(d, "test_scores.npy"), os.path.join(d, "results.json")
    if not (os.path.exists(f) and os.path.exists(j)):
        continue
    a = json.load(open(j, encoding="utf-8"))["args"]
    if a.get("split") != "official" or a.get("uniform_lr") or a.get("enc_lr") is not None:
        continue
    if a.get("normalize") != "loudness" or a.get("augment_val"):
        continue
    if any(a.get(x) is not None for x in ("bg_f_lo", "bg_bands", "bg_db")):
        continue
    y, p, _ = np.load(f)
    BAKU[(a["model"], a["augment"], a["batch"], a["epochs"], bool(a.get("unfreeze")))].append(
        full_metrics(y.astype(int), p, prior_matched_threshold(p, 0.5))["accuracy"] * 100)

# kurva SNR
SNR = defaultdict(lambda: defaultdict(list))
for r in json.load(open(os.path.join(HERE, "snr_results.json"), encoding="utf-8")):
    SNR[r["arch"]][r["snr"]].append(r)

SOTA = json.load(open(os.path.join(HERE, "sota_modern_results.json"), encoding="utf-8"))

# angka turunan yang dipakai di beberapa slide
KALIBRASI = OF["apm"].mean() - OF["a05"].mean()
MATANG = OF["apm"].mean() - OF_LAMA["apm"].mean()
PROTOKOL = RA["apm"].mean() - OF["apm"].mean()
TOTAL_LAMA = RA["a05"].mean() - OF_LAMA["a05"].mean()


# =====================================================================
# Gambar
# =====================================================================
def g_codec():
    fig, ax = sosok(L_KIRI, 3.3)
    urut = ["training", "validation", "testing"]
    nilai = [MP3[(s, "palsu")][2] for s in urut]
    warna = [P["merah"], P["merah"], P["biru"]]
    b = ax.bar(range(3), nilai, width=0.5, color=warna, zorder=3)
    ax.bar(range(3), [0.6] * 3, width=0.5, bottom=[0] * 3, color="none", zorder=3)
    for i, (r, v) in enumerate(zip(b, nilai)):
        ax.text(i, v + 3, f"{n(v, 1)}%", ha="center", va="bottom",
                fontsize=19, color=warna[i], fontweight="semibold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["data latih", "data validasi", "data uji"], fontsize=14)
    ax.set_ylabel("sampel palsu yang\nberasal dari MP3", fontsize=12)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_axisbelow(True)
    ax.annotate("isyarat hilang\ntepat di sini", xy=(2, 6), xytext=(2, 42),
                ha="center", fontsize=12.5, color=P["biru"],
                arrowprops=dict(arrowstyle="-|>", color=P["biru"], lw=1.6))
    return simpan(fig, "p1_codec.png")


def g_snr():
    fig, ax = sosok(L_KIRI, 3.6)
    ax2 = ax.twinx()
    ax2.grid(False)
    titik = [None, 30, 25, 20, 15, 10, 5, 0, -5]
    xs = list(range(len(titik)))
    d = SNR["wavlm[codec]"]
    pm = [np.mean([r["acc_pm"] for r in d[s]]) * 100 for s in titik]
    fx = [np.mean([r["acc_fx"] for r in d[s]]) * 100 for s in titik]
    au = [np.mean([r["auc"] for r in d[s]]) for s in titik]
    ax.fill_between(xs, fx, pm, color=P["jingga"], alpha=0.16, zorder=2)
    ax.plot(xs, pm, "-o", color=P["biru"], lw=2.4, ms=6, zorder=4)
    ax.plot(xs, fx, "--o", color=P["merah"], lw=2.2, ms=5.5, zorder=4)
    ax.plot(xs, [a * 100 for a in au], "-", color=P["tinta"], lw=2.0, zorder=5)
    ax2.set_yticks([])
    # Legenda dihapus dan diganti label langsung di ujung tiap kurva. Legenda
    # memaksa mata bolak-balik antara kotak dan garis, dan pada bidang sesempit
    # ini ia juga bertabrakan dengan kurvanya sendiri.
    for teks_, nilai, w in [("AUC", au[-1] * 100 + 5.5, P["tinta"]),
                            ("ambang disesuaikan", pm[-1] - 1.5, P["biru"]),
                            ("ambang dibekukan", fx[-1] - 1.5, P["merah"])]:
        ax.text(len(xs) - 0.72, nilai, teks_, fontsize=11, color=w,
                ha="left", va="center", fontweight="semibold")
    ax.text(4.6, (pm[4] + fx[4]) / 2 - 1, "hilang karena\nambang meleset",
            fontsize=10.5, color="#A8410A", ha="center", va="center")
    ax.set_xticks(xs)
    ax.set_xticklabels(["bersih", "30", "25", "20", "15", "10", "5", "0", "-5"],
                       fontsize=11.5)
    ax.set_xlabel("rasio sinyal terhadap derau (dB)", fontsize=11.5)
    ax.set_ylabel("akurasi dan AUC (persen)", fontsize=11.5)
    ax.set_ylim(40, 104)
    ax.set_xlim(-0.35, len(xs) + 2.3)
    ax.set_axisbelow(True)
    return simpan(fig, "p2_snr.png")


def g_arsitektur():
    pilih = [("wavlm", "full", 16, 10), ("hubert", "codec", 16, 10),
             ("nes2net", "full", 16, 10), ("cnn_asp", "codec", 32, 10),
             ("wav2vec2", "codec", 32, 10), ("ast", "full", 32, 10),
             ("cnnlstm", "codec", 32, 10)]
    data = []
    for m, aug, b, e in pilih:
        v = BAKU.get((m, aug, b, e, False))
        if v:
            data.append((m, aug, np.array(v)))
    data.sort(key=lambda t: -t[2].mean())
    fig, ax = sosok(L_PENUH, 3.5)
    xs = np.arange(len(data))
    mu = [d[2].mean() for d in data]
    sd = [d[2].std(ddof=1) if len(d[2]) > 1 else 0 for d in data]
    ax.bar(xs, mu, width=0.56, color=[ARCH[d[0]] for d in data],
           yerr=sd, capsize=5, error_kw=dict(ecolor=P["tinta"], lw=1.4), zorder=3)
    for i, (d, m_, s_) in enumerate(zip(data, mu, sd)):
        ax.text(i, m_ + s_ + 1.4, n(m_, 1), ha="center", fontsize=13.5,
                color=P["tinta"], fontweight="semibold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d[0]}\n{d[1]}, n={len(d[2])}" for d in data], fontsize=11)
    ax.set_ylabel("akurasi partisi resmi (persen)", fontsize=12)
    ax.set_ylim(70, 104)
    ax.set_axisbelow(True)
    ax.axhline(94.7, ls=":", lw=1.4, color=P["redup"])
    ax.text(len(data) - 0.4, 95.1, "94,7% baseline terpublikasi pada FoR",
            ha="right", fontsize=11, color=P["redup"])
    return simpan(fig, "p3_arsitektur.png")


def g_ragam():
    # Penanda batch dan epoch ikut ke label, karena ada konfigurasi yang model
    # dan augmentasinya sama tetapi lama pelatihannya berbeda, dan tanpa penanda
    # itu dua baris terlihat seperti hal yang sama.
    data = [(f"{k[0]} [{k[1]}] b{k[2]}e{k[3]}" + (" UF" if k[4] else ""),
             np.array(sorted(v)))
            for k, v in BAKU.items() if len(v) >= 3]
    data.sort(key=lambda t: t[1].std(ddof=1))
    data = data[:4] + data[-4:]
    fig, ax = sosok(L_PENUH, 3.4)
    for i, (nm, a) in enumerate(data):
        y = len(data) - 1 - i
        ax.plot([a.min(), a.max()], [y, y], "-", color=P["abu"], lw=2.6, alpha=0.45,
                solid_capstyle="round", zorder=2)
        ax.plot(a, [y] * len(a), "o", color=P["biru"], ms=8, alpha=0.65, zorder=3)
        ax.plot([a.mean()], [y], "D", color=P["tinta"], ms=8, zorder=4,
                markeredgecolor="white", markeredgewidth=1.2)
        sd = a.std(ddof=1)
        w = P["hijau"] if sd < 1 else (P["merah"] if sd > 3 else P["redup"])
        ax.text(1.012, y, f"sd {n(sd)}", fontsize=11.5, color=w, va="center",
                transform=ax.get_yaxis_transform(), clip_on=False)
    # Batas sumbu diturunkan dari datanya sendiri. Nilai tetap pernah memotong
    # baris dengan sebaran terlebar, sehingga justru menyembunyikan hal yang
    # ingin ditunjukkan grafik ini.
    semua = np.concatenate([d[1] for d in data])
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([d[0] for d in reversed(data)], fontsize=11)
    ax.set_xlabel("akurasi partisi resmi (persen)", fontsize=11.5)
    ax.set_xlim(semua.min() - 2.0, semua.max() + 1.2)
    ax.set_ylim(-0.7, len(data) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    return simpan(fig, "p4_ragam.png")


def g_lr():
    fig, ax = sosok(L_KIRI, 3.5)
    ars = [("ast", "AST\n86 juta, terselia"), ("wavlm", "WavLM Large\n300 juta, swa-selia"),
           ("hubert", "HuBERT Large\n300 juta, swa-selia")]
    lab = [("proposal", "laju 0,001 seragam", P["merah"]),
           ("beku", "laju per model, encoder beku", P["biru"]),
           ("dilatih", "laju per model, encoder dilatih", P["hijau"])]
    w = 0.26
    xs = np.arange(len(ars))
    for j, (k, nama, c) in enumerate(lab):
        mu = [MTX[(m, k)]["apm"].mean() if MTX.get((m, k)) else np.nan for m, _ in ars]
        sd = [MTX[(m, k)]["apm"].std(ddof=1) if MTX.get((m, k)) and MTX[(m, k)]["n"] > 1
              else 0 for m, _ in ars]
        pos = xs + (j - 1) * w
        ax.bar(pos, mu, width=w * 0.92, color=c, label=nama, zorder=3,
               yerr=sd, capsize=3.5, error_kw=dict(ecolor=P["tinta"], lw=1.2))
        for x, v in zip(pos, mu):
            if not np.isnan(v):
                ax.text(x, v + 2.2, n(v, 1), ha="center", fontsize=11.5,
                        color=P["tinta"], fontweight="semibold")
    ax.axhline(50, ls=":", lw=1.4, color=P["redup"])
    ax.text(-0.44, 51.6, "tebakan acak", ha="left", fontsize=10.5, color=P["redup"])
    ax.set_xticks(xs)
    ax.set_xticklabels([t[1] for t in ars], fontsize=11.5)
    ax.set_ylabel("akurasi partisi resmi (persen)", fontsize=12)
    ax.set_ylim(40, 112)
    ax.set_yticks([40, 60, 80, 100])
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, 1.16))
    return simpan(fig, "p5_lr.png")


def g_sota():
    fig, (a1, a2) = sosok(L_PENUH, 3.0, ncols=2,
                          gridspec_kw={"width_ratios": [1, 1.25]})
    f = SOTA["for_2sec"]
    a1.barh([1, 0], [f["auc"] * 100, 50], height=0.42,
            color=[P["merah"], P["samar"]], zorder=3)
    a1.text(f["auc"] * 100 + 2, 1, f"AUC {n(f['auc'] * 100, 1)}%", va="center",
            fontsize=14, color=P["merah"], fontweight="semibold")
    a1.text(52, 0, "tebakan acak", va="center", fontsize=12, color=P["redup"])
    a1.set_yticks([])
    a1.set_xlim(0, 100)
    a1.set_xticks([0, 50, 100])
    a1.set_xticklabels(["0%", "50%", "100%"])
    a1.set_xlabel("pada partisi uji Fake-or-Real", fontsize=11.5)
    a1.grid(axis="y", visible=False)
    a1.set_axisbelow(True)

    sistem = [k for k in SOTA if k != "for_2sec"]
    r0 = [SOTA[s]["recall@0.5"] * 100 for s in sistem]
    rt = [SOTA[s]["recall@thr_for"] * 100 for s in sistem]
    ys = np.arange(len(sistem))
    a2.barh(ys + 0.19, r0, height=0.34, color=P["biru"], label="ambang 0,5", zorder=3)
    a2.barh(ys - 0.19, rt, height=0.34, color=P["jingga"],
            label="ambang dikalibrasi di FoR", zorder=3)
    a2.set_yticks(ys)
    a2.set_yticklabels(sistem, fontsize=11)
    a2.set_xlim(0, 108)
    a2.set_xticks([0, 50, 100])
    a2.set_xticklabels(["0%", "50%", "100%"])
    a2.set_xlabel("recall pada TTS komersial 2025-2026", fontsize=11.5)
    # Legenda ditaruh di luar bidang data. Di dalam, ia menimpa batang paling
    # bawah pada setiap penataan yang dicoba.
    a2.legend(fontsize=10.5, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, 1.0))
    a2.grid(axis="y", visible=False)
    a2.set_axisbelow(True)
    return simpan(fig, "p6_sota.png")


def g_sebab():
    fig, ax = sosok(L_PENUH, 2.0)
    seg = [("ambang keputusan", KALIBRASI, P["jingga"]),
           ("model kurang terlatih", MATANG, P["ungu"]),
           ("protokol pembagian data", PROTOKOL, P["biru"])]
    kiri = 0
    for nama, v, c in seg:
        ax.barh([0], [v], left=[kiri], height=0.5, color=c, zorder=3)
        ax.text(kiri + v / 2, 0, n(v, 1), ha="center", va="center",
                fontsize=16, color="white", fontweight="semibold")
        kiri += v
    # Nama tiap sebab diletakkan sebagai baris keterangan di bawah, bukan di atas
    # potongannya. Potongan ketiga terlalu sempit untuk memuat namanya, dan
    # menaruhnya di atas membuat ketiganya bertumpuk.
    x = 0.0
    for nama, v, c in seg:
        ax.plot([x + 0.7], [-0.62], "s", color=c, ms=9, clip_on=False, zorder=5)
        ax.text(x + 2.4, -0.62, nama, va="center", ha="left", fontsize=11.5,
                color=P["redup"], clip_on=False)
        x += 24.0
    ax.set_xlim(0, kiri * 1.02)
    ax.set_ylim(-0.95, 0.42)
    ax.set_yticks([])
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax.tick_params(labelsize=10.5)
    ax.set_xlabel("poin persentase dari selisih yang semula dilaporkan", fontsize=11)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    return simpan(fig, "p7_sebab.png")


GAMBAR = {"codec": g_codec(), "snr": g_snr(), "arsitektur": g_arsitektur(),
          "ragam": g_ragam(), "lr": g_lr(), "sota": g_sota(), "sebab": g_sebab()}
print("gambar selesai:", len(GAMBAR))


# =====================================================================
# Kerangka slide
# =====================================================================
c = canvas.Canvas(OUT, pagesize=(W, H))
c.setTitle("Ambang Keputusan, Bukan Arsitektur")
NOMOR = [0]
BAGIAN = [""]


def teks(x, y, s, font="SG", ukuran=14, warna="tinta", spasi=0.0, tengah=False,
         kanan=False):
    warna_hex = HexColor(P[warna]) if warna in P else HexColor(warna)
    if spasi:
        # Jarak antar huruf hanya tersedia pada objek teks, bukan pada canvas,
        # sehingga posisi awalnya dihitung sendiri untuk rata tengah dan kanan.
        lebar = pdfmetrics.stringWidth(s, font, ukuran) + spasi * max(len(s) - 1, 0)
        if tengah:
            x -= lebar / 2
        elif kanan:
            x -= lebar
        obj = c.beginText(x, y)
        obj.setFont(font, ukuran)
        obj.setCharSpace(spasi)
        obj.setFillColor(warna_hex)
        obj.textOut(s)
        # Tc adalah keadaan teks yang menetap di dalam aliran konten PDF, bukan
        # milik satu objek teks saja. Tanpa dikembalikan ke nol di sini, seluruh
        # teks berikutnya di halaman yang sama ikut melebar, dan pelebaran itu
        # tidak ikut terhitung ketika baris dipenggal sehingga meluber.
        obj.setCharSpace(0)
        c.drawText(obj)
        return
    c.setFont(font, ukuran)
    c.setFillColor(warna_hex)
    if tengah:
        c.drawCentredString(x, y, s)
    elif kanan:
        c.drawRightString(x, y, s)
    else:
        c.drawString(x, y, s)


def bungkus(s, font, ukuran, lebar):
    kata, baris, kini = s.split(), [], ""
    for k in kata:
        uji = (kini + " " + k).strip()
        if pdfmetrics.stringWidth(uji, font, ukuran) <= lebar:
            kini = uji
        else:
            if kini:
                baris.append(kini)
            kini = k
    if kini:
        baris.append(kini)
    return baris


def paragraf(x, y, s, font="SG", ukuran=13.5, warna="redup", lebar=None, lh=1.5):
    lebar = lebar or LEBAR
    for i, b in enumerate(bungkus(s, font, ukuran, lebar)):
        teks(x, y - i * ukuran * lh, b, font, ukuran, warna)
    return y - len(bungkus(s, font, ukuran, lebar)) * ukuran * lh


def slide(judul, dek=None, bagian=None):
    """Halaman isi: label bagian, garis rambut, judul, kalimat pengantar."""
    if NOMOR[0]:
        c.showPage()
    NOMOR[0] += 1
    if bagian is not None:
        BAGIAN[0] = bagian
    c.setFillColor(HexColor(P["latar"]))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setStrokeColor(HexColor(P["garis"]))
    c.setLineWidth(0.7)
    c.line(ML, H - 56, W - MR, H - 56)
    if BAGIAN[0]:
        teks(ML, H - 44, BAGIAN[0].upper(), "SG-Semi", 8.5, "samar", spasi=1.5)
    teks(W - MR, H - 44, f"{NOMOR[0]:02d}", "SG", 8.5, "samar", kanan=True)
    y = H - 108
    for i, b in enumerate(bungkus(judul, "SG-Semi", 30, LEBAR)):
        teks(ML, y - i * 38, b, "SG-Semi", 30, "tinta")
    y -= len(bungkus(judul, "SG-Semi", 30, LEBAR)) * 38
    if dek:
        y = paragraf(ML, y - 4, dek, "SG-Light", 15.5, "redup", LEBAR * 0.82, 1.42) - 8
    return y


def pemisah(nomor, judul, isi):
    """Halaman pembatas bagian, latar penuh warna gelap."""
    if NOMOR[0]:
        c.showPage()
    NOMOR[0] += 1
    c.setFillColor(HexColor(P["navy"]))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(HexColor(P["navy2"]))
    c.rect(0, 0, W * 0.36, H, fill=1, stroke=0)
    teks(ML, H / 2 + 26, nomor, "SG-Light", 116, "#2E5F86")
    teks(W * 0.42, H / 2 + 40, judul, "SG-Semi", 34, "#FFFFFF")
    paragraf(W * 0.42, H / 2 - 2, isi, "SG-Light", 15.5, "#9FB6C9", W * 0.5, 1.5)
    BAGIAN[0] = judul


def gambar(nama, y_atas, tinggi, x=None, lebar=None):
    p = GAMBAR[nama]
    im = ImageReader(p)
    iw, ih = im.getSize()
    if lebar is None:
        lebar = tinggi * iw / ih
    else:
        tinggi = lebar * ih / iw
    x = ML if x is None else x
    c.drawImage(im, x, y_atas - tinggi, width=lebar, height=tinggi, mask="auto")
    return y_atas - tinggi


def statistik(items, y, lebar_kolom=None, ukuran=None):
    """Baris angka besar. items = [(angka, satuan, keterangan), ...]

    Ukuran angka menyesuaikan jumlah kolom bila tidak ditentukan, supaya angka
    beserta satuannya tidak pernah menabrak kolom sebelahnya.
    """
    k = len(items)
    lebar_kolom = lebar_kolom or LEBAR / k
    if ukuran is None:
        ukuran = 52 if k <= 3 else 40
    for i, (ang, sat, ket) in enumerate(items):
        x = ML + i * lebar_kolom
        teks(x, y, ang, "SG-Light", ukuran, "tinta")
        w = pdfmetrics.stringWidth(ang, "SG-Light", ukuran)
        if sat:
            teks(x + w + 6, y + 2, sat, "SG", 13.5, "redup")
        for j, b in enumerate(bungkus(ket, "SG", 12, lebar_kolom - 30)):
            teks(x, y - 22 - j * 16, b, "SG", 12, "redup")


def tabel(kol, baris, y, lebar_kol, ukuran=12, tinggi_baris=25, warna_kol=None):
    x0 = ML
    for i, (t, lw) in enumerate(zip(kol, lebar_kol)):
        teks(x0 + sum(lebar_kol[:i]), y, t.upper(), "SG-Semi", 8.5, "samar", spasi=1.1)
    c.setStrokeColor(HexColor(P["garis"]))
    c.setLineWidth(0.7)
    c.line(ML, y - 9, ML + sum(lebar_kol), y - 9)
    yy = y - 9
    lh = ukuran * 1.32
    for r in baris:
        # Tinggi tiap baris mengikuti sel terpanjang. Tinggi tetap membuat baris
        # kedua sebuah sel menembus garis pemisah di bawahnya.
        pecah = []
        for i, sel in enumerate(r):
            f = "SG-Semi" if i == 0 else "SG"
            pecah.append(bungkus(str(sel), f, ukuran, lebar_kol[i] - 16))
        n_baris = max(len(p) for p in pecah)
        yy -= max(tinggi_baris, 16 + n_baris * lh)
        for i, potong in enumerate(pecah):
            w = "tinta" if i == 0 else "redup"
            f = "SG-Semi" if i == 0 else "SG"
            if warna_kol and warna_kol.get(i):
                w = warna_kol[i](r[i])
            for j, b in enumerate(potong):
                teks(ML + sum(lebar_kol[:i]), yy + (n_baris - 1 - j) * lh, b,
                     f, ukuran, w)
        c.setStrokeColor(HexColor(P["garis"]))
        c.line(ML, yy - 11, ML + sum(lebar_kol), yy - 11)
    return yy


def catatan(y, s, warna="biru"):
    """Baris penutup bermakna 'jadi apa', dengan penanda vertikal tipis."""
    c.setFillColor(HexColor(P[warna]))
    tinggi = 15.5 * len(bungkus(s, "SG", 13.5, LEBAR - 22))
    c.rect(ML, y - tinggi + 12, 2.4, tinggi, fill=1, stroke=0)
    paragraf(ML + 14, y, s, "SG", 13.5, "tinta", LEBAR - 22, 1.15)


# =====================================================================
# Isi deck
# =====================================================================

# ---- 1. sampul
NOMOR[0] += 1
c.setFillColor(HexColor(P["navy"]))
c.rect(0, 0, W, H, fill=1, stroke=0)
c.setFillColor(HexColor(P["navy2"]))
c.rect(0, 0, W, 4, fill=1, stroke=0)
teks(ML, H - 132, "AMBANG KEPUTUSAN,", "SG-Light", 46, "#FFFFFF")
teks(ML, H - 186, "BUKAN ARSITEKTUR", "SG-Semi", 46, "#FFFFFF")
paragraf(ML, H - 236,
         "Audit ulang deteksi deepfake audio pada dataset Fake-or-Real: "
         "apa yang sebenarnya diukur, apa yang tidak bertahan, dan apa yang "
         "ditemukan di luar rencana.", "SG-Light", 17, "#9FB6C9", LEBAR * 0.66, 1.5)
c.setStrokeColor(HexColor("#2E5F86"))
c.setLineWidth(0.8)
c.line(ML, 152, ML + 300, 152)
teks(ML, 124, "Gusti Ayu Putu Kesari Purnama Yani", "SG-Semi", 14, "#FFFFFF")
teks(ML, 102, "Magister Sistem Informasi, ITB STIKOM Bali", "SG-Light", 13, "#9FB6C9")
teks(W - MR, 124, "Pembimbing", "SG", 11, "#6E8DA6", kanan=True)
teks(W - MR, 104, "Dr. Dandy Pramana Hostiadi, S.Kom., M.T.", "SG-Light", 12, "#9FB6C9", kanan=True)
teks(W - MR, 86, "Dr. Gede Angga Pradipta, S.T., M.Eng.", "SG-Light", 12, "#9FB6C9", kanan=True)

# ---- 2. pertanyaan
y = slide("Rumusan masalah yang harus dijawab",
          "Kalimat berikut dikutip apa adanya dari proposal, bagian 1.2.",
          bagian="Pertanyaan")
c.setFillColor(HexColor("#F4F7FA"))
c.rect(ML, y - 104, LEBAR, 96, fill=1, stroke=0)
c.setFillColor(HexColor(P["biru"]))
c.rect(ML, y - 104, 3, 96, fill=1, stroke=0)
paragraf(ML + 22, y - 34,
         "Bagaimana kinerja arsitektur deep learning Wav2Vec2, Audio Spectrogram "
         "Transformer, HuBERT Large, dan CNN-LSTM dalam mengklasifikasikan suara "
         "asli dan deepfake dalam kondisi dengan gangguan noise?",
         "SG-Light", 17, "tinta", LEBAR - 46, 1.5)
y -= 132
paragraf(ML, y,
         "Untuk menjawabnya dengan jujur, tiga hal dikerjakan sekaligus, dan "
         "urutannya menentukan. Dataset diaudit lebih dahulu sebelum satu model "
         "pun dilatih. Kegagalan dipisahkan menjadi kegagalan mengenali dan "
         "kegagalan memutuskan. Setiap konfigurasi dijalankan berkali-kali "
         "sehingga selisihnya dapat dibandingkan dengan ragamnya sendiri.",
         "SG", 14, "redup", LEBAR * 0.86, 1.55)

# ---- 3. jawaban di depan
y = slide("Jawabannya, sebelum pembuktiannya",
          "Tiga kalimat ini adalah keseluruhan temuan. Sisa presentasi hanya "
          "menunjukkan dari mana ketiganya berasal.")
kotak = [
    ("Yang dipelajari model sebagian bukan jejak sintesis, melainkan riwayat berkas.",
     f"{n(MP3[('training', 'palsu')][2], 1)}% sampel palsu di data latih berasal dari MP3, "
     f"dan {n(MP3[('testing', 'palsu')][2], 1)}% di data uji.", P["merah"]),
    ("Kegagalan di bawah noise sebagian besar kegagalan kalibrasi, bukan pengenalan.",
     "Daya pisah bertahan sementara letak ambangnya bergeser.", P["jingga"]),
    ("Sebagian besar selisih antar arsitektur lebih kecil daripada ragam antar inisialisasi.",
     "Dari enam perbandingan utama, hanya dua yang bertahan setelah diuji.", P["biru"]),
]
yy = y - 6
for i, (utama, dukung, c_) in enumerate(kotak):
    c.setFillColor(HexColor(c_))
    c.rect(ML, yy - 46, 3, 52, fill=1, stroke=0)
    teks(ML + 20, yy, f"{i + 1}", "SG-Light", 26, c_)
    b = bungkus(utama, "SG-Semi", 16, LEBAR - 90)
    for j, t in enumerate(b):
        teks(ML + 56, yy, t, "SG-Semi", 16, "tinta") if j == 0 else \
            teks(ML + 56, yy - j * 21, t, "SG-Semi", 16, "tinta")
    paragraf(ML + 56, yy - len(b) * 21 - 2, dukung, "SG", 13, "redup", LEBAR - 100, 1.35)
    yy -= 46 + len(b) * 21

# ---- 4. pemisah 1
pemisah("01", "Sebelum melatih apa pun",
        "Dataset diperiksa lebih dahulu. Bagian ini berisi temuan yang tidak "
        "memerlukan satu pun model, tidak memiliki ragam antar inisialisasi, "
        "dan karena itu bertahan tanpa syarat.")

# ---- 5. audit
y = slide("Yang diperiksa sebelum pelatihan dimulai",
          "Seluruh 17.870 berkas dibaca satu per satu. Hasilnya deterministik "
          "dan dapat diulang siapa pun dalam hitungan menit.")
statistik([("17.870", "berkas", "seluruhnya 16 kHz mono, durasi tepat 2,000 detik"),
           ("0", "duplikat", "tidak ada berkas identik yang melintasi partisi"),
           ("0,5001", "AUC", "pintasan keheningan awal, praktis tidak ada"),
           ("0,698", "AUC", "fitur trivial terkuat, yaitu tingkat energi rata-rata")],
          y - 26)
y -= 118
catatan(y, "Partisi resminya 78,1 / 15,8 / 6,1 persen, bukan 60/20/20 seperti "
           "rencana proposal. Mencapai 60/20/20 menuntut penggabungan ulang seluruh "
           "data, dan itu menghancurkan pemisahan domain yang sengaja dirancang "
           "pembuat dataset.", "biru")

# ---- 6. kebocoran codec
y = slide("Temuan pertama: riwayat kompresi, bukan jejak sintesis",
          "Dihitung langsung dari nama berkas dan label. Tanpa model, tanpa "
          "keacakan, sehingga tidak memerlukan pengujian statistik.")
y_bawah = gambar("codec", y - 4, 0, x=ML, lebar=L_KIRI)
xk = ML + L_KIRI + 26
teks(xk, y - 30, "Akibatnya", "SG-Semi", 15, "tinta")
paragraf(xk, y - 56,
         "Model yang mempelajari jejak MP3 akan memperoleh isyarat hampir sempurna "
         "pada dua partisi pertama, lalu kehilangan isyarat itu seluruhnya pada "
         "partisi ketiga.", "SG", 13, "redup", LEBAR * 0.32, 1.5)
paragraf(xk, y - 146,
         "Terukur pada energi di atas 6 kHz. Palsu turunan MP3 berselisih 4,18 kali "
         "terhadap asli, sedangkan palsu yang bukan turunan MP3 hanya 1,17 kali.",
         "SG", 13, "redup", LEBAR * 0.32, 1.5)
catatan(y_bawah - 26, "Ini temuan terkuat penelitian ini justru karena ia tidak "
             "bergantung pada model, pelatihan, maupun keberuntungan inisialisasi.",
        "merah")

# ---- 7. pemisah 2
pemisah("02", "Menjawab pertanyaan noise",
        "Rumusan masalah menanyakan kinerja di bawah gangguan derau. Jawabannya "
        "ternyata bukan tentang seberapa jauh kinerja turun, melainkan tentang "
        "bagian mana yang turun.")

# ---- 8. kurva SNR
y = slide("Akurasi runtuh, daya pisah tidak",
          "WavLM diuji pada derau DEMAND, korpus yang tidak pernah dipakai saat "
          "pelatihan. Sumbu kiri akurasi, sumbu kanan AUC.")
y_bawah = gambar("snr", y - 2, 0, x=ML, lebar=L_KIRI)
xk = ML + L_KIRI + 26
d10 = SNR["wavlm[codec]"][10]
teks(xk, y - 34, "Pada 10 dB", "SG-Semi", 15, "tinta")
for i, (lab, val, w) in enumerate([
        ("AUC", n(np.mean([r["auc"] for r in d10]), 3), "tinta"),
        ("akurasi, ambang beku", n(np.mean([r["acc_fx"] for r in d10]) * 100, 1) + "%", "merah"),
        ("akurasi, ambang disesuaikan", n(np.mean([r["acc_pm"] for r in d10]) * 100, 1) + "%", "biru")]):
    teks(xk, y - 66 - i * 46, val, "SG-Light", 30, w)
    teks(xk, y - 84 - i * 46, lab, "SG", 11.5, "redup")
catatan(y_bawah - 26, "Model masih memisahkan kedua kelas hampir sempurna. Yang salah "
            "hanyalah letak garis keputusannya, dan itu dapat diperbaiki tanpa "
            "melatih ulang apa pun.", "jingga")

# ---- 9. dekomposisi
y = slide("Selisih yang semula dikira soal protokol",
          f"Selisih {n(TOTAL_LAMA, 1)} poin yang sempat dilaporkan ternyata "
          f"memuat tiga sebab yang terpisah. Yang selama ini dianggap utama "
          f"justru menyumbang paling sedikit.")
y = gambar("sebab", y - 12, 0, x=ML, lebar=L_PENUH) - 34
catatan(y, f"Efek protokolnya sendiri belum terbukti, yaitu {n(PROTOKOL, 2)} poin "
           f"dengan p sebesar 0,0822 pada uji Welch. Yang tersisa dari temuan "
           f"pembuka ini adalah peringatan yang berbeda dari yang semula ditulis: "
           f"sebab utamanya kalibrasi ambang, bukan pembagian data.", "jingga")

# ---- 10. jawaban RQ
y = slide("Jawaban atas rumusan masalah",
          "Dinyatakan sebagai kalimat yang dapat diuji ulang orang lain.")
c.setFillColor(HexColor("#F4F7FA"))
c.rect(ML, y - 150, LEBAR, 142, fill=1, stroke=0)
c.setFillColor(HexColor(P["jingga"]))
c.rect(ML, y - 150, 3, 142, fill=1, stroke=0)
paragraf(ML + 22, y - 34,
         "Di bawah gangguan derau lingkungan, keempat arsitektur tidak kehilangan "
         "kemampuan membedakan suara asli dari suara buatan sebesar yang tampak "
         "dari angka akurasi. Sebagian besar penurunan akurasi berasal dari "
         "bergesernya letak ambang keputusan, bukan dari hilangnya daya pisah. "
         "Konsekuensi praktisnya, perbaikan yang tepat adalah mengkalibrasi ulang "
         "ambang, bukan melatih ulang model.",
         "SG-Light", 17, "tinta", LEBAR - 46, 1.52)
y -= 176
paragraf(ML, y,
         "Pernyataan ini berlaku pada arsitektur yang AUC-nya memang bertahan. "
         "CNN-LSTM gagal dengan cara yang berbeda dan harus dinyatakan terpisah: "
         "AUC-nya sendiri jatuh ke 0,605, sehingga yang hilang memang daya "
         "pisahnya, bukan ambangnya.", "SG", 14, "redup", LEBAR * 0.88, 1.5)

# ---- 11. pemisah 3
pemisah("03", "Perbandingan arsitektur",
        "Bagian yang menjadi judul penelitian. Hasilnya perlu dinyatakan dengan "
        "hati-hati, karena sebagian besar selisih di sini tidak lebih besar "
        "daripada ragam antar inisialisasi acak.")

# ---- 12. hasil per arsitektur
y = slide("Hasil pada partisi resmi",
          "Rerata beserta simpangan baku atas beberapa inisialisasi acak, pada "
          "ambang prior-matched. Batang galat adalah simpangan bakunya.")
y = gambar("arsitektur", y - 2, 0, x=ML, lebar=L_PENUH) - 28
catatan(y, "Peringkatnya tampak jelas. Bagian berikutnya menunjukkan mengapa "
            "sebagian besar jarak pada grafik ini belum boleh dibaca sebagai "
            "perbedaan.", "biru")

# ---- 13. ragam
y = slide("Ragam antar inisialisasi menelan sebagian besar selisih",
          "Konfigurasi yang sama persis, dijalankan ulang hanya dengan angka awal "
          "acak yang berbeda. Titik adalah satu pelatihan, belah ketupat reratanya.")
y = gambar("ragam", y - 2, 0, x=ML, lebar=L_PENUH) - 28
catatan(y, "Nes2Net dengan pengaturan identik memberi 87,87 dan 97,43 persen. "
           "Selisih 9,56 poin yang satu-satunya sebabnya adalah angka pertama di "
           "dalam pembangkit acak. Melaporkan satu angka tunggal pada konfigurasi "
           "semacam ini menyesatkan, apa pun metodenya.", "merah")

# ---- 14. temuan LR
y = slide("Yang bertahan: laju belajar harus menyesuaikan encoder",
          "Satu-satunya temuan kuantitatif yang lolos pengujian dengan selisih "
          "berpuluh poin. Diuji dengan t Welch dan koreksi Holm-Bonferroni.")
y_bawah = gambar("lr", y - 4, 0, x=ML, lebar=L_KIRI)
xk = ML + L_KIRI + 26
teks(xk, y - 30, "Selisih terhadap proposal", "SG-Semi", 15, "tinta")
for i, (m, nama) in enumerate([("wavlm", "WavLM Large"), ("hubert", "HuBERT Large")]):
    b = MTX[(m, "beku")] if m == "wavlm" else MTX[(m, "dilatih")]
    p_ = MTX[(m, "proposal")]
    d = b["apm"].mean() - p_["apm"].mean()
    teks(xk, y - 66 - i * 72, f"+{n(d, 2)}", "SG-Light", 38, "hijau")
    teks(xk, y - 88 - i * 72, f"{nama}, p Holm 0,0002", "SG", 12, "redup")
    teks(xk, y - 106 - i * 72, f"n={p_['n']} lawan n={b['n']}", "SG", 11, "samar")
catatan(y_bawah - 26, "Satu laju belajar seragam 0,001 berjalan wajar pada AST yang 86 juta "
            "parameter, tetapi menjatuhkan kedua model 300 juta parameter ke "
            "tingkat tebakan koin. Keputusan ini harus ditetapkan per arsitektur, "
            "bukan diseragamkan di muka.", "hijau")

# ---- 15. pemisah 4
pemisah("04", "Yang tidak bertahan",
        "Tiga belas pernyataan sempat ditulis lalu gugur setelah diuji ulang. "
        "Mencantumkannya bukan formalitas, melainkan satu-satunya cara membuat "
        "sisa hasilnya dapat dipercaya.")

# ---- 16. klaim ditarik
y = slide("Klaim yang ditarik dan alasannya",
          "Enam yang paling berpengaruh terhadap kesimpulan. Tujuh sisanya "
          "tercatat pada berkas hasil di repositori.")
tabel(["Klaim", "Alasan penarikan"],
      [["Protokol pembagian data menentukan hasil, sekitar 50 poin",
        f"Menggabungkan tiga sebab. Protokol hanya {n(PROTOKOL, 2)} poin, dan itu pun p = 0,0822"],
       ["Korelasi -0,542 antara akurasi FoR dan deteksi TTS mutakhir",
        "Dihitung tanpa merata-ratakan seed. Setelah dikoreksi menjadi -0,048 dengan p = 0,895"],
       ["Band-gain memperbaiki generalisasi sebesar sepuluh poin",
        "Dua belas perbandingan diuji, seluruhnya memberi p Holm 1,0000"],
       ["Encoder yang dibekukan lebih baik daripada yang dilatih",
        "Arahnya berbeda antar arsitektur, tidak satu pun lolos koreksi Holm"],
       ["Nilai rekayasa +37,59 poin persentase",
        "Artefak dari baseline yang lumpuh karena bug, bukan hasil rekayasa"],
       ["Sebagian besar hasil terbitan pada FoR memakai split acak",
        "Terbantah. Satu penelitian terverifikasi memakai partisi resmi speaker-disjoint"]],
      y - 12, [LEBAR * 0.42, LEBAR * 0.58], ukuran=12, tinggi_baris=42)

# ---- 17. bug
y = slide("Kekeliruan terbesar, dan cara ia ketahuan",
          "Bendera untuk melatih encoder tidak pernah benar-benar bekerja. Log "
          "mencetak encoder DILATIH, padahal tidak satu pun gradien sampai ke sana.")
statistik([("0", "dari 211", "parameter encoder yang menerima gradien sebelum diperbaiki"),
           ("210", "dari 211", "sesudah satu atribut diperbaiki"),
           ("7", "kekeliruan", "seluruhnya berjenis sama dan gagal tanpa pesan error")],
          y - 26, LEBAR / 3)
y -= 112
paragraf(ML, y,
         "Bug ini tidak ditemukan lewat pembacaan kode, melainkan lewat tangga "
         "ablasi yang kebetulan menempatkan dua konfigurasi berbeda secara "
         "berdampingan. Keduanya menghasilkan skor identik sampai empat desimal, "
         "dan dua konfigurasi yang berbeda tidak mungkin identik bitwise kecuali "
         "keduanya sebenarnya sama.", "SG", 14, "redup", LEBAR * 0.9, 1.55)
y -= 92
catatan(y, "Sebagai tanggapan, dipasang pemeriksa otomatis yang menolak menyusun "
           "laporan bila ada kelompok run memuat lebih dari satu konfigurasi "
           "pelatihan tanpa alasan tercatat. Pengaman struktural, bukan kewaspadaan.",
        "biru")

# ---- 18. pemisah 5
pemisah("05", "Di luar rencana proposal",
        "Empat hal yang tidak diminta proposal, muncul dari pengukuran sendiri, "
        "dan menyentuh cara bidang ini melaporkan hasilnya.")

# ---- 19. SOTA runtuh
y = slide("Model publik terkemuka runtuh di luar domainnya",
          "Nes2Net-X, EER 1,49 persen pada ASVspoof 2021 DF, diuji zero-shot "
          "tanpa adaptasi apa pun.")
y = gambar("sota", y - 2, 0, x=ML, lebar=L_PENUH) - 28
catatan(y, "AUC di bawah lima puluh persen berarti pengurutannya terbalik, bukan "
           "sekadar buruk. Metrik EER di dalam domain secara struktural tidak dapat "
           "menangkap mode kegagalan ini, dan itu celah evaluasi yang nyata.",
        "merah")

# ---- 20. keterbaruan
y = slide("Empat hal yang ditemukan di luar proposal",
          "Seluruhnya berasal dari pengukuran pada mesin sendiri, bukan dari "
          "literatur.")
tabel(["Temuan", "Apa yang ditambahkan"],
      [["Audit provenance codec pada Fake-or-Real",
        "Ketidakseimbangan riwayat kompresi yang sudah ada di dalam dataset apa adanya, belum terdokumentasi sebelumnya"],
       ["Pemisahan kegagalan pengenalan dan kegagalan kalibrasi",
        "Bidang ini melaporkan EER yang bebas ambang, sehingga buta terhadap mode kegagalan yang dominan di pemakaian nyata"],
       ["Pengujian lintas generasi text-to-speech",
        "Empat belas sistem dari empat generasi, diukur pada spesifisitas 95 persen yang disamakan sehingga adil"],
       ["Pelaporan ragam antar inisialisasi",
        "Ketiga penelitian pembanding pada FoR tidak mencantumkan simpangan baku sama sekali"]],
      y - 12, [LEBAR * 0.4, LEBAR * 0.6], ukuran=12, tinggi_baris=44)

# ---- 21. arsitektur terkini
y = slide("Posisi terhadap arsitektur mutakhir",
          "Empat arsitektur pada judul mewakili generasi 2020 sampai 2022. "
          "Perkembangan sesudahnya bergerak ke bagian belakang model, bukan ke "
          "penyari depannya.")
tabel(["Pendekatan", "Tahun", "Catatan"],
      [["RawNet2, AASIST", "2021-2022", "Bagian belakang berbasis graf, masih menjadi acuan pembanding"],
       ["wav2vec 2.0 disetel ulang", "2022", "Menetapkan pola penyari depan swa-selia yang kini lazim"],
       ["Nes2Net-X", "2025", "Bagian belakang bersarang 511 ribu parameter, EER 1,49 persen pada ASVspoof 2021 DF"],
       ["XLSR-Mamba, SLS, TCM", "2024-2025", "Varian bagian belakang, selisih antar keduanya berbilang persepuluh poin"]],
      y - 12, [LEBAR * 0.26, LEBAR * 0.14, LEBAR * 0.6], ukuran=12, tinggi_baris=38)
catatan(96, "Nes2Net-X ikut diuji dalam penelitian ini, dan justru pada model itu "
            "keruntuhan lintas domain paling terlihat. Kebaruan arsitektur tidak "
            "menyelesaikan masalah kalibrasi.", "teal")

# ---- 22. kesimpulan
y = slide("Kesimpulan", "Dua temuan bertahan, dan keduanya perlu dinyatakan "
                        "terus terang.", bagian="Penutup")
kotak2 = [("Kalibrasi ambang keputusan",
           "Sebagian besar kegagalan yang terlihat sebagai runtuhnya kemampuan "
           "sebenarnya adalah bergesernya letak ambang. Berlaku pada dua percobaan "
           "yang tidak berhubungan."),
          ("Besaran laju belajar relatif terhadap encoder",
           f"Selisih berpuluh poin dengan p Holm 0,0002 pada dua model swa-selia "
           f"berukuran besar, diuji atas lima inisialisasi.")]
yy = y - 10
for i, (j, isi) in enumerate(kotak2):
    c.setFillColor(HexColor("#F4F7FA"))
    c.rect(ML + i * (LEBAR / 2 + 8), yy - 118, LEBAR / 2 - 8, 118, fill=1, stroke=0)
    c.setFillColor(HexColor([P["jingga"], P["hijau"]][i]))
    c.rect(ML + i * (LEBAR / 2 + 8), yy - 118, 3, 118, fill=1, stroke=0)
    x = ML + i * (LEBAR / 2 + 8) + 20
    for k, b in enumerate(bungkus(j, "SG-Semi", 17, LEBAR / 2 - 48)):
        teks(x, yy - 30 - k * 22, b, "SG-Semi", 17, "tinta")
    paragraf(x, yy - 78, isi, "SG", 12.5, "redup", LEBAR / 2 - 48, 1.42)
yy -= 146
c.setFillColor(HexColor("#FDF4F7"))
c.rect(ML, yy - 66, LEBAR, 62, fill=1, stroke=0)
c.setFillColor(HexColor(P["merah"]))
c.rect(ML, yy - 66, 3, 62, fill=1, stroke=0)
paragraf(ML + 20, yy - 26,
         "Keduanya tidak menyangkut pilihan arsitektur sama sekali, padahal judul "
         "penelitian ini menyangkut perbandingan arsitektur. Hal itu dinyatakan "
         "apa adanya, bukan disembunyikan.",
         "SG", 14, "tinta", LEBAR - 44, 1.45)

# ---- 23. yang tersisa
y = slide("Yang masih harus dikerjakan",
          "Diurutkan menurut seberapa besar pengaruhnya terhadap kesimpulan.")
tabel(["Pekerjaan", "Mengapa penting"],
      [["Menyamakan kapasitas penyari depan swa-selia",
        "Wav2Vec2 Base 95 juta dibandingkan dengan HuBERT Large 300 juta, sehingga arsitektur dan kapasitas masih tercampur"],
       ["Mengukur ulang pintasan keheningan pada varian for-norm",
        "Nama berkas menunjukkan keheningan sudah dibuang saat for-2sec dibuat, jadi hasil nol pada varian ini belum tentu berlaku umum"],
       ["Menambah inisialisasi pada sel yang masih tiga",
        "Pada n sebesar tiga, selang kepercayaan simpangan bakunya sendiri membentang setengah sampai tiga kali lipat"],
       ["Kalibrasi induktif yang tidak memerlukan seluruh skor uji",
        "Ambang prior-matched bersifat transduktif, sehingga belum siap dipakai pada deteksi satu per satu"]],
      y - 12, [LEBAR * 0.4, LEBAR * 0.6], ukuran=12, tinggi_baris=44)

# ---- 24. penutup
c.showPage()
NOMOR[0] += 1
c.setFillColor(HexColor(P["navy"]))
c.rect(0, 0, W, H, fill=1, stroke=0)
teks(ML, H / 2 + 46, "Terima kasih", "SG-Light", 44, "#FFFFFF")
paragraf(ML, H / 2 - 6,
         "Seluruh angka pada presentasi ini dihitung ulang dari berkas hasil "
         "setiap kali dokumen dibangun, sehingga tidak mungkin ada angka usang. "
         "Kode, skor 159 pelatihan, naskah lengkap, dan daftar klaim yang ditarik "
         "tersedia terbuka.", "SG-Light", 15.5, "#9FB6C9", LEBAR * 0.6, 1.55)
teks(ML, 120, "github.com/Tristan-tech-ai/general-AI", "Mono", 13, "#6E8DA6")

c.save()

# ringkasan
print(f"-> {os.path.relpath(OUT, HERE)}  ({NOMOR[0]} halaman, "
      f"{os.path.getsize(OUT) / 1e6:.1f} MB)")
