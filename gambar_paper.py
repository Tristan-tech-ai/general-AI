"""
Gambar untuk naskah, dirancang ulang untuk terbit.

Rangkaian gambar sebelumnya memiliki beberapa cacat yang membuatnya tidak layak
dipakai: satu gambar kosong sama sekali karena kunci pencarian tidak lagi cocok
setelah penamaan diubah, satu kehilangan seluruh kolom pertamanya karena sebab
yang sama, dan satu lagi memuat legenda dua puluh tujuh baris berisi nama
direktori mentah yang menutupi judul beserta separuh bidang gambarnya.

Modul ini menggantikannya. Beberapa aturan diterapkan pada seluruh gambar:

  * Tata letak diatur constrained_layout supaya label tidak pernah terpotong.
  * Legenda diletakkan di luar bidang data bila jumlah entrinya lebih dari tiga.
  * Label titik data digeser menurut kepadatan sekitarnya, bukan digeser tetap,
    sehingga dua titik yang berdekatan tidak saling menimpa.
  * Batang galat selalu ditampilkan bila jumlah inisialisasi lebih dari satu,
    dan jumlah inisialisasi dicantumkan supaya pembaca tahu bobot tiap titik.
  * Tidak ada gambar tiga dimensi. Seluruh hubungan dalam penelitian ini
    berdimensi dua, dan proyeksi tiga dimensi hanya akan menambah kesalahan baca
    tanpa menambah informasi.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.metrics import full_metrics, prior_matched_threshold

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "gambar")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 220, "font.size": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.7,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})

# Palet konsisten untuk seluruh naskah.
C = {"asli": "#1864AB", "palsu": "#C2255C", "netral": "#868E96",
     "baik": "#2F9E44", "buruk": "#E03131", "aksen": "#E8590C",
     "gelap": "#212529"}
ARCH = {"wavlm": "#1864AB", "hubert": "#2F9E44", "ast": "#C2255C",
        "wav2vec2": "#E8590C", "cnn_asp": "#7048E8", "cnnlstm": "#868E96",
        "nes2net": "#0B7285"}
NAMA = {"wavlm": "WavLM Large", "hubert": "HuBERT Large", "ast": "AST",
        "wav2vec2": "Wav2Vec2 Base", "cnn_asp": "CNN + ASP",
        "cnnlstm": "CNN-BiLSTM", "nes2net": "Nes2Net-X"}


def simpan(fig, nama):
    p = os.path.join(OUT, nama)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ->", os.path.relpath(p, HERE))


def skor(pola):
    """Akurasi pada kedua ambang, AUC, dan EER untuk tiap run yang cocok."""
    a05, apm, auc, eer = [], [], [], []
    for d in sorted(glob.glob(os.path.join(HERE, pola))):
        f = os.path.join(d, "test_scores.npy")
        if not os.path.exists(f):
            continue
        y, p, _ = np.load(f)
        y = y.astype(int)
        m0 = full_metrics(y, p, 0.5)
        a05.append(m0["accuracy"] * 100)
        auc.append(m0["auc"])
        eer.append(m0["eer"] * 100)
        apm.append(full_metrics(y, p, prior_matched_threshold(p, 0.5))["accuracy"] * 100)
    if not a05:
        return None
    return {"a05": np.array(a05), "apm": np.array(apm),
            "auc": np.array(auc), "eer": np.array(eer), "n": len(a05)}


def sd(a):
    return float(a.std(ddof=1)) if len(a) > 1 else 0.0


# =====================================================================
# Gambar 1. Kebocoran provenance codec pada dataset
# =====================================================================
def gambar1():
    import csv
    rows = list(csv.DictReader(open(os.path.join(HERE, "manifest.csv"),
                                    encoding="utf-8")))
    c = defaultdict(lambda: [0, 0])
    for r in rows:
        kelas = "asli" if r["label"] == "0" else "palsu"
        c[(r["split_official"], kelas)][0] += 1
        if r["is_mp3"] in ("1", "True", "true"):
            c[(r["split_official"], kelas)][1] += 1

    urut = ["training", "validation", "testing"]
    label = ["Latih\n(13.956 berkas)", "Validasi\n(2.826)", "Uji\n(1.088)"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(urut))
    w = 0.36
    for i, (kelas, warna) in enumerate([("asli", C["asli"]), ("palsu", C["palsu"])]):
        pct = [100 * c[(s, kelas)][1] / c[(s, kelas)][0] for s in urut]
        b = ax.bar(x + (i - 0.5) * w, pct, w, color=warna, zorder=3,
                   label=f"kelas {kelas}")
        for xi, v in zip(x + (i - 0.5) * w, pct):
            ax.text(xi, v + 2.2, f"{v:.1f}%", ha="center", fontsize=10,
                    weight="bold", color=warna)
    ax.set_xticks(x)
    ax.set_xticklabels(label)
    ax.set_ylabel("Berkas berasal dari MP3 (persen)")
    ax.set_ylim(0, 106)
    ax.legend(loc="upper right", ncol=2, fontsize=10)
    ax.set_title("Riwayat kompresi berkorelasi dengan label, "
                 "tetapi hanya pada partisi latih dan validasi", loc="left")
    ax.text(0.5, -0.30, "Perhitungan langsung dari nama berkas pada manifest. "
            "Tidak melibatkan model, pelatihan, maupun keacakan.",
            transform=ax.transAxes, ha="center", fontsize=9, color=C["netral"])
    simpan(fig, "gambar1_kebocoran_codec.png")


# =====================================================================
# Gambar 2. Ambang keputusan, bukan protokol
# =====================================================================
def gambar2():
    r = skor("runs/cnn_asp_random_none_b32e10_s*")
    o = skor("runs/cnn_asp_official_none_b32e10_s*")
    if not (r and o):
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6),
                                  gridspec_kw={"width_ratios": [1.35, 1]})

    # kiri: dua garis kemiringan
    for a, b, nm, cl, ls in [(r["a05"], o["a05"], "Ambang tetap 0,5",
                              C["buruk"], "--"),
                             (r["apm"], o["apm"], "Ambang prior-matched",
                              C["asli"], "-")]:
        ax.errorbar([0, 1], [a.mean(), b.mean()], yerr=[sd(a), sd(b)],
                    fmt="o", ls=ls, color=cl, lw=2.6, ms=9, capsize=4,
                    ecolor=C["gelap"], elinewidth=1.1, label=nm, zorder=3)
        ax.annotate(f"{a.mean():.1f}", (0, a.mean()),
                    xytext=(-12, 8 if ls == "--" else -8),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=10.5, color=cl, weight="bold")
        ax.annotate(f"{b.mean():.1f}", (1, b.mean()), xytext=(12, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=10.5, color=cl, weight="bold")
    ax.annotate("", xy=(1.0, o["a05"].mean()), xytext=(1.0, o["apm"].mean()),
                arrowprops=dict(arrowstyle="<->", color=C["aksen"], lw=2))
    ax.text(1.06, (o["a05"].mean() + o["apm"].mean()) / 2,
            f"kesalahan\nkalibrasi\n{o['apm'].mean() - o['a05'].mean():.1f} pp",
            fontsize=9.5, color=C["aksen"], weight="bold", va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Split acak\n60/20/20", "Partisi resmi\nFoR"])
    ax.set_xlim(-0.35, 1.55)
    ax.set_ylim(45, 105)
    ax.set_ylabel("Akurasi uji (persen)")
    ax.legend(loc="lower left", fontsize=9.5)
    ax.set_title("(a) Selisih pada dua ambang", loc="left")

    # kanan: pemecahan tiga sebab
    lama_o = skor("runs/cnn_asp_official_none_s42")
    if lama_o:
        sebab = [("Ambang\nkeputusan", o["apm"].mean() - o["a05"].mean(), C["aksen"]),
                 ("Model kurang\nterlatih", o["apm"].mean() - lama_o["apm"].mean(),
                  C["netral"]),
                 ("Protokol\npembagian data", r["apm"].mean() - o["apm"].mean(),
                  C["asli"])]
        nm = [s[0] for s in sebab]
        val = [s[1] for s in sebab]
        cl = [s[2] for s in sebab]
        b = ax2.barh(range(len(nm)), val, color=cl, zorder=3, height=0.6)
        for i, v in enumerate(val):
            ax2.text(v + 1.2, i, f"{v:.2f} pp", va="center", fontsize=10.5,
                     weight="bold", color=cl[i])
        ax2.set_yticks(range(len(nm)))
        ax2.set_yticklabels(nm, fontsize=10)
        ax2.invert_yaxis()
        ax2.set_xlim(0, max(val) * 1.35)
        ax2.set_xlabel("Sumbangan terhadap selisih 49,94 poin")
        ax2.set_title("(b) Pemecahan selisih menjadi sebab", loc="left")
    fig.suptitle("Sebagian besar selisih berasal dari ambang "
                 "keputusan, bukan dari protokol pembagian data",
                 fontsize=12, weight="bold", ha="left", x=0.01)
    simpan(fig, "gambar2_ambang_vs_protokol.png")


# =====================================================================
# Gambar 3. Mekanisme yang sama pada dua eksperimen terpisah
# =====================================================================
def gambar3():
    sp = os.path.join(HERE, "snr_results.json")
    if not os.path.exists(sp):
        return
    g = defaultdict(list)
    for r in json.load(open(sp, encoding="utf-8")):
        g[(r["arch"], r["snr"])].append(r)

    snrs = [None, 20, 15, 10, 5, 0, -5]
    xs = list(range(len(snrs)))
    lbl = ["bersih" if s is None else f"{s}" for s in snrs]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6))
    arch = "wavlm[codec]"
    auc, fx, pm = [], [], []
    for s in snrs:
        v = g.get((arch, s), [])
        if not v:
            auc.append(np.nan); fx.append(np.nan); pm.append(np.nan); continue
        auc.append(np.mean([x["auc"] for x in v]) * 100)
        fx.append(np.mean([x["acc_fx"] for x in v]) * 100)
        pm.append(np.mean([x["acc_pm"] for x in v]) * 100)
    ax.plot(xs, auc, "-s", color=C["gelap"], lw=2.4, ms=7,
            label="AUC (tidak bergantung ambang)", zorder=4)
    ax.plot(xs, pm, "-o", color=C["asli"], lw=2.4, ms=7,
            label="Akurasi, ambang prior-matched", zorder=3)
    ax.plot(xs, fx, "--o", color=C["buruk"], lw=2.4, ms=7,
            label="Akurasi, ambang beku dari kondisi bersih", zorder=3)
    ax.fill_between(xs, fx, pm, color=C["aksen"], alpha=0.16, zorder=2)
    i10 = snrs.index(10)
    ax.annotate(f"{pm[i10] - fx[i10]:.1f} pp", (i10, (pm[i10] + fx[i10]) / 2),
                xytext=(14, 0), textcoords="offset points", fontsize=10,
                color=C["aksen"], weight="bold", va="center")
    ax.set_xticks(xs); ax.set_xticklabels(lbl)
    ax.set_xlabel("Signal to noise ratio (dB)")
    ax.set_ylabel("Persen")
    ax.set_ylim(40, 103)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("(a) WavLM di bawah noise", loc="left")

    # kanan: partisi resmi, sumbu yang sama
    o = skor("runs/cnn_asp_official_none_b32e10_s*")
    r = skor("runs/cnn_asp_random_none_b32e10_s*")
    if o and r:
        kat = ["Split acak", "Partisi resmi"]
        x2 = np.arange(2)
        w = 0.26
        for i, (nm, vals, cl) in enumerate([
                ("AUC", [r["auc"].mean() * 100, o["auc"].mean() * 100], C["gelap"]),
                ("Akurasi @prior", [r["apm"].mean(), o["apm"].mean()], C["asli"]),
                ("Akurasi @0,5", [r["a05"].mean(), o["a05"].mean()], C["buruk"])]):
            ax2.bar(x2 + (i - 1) * w, vals, w, color=cl, label=nm, zorder=3)
            for xi, v in zip(x2 + (i - 1) * w, vals):
                ax2.text(xi, v + 1.4, f"{v:.1f}", ha="center", fontsize=9,
                         weight="bold", color=cl)
        ax2.set_xticks(x2); ax2.set_xticklabels(kat)
        ax2.set_ylim(40, 112)
        ax2.set_ylabel("Persen")
        ax2.legend(loc="lower left", fontsize=9, ncol=1)
        ax2.set_title("(b) CNN + ASP lintas protokol", loc="left")
    fig.suptitle("Dua eksperimen terpisah, satu mekanisme: daya pisah "
                 "bertahan sementara ambang bergeser",
                 fontsize=12, weight="bold", ha="left", x=0.01)
    simpan(fig, "gambar3_mekanisme_kalibrasi.png")


# =====================================================================
# Penataan label supaya tidak bertumpuk
# =====================================================================
def rapikan(ys, jarak, batas):
    """Geser label sesedikit mungkin sampai jarak antar label terpenuhi.

    Masalah pada rangkaian gambar sebelumnya adalah label diletakkan tepat di
    ordinat datanya. Ketika dua seri berdekatan, kedua labelnya saling menimpa
    dan keduanya menjadi tidak terbaca. Fungsi ini menyelesaikan urutan label
    dari bawah ke atas, mendorong setiap label ke atas hanya bila tetangga di
    bawahnya terlalu dekat, lalu menggeser seluruh blok ke bawah bila hasilnya
    melewati batas atas bidang gambar. Pergeserannya minimal, sehingga urutan
    label tetap mencerminkan urutan nilai sebenarnya.
    """
    idx = sorted(range(len(ys)), key=lambda i: ys[i])
    hasil = list(ys)
    for k in range(1, len(idx)):
        a, b = idx[k - 1], idx[k]
        if hasil[b] - hasil[a] < jarak:
            hasil[b] = hasil[a] + jarak
    lebih = hasil[idx[-1]] - batas[1]
    if lebih > 0:
        for i in idx:
            hasil[i] -= lebih
    kurang = batas[0] - hasil[idx[0]]
    if kurang > 0:
        for i in idx:
            hasil[i] += kurang
    return hasil


# =====================================================================
# Gambar 4. Perlakuan encoder tidak punya satu jawaban benar
# =====================================================================
PERLAKUAN = [
    ("Encoder dibekukan", "runs/{m}_official_full_b{b}e10_s*", C["asli"]),
    ("Encoder dilatih,\nlaju wajar per model", "runs/{m}_official_fullUF_b{b}e10_s*",
     C["baik"]),
    ("Encoder dilatih,\nlaju 0,001", "runs/{m}_official_fullUFENC0.001_b{b}e10_s*",
     C["aksen"]),
    ("Proposal apa adanya,\nlaju 0,001 seragam",
     "runs/{m}_official_proposalULRPK_b{b}e20_s*", C["buruk"]),
]
ARS4 = [("ast", 32, "(a) AST\n86 juta parameter, pra-latih terselia"),
        ("wavlm", 16, "(b) WavLM Large\n300 juta parameter, swa-selia"),
        ("hubert", 32, "(c) HuBERT Large\n300 juta parameter, swa-selia")]


def gambar4():
    fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.9), sharex=True)
    for ax, (m, b, judul) in zip(axs, ARS4):
        nm, mu, er, cl, nn = [], [], [], [], []
        for label, pola, warna in PERLAKUAN:
            s = skor(pola.format(m=m, b=b))
            if not s:
                continue
            nm.append(label); mu.append(s["apm"].mean())
            er.append(sd(s["apm"])); cl.append(warna); nn.append(s["n"])
        y = np.arange(len(nm))
        ax.barh(y, mu, xerr=er, color=cl, height=0.62, zorder=3,
                error_kw=dict(ecolor=C["gelap"], elinewidth=1.2, capsize=4))
        for i, (v, e, n) in enumerate(zip(mu, er, nn)):
            ax.text(v + e + 1.6, i, f"{v:.1f}", va="center", fontsize=10,
                    weight="bold", color=cl[i])
            ax.text(1.5, i, f"n={n}", va="center", fontsize=8.5,
                    color="white", weight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(nm if ax is axs[0] else [""] * len(nm), fontsize=9.2)
        ax.invert_yaxis()
        ax.set_xlim(0, 118)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.axvline(50, color=C["netral"], ls=":", lw=1.2, zorder=1)
        ax.set_title(judul, loc="left", fontsize=10.5)
        ax.set_xlabel("Akurasi partisi resmi (persen)")
    # Keterangan garis acuan diletakkan di dalam bidang data pada panel
    # pertama saja. Bila diletakkan di atas sumbu, ia menabrak judul panel.
    axs[0].text(51.5, 3.3, "tebak acak", fontsize=8.5, color=C["netral"],
                ha="left", va="center")
    fig.suptitle("Arah pengaruh perlakuan encoder berbeda antar "
                 "arsitektur, sehingga tidak ada satu pilihan yang benar untuk "
                 "semuanya", fontsize=12, weight="bold", ha="left", x=0.01)
    simpan(fig, "gambar4_perlakuan_encoder.png")


# =====================================================================
# Gambar 5. Peta hutan: apa yang bertahan setelah ragam diukur
# =====================================================================
PASANG = [
    ("AST: encoder dilatih vs dibekukan",
     "runs/ast_official_fullUF_b32e10_s*", "runs/ast_official_full_b32e10_s*"),
    ("AST: encoder dilatih vs proposal",
     "runs/ast_official_fullUF_b32e10_s*",
     "runs/ast_official_proposalULRPK_b32e20_s*"),
    ("WavLM: encoder dibekukan vs dilatih",
     "runs/wavlm_official_full_b16e10_s*", "runs/wavlm_official_fullUF_b16e10_s*"),
    ("HuBERT: encoder dilatih vs dibekukan",
     "runs/hubert_official_fullUF_b32e10_s*", "runs/hubert_official_full_b32e10_s*"),
    ("WavLM: rekayasa vs proposal",
     "runs/wavlm_official_full_b16e10_s*",
     "runs/wavlm_official_proposalULRPK_b16e20_s*"),
    ("HuBERT: rekayasa vs proposal",
     "runs/hubert_official_fullUF_b32e10_s*",
     "runs/hubert_official_proposalULRPK_b32e20_s*"),
]


def gambar5():
    from scipy import stats
    baris = []
    for nama, pa, pb in PASANG:
        a, b = skor(pa), skor(pb)
        if not (a and b):
            continue
        x, y = a["apm"], b["apm"]
        d = x.mean() - y.mean()
        se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
        dfree = se ** 4 / ((x.var(ddof=1) / len(x)) ** 2 / (len(x) - 1) +
                           (y.var(ddof=1) / len(y)) ** 2 / (len(y) - 1))
        t = stats.t.ppf(0.975, dfree)
        p = stats.ttest_ind(x, y, equal_var=False).pvalue
        baris.append([nama, d, t * se, p, len(x), len(y)])

    # Koreksi Holm-Bonferroni, urutan sama seperti pada berkas hasil.
    urut = sorted(range(len(baris)), key=lambda i: baris[i][3])
    mx, k = 0.0, len(baris)
    for j, i in enumerate(urut):
        mx = max(mx, min(1.0, (k - j) * baris[i][3]))
        baris[i].append(mx)

    baris.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    y = np.arange(len(baris))
    for i, r in enumerate(baris):
        kuat = r[6] < 0.05
        cl = C["baik"] if kuat else C["netral"]
        ax.errorbar(r[1], i, xerr=r[2], fmt="o", ms=10 if kuat else 8,
                    color=cl, ecolor=cl, elinewidth=2.2, capsize=5, zorder=3)
    ax.axvline(0, color=C["gelap"], lw=1.4, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[0]}\nn = {r[4]} lawan {r[5]}" for r in baris],
                       fontsize=9.4)
    ax.set_xlim(-14, 72)
    ax.set_xlabel("Selisih akurasi, poin persentase, dengan selang "
                  "kepercayaan 95 persen")
    for i, r in enumerate(baris):
        ax.text(66, i, f"p Holm\n{r[6]:.4f}", fontsize=8.8, va="center",
                ha="right", color=C["baik"] if r[6] < 0.05 else C["netral"])
    # Keterangan garis nol diletakkan di bawah baris terakhir, bukan di atas
    # baris pertama, supaya tidak menabrak judul gambar.
    ax.set_ylim(-1.15, len(baris) - 0.4)
    ax.text(0.7, -0.95, "tidak ada selisih", fontsize=9, color=C["gelap"],
            va="center", ha="left")
    ax.set_title("Hanya dua dari enam perbandingan yang selisihnya "
                 "melampaui ragam antar inisialisasi", loc="left",
                 fontsize=12, weight="bold")
    simpan(fig, "gambar5_peta_hutan.png")


# =====================================================================
# Gambar 6. Ketahanan terhadap noise
# =====================================================================
def gambar6():
    sp = os.path.join(HERE, "snr_results.json")
    if not os.path.exists(sp):
        return
    g = defaultdict(lambda: defaultdict(list))
    for r in json.load(open(sp, encoding="utf-8")):
        g[r["arch"]][r["snr"]].append(r["acc_pm"] * 100)

    snrs = [None, 20, 15, 10, 5, 0, -5]
    xs = list(range(len(snrs)))
    fig, ax = plt.subplots(figsize=(9.8, 5.4))
    kurva = []
    for arch, d in g.items():
        v = [float(np.mean(d[s])) if d.get(s) else np.nan for s in snrs]
        if np.isnan(v[0]):
            continue
        m = re.match(r"([a-z0-9_]+)\[(\w+)\]", arch)
        base = m.group(1) if m else arch
        aug = m.group(2) if m else ""
        kurva.append((NAMA.get(base, base), aug, v, ARCH.get(base, C["netral"])))
    kurva.sort(key=lambda t: -t[2][-1])

    for nama, aug, v, cl in kurva:
        ax.plot(xs, v, "-o" if aug == "full" else "--s", color=cl, lw=2.2,
                ms=5.5, alpha=0.95 if aug == "full" else 0.75, zorder=3)

    # Label langsung di ujung kanan, digeser hanya bila bertabrakan.
    akhir = [k[2][-1] for k in kurva]
    pos = rapikan(akhir, 3.6, (48, 102))
    for (nama, aug, v, cl), yy in zip(kurva, pos):
        ax.annotate(f"{nama} [{aug}]", (xs[-1], v[-1]),
                    xytext=(xs[-1] + 0.35, yy), fontsize=9, color=cl,
                    weight="bold", va="center",
                    arrowprops=dict(arrowstyle="-", color=cl, lw=0.9,
                                    shrinkA=2, shrinkB=2, alpha=0.6))
    ax.set_xticks(xs)
    ax.set_xticklabels(["bersih" if s is None else str(s) for s in snrs])
    ax.set_xlim(-0.25, len(snrs) + 2.6)
    ax.set_ylim(46, 104)
    ax.set_xlabel("Signal to noise ratio (dB), noise DEMAND yang tidak dipakai "
                  "saat pelatihan")
    ax.set_ylabel("Akurasi pada ambang prior-matched (persen)")
    ax.axhline(50, color=C["netral"], ls=":", lw=1.2)
    ax.text(0, 51, "tebak acak", fontsize=8.5, color=C["netral"])
    ax.set_title("Augmentasi penuh mempertahankan akurasi jauh lebih "
                 "lama daripada augmentasi codec saja", loc="left",
                 fontsize=12, weight="bold")
    simpan(fig, "gambar6_ketahanan_noise.png")


# =====================================================================
# Gambar 7. Pertukaran antara akurasi FoR dan deteksi TTS mutakhir
# =====================================================================
def gambar7():
    gp = os.path.join(HERE, "generations_results.json")
    if not os.path.exists(gp):
        return
    G = json.load(open(gp, encoding="utf-8"))
    pola = re.compile(r"^(.+?)_official_([a-zA-Z]+)_(b\d+e\d+)_s(\d+)$")
    kel = defaultdict(list)
    for tag, r in G.items():
        m = pola.match(tag)
        if not m:
            continue
        md = [v["recall"] for v in r["tts"].values()
              if v["era"] == "2025-2026 komersial"]
        if md:
            kel[(m.group(1), m.group(2), m.group(3))].append(np.mean(md) * 100)

    titik = []
    for (arch, aug, cfg), rec in kel.items():
        s = skor(f"runs/{arch}_official_{aug}_{cfg}_s*")
        if not s or len(rec) < 1:
            continue
        titik.append((arch, aug, s["apm"].mean(), sd(s["apm"]),
                      float(np.mean(rec)), float(np.std(rec, ddof=1))
                      if len(rec) > 1 else 0.0, len(rec)))
    if len(titik) < 3:
        return

    xa = np.array([t[2] for t in titik])
    ya = np.array([t[4] for t in titik])
    r = float(np.corrcoef(xa, ya)[0, 1])

    # Uji permutasi tepat pada tempatnya, supaya angka pada gambar tidak
    # bergantung pada berkas lain yang mungkin sudah berubah.
    rng = np.random.default_rng(0)
    nol = np.array([abs(np.corrcoef(xa, rng.permutation(ya))[0, 1])
                    for _ in range(20000)])
    p = float((nol >= abs(r)).mean())

    # Sebar dua sumbu sudah dicoba dan gagal. Kesepuluh titik berdesakan pada
    # rentang lima poin persentase, sehingga labelnya harus dipindahkan ke
    # tepi kanan, dan garis penunjuknya kemudian menyilang seluruh bidang
    # gambar. Rancangan di bawah memakai satu baris per konfigurasi. Nama
    # konfigurasi menjadi label sumbu tegak sehingga tidak mungkin bertabrakan,
    # dan pertanyaan pertukaran terbaca dari apakah kedua penanda bergerak
    # berlawanan arah ketika baris ditelusuri dari atas ke bawah.
    titik.sort(key=lambda t: -t[2])
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    y = np.arange(len(titik))
    for i, (arch, aug, fx, fe, ry, re_, n) in enumerate(titik):
        ax.plot([fx, ry], [i, i], color=C["netral"], lw=1.4, alpha=0.45,
                zorder=2)
        ax.errorbar(fx, i, xerr=fe, fmt="o", ms=10, color=C["asli"],
                    ecolor=C["asli"], elinewidth=1.4, capsize=3, zorder=3)
        ax.errorbar(ry, i, xerr=re_, fmt="s", ms=9, color=C["aksen"],
                    ecolor=C["aksen"], elinewidth=1.4, capsize=3, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{NAMA.get(t[0], t[0])} [{t[1]}]" for t in titik],
                       fontsize=9.4)
    ax.invert_yaxis()
    ax.set_ylim(len(titik) - 0.3, -1.5)
    ax.set_xlim(74, 104)
    ax.set_xlabel("Persen")
    ax.plot([], [], "o", color=C["asli"], ms=9,
            label="Akurasi partisi resmi Fake-or-Real")
    ax.plot([], [], "s", color=C["aksen"], ms=8,
            label="Recall TTS komersial 2025-2026 pada spesifisitas 95 persen")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02), fontsize=9.2, ncol=1)
    ax.text(0.0, -0.185, f"r = {r:+.3f} pada {len(titik)} konfigurasi. "
            f"Uji permutasi 20.000 kali memberi p = {p:.3f}.",
            transform=ax.transAxes, fontsize=9.6, color=C["netral"],
            va="top", ha="left")
    ax.set_title("Tidak ada pertukaran yang terbukti antara akurasi "
                 "Fake-or-Real dan deteksi TTS mutakhir", loc="left",
                 fontsize=12, weight="bold", pad=34)
    simpan(fig, "gambar7_pertukaran.png")


# =====================================================================
# Gambar 8. Model publik terkemuka pada TTS 2025-2026
# =====================================================================
def gambar8():
    sp = os.path.join(HERE, "sota_modern_results.json")
    if not os.path.exists(sp):
        return
    d = json.load(open(sp, encoding="utf-8"))
    sistem = [k for k in d if k != "for_2sec"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.6),
                                  gridspec_kw={"width_ratios": [1, 1.5]})

    f = d["for_2sec"]
    nm = ["Akurasi", "AUC", "Recall palsu\n@0,5"]
    val = [f["acc"] * 100, f["auc"] * 100, f["recall_spoof@0.5"] * 100]
    cl = [C["buruk"], C["buruk"], C["netral"]]
    ax.bar(range(3), val, color=cl, width=0.6, zorder=3)
    for i, v in enumerate(val):
        ax.text(i, v + 2.5, f"{v:.1f}", ha="center", fontsize=10.5,
                weight="bold", color=cl[i])
    ax.axhline(50, color=C["gelap"], ls=":", lw=1.3)
    ax.text(0.55, 53, "tebak acak", fontsize=8.5, color=C["gelap"], ha="left")
    ax.set_xticks(range(3)); ax.set_xticklabels(nm, fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Persen")
    ax.set_title("(a) Pada partisi uji Fake-or-Real", loc="left")

    r05 = [d[s]["recall@0.5"] * 100 for s in sistem]
    rth = [d[s]["recall@thr_for"] * 100 for s in sistem]
    o = np.argsort(r05)[::-1]
    sistem = [sistem[i] for i in o]
    r05 = [r05[i] for i in o]; rth = [rth[i] for i in o]
    y = np.arange(len(sistem))
    ax2.barh(y - 0.19, r05, 0.36, color=C["baik"], zorder=3,
             label="Ambang 0,5 apa adanya")
    ax2.barh(y + 0.19, rth, 0.36, color=C["buruk"], zorder=3,
             label="Ambang yang dikalibrasi pada Fake-or-Real")
    for i in range(len(sistem)):
        ax2.text(r05[i] + 1.5, i - 0.19, f"{r05[i]:.1f}", va="center",
                 fontsize=9, weight="bold", color=C["baik"])
        ax2.text(rth[i] + 1.5, i + 0.19, f"{rth[i]:.1f}", va="center",
                 fontsize=9, weight="bold", color=C["buruk"])
    ax2.set_yticks(y); ax2.set_yticklabels(sistem, fontsize=9.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 118)
    ax2.set_xlabel("Recall pada TTS komersial (persen)")
    # Legenda diletakkan di bawah bidang data. Pada percobaan sebelumnya ia
    # menutupi batang sistem terakhir bila diletakkan di dalam.
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
               fontsize=9)
    ax2.set_title("(b) Pada TTS komersial 2025-2026", loc="left")
    fig.suptitle("Model publik terkemuka runtuh pada partisi uji "
                 "Fake-or-Real, dan ambangnya sendiri yang merusak deteksi "
                 "TTS mutakhir", fontsize=12, weight="bold", ha="left", x=0.01)
    simpan(fig, "gambar8_sota_runtuh.png")


# =====================================================================
# Gambar 9. Besar efek dibandingkan ragam antar inisialisasi
# =====================================================================
def gambar9():
    kel = defaultdict(list)
    pola = re.compile(r"^(.+?)_(official|random)_([a-zA-Z0-9.]+?)"
                      r"(_b\d+e\d+)?_s(\d+)$")
    for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
        m = pola.match(os.path.basename(d))
        f = os.path.join(d, "test_scores.npy")
        if not (m and os.path.exists(f)):
            continue
        y, p, _ = np.load(f)
        a = full_metrics(y.astype(int), p,
                         prior_matched_threshold(p, 0.5))["accuracy"] * 100
        kel[(m.group(1), m.group(2), m.group(3), m.group(4) or "")].append(a)

    banyak = [float(np.std(v, ddof=1)) for v in kel.values() if len(v) >= 3]
    if len(banyak) < 3:
        return

    # Versi pertama gambar ini adalah sebar akurasi terhadap simpangan baku,
    # dengan label pada tiap titik. Rancangan itu gagal karena dua sebab.
    # Pertama, dua puluh dari tiga puluh dua titik berdesakan pada sudut kiri
    # atas sehingga labelnya saling menimpa. Kedua, dan ini lebih penting,
    # sebar tersebut tidak menjawab pertanyaannya. Pertanyaannya bukan berapa
    # simpangan baku tiap konfigurasi, melainkan apakah efek yang diukur lebih
    # besar daripada simpangan baku itu. Rancangan di bawah membandingkan
    # keduanya pada satu sumbu yang sama.
    from scipy import stats
    efek = []
    for nama, pa, pb in PASANG:
        a, b = skor(pa), skor(pb)
        if not (a and b):
            continue
        p = stats.ttest_ind(a["apm"], b["apm"], equal_var=False).pvalue
        efek.append([nama, abs(a["apm"].mean() - b["apm"].mean()), p])
    urut = sorted(range(len(efek)), key=lambda i: efek[i][2])
    mx = 0.0
    for j, i in enumerate(urut):
        mx = max(mx, min(1.0, (len(efek) - j) * efek[i][2]))
        efek[i].append(mx)
    efek.sort(key=lambda r: r[1])

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10.2, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 1.5]})
    tepi = np.logspace(np.log10(0.1), np.log10(60), 22)
    ax.hist(banyak, bins=tepi, color=C["netral"], zorder=3)
    med = float(np.median(banyak))
    ax.axvline(med, color=C["gelap"], lw=1.6, zorder=4)
    ax.annotate(f"median {med:.2f} pp", (med, ax.get_ylim()[1] * 0.82),
                xytext=(8, 0), textcoords="offset points", fontsize=9,
                weight="bold", color=C["gelap"], va="center")
    ax.set_ylabel("Jumlah\nkonfigurasi")
    ax.set_title(f"(a) Sebaran ragam antar inisialisasi pada {len(banyak)} "
                 "konfigurasi yang dijalankan minimal tiga kali", loc="left",
                 fontsize=10.5)

    y = np.arange(len(efek))
    for i, r in enumerate(efek):
        kuat = r[3] < 0.05
        cl = C["baik"] if kuat else C["netral"]
        ax2.plot([0.1, r[1]], [i, i], color=cl, lw=2.4, alpha=0.5, zorder=3)
        ax2.plot(r[1], i, "o", ms=11, color=cl, zorder=4)
        # Label digeser melewati garis median bila titiknya jatuh tepat di
        # sekitarnya, supaya teks tidak tertusuk garis.
        ax2.text(max(r[1] * 1.14, med * 1.16), i,
                 f"{r[1]:.2f} pp   p Holm {r[3]:.4f}",
                 fontsize=9, va="center", color=cl, weight="bold")
    ax2.axvspan(0.1, med, color=C["buruk"], alpha=0.09, zorder=1)
    ax2.axvline(med, color=C["gelap"], lw=1.6, zorder=2)
    ax2.set_yticks(y)
    ax2.set_yticklabels([r[0] for r in efek], fontsize=9.2)
    ax2.set_ylim(-0.7, len(efek) - 0.3)
    ax2.set_xscale("log")
    ax2.set_xlim(0.1, 460)
    ax2.set_xticks([0.1, 0.3, 1, 3, 10, 30])
    ax2.set_xticklabels(["0,1", "0,3", "1", "3", "10", "30"])
    ax2.set_xlabel("Poin persentase, sumbu logaritmik")
    ax2.set_title("(b) Besar enam efek yang diuji, pada sumbu yang sama",
                  loc="left", fontsize=10.5)
    ax2.text(0.105, -0.55, "wilayah tempat efek lebih kecil daripada ragam "
             "khas antar inisialisasi", fontsize=8.6, color=C["buruk"],
             va="center")
    fig.suptitle("Empat dari enam efek yang diuji tidak melampaui "
                 "ragam antar inisialisasi",
                 fontsize=12, weight="bold", ha="left", x=0.01)
    simpan(fig, "gambar9_ragam_inisialisasi.png")


if __name__ == "__main__":
    print("membuat gambar naskah ...")
    for f in (gambar1, gambar2, gambar3, gambar4, gambar5, gambar6, gambar7,
              gambar8, gambar9):
        try:
            f()
        except Exception as e:
            print(f"  !! {f.__name__} gagal: {type(e).__name__}: {e}")
    print("selesai ->", os.path.relpath(OUT, HERE))
