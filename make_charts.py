"""
Grafik profesional untuk seluruh hasil eksperimen.

Menghasilkan PNG siap-tesis dari berkas hasil yang sudah ada:
  runs/*/results.json     -> akurasi & EER per model/augmentasi/seed (FoR-2sec)
  snr_results.json        -> kurva degradasi vs SNR
  rerec_results.json      -> lintas kondisi rekaman
  itw_results.json        -> lintas korpus
"""
from __future__ import annotations

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
from matplotlib import gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.metrics import full_metrics, prior_matched_threshold

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "charts")
os.makedirs(OUT, exist_ok=True)

# ---- gaya konsisten -------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
PAL = {
    "nes2net": "#0B7285", "nes2net_lastlayer": "#5BA3B0",
    "wavlm": "#1864AB", "hubert": "#2F9E44", "wav2vec2": "#F08C00",
    "ast": "#C2255C", "cnn_asp": "#7048E8", "cnnlstm": "#868E96",
}
def col(m):
    return PAL.get(m.split("[")[0], "#495057")

LABEL = {
    "nes2net": "Nes2Net-X (fork, +layer-weighting)",
    "nes2net_lastlayer": "Nes2Net-X (rancangan asli)",
    "wavlm": "WavLM Large", "hubert": "HuBERT Large",
    "wav2vec2": "Wav2Vec2 Base", "ast": "AST",
    "cnn_asp": "CNN + ASP", "cnnlstm": "CNN-BiLSTM",
}
def lab(m):
    return LABEL.get(m, m)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("  ->", os.path.relpath(p, HERE))


# ---------------------------------------------------------------- data FoR
def load_for():
    """Kumpulkan hasil FoR-2sec dari runs/."""
    rec = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
        f = os.path.join(d, "test_scores.npy")
        if not os.path.exists(f):
            continue
        m = re.match(r"^(.+?)_(official|random|clean_val|wavval)_([a-z]+?)(AV)?"
                     r"(?:_b\d+e\d+)?_s(\d+)$", os.path.basename(d))
        if not m:
            continue
        y, p, _ = np.load(f)
        y = y.astype(int)
        met = full_metrics(y, p, prior_matched_threshold(p, 0.5))
        rec[(m.group(1), m.group(2), m.group(3))].append(met)
    return rec


def chart_for(rec):
    """Grafik 1: akurasi per model x augmentasi pada split resmi."""
    augs = ["none", "codec", "full"]
    models = sorted({k[0] for k in rec if k[1] == "official"},
                    key=lambda m: -np.mean([r["accuracy"] for k, v in rec.items()
                                            if k[0] == m and k[1] == "official"
                                            for r in v]))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    w = 0.26
    x = np.arange(len(models))
    hatch = {"none": "///", "codec": "", "full": ".."}
    for j, a in enumerate(augs):
        vals, errs = [], []
        for m in models:
            v = rec.get((m, "official", a), [])
            vals.append(np.mean([r["accuracy"] for r in v]) * 100 if v else np.nan)
            errs.append(np.std([r["accuracy"] for r in v], ddof=1) * 100
                        if len(v) > 1 else 0)
        ax.bar(x + (j - 1) * w, vals, w, yerr=errs, capsize=3,
               color=[col(m) for m in models],
               alpha={"none": 0.35, "codec": 0.65, "full": 1.0}[a],
               hatch=hatch[a], edgecolor="white", linewidth=0.6,
               label=f"augmentasi: {a}")
    ax.set_xticks(x)
    ax.set_xticklabels([lab(m) for m in models], rotation=18, ha="right")
    ax.set_ylabel("Akurasi test (%)")
    ax.set_ylim(45, 101)
    ax.axhline(94.7, ls="--", lw=1, color="#E03131")
    ax.text(len(models) - 0.4, 94.9, "SOTA FoR terpublikasi 94,7%",
            color="#E03131", fontsize=8, ha="right")
    ax.set_title("FoR-2sec (split resmi) — akurasi per arsitektur dan strategi augmentasi\n"
                 "rerata ± simpangan baku atas 3 seed, ambang prior-matched",
                 loc="left", fontsize=11)
    ax.legend(ncol=3, loc="lower right")
    save(fig, "01_for_akurasi.png")


def chart_split(rec):
    """Grafik 2: efek protokol split — temuan metodologis utama."""
    models = sorted({k[0] for k in rec if k[1] == "random"})
    if not models:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for i, m in enumerate(models):
        r = np.mean([x["accuracy"] for x in rec.get((m, "random", "none"), [])] or [np.nan]) * 100
        o = np.mean([x["accuracy"] for x in rec.get((m, "official", "none"), [])] or [np.nan]) * 100
        ax.plot([0, 1], [r, o], "-o", color=col(m), lw=2.5, ms=9, label=lab(m))
        ax.annotate(f"{r:.1f}%", (0, r), textcoords="offset points",
                    xytext=(-8, 6), ha="right", fontsize=10, color=col(m))
        ax.annotate(f"{o:.1f}%", (1, o), textcoords="offset points",
                    xytext=(8, 6), ha="left", fontsize=10, color=col(m))
        ax.annotate(f"−{r-o:.1f} pp", (0.5, (r + o) / 2), ha="center",
                    fontsize=11, color="#E03131", fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Split acak 60/20/20\n(rencana proposal)",
                        "Partisi resmi FoR\n(lintas-domain)"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylabel("Akurasi test (%)")
    ax.set_title("Protokol split menentukan hasil, bukan model\n"
                 "arsitektur, data, dan hyperparameter identik",
                 loc="left", fontsize=11)
    save(fig, "02_efek_split.png")


def chart_snr():
    p = os.path.join(HERE, "snr_results.json")
    if not os.path.exists(p):
        return
    res = json.load(open(p, encoding="utf-8"))
    g = defaultdict(list)
    for r in res:
        g[(r["arch"], r["snr"])].append(r)
    archs = sorted({r["arch"] for r in res})
    snrs = [None, 30, 25, 20, 15, 10, 5, 0, -5]
    xs = list(range(len(snrs)))

    fig = plt.figure(figsize=(12.5, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.25, 1], wspace=0.25)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    for a in archs:
        acc = [np.mean([r["acc_pm"] for r in g.get((a, s), [])] or [np.nan]) * 100
               for s in snrs]
        eer = [np.mean([r["eer"] for r in g.get((a, s), [])] or [np.nan]) * 100
               for s in snrs]
        full = a.endswith("[full]")
        ax1.plot(xs, acc, "-o" if full else "--s", color=col(a), lw=2.4 if full else 1.4,
                 ms=5, alpha=1.0 if full else 0.55, label=a)
        ax2.plot(xs, eer, "-o" if full else "--s", color=col(a), lw=2.4 if full else 1.4,
                 ms=5, alpha=1.0 if full else 0.55)
    for ax, yl, t in [(ax1, "Akurasi (%)", "Akurasi vs SNR"),
                      (ax2, "EER (%)", "EER vs SNR")]:
        ax.set_xticks(xs)
        ax.set_xticklabels(["bersih"] + [str(s) for s in snrs[1:]])
        ax.set_xlabel("SNR (dB) — noise DEMAND, korpus tak terlihat saat latih")
        ax.set_ylabel(yl)
        ax.set_title(t, loc="left", fontsize=11)
    ax1.legend(fontsize=7.5, ncol=2, loc="lower left")
    fig.suptitle("Ketahanan terhadap noise: garis tebal = dilatih dengan augmentasi noise, "
                 "garis putus = tanpa", x=0.01, ha="left", fontsize=11)
    save(fig, "03_kurva_snr.png")


def chart_datasets(rec):
    """Grafik 4: konsistensi model lintas dataset."""
    src = {}
    for name, f, key in [("for-rerec", "rerec_results.json", None),
                         ("In-the-Wild", "itw_results.json", None)]:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            src[name] = json.load(open(p, encoding="utf-8"))
    if not src:
        return

    rows = defaultdict(dict)
    for m, aug, a in [(k[0], k[2], v) for k, v in rec.items() if k[1] == "official"]:
        if aug in ("codec", "full"):
            rows[f"{m}[{aug}]"]["FoR-2sec"] = np.mean([r["eer"] for r in a]) * 100
    for name, res in src.items():
        agg = defaultdict(list)
        for r in res:
            k = f"{r.get('arch', r.get('model','?'))}[{r.get('aug','codec')}]"
            agg[k].append(r.get("eer", np.nan))
        for k, v in agg.items():
            rows[k][name] = np.nanmean(v) * 100

    keys = [k for k in rows if len(rows[k]) >= 2]
    if not keys:
        return
    cols = ["FoR-2sec", "for-rerec", "In-the-Wild"]
    keys.sort(key=lambda k: rows[k].get("FoR-2sec", 99))

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    xs = np.arange(len(cols))
    for k in keys:
        ys = [rows[k].get(c, np.nan) for c in cols]
        ax.plot(xs, ys, "-o", color=col(k), lw=2, ms=7, label=k, alpha=0.9)
    ax.set_xticks(xs); ax.set_xticklabels(cols)
    ax.set_ylabel("EER (%)  — makin rendah makin baik")
    ax.set_yscale("log")
    ax.set_title("Konsistensi lintas dataset: unggul di FoR tidak menjamin unggul di korpus lain\n"
                 "sumbu Y logaritmik", loc="left", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    save(fig, "04_lintas_dataset.png")


def chart_2x2():
    """Grafik kemiringan: seberapa jauh hasil runtuh saat protokol diperketat.

    Garis yang landai berarti hasilnya tidak bergantung pada protokol evaluasi,
    dan itulah yang dimaksud dengan generalisasi lintas domain. Garis yang curam
    berarti angka yang dilaporkan sebagian besar ditentukan oleh cara data
    dibagi.
    """
    import glob as _g
    from forlib.metrics import full_metrics, prior_matched_threshold

    data = {}
    for m in ["ast", "wavlm", "hubert", "nes2net"]:
        sel = {}
        for cfg in ["proposal", "diperbaiki"]:
            pat = (f"runs/{m}_%s_proposalULRPK_*" if cfg == "proposal"
                   else f"runs/{m}_%s_full_*")
            for sp in ["random", "official"]:
                a = []
                for d in sorted(_g.glob(os.path.join(HERE, pat % sp))):
                    f = os.path.join(d, "test_scores.npy")
                    if not os.path.exists(f):
                        continue
                    y, p, _ = np.load(f)
                    y = y.astype(int)
                    t = 0.5 if cfg == "proposal" else prior_matched_threshold(p, 0.5)
                    a.append(full_metrics(y, p, t)["accuracy"] * 100)
                sel[(cfg, sp)] = a
        if all(sel.values()):
            data[m] = sel
    if not data:
        return

    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 5.4),
                             sharey=True, squeeze=False)
    for ax, (m, sel) in zip(axes[0], data.items()):
        for cfg, sty, cl in [("proposal", "--", "#ADB5BD"),
                             ("diperbaiki", "-", col(m))]:
            r, o = (np.mean(sel[(cfg, "random")]),
                    np.mean(sel[(cfg, "official")]))
            ax.plot([0, 1], [r, o], sty, color=cl, lw=3, marker="o", ms=9,
                    zorder=3,
                    label=("Proposal apa adanya" if cfg == "proposal"
                           else "Diperbaiki (rekayasa)"))
            # kedua garis berangkat dari titik yang hampir sama pada split acak,
            # sehingga labelnya digeser vertikal agar tidak saling menimpa
            dy = 8 if cfg == "proposal" else -8
            ax.annotate(f"{r:.2f}", (0, r), xytext=(-9, dy),
                        textcoords="offset points", ha="right", va="center",
                        fontsize=9.5, color=cl, weight="bold")
            ax.annotate(f"{o:.2f}", (1, o), xytext=(9, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=9.5, color=cl, weight="bold")
            ax.annotate(f"turun {r - o:.2f} pp", (0.5, (r + o) / 2),
                        xytext=(0, -14 if cfg == "proposal" else 10),
                        textcoords="offset points", ha="center",
                        fontsize=8.5, color=cl, style="italic")
        ax.set_xlim(-0.42, 1.42)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Split acak\n60/20/20", "Partisi resmi\nFoR"])
        ax.set_title(lab(m), fontsize=11, weight="bold")
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.axhline(50, color="#E03131", lw=1, ls=":", zorder=1)
    axes[0][0].set_ylabel("Akurasi (persen)")
    axes[0][0].set_ylim(40, 104)
    axes[0][0].legend(loc="lower left", fontsize=9, framealpha=0.95)
    fig.suptitle("Ketergantungan hasil pada protokol evaluasi\n"
                 "garis landai berarti kemampuan bertahan saat domain berubah",
                 fontsize=12.5, weight="bold", y=1.03)
    save(fig, "10_matriks_2x2.png")


def chart_ablasi():
    """Tangga ablasi: sumbangan tiap perbaikan, termasuk yang negatif.

    Batang hijau menaikkan akurasi, batang merah menurunkannya. Batang yang
    merah sengaja ditampilkan dan tidak disembunyikan, karena perbaikan yang
    merugikan bila berdiri sendiri merupakan bagian dari temuan.
    """
    from forlib.metrics import full_metrics, prior_matched_threshold

    TANGGA = [
        ("Proposal\napa adanya", "runs/ast_official_proposalULRPK_b32e20_s42"),
        ("Normalisasi\nloudness", "runs/ast_official_proposalULR_b32e20_s42"),
        ("LR per model,\nencoder beku", "runs/ast_official_proposal_b32e20_s42"),
        ("Early stopping\npada EER", "runs/ast_official_proposal_b32e10_s42"),
        ("Augmentasi\npenuh", "runs/ast_official_full_b32e10_s42"),
    ]
    nm, acc = [], []
    for label, d in TANGGA:
        f = os.path.join(HERE, d, "test_scores.npy")
        if not os.path.exists(f):
            continue
        y, p, _ = np.load(f)
        acc.append(full_metrics(y.astype(int), p,
                                prior_matched_threshold(p, 0.5))["accuracy"] * 100)
        nm.append(label)
    if len(acc) < 2:
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.bar(0, acc[0], color="#868E96", width=0.62, zorder=3)
    ax.text(0, acc[0] + 1.2, f"{acc[0]:.2f}", ha="center", fontsize=10,
            weight="bold", color="#495057")
    for i in range(1, len(acc)):
        d = acc[i] - acc[i - 1]
        lo, hi = min(acc[i - 1], acc[i]), max(acc[i - 1], acc[i])
        ax.bar(i, hi - lo, bottom=lo, width=0.62, zorder=3,
               color="#2F9E44" if d >= 0 else "#E03131")
        ax.plot([i - 0.69, i - 0.31], [acc[i - 1]] * 2, color="#ADB5BD",
                lw=1.1, ls="--", zorder=2)
        ax.text(i, hi + 1.2, f"{d:+.2f}", ha="center", fontsize=10.5,
                weight="bold", color="#2F9E44" if d >= 0 else "#E03131")
        ax.text(i, lo - 3.0, f"{acc[i]:.2f}", ha="center", fontsize=9,
                color="#495057")
    ax.set_xticks(range(len(nm)))
    ax.set_xticklabels(nm, fontsize=9.5)
    ax.set_ylabel("Akurasi pada partisi resmi (persen)")
    ax.set_ylim(min(acc) - 8, max(acc) + 7)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_title("Sumbangan tiap perbaikan pada AST, partisi resmi\n"
                 "batang merah menandai perbaikan yang merugikan bila "
                 "berdiri sendiri",
                 fontsize=12, weight="bold")
    save(fig, "11_tangga_ablasi.png")


def chart_matriks_lr():
    """Arah pengaruh perlakuan encoder berbeda antar arsitektur.

    Tiga batang pertama tiap kelompok memakai paket rekayasa yang sama persis
    dan hanya berbeda pada perlakuan encoder, sehingga tingginya dapat
    dibandingkan langsung. Batang keempat adalah konfigurasi proposal apa
    adanya, disertakan sebagai acuan.
    """
    import glob as _g
    from forlib.metrics import full_metrics, prior_matched_threshold

    PERLAKUAN = [
        ("Encoder beku", "runs/{m}_official_full_b{b}e10_s*", "#1864AB"),
        ("Dilatih, laju wajar", "runs/{m}_official_fullUF_b{b}e10_s*", "#2F9E44"),
        ("Dilatih, laju 0,001", "runs/{m}_official_fullUFENC0.001_b{b}e10_s*",
         "#E8590C"),
        ("Proposal apa adanya", "runs/{m}_official_proposalULRPK_b{b}e20_s*",
         "#868E96"),
    ]
    ARS = [("ast", 32, "AST\n86 M, terselia"),
           ("wavlm", 16, "WavLM Large\n300 M, swa-selia"),
           ("hubert", 32, "HuBERT Large\n300 M, swa-selia")]

    data = []
    for m, b, lbl in ARS:
        vals = []
        for _, pat, _ in PERLAKUAN:
            acc = []
            for d in sorted(_g.glob(os.path.join(HERE, pat.format(m=m, b=b)))):
                f = os.path.join(d, "test_scores.npy")
                if not os.path.exists(f):
                    continue
                y, p, _x = np.load(f)
                acc.append(full_metrics(y.astype(int), p,
                                        prior_matched_threshold(p, 0.5))["accuracy"] * 100)
            vals.append(np.mean(acc) if acc else None)
        if any(v is not None for v in vals):
            data.append((lbl, vals))
    if not data:
        return

    fig, ax = plt.subplots(figsize=(12, 5.8))
    n, w = len(PERLAKUAN), 0.2
    for j, (nm, _, cl) in enumerate(PERLAKUAN):
        xs = [i + (j - (n - 1) / 2) * w for i in range(len(data))]
        ys = [(d[1][j] if d[1][j] is not None else 0) for d in data]
        ax.bar(xs, ys, width=w * 0.92, color=cl, label=nm, zorder=3)
        for x, y in zip(xs, ys):
            if y:
                ax.text(x, y + 1.0, f"{y:.1f}", ha="center", fontsize=8.5,
                        weight="bold", color=cl)
    ax.axhline(50, color="#E03131", lw=1.1, ls=":", zorder=2)
    ax.text(-0.48, 51.2, "tebakan acak", fontsize=8.5, color="#E03131",
            ha="left")
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d[0] for d in data], fontsize=10)
    ax.set_ylabel("Akurasi pada partisi resmi (persen)")
    ax.set_ylim(40, 108)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(fontsize=9, ncol=2, loc="lower left", framealpha=0.95)
    ax.set_title("Perlakuan encoder terbaik berbeda menurut arsitektur\n"
                 "laju seragam 0,001 menghancurkan kedua model swa-selia besar",
                 fontsize=12.5, weight="bold")
    save(fig, "12_matriks_encoder.png")


def main():
    print("membuat grafik ...")
    rec = load_for()
    print(f"  {len(rec)} kombinasi (model x split x augmentasi) dari runs/")
    chart_for(rec)
    chart_split(rec)
    chart_snr()
    chart_datasets(rec)
    chart_2x2()
    chart_ablasi()
    chart_matriks_lr()
    print(f"\nselesai -> {os.path.relpath(OUT, HERE)}/")


if __name__ == "__main__":
    main()
