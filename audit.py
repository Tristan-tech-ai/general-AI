"""
Audit integritas dataset FoR-2sec — ground truth sebelum training apa pun.

Menjalankan tes kontrol negatif dari ANALISIS_KRITIS.md §5:
  T1  fraksi sampel digital-silence per kelas  -> AUC fitur tunggal
  T2  fitur trivial (durasi, RMS, peak, dst)   -> akurasi LogReg
  T3  duplikat eksak lintas split (kebocoran)
  T4  energi pita tinggi (>6 kHz) per kelas    -> menguji klaim artefak HF
  T5  analisis pola nama berkas                -> jejak sumber/pembicara

Output: audit_report.md + audit_features.csv
"""
import os
import sys
import csv
import json
import hashlib
import collections
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "for-2seconds")
SPLITS = ["training", "validation", "testing"]
CLASSES = ["real", "fake"]   # real=0, fake=1


def list_files():
    rows = []
    for sp in SPLITS:
        for cl in CLASSES:
            d = os.path.join(ROOT, sp, cl)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith(".wav"):
                    rows.append((os.path.join(d, fn), sp, cl, fn))
    return rows


def analyze(args):
    """Ekstrak fitur murah + hash. Dijalankan di proses terpisah."""
    path, sp, cl, fn = args
    try:
        x, sr = sf.read(path, dtype="float64", always_2d=True)
    except Exception as e:
        return {"path": path, "split": sp, "cls": cl, "fname": fn, "error": repr(e)}

    n_ch = x.shape[1]
    x = x.mean(axis=1)                       # ke mono
    n = len(x)
    dur = n / sr if sr else 0.0
    ax = np.abs(x)

    peak = float(ax.max()) if n else 0.0
    rms = float(np.sqrt((x ** 2).mean())) if n else 0.0
    dc = float(x.mean()) if n else 0.0

    # --- T1: digital silence ---
    zero_frac = float((ax < 1e-6).mean()) if n else 0.0
    # true digital zero (bit-exact)
    exact_zero = float((x == 0.0).mean()) if n else 0.0

    # leading / trailing silence (ambang relatif)
    thr = max(peak * 0.01, 1e-5)
    nz = np.flatnonzero(ax > thr)
    if len(nz):
        lead = float(nz[0] / sr)
        trail = float((n - 1 - nz[-1]) / sr)
        speech = float((nz[-1] - nz[0] + 1) / sr)
    else:
        lead = trail = dur
        speech = 0.0

    # clipping
    clip_frac = float((ax > 0.999).mean()) if n else 0.0

    # --- T4: distribusi energi spektral ---
    seg = x[: min(n, sr * 2)]
    if len(seg) >= 512:
        w = np.hanning(len(seg))
        S = np.abs(np.fft.rfft(seg * w)) ** 2
        freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
        tot = S.sum() + 1e-20
        def band(lo, hi):
            m = (freqs >= lo) & (freqs < hi)
            return float(S[m].sum() / tot)
        e_0_1k, e_1_4k, e_4_6k, e_6_8k = band(0, 1000), band(1000, 4000), band(4000, 6000), band(6000, 8000)
        centroid = float((freqs * S).sum() / tot)
        cs = np.cumsum(S) / tot
        idx = int(np.searchsorted(cs, 0.95))
        rolloff95 = float(freqs[min(idx, len(freqs) - 1)])
        # kemiringan spektral pita tinggi (indikator batas vocoder)
        hi = (freqs >= 5000) & (freqs <= 7800)
        if hi.sum() > 10:
            slope = float(np.polyfit(freqs[hi], 10 * np.log10(S[hi] + 1e-20), 1)[0])
        else:
            slope = 0.0
    else:
        e_0_1k = e_1_4k = e_4_6k = e_6_8k = centroid = rolloff95 = slope = 0.0

    # zero crossing rate
    zcr = float((np.diff(np.signbit(x)) != 0).mean()) if n > 1 else 0.0

    with open(path, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()

    return {
        "path": path, "split": sp, "cls": cl, "fname": fn, "error": "",
        "sr": sr, "n_ch": n_ch, "n_samp": n, "dur": dur,
        "peak": peak, "rms": rms, "dc": dc,
        "zero_frac": zero_frac, "exact_zero": exact_zero,
        "lead_sil": lead, "trail_sil": trail, "speech_dur": speech,
        "clip_frac": clip_frac, "zcr": zcr,
        "e_0_1k": e_0_1k, "e_1_4k": e_1_4k, "e_4_6k": e_4_6k, "e_6_8k": e_6_8k,
        "centroid": centroid, "rolloff95": rolloff95, "hf_slope": slope,
        "md5": md5,
    }


def auc(y, s):
    """AUC via peringkat (Mann-Whitney). y: 1=fake, 0=real."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # rata-ratakan peringkat untuk nilai seri
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    if not os.path.isdir(ROOT):
        print("Dataset belum ada di", ROOT); sys.exit(1)

    files = list_files()
    print(f"Menganalisis {len(files)} berkas ...")
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 4)) as ex:
        recs = list(ex.map(analyze, files, chunksize=64))

    errs = [r for r in recs if r.get("error")]
    recs = [r for r in recs if not r.get("error")]
    print(f"selesai. rusak/gagal-baca: {len(errs)}")

    keys = [k for k in recs[0].keys() if k != "path"]
    with open("audit_features.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)

    y = np.array([1 if r["cls"] == "fake" else 0 for r in recs])
    L = []
    def out(s=""):
        print(s); L.append(s)

    out("# Audit Integritas Dataset FoR-2sec\n")
    out(f"Total berkas dianalisis: **{len(recs)}** (gagal dibaca: {len(errs)})\n")

    # ---------- ringkasan split ----------
    out("## 1. Komposisi split\n")
    out("| split | real | fake | total |")
    out("|---|---|---|---|")
    cnt = collections.Counter((r["split"], r["cls"]) for r in recs)
    for sp in SPLITS:
        a, b = cnt[(sp, "real")], cnt[(sp, "fake")]
        out(f"| {sp} | {a} | {b} | {a+b} |")
    out("")

    # ---------- format ----------
    out("## 2. Keseragaman format\n")
    srs = collections.Counter(r["sr"] for r in recs)
    chs = collections.Counter(r["n_ch"] for r in recs)
    durs = np.array([r["dur"] for r in recs])
    out(f"- Sampling rate: `{dict(srs)}`")
    out(f"- Kanal: `{dict(chs)}`")
    out(f"- Durasi: min={durs.min():.3f}s  median={np.median(durs):.3f}s  "
        f"maks={durs.max():.3f}s  std={durs.std():.4f}s")
    for cl in CLASSES:
        d = durs[y == (1 if cl == "fake" else 0)]
        out(f"  - {cl}: mean={d.mean():.4f}s std={d.std():.4f}s")
    out("")

    # ---------- T3 duplikat ----------
    out("## 3. T3 — Duplikat eksak & kebocoran lintas split\n")
    bymd5 = collections.defaultdict(list)
    for r in recs:
        bymd5[r["md5"]].append(r)
    dups = {k: v for k, v in bymd5.items() if len(v) > 1}
    cross, xclass = 0, 0
    for v in dups.values():
        if len({r["split"] for r in v}) > 1:
            cross += 1
        if len({r["cls"] for r in v}) > 1:
            xclass += 1
    out(f"- Grup duplikat byte-identik: **{len(dups)}** "
        f"(total {sum(len(v) for v in dups.values())} berkas)")
    out(f"- Duplikat yang MELINTASI split (kebocoran train/test): **{cross}**")
    out(f"- Duplikat dengan label berbeda (konflik label): **{xclass}**")
    if cross:
        out("\n  ⚠️ **KEBOCORAN TERDETEKSI.** Contoh:")
        shown = 0
        for v in dups.values():
            if len({r["split"] for r in v}) > 1 and shown < 5:
                out("  - " + " | ".join(f"{r['split']}/{r['cls']}/{r['fname']}" for r in v))
                shown += 1
    else:
        out("\n  ✅ Tidak ada duplikat byte-identik yang melintasi split.")
    out("")

    # ---------- T1/T2/T4 daya diskriminatif fitur trivial ----------
    out("## 4. T1/T2/T4 — Daya diskriminatif fitur non-semantik\n")
    out("AUC 0,5 = tidak informatif; AUC → 1,0 atau → 0,0 = sangat memisahkan kelas.")
    out("**AUC tinggi pada fitur trivial = pintasan (shortcut) dataset.**\n")
    FEATS = ["exact_zero", "zero_frac", "lead_sil", "trail_sil", "speech_dur",
             "peak", "rms", "dc", "clip_frac", "zcr",
             "e_0_1k", "e_1_4k", "e_4_6k", "e_6_8k",
             "centroid", "rolloff95", "hf_slope", "dur"]
    out("| fitur | AUC | mean(real) | mean(fake) | tafsir |")
    out("|---|---|---|---|---|")
    aucs = {}
    for k in FEATS:
        v = np.array([r[k] for r in recs], dtype=float)
        a = auc(y, v)
        aucs[k] = a
        mr, mf = v[y == 0].mean(), v[y == 1].mean()
        dev = abs(a - 0.5)
        tag = ("🔴 pintasan kuat" if dev > 0.30 else
               "🟠 pintasan sedang" if dev > 0.20 else
               "🟡 lemah" if dev > 0.10 else "✅ netral")
        out(f"| `{k}` | {a:.4f} | {mr:.5g} | {mf:.5g} | {tag} |")
    out("")

    # LogReg pada fitur trivial (uji T2)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.ensemble import RandomForestClassifier

        tr = [i for i, r in enumerate(recs) if r["split"] == "training"]
        te = [i for i, r in enumerate(recs) if r["split"] == "testing"]
        X = np.array([[r[k] for k in FEATS] for r in recs], dtype=float)
        X = np.nan_to_num(X)

        out("### Klasifikasi HANYA dari fitur trivial (tanpa melihat isi wicara)\n")
        out("Dilatih pada split `training` resmi, diuji pada split `testing` resmi.\n")
        out("| model | akurasi test | tafsir |")
        out("|---|---|---|")
        for name, clf in [
            ("LogReg (18 fitur trivial)", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
            ("RandomForest (18 fitur trivial)", RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)),
        ]:
            clf.fit(X[tr], y[tr])
            acc = float((clf.predict(X[te]) == y[te]).mean())
            tag = ("🔴 dataset sangat mudah ditembus" if acc > 0.90 else
                   "🟠 pintasan signifikan" if acc > 0.75 else
                   "🟡 ada sinyal trivial" if acc > 0.65 else "✅ wajar")
            out(f"| {name} | **{acc*100:.2f}%** | {tag} |")

        # fitur tunggal terbaik
        best = max(aucs, key=lambda k: abs(aucs[k] - 0.5))
        out(f"\nFitur trivial tunggal paling diskriminatif: **`{best}`** (AUC {aucs[best]:.4f})")
    except Exception as e:
        out(f"\n(sklearn gagal: {type(e).__name__}: {e})")
    out("")

    # ---------- T5 pola nama berkas ----------
    out("## 5. T5 — Pola nama berkas (jejak sumber / pembicara)\n")
    for cl in CLASSES:
        names = [r["fname"] for r in recs if r["cls"] == cl]
        out(f"**{cl}** — {len(names)} berkas. Contoh: `{names[0]}`, `{names[1]}`, `{names[2]}`")
        pref = collections.Counter(n.split("_")[0][:24] for n in names)
        out(f"  prefiks unik: {len(pref)}; 8 terbanyak: {pref.most_common(8)}")
    out("")

    # ---------- resolusi statistik ----------
    out("## 6. Resolusi statistik test set\n")
    nte = int(sum(1 for r in recs if r["split"] == "testing"))
    out(f"Ukuran test set resmi: **{nte}** berkas.\n")
    out("| akurasi | jumlah error | CI 95% (±) |")
    out("|---|---|---|")
    for p in [0.95, 0.97, 0.98, 0.99, 0.995, 1.0]:
        err = int(round((1 - p) * nte))
        ci = 1.96 * np.sqrt(p * (1 - p) / nte) * 100
        out(f"| {p*100:.1f}% | {err} berkas | ±{ci:.2f} pp |")
    out("")
    out(f"**Implikasi:** pada {nte} berkas uji, selisih 1 berkas = "
        f"{100.0/nte:.3f} pp. Dua model yang berbeda < {1.96*np.sqrt(0.98*0.02/nte)*200:.1f} pp "
        f"tidak dapat dibedakan secara statistik tanpa uji berpasangan (McNemar).")
    out("")

    with open("audit_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    json.dump({k: float(v) for k, v in aucs.items()}, open("audit_aucs.json", "w"), indent=1)
    print("\n-> audit_report.md, audit_features.csv, audit_aucs.json")


if __name__ == "__main__":
    main()
