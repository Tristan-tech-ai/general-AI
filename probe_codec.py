"""
Verifikasi hipotesis: apakah pemisahan real/fake di FoR-2sec sebagian besar
berasal dari JEJAK CODEC (fake berasal dari MP3, real dari WAV), bukan dari
artefak sintesis TTS?

Uji:
  P1  provenance nama berkas: berapa banyak fake vs real mengandung '.mp3'
  P2  cutoff spektral per berkas (frekuensi tempat energi runtuh)
  P3  AUC cutoff sebagai fitur tunggal
  P4  akurasi classifier HANYA dari profil pita frekuensi (32 bin energi)
  P5  apa yang terjadi bila kedua kelas di-lowpass ke 4 kHz (pintasan dinetralkan)
"""
import os
import collections
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "for-2seconds")
SPLITS = ["training", "validation", "testing"]
CLASSES = ["real", "fake"]
NB = 32                      # jumlah bin pita untuk profil spektral


def list_files():
    rows = []
    for sp in SPLITS:
        for cl in CLASSES:
            d = os.path.join(ROOT, sp, cl)
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith(".wav"):
                    rows.append((os.path.join(d, fn), sp, cl, fn))
    return rows


def feats(args):
    path, sp, cl, fn = args
    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    w = np.hanning(len(x))
    S = np.abs(np.fft.rfft(x * w)) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    tot = S.sum() + 1e-20

    # profil pita: 32 bin linear 0-8000 Hz, dinormalisasi ke energi total
    edges = np.linspace(0, sr / 2, NB + 1)
    prof = np.array([S[(f >= edges[i]) & (f < edges[i + 1])].sum() / tot for i in range(NB)])

    # cutoff spektral: frekuensi tertinggi yang energinya masih > 1e-5 dari puncak
    Sdb = 10 * np.log10(S / (S.max() + 1e-20) + 1e-20)
    above = np.flatnonzero(Sdb > -60)
    cutoff = float(f[above[-1]]) if len(above) else 0.0
    # cutoff yang lebih ketat (-40 dB)
    above40 = np.flatnonzero(Sdb > -40)
    cutoff40 = float(f[above40[-1]]) if len(above40) else 0.0

    return {
        "split": sp, "cls": cl, "fname": fn,
        "has_mp3": int(".mp3" in fn.lower()),
        "cutoff60": cutoff, "cutoff40": cutoff40,
        "prof": prof,
    }


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float); ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    files = list_files()
    print(f"Memproses {len(files)} berkas ...")
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 4)) as ex:
        R = list(ex.map(feats, files, chunksize=64))

    y = np.array([1 if r["cls"] == "fake" else 0 for r in R])
    L = []
    def out(s=""):
        print(s); L.append(s)

    out("# Probe: Apakah FoR-2sec Memisahkan Codec, Bukan Deepfake?\n")

    # ---------- P1 ----------
    out("## P1 — Provenance nama berkas\n")
    out("| kelas | total | mengandung `.mp3` | persen |")
    out("|---|---|---|---|")
    for cl in CLASSES:
        sub = [r for r in R if r["cls"] == cl]
        m = sum(r["has_mp3"] for r in sub)
        out(f"| {cl} | {len(sub)} | {m} | **{100*m/len(sub):.2f}%** |")
    a_mp3 = auc(y, np.array([r["has_mp3"] for r in R], dtype=float))
    out(f"\nAUC dari fitur tunggal `'.mp3' ada di nama berkas`: **{a_mp3:.4f}**")
    if a_mp3 > 0.99:
        out("\n🔴 **Nama berkas SENDIRI mengungkap label secara sempurna.** "
            "Kelas fake dan real punya riwayat format berbeda (MP3 vs WAV).")
    out("")

    # ---------- P2/P3 ----------
    out("## P2/P3 — Cutoff spektral (batas atas pita frekuensi)\n")
    out("| kelas | cutoff −60 dB (Hz) | cutoff −40 dB (Hz) |")
    out("|---|---|---|")
    for cl in CLASSES:
        sub = [r for r in R if r["cls"] == cl]
        c60 = np.array([r["cutoff60"] for r in sub]); c40 = np.array([r["cutoff40"] for r in sub])
        out(f"| {cl} | {c60.mean():.0f} ± {c60.std():.0f} | {c40.mean():.0f} ± {c40.std():.0f} |")
    for k in ["cutoff60", "cutoff40"]:
        out(f"\nAUC `{k}` sebagai fitur tunggal: **{auc(y, np.array([r[k] for r in R])):.4f}**")
    out("")

    # ---------- P4 ----------
    out("## P4 — Klasifikasi HANYA dari profil 32-bin energi spektral\n")
    out("Tidak ada informasi fonetik, tidak ada fase, tidak ada temporal — "
        "hanya *bentuk* spektrum rata-rata.\n")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import RandomForestClassifier

    X = np.log10(np.array([r["prof"] for r in R]) + 1e-12)
    tr = np.array([i for i, r in enumerate(R) if r["split"] == "training"])
    te = np.array([i for i, r in enumerate(R) if r["split"] == "testing"])

    out("| model | akurasi test |")
    out("|---|---|")
    res = {}
    for name, clf in [
        ("LogReg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))),
        ("RandomForest", RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)),
    ]:
        clf.fit(X[tr], y[tr])
        acc = float((clf.predict(X[te]) == y[te]).mean())
        res[name] = acc
        out(f"| {name} (32 bin spektral) | **{acc*100:.2f}%** |")
    out("")

    # ---------- P5 ----------
    out("## P5 — Setelah pintasan dinetralkan (buang seluruh pita > 4 kHz)\n")
    nb_half = NB // 2      # 32 bin atas 0-8 kHz -> 16 bin pertama = 0-4 kHz
    Xlow = X[:, :nb_half]
    out("| model | akurasi test (hanya 0–4 kHz) | Δ vs pita penuh |")
    out("|---|---|---|")
    for name, clf in [
        ("LogReg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))),
        ("RandomForest", RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)),
    ]:
        clf.fit(Xlow[tr], y[tr])
        acc = float((clf.predict(Xlow[te]) == y[te]).mean())
        out(f"| {name} | **{acc*100:.2f}%** | {(acc-res[name])*100:+.2f} pp |")
    out("")

    # profil rata-rata per kelas untuk dilaporkan
    out("## Profil energi rata-rata per pita (fraksi energi total)\n")
    edges = np.linspace(0, 8000, NB + 1)
    Pm = {cl: np.array([r["prof"] for r in R if r["cls"] == cl]).mean(axis=0) for cl in CLASSES}
    out("| pita (Hz) | real | fake | rasio real/fake |")
    out("|---|---|---|---|")
    for i in range(NB):
        if i % 2 == 0 or i >= NB - 8:
            rr, ff = Pm["real"][i], Pm["fake"][i]
            out(f"| {edges[i]:.0f}–{edges[i+1]:.0f} | {rr:.3e} | {ff:.3e} | {rr/(ff+1e-20):.2f}× |")
    out("")

    open("probe_codec_report.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n-> probe_codec_report.md")


if __name__ == "__main__":
    main()
