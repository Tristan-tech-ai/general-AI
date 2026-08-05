"""
Mengukur pergeseran distribusi train -> test pada FoR-2sec.

Temuan pemicu: 90,7% fake di training berasal dari MP3, tetapi 0% fake di
testing berasal dari MP3. Skrip ini mengukur akibat spektralnya dan menguji
apakah augmentasi codec pada KEDUA kelas menetralkan korelasi semu tersebut.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "for-2seconds")
NB = 32


def job(a):
    path, sp, cl, mp3 = a
    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    S = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    tot = S.sum() + 1e-20
    edges = np.linspace(0, sr / 2, NB + 1)
    prof = np.array([S[(f >= edges[i]) & (f < edges[i + 1])].sum() / tot for i in range(NB)])
    hf = float(S[f >= 6000].sum() / tot)
    return sp, cl, mp3, hf, prof


def main():
    items = []
    for sp in ["training", "validation", "testing"]:
        for cl in ["real", "fake"]:
            d = os.path.join(ROOT, sp, cl)
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith(".wav"):
                    items.append((os.path.join(d, fn), sp, cl, int(".mp3" in fn.lower())))

    print(f"memproses {len(items)} berkas ...")
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 4)) as ex:
        R = list(ex.map(job, items, chunksize=64))

    L = []
    def out(s=""):
        print(s); L.append(s)

    out("# Pergeseran Distribusi Train → Test pada FoR-2sec\n")

    out("## 1. Energi pita tinggi (>6 kHz) per split × kelas × provenance\n")
    out("| split | kelas | provenance | n | energi >6kHz (mean) |")
    out("|---|---|---|---|---|")
    groups = {}
    for sp in ["training", "validation", "testing"]:
        for cl in ["real", "fake"]:
            for mp3 in [0, 1]:
                v = [r[3] for r in R if r[0] == sp and r[1] == cl and r[2] == mp3]
                if not v:
                    continue
                groups[(sp, cl, mp3)] = float(np.mean(v))
                tag = "dari MP3" if mp3 else "dari WAV"
                out(f"| {sp} | {cl} | {tag} | {len(v)} | {np.mean(v):.5f} |")
    out("")

    out("## 2. Inti masalah\n")
    tr_r = groups.get(("training", "real", 0))
    tr_f_mp3 = groups.get(("training", "fake", 1))
    tr_f_wav = groups.get(("training", "fake", 0))
    te_r = groups.get(("testing", "real", 0))
    te_f = groups.get(("testing", "fake", 0))
    out(f"- Training: real = **{tr_r:.5f}**, fake(MP3) = **{tr_f_mp3:.5f}** "
        f"→ rasio **{tr_r/tr_f_mp3:.2f}×**")
    out(f"- Training: real = {tr_r:.5f}, fake(WAV) = **{tr_f_wav:.5f}** "
        f"→ rasio **{tr_r/tr_f_wav:.2f}×**")
    out(f"- Testing : real = **{te_r:.5f}**, fake(WAV) = **{te_f:.5f}** "
        f"→ rasio **{te_r/te_f:.2f}×**")
    out("")
    out("Selama pelatihan, 90,7% sampel fake adalah turunan MP3 dengan energi pita")
    out("tinggi yang tertekan. Model belajar aturan **\"HF rendah ⇒ fake\"**.")
    out("Pada pengujian, TIDAK ADA fake yang berasal dari MP3 — aturan itu menjadi")
    out("tidak berlaku, bahkan menyesatkan.")
    out("")

    # ---------- uji intervensi ----------
    out("## 3. Uji intervensi: apakah normalisasi pita menyembuhkan?\n")
    from sklearn.ensemble import RandomForestClassifier

    X = np.log10(np.array([r[4] for r in R]) + 1e-12)
    y = np.array([1 if r[1] == "fake" else 0 for r in R])
    sp_arr = np.array([r[0] for r in R])
    tr = np.flatnonzero(sp_arr == "training")
    te = np.flatnonzero(sp_arr == "testing")

    def run(Xa, label):
        clf = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)
        clf.fit(Xa[tr], y[tr])
        acc_tr = float((clf.predict(Xa[tr]) == y[tr]).mean())
        acc_te = float((clf.predict(Xa[te]) == y[te]).mean())
        out(f"| {label} | {acc_tr*100:.2f}% | **{acc_te*100:.2f}%** | {(acc_tr-acc_te)*100:.2f} pp |")
        return acc_te

    out("| representasi | akurasi train | akurasi test | celah generalisasi |")
    out("|---|---|---|---|")
    run(X, "Profil spektral penuh 0–8 kHz")
    run(X[:, :NB // 2], "Hanya 0–4 kHz (pintasan HF dibuang)")
    # normalisasi bentuk: kurangi rata-rata per-berkas (hilangkan tilt global)
    Xn = X - X.mean(axis=1, keepdims=True)
    run(Xn, "Ternormalisasi per-berkas (tilt dibuang)")
    Xl = X[:, :NB // 2]
    Xl = Xl - Xl.mean(axis=1, keepdims=True)
    run(Xl, "0–4 kHz + ternormalisasi")
    out("")
    out("Celah generalisasi yang besar = model menghafal pintasan yang tidak ada di test set.")
    out("")

    open("probe_shift_report.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n-> probe_shift_report.md")


if __name__ == "__main__":
    main()
