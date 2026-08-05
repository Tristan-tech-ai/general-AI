"""
Analisis error tersisa pada ensemble terbaik.

Menjawab pertanyaan "apakah sudah maksimal secara struktural?": bila error yang
tersisa adalah berkas rusak / salah label / hening, maka plafon dataset sudah
tercapai dan tidak ada arsitektur yang bisa memperbaikinya. Bila error itu
audio normal, masih ada ruang pemodelan.
"""
import glob
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.data import load_manifest, make_splits
from forlib.metrics import full_metrics, prior_matched_threshold

rows = load_manifest("manifest.csv")
_, _, te = make_splits(rows, "official")

# ensemble seluruh run hubert augmentasi-per-epoch
files = [os.path.join(d, "test_scores.npy")
         for d in sorted(glob.glob("runs/hubert*")) if "b16e10" in d]
files = [f for f in files if os.path.exists(f)]
print(f"menggabungkan {len(files)} run HuBERT")

arrs = [np.load(f) for f in files]
y = arrs[0][0].astype(int)
P = np.mean([a[1] for a in arrs], axis=0)
thr = prior_matched_threshold(P, 0.5)
pred = (P >= thr).astype(int)
m = full_metrics(y, P, thr)
print(f"ensemble: acc={m['accuracy']*100:.2f}%  EER={m['eer']*100:.2f}%  "
      f"salah={m['n_errors']}/{m['n']}")

assert len(te) == len(y), f"urutan test tidak cocok: {len(te)} vs {len(y)}"

wrong = np.flatnonzero(pred != y)
L = []
def out(s=""):
    print(s); L.append(s)

out("# Analisis Error Tersisa — Ensemble HuBERT Terbaik\n")
out(f"Ensemble {len(files)} run HuBERT Large (augmentasi per-epoch), split resmi.")
out(f"Akurasi **{m['accuracy']*100:.2f}%**, EER {m['eer']*100:.2f}%, "
    f"**{len(wrong)} berkas salah dari {len(y)}**.\n")
out("Pertanyaan: apakah sisa error ini *bisa* diperbaiki, atau plafon dataset?\n")

out("| # | berkas | label benar | prediksi | skor | durasi bicara | RMS | energi>6kHz | zero-frac |")
out("|---|---|---|---|---|---|---|---|---|")
details = []
for k, i in enumerate(wrong, 1):
    r = te[i]
    x, sr = sf.read(r["path"], dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    ax = np.abs(x)
    thr_s = max(ax.max() * 0.01, 1e-5)
    nz = np.flatnonzero(ax > thr_s)
    speech = (nz[-1] - nz[0] + 1) / sr if len(nz) else 0.0
    rms = float(np.sqrt((x ** 2).mean()))
    S = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    hf = float(S[f >= 6000].sum() / (S.sum() + 1e-20))
    zf = float((ax < 1e-6).mean())
    details.append(dict(fname=r["fname"], truth=r["cls"], pred="fake" if pred[i] else "real",
                        score=float(P[i]), speech=speech, rms=rms, hf=hf, zf=zf))
    out(f"| {k} | `{r['fname'][:44]}` | **{r['cls']}** | {'fake' if pred[i] else 'real'} | "
        f"{P[i]:.4f} | {speech:.3f}s | {rms:.4f} | {hf:.5f} | {zf:.3f} |")
out("")

# statistik pembanding dari berkas yang BENAR
ok = np.flatnonzero(pred == y)
samp = ok[:: max(1, len(ok) // 200)]
ref = {"speech": [], "rms": [], "hf": [], "zf": []}
for i in samp:
    r = te[i]
    x, sr = sf.read(r["path"], dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    ax = np.abs(x)
    t = max(ax.max() * 0.01, 1e-5)
    nz = np.flatnonzero(ax > t)
    ref["speech"].append((nz[-1] - nz[0] + 1) / sr if len(nz) else 0.0)
    ref["rms"].append(float(np.sqrt((x ** 2).mean())))
    S = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    ref["hf"].append(float(S[f >= 6000].sum() / (S.sum() + 1e-20)))
    ref["zf"].append(float((ax < 1e-6).mean()))

out("## Pembanding: berkas yang diklasifikasi BENAR (sampel n=%d)\n" % len(samp))
out("| properti | error (rerata) | benar (rerata) | benar (p5–p95) | anomali? |")
out("|---|---|---|---|---|")
for key, nm in [("speech", "durasi bicara (s)"), ("rms", "RMS"),
                ("hf", "energi >6 kHz"), ("zf", "fraksi hening")]:
    ev = np.mean([d[key] for d in details])
    rv = np.array(ref[key])
    lo, hi = np.percentile(rv, [5, 95])
    flag = "🔴 YA" if (ev < lo or ev > hi) else "tidak"
    out(f"| {nm} | {ev:.4f} | {rv.mean():.4f} | {lo:.4f} – {hi:.4f} | {flag} |")
out("")

out("## Keyakinan model pada error\n")
sc = np.array([d["score"] for d in details])
near = int(np.sum(np.abs(sc - thr) < 0.15))
out(f"- Ambang keputusan: {thr:.4f}")
out(f"- Skor error: {', '.join(f'{s:.4f}' for s in sc)}")
out(f"- Error yang **dekat ambang** (|skor−ambang| < 0,15): **{near}/{len(sc)}**")
out("")
if near == len(sc):
    out("Seluruh error berada dekat ambang keputusan. Artinya model **ragu**, bukan")
    out("**yakin-salah**. Ini pola khas sampel yang secara intrinsik ambigu, bukan")
    out("pola kegagalan sistematis yang dapat diperbaiki arsitektur.")
else:
    out(f"Ada {len(sc)-near} error dengan keyakinan tinggi — ini pola kegagalan")
    out("sistematis dan **masih ada ruang perbaikan pemodelan**.")
out("")

open("HASIL_ANALISIS_ERROR.md", "w", encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_ANALISIS_ERROR.md")
