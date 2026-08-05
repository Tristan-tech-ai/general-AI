"""
Cari kombinasi ensemble terbaik atas seluruh run yang tersimpan.

Fokus: apakah menambahkan WavLM (pra-pelatihan denoising) memperbaiki 4 sampel
batas yang gagal diurutkan HuBERT? Bila errornya terdekorelasi, gabungan akan
lebih baik daripada masing-masing.
"""
import glob
import itertools
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.metrics import full_metrics, prior_matched_threshold

runs = defaultdict(list)
for d in sorted(glob.glob("runs/*")):
    f = os.path.join(d, "test_scores.npy")
    if not os.path.exists(f):
        continue
    tag = os.path.basename(d)
    m = re.match(r"^(.+?)_official_codec(?:AV)?(?:_b\d+e\d+)?_s\d+$", tag)
    if not m:
        continue
    y, p, _ = np.load(f)
    if len(y) != 1088:
        continue
    runs[m.group(1)].append({"tag": tag, "y": y.astype(int), "p": p})

print("run tersedia:", {k: len(v) for k, v in runs.items()})
y = next(iter(runs.values()))[0]["y"]

L = []
def out(s=""):
    print(s); L.append(s)


def ev(ps):
    P = np.mean(ps, axis=0)
    return full_metrics(y, P, prior_matched_threshold(P, 0.5)), P


out("# Pencarian Ensemble Terbaik\n")
out("Seluruh run split resmi + augmentasi codec, ambang prior-matched, "
    f"test set {len(y)} berkas.\n")

out("## 1. Per arsitektur (ensemble antar-seed)\n")
out("| arsitektur | n run | akurasi | EER | AUC | salah |")
out("|---|---|---|---|---|---|")
per = {}
for k, rs in sorted(runs.items()):
    m, P = ev([r["p"] for r in rs])
    per[k] = P
    out(f"| `{k}` | {len(rs)} | **{m['accuracy']*100:.2f}%** | {m['eer']*100:.2f}% | "
        f"{m['auc']:.4f} | {m['n_errors']}/{m['n']} |")
out("")

out("## 2. Korelasi error antar arsitektur (pada ensemble per-arsitektur)\n")
out("| pasangan | φ | error bersama | hanya A | hanya B |")
out("|---|---|---|---|---|")
errs = {}
for k, P in per.items():
    errs[k] = ((P >= prior_matched_threshold(P, 0.5)).astype(int) != y).astype(int)
for a, b in itertools.combinations(sorted(per), 2):
    ea, eb = errs[a], errs[b]
    phi = float(np.corrcoef(ea, eb)[0, 1]) if ea.std() and eb.std() else float("nan")
    out(f"| {a} vs {b} | {phi:.3f} | {int((ea & eb).sum())} | "
        f"{int((ea & ~eb).sum())} | {int((~ea & eb).sum())} |")
out("")

out("## 3. Seluruh kombinasi arsitektur\n")
out("| kombinasi | akurasi | EER | AUC | salah |")
out("|---|---|---|---|---|")
best = None
names = sorted(per)
for r in range(1, len(names) + 1):
    for combo in itertools.combinations(names, r):
        m, _ = ev([per[c] for c in combo])
        rec = (m["accuracy"], combo, m)
        if best is None or rec[0] > best[0]:
            best = rec
        if r >= 2 or len(names) <= 3:
            out(f"| {' + '.join(combo)} | **{m['accuracy']*100:.2f}%** | "
                f"{m['eer']*100:.2f}% | {m['auc']:.4f} | {m['n_errors']}/{m['n']} |")
out("")

# seluruh run individual (bukan per-arsitektur)
allp = [r["p"] for rs in runs.values() for r in rs]
m_all, _ = ev(allp)
out(f"**Seluruh {len(allp)} run digabung:** {m_all['accuracy']*100:.2f}%  "
    f"EER {m_all['eer']*100:.2f}%  AUC {m_all['auc']:.4f}  "
    f"salah {m_all['n_errors']}/{m_all['n']}\n")

out("## 4. Terbaik\n")
acc, combo, m = best
out(f"**{' + '.join(combo)}** → **{acc*100:.2f}%**, EER {m['eer']*100:.2f}%, "
    f"AUC {m['auc']:.4f}, **{m['n_errors']} salah dari {m['n']}**\n")

# apakah sisa error masih terbalik urutannya?
_, Pb = ev([per[c] for c in combo])
thr = prior_matched_threshold(Pb, 0.5)
w = np.flatnonzero((Pb >= thr).astype(int) != y)
if len(w):
    out("### Sisa error\n")
    out("| label benar | skor | posisi vs ambang |")
    out("|---|---|---|")
    for i in w:
        out(f"| {'fake' if y[i] else 'real'} | {Pb[i]:.4f} | "
            f"{'di atas' if Pb[i] >= thr else 'di bawah'} (ambang {thr:.4f}) |")
    sr = Pb[w][y[w] == 0]
    sf = Pb[w][y[w] == 1]
    if len(sr) and len(sf):
        inv = float(sr.min()) > float(sf.max())
        out("")
        out(f"Skor `real` yang salah: {np.round(sr,4).tolist()}")
        out(f"Skor `fake` yang salah: {np.round(sf,4).tolist()}")
        out(f"**Terbalik urutannya: {'YA — tidak ada ambang yang bisa memperbaikinya' if inv else 'tidak'}**")
out("")

open("HASIL_ENSEMBLE.md", "w", encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_ENSEMBLE.md")
