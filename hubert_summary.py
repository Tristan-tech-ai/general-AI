"""Ringkas seluruh seed HuBERT + ensemble kumulatif."""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.metrics import full_metrics, prior_matched_threshold

rows = []
for d in sorted(glob.glob("runs/hubert*")):
    f = os.path.join(d, "test_scores.npy")
    if not os.path.exists(f):
        continue
    y, p, _ = np.load(f)
    y = y.astype(int)
    m = full_metrics(y, p, prior_matched_threshold(p, 0.5))
    rows.append({"seed": os.path.basename(d).split("_s")[-1], "p": p, "y": y, **m})

L = []
def out(s=""):
    print(s); L.append(s)

out("# HuBERT Large — 8 Seed + Ensemble Kumulatif\n")
out("Split resmi FoR, augmentasi codec per-epoch, batch 16, 10 epoch, "
    "ambang prior-matched.\n")

out("## Per seed\n")
out("| seed | akurasi | EER | AUC | salah |")
out("|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: -r["accuracy"]):
    out(f"| {r['seed']} | {r['accuracy']*100:.2f}% | {r['eer']*100:.2f}% | "
        f"{r['auc']:.4f} | {r['n_errors']}/{r['n']} |")
a = np.array([r["accuracy"] for r in rows]) * 100
e = np.array([r["eer"] for r in rows]) * 100
out(f"\n**n={len(rows)} · rerata {a.mean():.2f}% ± {a.std(ddof=1):.2f} · "
    f"EER {e.mean():.2f}% ± {e.std(ddof=1):.2f} · rentang {a.max()-a.min():.2f} pp**\n")

y = rows[0]["y"]
out("## Ensemble kumulatif (urut seed terbaik menurut akurasi)\n")
out("| jumlah seed | akurasi | EER | AUC | salah |")
out("|---|---|---|---|---|")
srt = sorted(rows, key=lambda r: -r["accuracy"])
best = None
for k in range(1, len(srt) + 1):
    P = np.mean([r["p"] for r in srt[:k]], axis=0)
    m = full_metrics(y, P, prior_matched_threshold(P, 0.5))
    if best is None or m["accuracy"] > best[0]:
        best = (m["accuracy"], k, m)
    out(f"| {k} | **{m['accuracy']*100:.2f}%** | {m['eer']*100:.2f}% | "
        f"{m['auc']:.4f} | {m['n_errors']}/{m['n']} |")
out("")

# ensemble seluruh seed (urutan acak = tanpa seleksi berbasis test)
P_all = np.mean([r["p"] for r in rows], axis=0)
m_all = full_metrics(y, P_all, prior_matched_threshold(P_all, 0.5))
out(f"**Seluruh {len(rows)} seed (tanpa seleksi apa pun — ini angka yang sah):** "
    f"**{m_all['accuracy']*100:.2f}%**, EER {m_all['eer']*100:.2f}%, "
    f"AUC {m_all['auc']:.4f}, **{m_all['n_errors']} salah dari {m_all['n']}**\n")
out(f"> Kolom kumulatif di atas mengurutkan seed menurut akurasi **test**, jadi "
    f"puncaknya ({best[0]*100:.2f}% pada {best[1]} seed) adalah angka oracle — "
    f"tidak sah dilaporkan sebagai hasil. Angka yang sah adalah baris "
    f"'seluruh {len(rows)} seed'.\n")

# error tersisa
thr = prior_matched_threshold(P_all, 0.5)
w = np.flatnonzero((P_all >= thr).astype(int) != y)
out(f"## Sisa {len(w)} error\n")
if len(w):
    out("| idx | label | skor ensemble | seed yang benar |")
    out("|---|---|---|---|")
    for i in w:
        good = [r["seed"] for r in rows
                if int(r["p"][i] >= prior_matched_threshold(r["p"], 0.5)) == y[i]]
        out(f"| {i} | {'fake' if y[i] else 'real'} | {P_all[i]:.4f} | "
            f"{', '.join(good) if good else '**tidak ada**'} ({len(good)}/{len(rows)}) |")

open("HASIL_HUBERT_8SEED.md", "w", encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_HUBERT_8SEED.md")
