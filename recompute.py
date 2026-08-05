"""Hitung ulang semua run dengan ambang prior-matched dari skor yang tersimpan."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.metrics import full_metrics, prior_matched_threshold

rows = []
for d in sorted(glob.glob("runs/*")):
    fs = os.path.join(d, "test_scores.npy")
    fj = os.path.join(d, "results.json")
    if not (os.path.exists(fs) and os.path.exists(fj)):
        continue
    y, p, _ = np.load(fs)
    y = y.astype(int)
    r = json.load(open(fj, encoding="utf-8"))
    thr = prior_matched_threshold(p, 0.5)
    m_prior = full_metrics(y, p, thr)
    m_val = r["test@val_threshold"]
    rows.append((r["tag"], m_val, m_prior, r["val"]))

print("| run | val acc | test EER | test AUC | test acc "
      "(ambang val) | **test acc (prior-matched)** | salah |")
print("|---|---|---|---|---|---|---|")
for tag, mv, mp, val in sorted(rows, key=lambda t: -t[2]["accuracy"]):
    print(f"| `{tag}` | {val['accuracy']*100:.2f}% | {mp['eer']*100:.2f}% | "
          f"{mp['auc']:.4f} | {mv['accuracy']*100:.2f}% | "
          f"**{mp['accuracy']*100:.2f}%** | {mp['n_errors']}/{mp['n']} |")

print("\nDetail (ambang prior-matched):")
for tag, mv, mp, val in sorted(rows, key=lambda t: -t[2]["accuracy"]):
    print(f"  {tag:36s} acc={mp['accuracy']*100:6.2f}%  F1={mp['f1']*100:6.2f}%  "
          f"P={mp['precision']*100:6.2f}%  R={mp['recall']*100:6.2f}%  "
          f"TP={mp['tp']:4d} TN={mp['tn']:4d} FP={mp['fp']:3d} FN={mp['fn']:3d}")
