"""Ringkas akurasi FoR per (arsitektur, augmentasi) atas seluruh seed."""
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.metrics import full_metrics, prior_matched_threshold

g = defaultdict(list)
for d in sorted(glob.glob("runs/*")):
    f = os.path.join(d, "test_scores.npy")
    # Konfigurasi ikut menjadi bagian kunci. Tanpa ini, run dengan jumlah epoch
    # atau batch yang berbeda tergabung menjadi satu kelompok dan sebarannya
    # terlaporkan sebagai ragam antar inisialisasi acak.
    m = re.match(r"^(.+?)_official_([a-z]+?)(_b\d+e\d+)?_s(\d+)$",
                 os.path.basename(d))
    if not (os.path.exists(f) and m):
        continue
    y, p, _ = np.load(f)
    if len(y) != 1088:
        continue
    met = full_metrics(y.astype(int), p, prior_matched_threshold(p, 0.5))
    kunci_aug = m.group(2) + "@" + (m.group(3) or "_lama").lstrip("_")
    g[(m.group(1), kunci_aug)].append((m.group(4), met["accuracy"] * 100,
                                        met["eer"] * 100))

print(f"{'arsitektur':22s} {'augmentasi':10s} {'n':>2s} {'akurasi':>9s} "
      f"{'std':>7s} {'EER':>7s}   seed")
print("-" * 88)
for (a, au), v in sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    acc = np.array([x[1] for x in v])
    eer = np.array([x[2] for x in v])
    sd = acc.std(ddof=1) if len(acc) > 1 else 0.0
    seeds = ", ".join(f"{x[1]:.2f}" for x in sorted(v, key=lambda t: -t[1]))
    print(f"{a:22s} {au:10s} {len(v):2d} {acc.mean():8.2f}% {sd:6.2f} "
          f"{eer.mean():6.2f}%   {seeds}")
