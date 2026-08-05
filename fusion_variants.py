"""
Varian fusi selain rata-rata probabilitas polos.

Temuan sebelumnya: rata-rata polos MENGENCERKAN model dominan (HuBERT sendirian
99,63% > gabungan semua 98,53%). Skrip ini menguji apakah skema fusi yang lebih
baik dapat memanfaatkan keberagaman tanpa dilusi.

Juga memeriksa: apakah setiap seed HuBERT secara individual salah pada 4 sampel
batas yang sama, atau berbeda-beda? Bila berbeda, menambah seed bisa membaliknya.
"""
import glob
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
    m = re.match(r"^(.+?)_official_codec(?:AV)?(?:_b\d+e\d+)?_s(\d+)$",
                 os.path.basename(d))
    if not m:
        continue
    y, p, _ = np.load(f)
    if len(y) != 1088:
        continue
    runs[m.group(1)].append({"seed": int(m.group(2)), "y": y.astype(int), "p": p})

y = next(iter(runs.values()))[0]["y"]
L = []
def out(s=""):
    print(s); L.append(s)


def sc(P):
    return full_metrics(y, P, prior_matched_threshold(P, 0.5))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def ranks(p):
    o = np.argsort(np.argsort(p))
    return o / (len(p) - 1.0)


out("# Varian Fusi: Mengatasi Dilusi Model Dominan\n")
out("Rata-rata probabilitas polos membuat HuBERT (99,63%) turun ketika digabung "
    "dengan model lemah. Di sini diuji skema fusi alternatif.\n")

hub = runs.get("hubert", [])
out(f"Run HuBERT tersedia: {len(hub)} (seed {sorted(r['seed'] for r in hub)})\n")

# ---------- 1. per-seed pada sampel batas ----------
if hub:
    P_h = np.mean([r["p"] for r in hub], axis=0)
    thr = prior_matched_threshold(P_h, 0.5)
    wrong = np.flatnonzero((P_h >= thr).astype(int) != y)
    out("## 1. Perilaku tiap seed HuBERT pada sampel yang salah di ensemble\n")
    if len(wrong):
        hdr = "| sampel | label | " + " | ".join(f"seed {r['seed']}" for r in hub) + " | ensemble |"
        out(hdr)
        out("|" + "---|" * (len(hub) + 3))
        for i in wrong:
            cells = []
            for r in hub:
                t = prior_matched_threshold(r["p"], 0.5)
                ok = int(r["p"][i] >= t) == y[i]
                cells.append(("✅" if ok else "❌") + f" {r['p'][i]:.3f}")
            out(f"| {i} | **{'fake' if y[i] else 'real'}** | " + " | ".join(cells) +
                f" | ❌ {P_h[i]:.3f} |")
        out("")
        n_all_wrong = 0
        for i in wrong:
            allw = all(int(r["p"][i] >= prior_matched_threshold(r["p"], 0.5)) != y[i]
                       for r in hub)
            n_all_wrong += int(allw)
        out(f"Sampel yang salah di **SEMUA** seed: **{n_all_wrong}/{len(wrong)}**")
        if n_all_wrong == len(wrong):
            out("\n→ Setiap seed gagal pada sampel yang sama. Menambah seed HuBERT "
                "**tidak akan** memperbaikinya. Ini plafon kelas model, bukan derau seed.")
        else:
            out(f"\n→ {len(wrong)-n_all_wrong} sampel diperbaiki oleh sebagian seed. "
                "Menambah seed berpeluang membaliknya.")
    out("")

# ---------- 2. skema fusi ----------
out("## 2. Perbandingan skema fusi (seluruh arsitektur)\n")
per = {k: np.mean([r["p"] for r in rs], axis=0) for k, rs in runs.items()}
names = sorted(per)
out("| skema | akurasi | EER | AUC | salah |")
out("|---|---|---|---|---|")

schemes = {}
schemes["rata-rata probabilitas (baseline)"] = np.mean([per[k] for k in names], axis=0)
schemes["rata-rata logit"] = 1 / (1 + np.exp(-np.mean([logit(per[k]) for k in names], axis=0)))
schemes["rata-rata peringkat"] = np.mean([ranks(per[k]) for k in names], axis=0)
schemes["median probabilitas"] = np.median([per[k] for k in names], axis=0)
schemes["maksimum probabilitas"] = np.max([per[k] for k in names], axis=0)
# berbobot AUC^k
aucs = {k: sc(per[k])["auc"] for k in names}
for pw in [4, 16, 64]:
    w = np.array([aucs[k] ** pw for k in names]); w = w / w.sum()
    schemes[f"berbobot AUC^{pw}"] = np.sum([w[i] * per[k] for i, k in enumerate(names)], axis=0)
# hanya top-k menurut AUC
top = sorted(names, key=lambda k: -aucs[k])
for k in [1, 2, 3]:
    schemes[f"top-{k} menurut AUC ({', '.join(top[:k])})"] = np.mean([per[t] for t in top[:k]], axis=0)

best = None
for nm, P in schemes.items():
    m = sc(P)
    if best is None or m["accuracy"] > best[0]:
        best = (m["accuracy"], nm, m)
    out(f"| {nm} | **{m['accuracy']*100:.2f}%** | {m['eer']*100:.2f}% | "
        f"{m['auc']:.4f} | {m['n_errors']}/{m['n']} |")
out("")
out(f"**Terbaik: {best[1]} → {best[0]*100:.2f}%, "
    f"{best[2]['n_errors']} salah dari {best[2]['n']}**\n")

open("HASIL_FUSI.md", "w", encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_FUSI.md")
