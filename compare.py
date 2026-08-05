"""
Perbandingan antar-arsitektur yang sah secara statistik.

Menjawab masalah yang diidentifikasi di ANALISIS_KRITIS.md §F4: test set hanya
1.088 berkas, 1 berkas = 0,092 pp, dan variansi antar-seed (+-3,5 pp) jauh
melampaui CI statistik (+-0,83 pp). Membandingkan run tunggal tidak sah.

Yang dilakukan:
  1. agregasi rerata +- std per model atas seluruh seed
  2. uji McNemar berpasangan (test set identik -> data berpasangan)
  3. koreksi Holm-Bonferroni untuk perbandingan ganda
  4. korelasi error antar model (memprediksi apakah ensembling akan berguna)
  5. ensemble sederhana: rata-rata skor atas model & seed
"""
from __future__ import annotations

import glob
import itertools
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.metrics import (full_metrics, prior_matched_threshold,
                            mcnemar, holm_bonferroni)

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "official"
AUG = sys.argv[2] if len(sys.argv) > 2 else "codec"


def parse(tag):
    """model_split_aug[AV]_[bXeY_]sZ"""
    m = re.match(r"^(.+?)_(official|random|clean_val|wavval)_([a-z]+?)(AV)?"
                 r"(?:_b(\d+)e(\d+))?_s(\d+)$", tag)
    if not m:
        return None
    return {"model": m.group(1), "split": m.group(2), "aug": m.group(3),
            "augval": bool(m.group(4)), "batch": m.group(5),
            "epochs": m.group(6), "seed": int(m.group(7))}


runs = []
for d in sorted(glob.glob("runs/*")):
    fs, fj = os.path.join(d, "test_scores.npy"), os.path.join(d, "results.json")
    if not (os.path.exists(fs) and os.path.exists(fj)):
        continue
    tag = os.path.basename(d)
    info = parse(tag)
    if not info or info["split"] != SPLIT or info["aug"] != AUG or info["augval"]:
        continue
    y, p, _ = np.load(fs)
    runs.append({**info, "tag": tag, "y": y.astype(int), "p": p})

if not runs:
    print(f"tidak ada run untuk split={SPLIT} aug={AUG}")
    sys.exit()

n_test = len(runs[0]["y"])
runs = [r for r in runs if len(r["y"]) == n_test]
by_model = defaultdict(list)
for r in runs:
    by_model[r["model"]].append(r)

L = []
def out(s=""):
    print(s); L.append(s)

out(f"# Perbandingan Arsitektur — split `{SPLIT}`, augmentasi `{AUG}`\n")
out(f"Test set: **{n_test}** berkas. 1 berkas = {100/n_test:.3f} pp.\n")
out("Ambang: prior-matched (transduktif, tanpa label test).\n")

# ---------- 1. agregasi ----------
out("## 1. Rerata ± simpangan baku atas seed\n")
out("| model | n seed | akurasi | EER | AUC | seed individual |")
out("|---|---|---|---|---|---|")
agg = {}
for mdl, rs in sorted(by_model.items()):
    accs, eers, aucs = [], [], []
    for r in rs:
        m = full_metrics(r["y"], r["p"], prior_matched_threshold(r["p"], 0.5))
        accs.append(m["accuracy"] * 100); eers.append(m["eer"] * 100); aucs.append(m["auc"])
        r["pred"] = (r["p"] >= prior_matched_threshold(r["p"], 0.5)).astype(int)
    a, e, u = np.array(accs), np.array(eers), np.array(aucs)
    sd = a.std(ddof=1) if len(a) > 1 else float("nan")
    agg[mdl] = {"acc": a.mean(), "std": sd, "eer": e.mean(), "auc": u.mean(), "n": len(a)}
    indiv = ", ".join(f"{x:.2f}" for x in sorted(accs, reverse=True))
    sdtxt = f"±{sd:.2f}" if len(a) > 1 else "n/a"
    out(f"| `{mdl}` | {len(a)} | **{a.mean():.2f}%** {sdtxt} | {e.mean():.2f}% | "
        f"{u.mean():.4f} | {indiv} |")
out("")

order = sorted(agg, key=lambda k: -agg[k]["acc"])
if len(order) > 1:
    top, second = order[0], order[1]
    gap = agg[top]["acc"] - agg[second]["acc"]
    pooled = np.sqrt(np.nanmean([agg[top]["std"] ** 2, agg[second]["std"] ** 2]))
    out(f"Selisih peringkat 1 (`{top}`) dan 2 (`{second}`): **{gap:.2f} pp**, "
        f"sedangkan std gabungan **±{pooled:.2f} pp**.")
    out(f"→ {'**Selisih berada DI DALAM derau — peringkat tidak dapat dipertahankan.**' if gap < pooled else 'Selisih melampaui satu simpangan baku.'}\n")

# ---------- 2. McNemar ----------
out("## 2. Uji McNemar berpasangan (seed terbaik per model)\n")
out("Test set identik untuk semua model → data berpasangan. "
    "n01 = A benar & B salah; n10 = A salah & B benar.\n")
best = {}
for mdl, rs in by_model.items():
    best[mdl] = max(rs, key=lambda r: full_metrics(
        r["y"], r["p"], prior_matched_threshold(r["p"], 0.5))["accuracy"])

pvals, details = {}, {}
for a, b in itertools.combinations(sorted(best), 2):
    ra, rb = best[a], best[b]
    res = mcnemar(ra["y"], ra["pred"], rb["pred"])
    key = f"{a} vs {b}"
    pvals[key] = res["p_value"]
    details[key] = res

if pvals:
    adj = holm_bonferroni(pvals)
    out("| perbandingan | n01 | n10 | p mentah | p terkoreksi | signifikan? |")
    out("|---|---|---|---|---|---|")
    for k in sorted(adj, key=lambda k: adj[k]["p_raw"]):
        d = details[k]
        sig = "✅ ya" if adj[k]["significant"] else "❌ tidak"
        out(f"| {k} | {d['n01']} | {d['n10']} | {adj[k]['p_raw']:.4g} | "
            f"{adj[k]['p_adj']:.4g} | {sig} |")
    out("\nKoreksi Holm-Bonferroni, α = 0,05.\n")

# ---------- 3. korelasi error ----------
out("## 3. Korelasi error antar model\n")
out("φ rendah (< 0,5) → model gagal pada berkas berbeda → ensembling berguna.\n")
out("| pasangan | φ | Jaccard error |")
out("|---|---|---|")
for a, b in itertools.combinations(sorted(best), 2):
    ea = (best[a]["pred"] != best[a]["y"]).astype(int)
    eb = (best[b]["pred"] != best[b]["y"]).astype(int)
    if ea.std() == 0 or eb.std() == 0:
        phi = float("nan")
    else:
        phi = float(np.corrcoef(ea, eb)[0, 1])
    jac = float((ea & eb).sum() / max((ea | eb).sum(), 1))
    out(f"| {a} vs {b} | {phi:.3f} | {jac:.3f} |")
out("")

# ---------- 4. ensemble ----------
out("## 4. Ensemble (rata-rata skor)\n")
out("| ensemble | anggota | akurasi | EER | AUC |")
out("|---|---|---|---|---|")
y0 = runs[0]["y"]

def ens(rs, label):
    P = np.mean([r["p"] for r in rs], axis=0)
    m = full_metrics(y0, P, prior_matched_threshold(P, 0.5))
    out(f"| {label} | {len(rs)} | **{m['accuracy']*100:.2f}%** | "
        f"{m['eer']*100:.2f}% | {m['auc']:.4f} |")
    return m

for mdl, rs in sorted(by_model.items()):
    if len(rs) > 1:
        ens(rs, f"`{mdl}` (semua seed)")
ens(list(best.values()), "**semua model** (seed terbaik)")
ens(runs, "**semua run**")
out("")

open("PERBANDINGAN.md", "w", encoding="utf-8").write("\n".join(L))
print("-> PERBANDINGAN.md")
