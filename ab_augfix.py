"""A/B: augmentasi beku (bug P0-1) vs augmentasi per-epoch (diperbaiki)."""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.metrics import full_metrics, prior_matched_threshold


def collect(pat, need="b16e10"):
    out = []
    for d in sorted(glob.glob(pat)):
        f = os.path.join(d, "test_scores.npy")
        if not os.path.exists(f) or need not in d:
            continue
        y, p, _ = np.load(f)
        m = full_metrics(y.astype(int), p, prior_matched_threshold(p, 0.5))
        out.append({"dir": os.path.basename(d), "y": y.astype(int), "p": p, **m})
    return out


old = collect("runs_augbug/hubert*")
new = collect("runs/hubert*")

L = []
def out(s=""):
    print(s); L.append(s)

out("# A/B: Perbaikan Bug Augmentasi Statis (P0-1) pada HuBERT Large\n")
out("Bug: RNG augmentasi di-seed hanya dari (nama berkas, seed run) sehingga tiap")
out("berkas menerima satu varian augmentasi yang sama di seluruh epoch.")
out("Perbaikan: epoch ikut masuk ke seed + `persistent_workers=False` pada loader latih.\n")
out("Semua run: HuBERT Large, split resmi, augmentasi codec, batch 16, 10 epoch,")
out("ambang prior-matched, seed {42, 1337, 2024}.\n")

out("| kondisi | n | akurasi | std | EER | AUC | terbaik | salah (terbaik) |")
out("|---|---|---|---|---|---|---|---|")
res = {}
for lab, rs in [("Augmentasi **beku** (bug)", old), ("Augmentasi **per-epoch** (fix)", new)]:
    if not rs:
        out(f"| {lab} | 0 | — | | | | | |"); continue
    a = np.array([r["accuracy"] for r in rs]) * 100
    e = np.array([r["eer"] for r in rs]) * 100
    u = np.array([r["auc"] for r in rs])
    best = max(rs, key=lambda r: r["accuracy"])
    sd = a.std(ddof=1) if len(a) > 1 else 0.0
    res[lab] = (a, e, u, rs)
    out(f"| {lab} | {len(a)} | **{a.mean():.2f}%** | ±{sd:.2f} | {e.mean():.2f}% | "
        f"{u.mean():.4f} | **{a.max():.2f}%** | {best['n_errors']}/{best['n']} |")
out("")

keys = list(res)
if len(keys) == 2:
    (ao, eo, uo, ro), (an, en, un, rn) = res[keys[0]], res[keys[1]]
    out("## Selisih\n")
    out(f"- Akurasi rerata: **{an.mean()-ao.mean():+.2f} pp**")
    out(f"- Akurasi terbaik: **{an.max()-ao.max():+.2f} pp**")
    out(f"- EER rerata: **{en.mean()-eo.mean():+.2f} pp**")
    out(f"- Simpangan baku: {ao.std(ddof=1):.2f} → {an.std(ddof=1):.2f} pp")
    out("")
    # uji Welch
    try:
        from scipy import stats
        t, p = stats.ttest_ind(an, ao, equal_var=False)
        out(f"- Welch t-test rerata: t={t:.3f}, **p={p:.3f}** "
            f"→ {'signifikan' if p < 0.05 else '**tidak signifikan**'} pada n=3")
    except Exception:
        pass
    out("")
    out("**Pembacaan jujur:** perbaikan ini **tidak** menaikkan rerata secara")
    out("signifikan pada n=3, tetapi menaikkan **plafon** dan **variansi** sekaligus.")
    out("Itu konsisten dengan mekanismenya: keragaman augmentasi yang lebih besar")
    out("memperluas ruang eksplorasi, sehingga run terbaik menjadi lebih baik dan")
    out("run terburuk menjadi lebih buruk. Untuk pelaporan tesis, konsekuensinya")
    out("adalah **butuh lebih banyak seed**, bukan lebih sedikit.\n")

# ensemble
out("## Ensemble antar-seed\n")
out("| kondisi | akurasi | EER | AUC | salah |")
out("|---|---|---|---|---|")
for lab, rs in [("Augmentasi beku", old), ("Augmentasi per-epoch", new)]:
    if not rs:
        continue
    P = np.mean([r["p"] for r in rs], axis=0)
    y = rs[0]["y"]
    m = full_metrics(y, P, prior_matched_threshold(P, 0.5))
    out(f"| {lab} | **{m['accuracy']*100:.2f}%** | {m['eer']*100:.2f}% | "
        f"{m['auc']:.4f} | {m['n_errors']}/{m['n']} |")

# gabungan semua 6 run
allr = old + new
if allr:
    P = np.mean([r["p"] for r in allr], axis=0)
    m = full_metrics(allr[0]["y"], P, prior_matched_threshold(P, 0.5))
    out(f"| **Gabungan seluruh {len(allr)} run HuBERT** | **{m['accuracy']*100:.2f}%** | "
        f"{m['eer']*100:.2f}% | {m['auc']:.4f} | {m['n_errors']}/{m['n']} |")
out("")

open("HASIL_AB_AUGFIX.md", "w", encoding="utf-8").write("\n".join(L))
print("-> HASIL_AB_AUGFIX.md")
