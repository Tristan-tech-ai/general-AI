"""
Fusi bertumpuk (stacking) yang SAH: meta-classifier dilatih pada VALIDASI,
lalu diterapkan ke test. Tidak ada label test yang dipakai untuk memilih apa pun.

Latar: uji sampel-keras menunjukkan keempat error ensemble HuBERT dapat
diklasifikasi benar oleh setidaknya satu run. Jadi 100% terjangkau oleh
selektor oracle. Skrip ini mengukur berapa banyak dari potensi itu yang dapat
diraih **tanpa** melihat label test.

Tahap 1: skor ulang validasi + test untuk tiap checkpoint (butuh GPU).
Tahap 2: latih stacker pada skor validasi, terapkan ke skor test (CPU).

Pemakaian:
  py stack_fusion.py score     # tahap 1 (GPU) - simpan val_scores.npy per run
  py stack_fusion.py fuse      # tahap 2 (CPU) - latih & evaluasi
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.data import load_manifest, make_splits, FoRDataset, AugmentConfig, collate
from forlib.metrics import full_metrics, prior_matched_threshold

HERE = os.path.dirname(os.path.abspath(__file__))


def parse(tag):
    m = re.match(r"^(.+?)_official_codec(?:AV)?(?:_b\d+e\d+)?_s(\d+)$", tag)
    return None if not m else (m.group(1), int(m.group(2)))


# ---------------------------------------------------------------- tahap 1
def score_all():
    import torch
    from torch.utils.data import DataLoader
    from forlib.models import build_model

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if dev.type == "cuda" else torch.float32

    rows = load_manifest(os.path.join(HERE, "manifest.csv"))
    _, va, _ = make_splits(rows, "official")
    ds = FoRDataset(va, AugmentConfig.none(), "loudness")
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2,
                    collate_fn=collate, pin_memory=True)
    yv = np.array([r["label"] for r in va])

    @torch.no_grad()
    def run(model):
        model.eval(); L = []
        for b in dl:
            with torch.autocast("cuda", dtype=amp, enabled=dev.type == "cuda"):
                o = model(b["wav"].to(dev, non_blocking=True))
            L.append(o.float().cpu())
        return torch.softmax(torch.cat(L), 1).numpy()[:, 1]

    for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
        ck = os.path.join(d, "best.pt")
        outp = os.path.join(d, "val_scores.npy")
        if not os.path.exists(ck) or os.path.exists(outp):
            continue
        info = parse(os.path.basename(d))
        if not info:
            continue
        arch, seed = info
        try:
            kw = {}
            if arch in ("wav2vec2", "hubert", "wavlm", "ast"):
                kw = {"freeze": True, "layer_weighting": True}
            m = build_model(arch, **kw).to(dev)
            m.load_state_dict(torch.load(ck, map_location=dev))
            pv = run(m)
            np.save(outp, np.stack([yv.astype(float), pv]))
            print(f"  {os.path.basename(d):46s} val AUC "
                  f"{full_metrics(yv, pv, 0.5)['auc']:.4f}")
            del m
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {os.path.basename(d):46s} GAGAL {type(e).__name__}: {e}")


# ---------------------------------------------------------------- tahap 2
def fuse():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    V, T, names = [], [], []
    yv = yt = None
    for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
        fv = os.path.join(d, "val_scores.npy")
        ft = os.path.join(d, "test_scores.npy")
        if not (os.path.exists(fv) and os.path.exists(ft)):
            continue
        info = parse(os.path.basename(d))
        if not info:
            continue
        av = np.load(fv); at = np.load(ft)
        if len(at[0]) != 1088:
            continue
        yv = av[0].astype(int); yt = at[0].astype(int)
        V.append(av[1]); T.append(at[1]); names.append(os.path.basename(d))

    if not V:
        print("belum ada val_scores.npy — jalankan tahap 'score' dulu"); return

    Xv = np.stack(V, 1); Xt = np.stack(T, 1)
    print(f"{Xv.shape[1]} run · validasi {Xv.shape[0]} · test {Xt.shape[0]}")

    def lg(p, e=1e-6):
        p = np.clip(p, e, 1 - e); return np.log(p / (1 - p))

    L = []
    def out(s=""):
        print(s); L.append(s)

    out("# Fusi Bertumpuk (Stacking) — Dilatih pada Validasi\n")
    out(f"{Xv.shape[1]} run sebagai fitur. Meta-classifier **hanya melihat label "
        f"validasi**; label test tidak dipakai untuk melatih maupun memilih apa pun.\n")

    out("## Batas atas oracle (bukan hasil yang dapat dicapai)\n")
    ok_any = np.zeros(len(yt), bool)
    for j in range(Xt.shape[1]):
        t = prior_matched_threshold(Xt[:, j], 0.5)
        ok_any |= ((Xt[:, j] >= t).astype(int) == yt)
    out(f"Sampel yang benar pada **setidaknya satu** run: **{ok_any.sum()}/{len(yt)}** "
        f"= **{ok_any.mean()*100:.2f}%**")
    out("")
    out("Angka ini memerlukan label test untuk memilih run per sampel, jadi **tidak "
        "dapat dicapai dalam praktik**. Fungsinya hanya menunjukkan bahwa sisa error "
        "bukan keterbatasan informasi kumpulan model.\n")

    out("## Hasil yang SAH\n")
    out("| metode | dilatih pada | akurasi test | EER | AUC | salah |")
    out("|---|---|---|---|---|---|")

    # baseline: rata-rata logit seluruh run
    Pb = 1 / (1 + np.exp(-lg(Xt).mean(1)))
    m = full_metrics(yt, Pb, prior_matched_threshold(Pb, 0.5))
    out(f"| rata-rata logit (tanpa pelatihan) | — | **{m['accuracy']*100:.2f}%** | "
        f"{m['eer']*100:.2f}% | {m['auc']:.4f} | {m['n_errors']}/{m['n']} |")

    # run tunggal terbaik menurut VALIDASI
    vaucs = [full_metrics(yv, Xv[:, j], 0.5)["auc"] for j in range(Xv.shape[1])]
    jb = int(np.argmax(vaucs))
    Pj = Xt[:, jb]
    m = full_metrics(yt, Pj, prior_matched_threshold(Pj, 0.5))
    out(f"| run tunggal terbaik menurut val (`{names[jb][:30]}`) | validasi | "
        f"**{m['accuracy']*100:.2f}%** | {m['eer']*100:.2f}% | {m['auc']:.4f} | "
        f"{m['n_errors']}/{m['n']} |")

    # stacking
    for nm, C in [("stacking LogReg (C=1)", 1.0), ("stacking LogReg (C=0.1)", 0.1),
                  ("stacking LogReg (C=0.01)", 0.01)]:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=5000))
        clf.fit(lg(Xv), yv)
        Ps = clf.predict_proba(lg(Xt))[:, 1]
        m = full_metrics(yt, Ps, prior_matched_threshold(Ps, 0.5))
        out(f"| {nm} | validasi | **{m['accuracy']*100:.2f}%** | {m['eer']*100:.2f}% | "
            f"{m['auc']:.4f} | {m['n_errors']}/{m['n']} |")
    out("")

    open(os.path.join(HERE, "HASIL_STACKING.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n-> HASIL_STACKING.md")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fuse"
    (score_all if mode == "score" else fuse)()
