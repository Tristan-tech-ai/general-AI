"""
Evaluasi ZERO-SHOT lintas-kondisi: model dilatih pada for-2sec, diuji pada for-rerec.

TUJUAN: menjawab serangan penguji A1 (RISET_CELAH_DAN_KRITIK.md §3):
  "Augmentasi codec Anda dipilih SETELAH mengaudit test set. Itu test-set peeking."

for-rerec adalah kondisi KETIGA yang tidak pernah dipakai untuk merancang
intervensi apa pun. Bila augmentasi codec juga menolong di sini, tuduhan
peeking gugur.

Pemakaian:
  py eval_rerec.py                 # semua run official+codec vs official+none
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.data import FoRDataset, AugmentConfig, collate, CLASSES
from forlib.models import build_model
from forlib.metrics import full_metrics, prior_matched_threshold

HERE = os.path.dirname(os.path.abspath(__file__))
REREC = os.path.join(HERE, "data", "for-rerecorded")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = torch.bfloat16 if DEV.type == "cuda" else torch.float32


def rerec_rows(split="testing"):
    rows = []
    for cls, lab in CLASSES.items():
        d = os.path.join(REREC, split, cls)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(".wav"):
                rows.append({"path": os.path.join(d, fn), "fname": fn,
                             "split_official": split, "label": lab, "cls": cls,
                             "is_mp3": int(".mp3" in fn.lower())})
    return rows


@torch.no_grad()
def score(model, loader):
    model.eval()
    L, Y = [], []
    for b in loader:
        with torch.autocast("cuda", dtype=AMP, enabled=DEV.type == "cuda"):
            o = model(b["wav"].to(DEV, non_blocking=True))
        L.append(o.float().cpu()); Y.append(b["label"])
    lg = torch.cat(L); y = torch.cat(Y).numpy()
    return torch.softmax(lg, 1).numpy()[:, 1], y


def parse(tag):
    m = re.match(r"^(.+?)_(official|random|clean_val|wavval)_([a-z]+?)(AV)?"
                 r"(?:_b(\d+)e(\d+))?_s(\d+)$", tag)
    return None if not m else {"model": m.group(1), "split": m.group(2),
                               "aug": m.group(3), "augval": bool(m.group(4)),
                               "seed": int(m.group(7))}


def main():
    if not os.path.isdir(REREC):
        print("for-rerec belum diekstrak ke", REREC); sys.exit(1)

    rows = rerec_rows("testing")
    print(f"for-rerec testing: {len(rows)} berkas "
          f"(fake {sum(r['label'] for r in rows)})")
    mp3 = sum(r["is_mp3"] for r in rows)
    print(f"berasal MP3: {mp3} ({100*mp3/len(rows):.1f}%)")

    ds = FoRDataset(rows, AugmentConfig.none(), "loudness")
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4,
                    collate_fn=collate, pin_memory=True)

    cands = []
    for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
        ck = os.path.join(d, "best.pt")
        if not os.path.exists(ck):
            continue
        info = parse(os.path.basename(d))
        if not info or info["split"] != "official" or info["augval"]:
            continue
        if info["aug"] not in ("none", "codec", "full"):
            continue
        cands.append((os.path.basename(d), ck, info))

    print(f"\nmengevaluasi {len(cands)} checkpoint ...\n")
    res = []
    for tag, ck, info in cands:
        try:
            kw = {}
            if info["model"] in ("wav2vec2", "hubert", "wavlm", "ast"):
                kw = {"freeze": True, "layer_weighting": True}
            m = build_model(info["model"], **kw).to(DEV)
            m.load_state_dict(torch.load(ck, map_location=DEV))
            p, y = score(m, dl)
            met = full_metrics(y, p, prior_matched_threshold(p, 0.5))
            res.append({"tag": tag, **info, **met})
            print(f"  {tag:44s} acc={met['accuracy']*100:6.2f}%  "
                  f"EER={met['eer']*100:5.2f}%  AUC={met['auc']:.4f}")
            del m
            if DEV.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {tag:44s} GAGAL: {type(e).__name__}: {e}")

    if not res:
        print("tidak ada hasil"); return

    # agregasi per (model, aug)
    g = defaultdict(list)
    for r in res:
        g[(r["model"], r["aug"])].append(r)

    L = []
    def out(s=""):
        print(s); L.append(s)

    out("\n# Evaluasi Zero-Shot Lintas-Kondisi: for-2sec → for-rerec\n")
    out(f"Model dilatih pada **for-2sec** (klip bersih 2 detik), diuji tanpa "
        f"penyesuaian apa pun pada **for-rerec** ({len(rows)} berkas hasil "
        f"pemutaran-ulang di ruangan dengan degradasi mirip telepon).\n")
    out("Kondisi ini **tidak pernah dipakai** untuk merancang augmentasi, "
        "memilih hyperparameter, atau memilih checkpoint.\n")
    out("Ambang: prior-matched (kelas seimbang 408/408).\n")

    out("| model | augmentasi latih | n seed | akurasi | EER | AUC |")
    out("|---|---|---|---|---|---|")
    agg = {}
    for (mdl, aug), rs in sorted(g.items()):
        a = np.array([r["accuracy"] for r in rs]) * 100
        e = np.array([r["eer"] for r in rs]) * 100
        u = np.array([r["auc"] for r in rs])
        sd = f"±{a.std(ddof=1):.2f}" if len(a) > 1 else ""
        agg[(mdl, aug)] = a.mean()
        out(f"| `{mdl}` | {aug} | {len(rs)} | **{a.mean():.2f}%** {sd} | "
            f"{e.mean():.2f}% | {u.mean():.4f} |")
    out("")

    out("## Uji kunci: apakah augmentasi codec menolong di kondisi yang tak pernah dilihat?\n")
    out("| model | tanpa augmentasi | + codec | Δ |")
    out("|---|---|---|---|")
    any_pair = False
    for mdl in sorted({m for m, _ in agg}):
        if (mdl, "none") in agg and (mdl, "codec") in agg:
            n, c = agg[(mdl, "none")], agg[(mdl, "codec")]
            out(f"| `{mdl}` | {n:.2f}% | {c:.2f}% | **{c-n:+.2f} pp** |")
            any_pair = True
    if not any_pair:
        out("| — | belum ada pasangan none/codec untuk model yang sama | | |")
    out("")
    out("Bila Δ positif, intervensi codec tergeneralisasi ke kondisi yang tidak "
        "dipakai merancangnya → tuduhan *test-set peeking* (serangan A1) gugur.")

    open("HASIL_REREC.md", "w", encoding="utf-8").write("\n".join(L))
    json.dump(res, open("rerec_results.json", "w"), indent=1)
    print("\n-> HASIL_REREC.md")


if __name__ == "__main__":
    main()
