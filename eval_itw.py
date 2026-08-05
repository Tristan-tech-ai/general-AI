"""
Uji generalisasi lintas-korpus terberat: FoR-2sec -> In-the-Wild.

In-the-Wild berisi audio selebriti/politisi dari internet (54 pembicara,
19.963 bona-fide + 11.816 spoof). Ini korpus yang benar-benar berbeda:
sumber, sistem sintesis, kanal, dan bahasa penutur semuanya lain.
Literatur melaporkan model yang dilatih di korpus lain runtuh ke EER ~31% di sini.

Dua penyesuaian yang WAJIB, dan keduanya mudah luput:
  1. Kelas TIDAK seimbang (37,2% spoof). Ambang prior-matched harus memakai
     prior aslinya, bukan 0,5 — kalau tidak, akurasinya menyesatkan.
  2. Durasi bervariasi, model menerima tepat 2 detik. Skor dihitung dengan
     merata-ratakan probabilitas atas jendela 2 detik berurutan, bukan
     memotong satu potongan saja.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.data import loudness_normalize, NSAMP, SR
from forlib.models import build_model
from forlib.metrics import full_metrics, compute_eer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "data", "release_in_the_wild")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = torch.bfloat16 if DEV.type == "cuda" else torch.float32
MAXW = 5          # maksimum jendela 2 detik per berkas


def load_meta(limit=None):
    rows = []
    with open(os.path.join(ROOT, "meta.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"path": os.path.join(ROOT, r["file"]),
                         "speaker": r["speaker"],
                         "label": 1 if r["label"].strip() == "spoof" else 0})
    if limit:
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(rows))[:limit]
        rows = [rows[i] for i in sorted(idx)]
    return rows


class ITW(Dataset):
    """Satu item = satu jendela 2 detik; beberapa jendela per berkas."""

    def __init__(self, rows):
        self.index = []          # (row_idx, offset_window)
        self.rows = rows
        for i, r in enumerate(rows):
            try:
                info = sf.info(r["path"])
                nw = max(1, min(MAXW, int(info.frames // NSAMP)))
            except Exception:
                nw = 1
            for w in range(nw):
                self.index.append((i, w))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, w = self.index[k]
        r = self.rows[i]
        try:
            x, sr = sf.read(r["path"], dtype="float32",
                            start=w * NSAMP, frames=NSAMP)
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != SR:                      # jaga-jaga bila ada yang bukan 16 kHz
                n = int(len(x) * SR / sr)
                x = np.interp(np.linspace(0, len(x) - 1, n),
                              np.arange(len(x)), x).astype(np.float32)
        except Exception:
            x = np.zeros(NSAMP, dtype=np.float32)
        x = x.astype(np.float64)
        if len(x) < NSAMP:
            x = np.pad(x, (0, NSAMP - len(x)))
        x = loudness_normalize(x[:NSAMP])
        return {"wav": torch.from_numpy(x.astype(np.float32)),
                "fi": torch.tensor(i, dtype=torch.long)}


def coll(b):
    return {"wav": torch.stack([x["wav"] for x in b]),
            "fi": torch.stack([x["fi"] for x in b])}


@torch.no_grad()
def score(model, ds, nrows):
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4,
                    collate_fn=coll, pin_memory=True)
    acc = np.zeros(nrows); cnt = np.zeros(nrows)
    model.eval()
    for b in dl:
        with torch.autocast("cuda", dtype=AMP, enabled=DEV.type == "cuda"):
            o = model(b["wav"].to(DEV, non_blocking=True))
        p = torch.softmax(o.float(), 1)[:, 1].cpu().numpy()
        fi = b["fi"].numpy()
        np.add.at(acc, fi, p)
        np.add.at(cnt, fi, 1)
    return acc / np.maximum(cnt, 1)


def thr_prior(p, rate):
    return float(np.quantile(p, 1.0 - rate))


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    rows = load_meta(limit)
    y = np.array([r["label"] for r in rows])
    rate = float(y.mean())
    print(f"In-the-Wild: {len(rows)} berkas, spoof {y.sum()} ({rate*100:.1f}%), "
          f"{len({r['speaker'] for r in rows})} pembicara")

    ds = ITW(rows)
    print(f"total jendela 2 detik: {len(ds)}\n")

    cks = []
    for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
        ck = os.path.join(d, "best.pt")
        m = re.match(r"^(.+?)_official_(codec|full)(?:_b\d+e\d+)?_s(\d+)$",
                     os.path.basename(d))
        if os.path.exists(ck) and m and m.group(3) in {"42", "1337", "2024"}:
            only = os.environ.get("ITW_ONLY")   # mis. "wavlm:full,ast:codec"
            if only:
                want = {t.strip() for t in only.split(",")}
                if f"{m.group(1)}:{m.group(2)}" not in want:
                    continue
            cks.append((m.group(1), m.group(2), m.group(3), ck))

    res = []
    for arch, aug, seed, ck in cks:
        kw = {"freeze": True, "layer_weighting": True} if arch in (
            "wav2vec2", "hubert", "wavlm", "ast") else {}
        try:
            model = build_model(arch, **kw).to(DEV)
            model.load_state_dict(torch.load(ck, map_location=DEV))
        except Exception as e:
            print(f"  {arch}[{aug}]/s{seed} GAGAL: {e}"); continue
        p = score(model, ds, len(rows))
        m = full_metrics(y, p, thr_prior(p, rate))
        res.append({"arch": arch, "aug": aug, "seed": seed,
                    "acc": m["accuracy"], "eer": m["eer"], "auc": m["auc"]})
        print(f"  {arch:9s}[{aug:5s}] s{seed:<5s} acc={m['accuracy']*100:6.2f}%  "
              f"EER={m['eer']*100:5.2f}%  AUC={m['auc']:.4f}")
        del model
        if DEV.type == "cuda":
            torch.cuda.empty_cache()

    json.dump(res, open(os.path.join(HERE, "itw_results.json"), "w"), indent=1)

    L = ["# Generalisasi Lintas-Korpus: FoR-2sec → In-the-Wild\n",
         f"Model dilatih **hanya** pada FoR-2sec, diuji tanpa adaptasi apa pun pada "
         f"In-the-Wild ({len(rows)} berkas, {y.sum()} spoof = {rate*100:.1f}%, "
         f"{len({r['speaker'] for r in rows})} pembicara).\n",
         "Skor per berkas = rata-rata probabilitas atas hingga 5 jendela 2 detik. "
         f"Ambang memakai prior asli ({rate:.3f}), bukan 0,5.\n",
         "| arsitektur | augmentasi | n seed | akurasi | EER | AUC |",
         "|---|---|---|---|---|---|"]
    from collections import defaultdict
    g = defaultdict(list)
    for r in res:
        g[(r["arch"], r["aug"])].append(r)
    for (a, au), rs in sorted(g.items(), key=lambda kv: -np.mean([x["auc"] for x in kv[1]])):
        acc = np.array([x["acc"] for x in rs]) * 100
        eer = np.array([x["eer"] for x in rs]) * 100
        auc = np.array([x["auc"] for x in rs])
        sd = f"±{acc.std(ddof=1):.2f}" if len(acc) > 1 else ""
        L.append(f"| `{a}` | {au} | {len(rs)} | **{acc.mean():.2f}%** {sd} | "
                 f"{eer.mean():.2f}% | {auc.mean():.4f} |")
    L.append("")
    L.append("Konteks literatur: ref [14] dalam proposal melaporkan EER In-the-Wild "
             "**31,14%** untuk sistem baseline yang kuat di domainnya sendiri "
             "(EER 4,06% pada ASVspoof 2019).")
    open(os.path.join(HERE, "HASIL_ITW.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n-> HASIL_ITW.md")


if __name__ == "__main__":
    main()
