"""
Uji paling menentukan untuk pertanyaan "apakah detektor masih relevan?".

Menjalankan model SOTA publik (Nes2Net-X, EER 1,49% pada ASVspoof 2021 DF,
dilatih pada ASVspoof 2019 LA) secara ZERO-SHOT pada:
  a) FoR-2sec  — TTS era 2019
  b) MLAAD     — TTS komersial 2025-2026 (Chatterbox, ElevenLabs-v3, dst)

Metrik utama: DETECTION RATE (recall pada kelas spoof) per sistem TTS, pada
ambang yang ditetapkan dari data asli FoR. Recall dipakai — bukan akurasi —
karena MLAAD hanya berisi audio palsu, sehingga memasangkannya dengan audio
asli dari korpus lain akan menciptakan confound provenance (pelajaran dari
artefak MP3 di FoR).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.data import load_manifest, make_splits, loudness_normalize, NSAMP, SR
from forlib.metrics import full_metrics, prior_matched_threshold
from load_sota import SotaNes2Net

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = torch.bfloat16 if DEV.type == "cuda" else torch.float32
MAXW = 3


class WavList(Dataset):
    def __init__(self, paths):
        self.idx = []
        self.paths = paths
        for i, p in enumerate(paths):
            try:
                nw = max(1, min(MAXW, int(sf.info(p).frames // NSAMP)))
            except Exception:
                nw = 1
            for w in range(nw):
                self.idx.append((i, w))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i, w = self.idx[k]
        try:
            x, sr = sf.read(self.paths[i], dtype="float32",
                            start=w * NSAMP, frames=NSAMP)
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != SR:
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
def score(model, paths):
    ds = WavList(paths)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=3,
                    collate_fn=coll, pin_memory=True)
    acc = np.zeros(len(paths)); cnt = np.zeros(len(paths))
    for b in dl:
        with torch.autocast("cuda", dtype=AMP, enabled=DEV.type == "cuda"):
            o = model(b["wav"].to(DEV, non_blocking=True))
        p = torch.softmax(o.float(), 1)[:, 1].cpu().numpy()
        fi = b["fi"].numpy()
        np.add.at(acc, fi, p); np.add.at(cnt, fi, 1)
    return acc / np.maximum(cnt, 1)


def main():
    model = SotaNes2Net().to(DEV).eval()
    print("model SOTA dimuat.\n")

    # ---- a) FoR-2sec ----
    rows = load_manifest(os.path.join(HERE, "manifest.csv"))
    _, _, te = make_splits(rows, "official")
    y = np.array([r["label"] for r in te])
    p_for = score(model, [r["path"] for r in te])
    thr = prior_matched_threshold(p_for, 0.5)
    m = full_metrics(y, p_for, thr)
    print(f"FoR-2sec (zero-shot): acc={m['accuracy']*100:.2f}%  "
          f"EER={m['eer']*100:.2f}%  AUC={m['auc']:.4f}")

    # ambang alternatif yang tidak memakai FoR sama sekali: 0,5
    m05 = full_metrics(y, p_for, 0.5)
    rec_for = float(((p_for >= 0.5) & (y == 1)).sum() / max((y == 1).sum(), 1))
    print(f"  pada ambang 0,5: acc={m05['accuracy']*100:.2f}%  "
          f"recall-spoof={rec_for*100:.2f}%")

    # ---- b) MLAAD per sistem TTS ----
    res = {"for_2sec": {"acc": m["accuracy"], "eer": m["eer"], "auc": m["auc"],
                        "recall_spoof@0.5": rec_for}}
    mroot = os.path.join(HERE, "data", "mlaad", "fake", "en")
    if os.path.isdir(mroot):
        print("\nMLAAD — detection rate (recall spoof) per sistem TTS:")
        print(f"{'sistem TTS':40s} {'n':>6s} {'@0,5':>8s} {'@thr-FoR':>10s} {'skor rata2':>11s}")
        for d in sorted(glob.glob(os.path.join(mroot, "*"))):
            ws = sorted(glob.glob(os.path.join(d, "*.wav")))
            if len(ws) < 20:
                continue
            ws = ws[:400]
            p = score(model, ws)
            r05 = float((p >= 0.5).mean())
            rft = float((p >= thr).mean())
            nm = os.path.basename(d)
            res[nm] = {"n": len(ws), "recall@0.5": r05,
                       "recall@thr_for": rft, "mean_score": float(p.mean())}
            print(f"{nm:40s} {len(ws):6d} {r05*100:7.1f}% {rft*100:9.1f}% {p.mean():11.4f}")
    json.dump(res, open(os.path.join(HERE, "sota_modern_results.json"), "w"), indent=1)
    print("\n-> sota_modern_results.json")


if __name__ == "__main__":
    main()
