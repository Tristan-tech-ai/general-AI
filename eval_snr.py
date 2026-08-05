"""
Grid SNR: menjawab rumusan masalah tesis secara langsung.

"Bagaimana kinerja Wav2Vec2, AST, HuBERT Large, dan CNN-LSTM dalam
mengklasifikasikan suara asli dan deepfake dalam kondisi dengan gangguan noise?"

Rancangan yang membuat hasilnya bermakna:
  * Noise UJI dari DEMAND (direkam sendiri oleh penulisnya) — terpisah total dari
    noise LATIH (colored noise sintetis). Ini yang membuat "unseen" benar-benar unseen.
    Korpus seperti MUSAN/ESC-50/FSD50K sama-sama bersumber Freesound sehingga
    "unseen" antar mereka belum tentu benar-benar terpisah.
  * Segmen noise dipilih deterministik per berkas → seluruh model diuji pada
    audio terkorupsi yang IDENTIK → perbandingan berpasangan yang sah.
  * Dilaporkan akurasi pada DUA ambang: ambang tetap dari kondisi bersih, dan
    ambang prior-matched. Selisihnya memisahkan kegagalan KALIBRASI dari
    kegagalan DISKRIMINASI (AUC/EER tidak berubah oleh ambang).
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.data import load_manifest, make_splits, loudness_normalize, NSAMP, SR
from forlib.models import build_model
from forlib.metrics import full_metrics, prior_matched_threshold

HERE = os.path.dirname(os.path.abspath(__file__))
DEMAND = os.path.join(HERE, "noise", "demand")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = torch.bfloat16 if DEV.type == "cuda" else torch.float32

SNRS = [None, 30, 25, 20, 15, 10, 5, 0, -5]      # None = bersih
ENVS = sorted(os.path.basename(d) for d in glob.glob(os.path.join(DEMAND, "*"))
              if os.path.isdir(d))


def load_noise():
    """Muat satu kanal per lingkungan (ch01), 300 detik masing-masing."""
    out = {}
    for e in ENVS:
        p = os.path.join(DEMAND, e, "ch01.wav")
        if not os.path.exists(p):
            continue
        x, sr = sf.read(p, dtype="float32")
        assert sr == SR, f"{p}: sr={sr}"
        out[e] = x
    return out


NOISE = load_noise()
print(f"korpus noise DEMAND: {list(NOISE)} "
      f"({', '.join(f'{len(v)/SR:.0f}s' for v in NOISE.values())})")


class NoisyTest(Dataset):
    """Test set FoR-2sec dikorupsi pada SNR tertentu dengan noise DEMAND."""

    def __init__(self, rows, snr_db, env=None):
        self.rows = rows
        self.snr = snr_db
        self.env = env

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x, _ = sf.read(r["path"], dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x.astype(np.float64)
        if len(x) < NSAMP:
            x = np.pad(x, (0, NSAMP - len(x)))
        x = x[:NSAMP]

        if self.snr is not None:
            # pemilihan deterministik → identik untuk semua model
            env = self.env or ENVS[i % len(ENVS)]
            nz = NOISE[env]
            off = (i * 7919) % max(1, len(nz) - NSAMP)
            n = nz[off: off + NSAMP].astype(np.float64)
            if len(n) < NSAMP:
                n = np.pad(n, (0, NSAMP - len(n)))
            ps = float(np.mean(x ** 2)) + 1e-12
            pn = float(np.mean(n ** 2)) + 1e-12
            x = x + n * math.sqrt(ps / (pn * (10 ** (self.snr / 10.0))))

        x = loudness_normalize(x)
        return {"wav": torch.from_numpy(x.astype(np.float32)),
                "label": torch.tensor(r["label"], dtype=torch.long)}


def coll(b):
    return {"wav": torch.stack([x["wav"] for x in b]),
            "label": torch.stack([x["label"] for x in b])}


@torch.no_grad()
def score(model, ds):
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=3,
                    collate_fn=coll, pin_memory=True)
    model.eval(); L, Y = [], []
    for b in dl:
        with torch.autocast("cuda", dtype=AMP, enabled=DEV.type == "cuda"):
            o = model(b["wav"].to(DEV, non_blocking=True))
        L.append(o.float().cpu()); Y.append(b["label"])
    return torch.softmax(torch.cat(L), 1).numpy()[:, 1], torch.cat(Y).numpy()


def main():
    rows = load_manifest(os.path.join(HERE, "manifest.csv"))
    _, _, te = make_splits(rows, "official")

    seeds = {"42", "1337", "2024"}
    cks = []
    for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
        ck = os.path.join(d, "best.pt")
        m = re.match(r"^(.+?)_official_(codec|full)(?:_b\d+e\d+)?_s(\d+)$",
                     os.path.basename(d))
        if os.path.exists(ck) and m and m.group(3) in seeds:
            # tandai augmentasi latih supaya kurva codec vs full dapat dibandingkan
            cks.append((f"{m.group(1)}[{m.group(2)}]", m.group(3), ck))
    print(f"{len(cks)} checkpoint × {len(SNRS)} kondisi SNR\n")

    res = []
    for arch, seed, ck in cks:
        base = arch.split("[")[0]
        kw = {"freeze": True, "layer_weighting": True} if base in (
            "wav2vec2", "hubert", "wavlm", "ast") else {}
        try:
            model = build_model(base, **kw).to(DEV)
            model.load_state_dict(torch.load(ck, map_location=DEV))
        except Exception as e:
            print(f"  {arch}/s{seed} GAGAL muat: {e}"); continue

        clean_thr = None
        for snr in SNRS:
            p, y = score(model, NoisyTest(te, snr))
            if snr is None:
                clean_thr = prior_matched_threshold(p, 0.5)
            m_pm = full_metrics(y, p, prior_matched_threshold(p, 0.5))
            m_fx = full_metrics(y, p, clean_thr)
            res.append({"arch": arch, "seed": seed, "snr": snr,
                        "acc_pm": m_pm["accuracy"], "acc_fx": m_fx["accuracy"],
                        "eer": m_pm["eer"], "auc": m_pm["auc"]})
            print(f"  {arch:10s} s{seed:<5s} SNR {str(snr):>5s} dB  "
                  f"acc_pm={m_pm['accuracy']*100:6.2f}%  acc_fix={m_fx['accuracy']*100:6.2f}%  "
                  f"EER={m_pm['eer']*100:5.2f}%  AUC={m_pm['auc']:.4f}")
        del model
        if DEV.type == "cuda":
            torch.cuda.empty_cache()

    json.dump(res, open(os.path.join(HERE, "snr_results.json"), "w"), indent=1)
    print("\n-> snr_results.json")


if __name__ == "__main__":
    main()
