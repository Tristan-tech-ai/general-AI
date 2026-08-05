"""
Verifikasi perbaikan bug P0-1: apakah augmentasi benar-benar berbeda antar epoch,
baik di proses utama maupun lewat DataLoader dengan worker?
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from torch.utils.data import DataLoader

from forlib.data import (load_manifest, make_splits, FoRDataset,
                         AugmentConfig, collate)


def main():
    rows = load_manifest("manifest.csv")
    tr, _, _ = make_splits(rows, "official")
    tr = tr[:64]
    aug = AugmentConfig(codec=1.0, noise=1.0, reverb=0.5, gain=1.0)

    print("=== A. Langsung (tanpa worker) ===")
    ds = FoRDataset(tr, aug, "loudness", seed=42)
    sig = {}
    for ep in [1, 2, 3]:
        ds.set_epoch(ep)
        sig[ep] = float(np.abs(ds[0]["wav"].numpy()).sum())
        print(f"  epoch {ep}: checksum {sig[ep]:.6f}")
    uniq = len(set(round(v, 6) for v in sig.values()))
    print(f"  -> {uniq}/3 varian unik", "OK" if uniq == 3 else "GAGAL")

    print("\n=== B. Lewat DataLoader dengan worker (kasus nyata) ===")
    for pw in [True, False]:
        ds2 = FoRDataset(tr, aug, "loudness", seed=42)
        dl = DataLoader(ds2, batch_size=8, shuffle=False, num_workers=2,
                        collate_fn=collate, persistent_workers=pw)
        sums = []
        for ep in [1, 2, 3]:
            ds2.set_epoch(ep)
            b = next(iter(dl))
            sums.append(round(float(b["wav"].abs().sum()), 4))
        u = len(set(sums))
        print(f"  persistent_workers={pw!s:5s} checksum={sums}")
        print(f"    -> {u}/3 unik", "OK" if u == 3 else "GAGAL (augmentasi beku)")
        del dl, ds2

    print("\n=== C. Keterulangan: seed sama harus menghasilkan hasil sama ===")
    a = FoRDataset(tr, aug, "loudness", seed=7); a.set_epoch(2)
    b = FoRDataset(tr, aug, "loudness", seed=7); b.set_epoch(2)
    same = np.allclose(a[0]["wav"].numpy(), b[0]["wav"].numpy())
    print("  deterministik:", "OK" if same else "GAGAL")


if __name__ == "__main__":
    main()
