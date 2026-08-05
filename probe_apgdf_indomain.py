"""
probe_apgdf_indomain.py — kontrol IN-DOMAIN untuk memisahkan kualitas fitur
dari pergeseran domain codec.

Split acak DI DALAM training resmi (provenance seimbang secara acak), sehingga
yang terukur adalah daya diskriminasi + ketahanan noise fitur itu sendiri,
bukan artefak MP3 lintas-split.

Latih pada BERSIH, uji pada 4 kondisi noise. Ini persis pertanyaan riset tesis.
"""
import time
import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

from probe_apgdf import (EXTRACTORS, pool, add_noise, load_manifest, eer_of)

N_PER_CLASS = 700
SNRS = [None, 20, 10, 0]


def feats(paths, snr, seed):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in EXTRACTORS}
    for p in paths:
        x, _ = sf.read(p, dtype="float32")
        x = np.asarray(x, np.float64)
        x /= (np.max(np.abs(x)) + 1e-9)
        x = add_noise(x, snr, rng)
        for k, fn in EXTRACTORS.items():
            out[k].append(pool(fn(x)))
    return {k: np.asarray(v) for k, v in out.items()}


def main():
    rows = load_manifest()
    tr = [r for r in rows if r["split_official"] == "training"]
    # seimbangkan provenance: ambil fake WAV (non-MP3) sebanyak mungkin + fake MP3
    real = [r for r in tr if r["cls"] == "real"]
    fake = [r for r in tr if r["cls"] == "fake"]
    rng = np.random.default_rng(7)
    rng.shuffle(real); rng.shuffle(fake)
    real, fake = real[:N_PER_CLASS], fake[:N_PER_CLASS]
    allr = real + fake
    y = np.array([0] * len(real) + [1] * len(fake))
    paths = [r["path"].replace("\\", "/") for r in allr]

    idx = np.arange(len(y)); rng.shuffle(idx)
    cut = int(0.7 * len(idx))
    itr, ite = idx[:cut], idx[cut:]
    print(f"IN-DOMAIN (acak dalam training resmi): latih {len(itr)}, uji {len(ite)}")

    t0 = time.time()
    Fclean = feats(paths, None, 1)
    print(f"  fitur bersih selesai {time.time()-t0:.0f}s")
    Fnoisy = {}
    for snr in SNRS[1:]:
        Fnoisy[snr] = feats([paths[i] for i in ite], snr, 42)
        print(f"  fitur {snr}dB selesai {time.time()-t0:.0f}s")

    print(f"\n{'fitur':<8} | " + " | ".join(f"{('bersih' if s is None else str(s)+'dB'):^22}" for s in SNRS) + " | degradasi")
    print("-" * 118)
    summary = {}
    for k in EXTRACTORS:
        sc = StandardScaler().fit(Fclean[k][itr])
        clf = LogisticRegression(max_iter=3000).fit(sc.transform(Fclean[k][itr]), y[itr])
        cells, eers = [], []
        for snr in SNRS:
            X = Fclean[k][ite] if snr is None else Fnoisy[snr][k]
            s = clf.predict_proba(sc.transform(X))[:, 1]
            a = accuracy_score(y[ite], (s > 0.5).astype(int)) * 100
            u = roc_auc_score(y[ite], s)
            e = eer_of(y[ite], s)
            eers.append(e)
            cells.append(f"acc{a:5.1f} auc{u:.3f} eer{e:5.2f}")
        summary[k] = eers
        print(f"{k:<8} | " + " | ".join(f"{c:^22}" for c in cells) + f" | x{eers[-1]/max(eers[0],1e-6):5.2f}")

    print("\n=== EER relatif terhadap kondisi bersih (semakin kecil semakin tahan noise) ===")
    print(f"{'fitur':<8} | {'bersih':>8} | {'20dB':>8} | {'10dB':>8} | {'0dB':>8}")
    for k, e in summary.items():
        print(f"{k:<8} | " + " | ".join(f"{v:8.2f}" for v in e))


if __name__ == "__main__":
    main()
