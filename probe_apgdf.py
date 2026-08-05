"""
probe_apgdf.py — Uji empiris APGDF (All-Pole Group Delay Function) pada FoR-2sec.

Membandingkan 4 representasi di bawah noise aditif:
  - MFCC          (magnitudo, baseline)
  - STDGD         (group delay standar, Eq.3 Rajan et al. 2013)
  - MGD           (modified group delay, Eq.4-5, alpha=gamma=0.1, lifter=12)
  - APGDF         (group delay dari model all-pole LP, Sec 2.2 + 4.1)

Implementasi APGDF mengikuti persis Rajan, Kinnunen, Hanilci, Pohjalainen, Alku,
"Using group delay functions from all-pole models for speaker recognition",
Interspeech 2013, pp. 2489-2493. DOI 10.21437/Interspeech.2013-416
  frame 30 ms / shift 15 ms, orde LP p=20, G=1, DCT, buang koef ke-0, ambil 18.

Juga mengukur LANGSUNG klaim mekanisme: floor penyebut |X(w)|^2 (rapuh)
vs |A(w)|^2 (kokoh), sebagai fungsi SNR.
"""
import sys, time
import numpy as np
import soundfile as sf
from scipy.fft import dct
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

SR = 16000
FRAME = 480          # 30 ms
HOP = 240            # 15 ms
NFFT = 512
LPC_ORDER = 20
NCEP = 18
RNG = np.random.default_rng(1234)


# ---------------------------------------------------------------- framing
def frames_of(x, frame=FRAME, hop=HOP):
    n = 1 + max(0, (len(x) - frame) // hop)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    F = x[idx] * np.hamming(frame)[None, :]
    return F


def preemph(x, a=0.97):
    return np.concatenate([[x[0]], x[1:] - a * x[:-1]])


# ---------------------------------------------------------------- LPC (vectorised Levinson-Durbin)
def lpc_frames(F, order=LPC_ORDER):
    """F: (nframes, frame). Returns A: (nframes, order+1), A[:,0]=1."""
    nf, L = F.shape
    nfft = 1 << int(np.ceil(np.log2(2 * L)))
    S = np.abs(np.fft.rfft(F, nfft)) ** 2
    r = np.fft.irfft(S, nfft)[:, : order + 1]
    r[:, 0] *= 1.0 + 1e-9
    r[:, 0] += 1e-10

    a = np.zeros((nf, order + 1))
    a[:, 0] = 1.0
    e = r[:, 0].copy()
    for i in range(1, order + 1):
        acc = r[:, i].copy()
        if i > 1:
            acc += np.sum(a[:, 1:i] * r[:, i - 1 : 0 : -1], axis=1)
        k = -acc / np.maximum(e, 1e-12)
        k = np.clip(k, -0.999, 0.999)
        new = a[:, 1 : i + 1] + k[:, None] * a[:, i - 1 :: -1][:, : i]
        a[:, 1 : i + 1] = new
        e = e * (1.0 - k ** 2)
        e = np.maximum(e, 1e-12)
    return a


# ---------------------------------------------------------------- features
def apgdf(x, order=LPC_ORDER, ncep=NCEP, return_floor=False):
    """Group delay dari model all-pole. H(w)=G/A(w), G=1."""
    F = frames_of(preemph(x))
    A = lpc_frames(F, order)                      # (nf, p+1)
    k = np.arange(order + 1)[None, :]
    Aw = np.fft.rfft(A, NFFT, axis=1)             # A(w)
    Bw = np.fft.rfft(A * k, NFFT, axis=1)         # sum k*a_k e^{-jwk}
    den = (Aw.real ** 2 + Aw.imag ** 2)           # |A(w)|^2 -- tidak pernah ~0
    tau = -(Aw.real * Bw.real + Aw.imag * Bw.imag) / den
    C = dct(tau, type=2, axis=1, norm="ortho")[:, 1 : ncep + 1]
    if return_floor:
        return C, den
    return C


def std_gd(x, ncep=NCEP, return_floor=False):
    """Group delay standar, Eq.3: (Xr Yr + Xi Yi)/|X|^2, y(n)=n x(n)."""
    F = frames_of(preemph(x))
    n = np.arange(F.shape[1])[None, :]
    X = np.fft.rfft(F, NFFT, axis=1)
    Y = np.fft.rfft(F * n, NFFT, axis=1)
    den = (X.real ** 2 + X.imag ** 2)             # |X(w)|^2 -- BISA ~0
    tau = (X.real * Y.real + X.imag * Y.imag) / np.maximum(den, 1e-12)
    C = dct(tau, type=2, axis=1, norm="ortho")[:, 1 : ncep + 1]
    if return_floor:
        return C, den
    return C


def mgd(x, alpha=0.1, gamma=0.1, lifter=12, ncep=NCEP):
    """Modified group delay, Eq.4-5 dengan cepstral smoothing pada penyebut."""
    F = frames_of(preemph(x))
    n = np.arange(F.shape[1])[None, :]
    X = np.fft.rfft(F, NFFT, axis=1)
    Y = np.fft.rfft(F * n, NFFT, axis=1)
    logmag = np.log(np.abs(X) + 1e-12)
    cep = dct(logmag, type=2, axis=1, norm="ortho")
    cep[:, lifter:] = 0.0
    S = np.exp(dct(cep, type=3, axis=1, norm="ortho"))     # |S(w)| smoothed
    tp = (X.real * Y.real + X.imag * Y.imag) / np.maximum(S ** (2 * gamma), 1e-12)
    tau = np.sign(tp) * np.abs(tp) ** alpha
    C = dct(tau, type=2, axis=1, norm="ortho")[:, 1 : ncep + 1]
    return C


_MEL = None
def mel_fb(nmel=26):
    global _MEL
    if _MEL is not None:
        return _MEL
    def h2m(f): return 2595 * np.log10(1 + f / 700)
    def m2h(m): return 700 * (10 ** (m / 2595) - 1)
    pts = m2h(np.linspace(h2m(20), h2m(SR / 2), nmel + 2))
    bins = np.floor((NFFT + 1) * pts / SR).astype(int)
    fb = np.zeros((nmel, NFFT // 2 + 1))
    for i in range(nmel):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c > l: fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c: fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    _MEL = fb
    return fb


def mfcc(x, ncep=NCEP):
    F = frames_of(preemph(x))
    P = np.abs(np.fft.rfft(F, NFFT, axis=1)) ** 2 / NFFT
    E = np.log(P @ mel_fb().T + 1e-10)
    return dct(E, type=2, axis=1, norm="ortho")[:, 1 : ncep + 1]


def pool(C):
    """mean+std pooling + CMVN sederhana -> vektor utterance."""
    C = (C - C.mean(0, keepdims=True)) / (C.std(0, keepdims=True) + 1e-8)
    return np.concatenate([C.mean(0), C.std(0)])


# ---------------------------------------------------------------- noise
def add_noise(x, snr_db, rng):
    if snr_db is None:
        return x
    noise = rng.standard_normal(len(x))
    ps, pn = np.mean(x ** 2), np.mean(noise ** 2)
    scale = np.sqrt(ps / (pn * 10 ** (snr_db / 10)) + 1e-20)
    return x + scale * noise


# ---------------------------------------------------------------- driver
EXTRACTORS = {"MFCC": mfcc, "STDGD": std_gd, "MGD": mgd, "APGDF": apgdf}


def load_manifest(path="manifest.csv"):
    import csv
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def build(rows, snr, rng):
    out = {k: [] for k in EXTRACTORS}
    y = []
    for r in rows:
        x, sr = sf.read(r["path"].replace("\\", "/"), dtype="float32")
        x = np.asarray(x, dtype=np.float64)
        x = x / (np.max(np.abs(x)) + 1e-9)
        x = add_noise(x, snr, rng)
        for k, fn in EXTRACTORS.items():
            out[k].append(pool(fn(x)))
        y.append(int(r["label"]))
    return {k: np.asarray(v) for k, v in out.items()}, np.asarray(y)


def eer_of(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2 * 100


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    rows = load_manifest()
    tr = [r for r in rows if r["split_official"] == "training"]
    te = [r for r in rows if r["split_official"] == "testing"]
    rng0 = np.random.default_rng(7)
    tr_r = [r for r in tr if r["cls"] == "real"]
    tr_f = [r for r in tr if r["cls"] == "fake"]
    rng0.shuffle(tr_r); rng0.shuffle(tr_f)
    tr = tr_r[: n_train // 2] + tr_f[: n_train // 2]
    print(f"train n={len(tr)}  test n={len(te)} (split RESMI FoR)")

    t0 = time.time()
    Xtr, ytr = build(tr, None, np.random.default_rng(1))
    print(f"  fitur latih (bersih) selesai {time.time()-t0:.0f}s")

    # --- mechanism check: floor penyebut
    print("\n=== VERIFIKASI MEKANISME: floor penyebut group delay ===")
    print(f"{'SNR':>6} | {'min |X(w)|^2 (STDGD)':>22} | {'min |A(w)|^2 (APGDF)':>22}")
    samp = [r["path"].replace("\\", "/") for r in te[:60]]
    for snr in [None, 20, 10, 0]:
        rng = np.random.default_rng(99)
        mx, ma = [], []
        for p in samp:
            x, _ = sf.read(p, dtype="float32")
            x = np.asarray(x, np.float64); x /= (np.max(np.abs(x)) + 1e-9)
            x = add_noise(x, snr, rng)
            _, dX = std_gd(x, return_floor=True)
            _, dA = apgdf(x, return_floor=True)
            mx.append(dX.min()); ma.append(dA.min())
        lab = "bersih" if snr is None else f"{snr} dB"
        print(f"{lab:>6} | {np.median(mx):22.3e} | {np.median(ma):22.3e}")

    # --- classification under noise
    print("\n=== AKURASI / EER pada test resmi (latih BERSIH, uji BERNOISE) ===")
    results = {}
    for snr in [None, 20, 10, 0]:
        Xte, yte = build(te, snr, np.random.default_rng(42))
        lab = "bersih" if snr is None else f"{snr}dB"
        for k in EXTRACTORS:
            sc = StandardScaler().fit(Xtr[k])
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xtr[k]), ytr)
            s = clf.predict_proba(sc.transform(Xte[k]))[:, 1]
            acc = accuracy_score(yte, (s > 0.5).astype(int)) * 100
            auc = roc_auc_score(yte, s)
            results[(k, lab)] = (acc, auc, eer_of(yte, s))
        # fusion MFCC+APGDF
        Ftr = np.hstack([Xtr["MFCC"], Xtr["APGDF"]])
        Fte = np.hstack([Xte["MFCC"], Xte["APGDF"]])
        sc = StandardScaler().fit(Ftr)
        clf = LogisticRegression(max_iter=3000).fit(sc.transform(Ftr), ytr)
        s = clf.predict_proba(sc.transform(Fte))[:, 1]
        results[("MFCC+APGDF", lab)] = (
            accuracy_score(yte, (s > 0.5).astype(int)) * 100,
            roc_auc_score(yte, s), eer_of(yte, s))
        print(f"  [{lab}] selesai  ({time.time()-t0:.0f}s)")

    print(f"\n{'fitur':<12} | " + " | ".join(f"{c:^20}" for c in ["bersih", "20dB", "10dB", "0dB"]))
    print("-" * 100)
    for k in list(EXTRACTORS) + ["MFCC+APGDF"]:
        cells = []
        for c in ["bersih", "20dB", "10dB", "0dB"]:
            a, u, e = results[(k, c)]
            cells.append(f"acc{a:5.1f} auc{u:.3f} eer{e:5.1f}")
        print(f"{k:<12} | " + " | ".join(f"{c:^20}" for c in cells))


if __name__ == "__main__":
    main()
