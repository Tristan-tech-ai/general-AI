"""Benchmark tiap komponen augmentasi untuk menemukan bottleneck."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from forlib.data import codec_augment, colored_noise, add_noise_snr, apply_reverb, SR

rng = np.random.default_rng(0)
x = rng.standard_normal(32000) * 0.05
N = 40

def bench(fn, label):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(N):
        fn()
    dt = (time.perf_counter() - t0) / N * 1000
    print(f"  {label:34s} {dt:8.2f} ms/sampel")
    return dt

print("Komponen augmentasi (per sampel 2 detik):")
t_codec = bench(lambda: codec_augment(x, rng), "codec_augment")
t_noise = bench(lambda: add_noise_snr(x, colored_noise(len(x), rng, 1.0), 10.0),
                "colored_noise + add_noise_snr")
t_rev = bench(lambda: apply_reverb(x, rng), "apply_reverb (np.convolve)")

# alternatif FFT
from scipy.signal import fftconvolve

def reverb_fft(sig, r, sr=SR):
    rt60 = float(r.uniform(0.15, 0.7))
    L = int(sr * rt60)
    if L < 8:
        return sig
    t = np.arange(L) / sr
    ir = r.standard_normal(L) * np.exp(-6.9 * t / rt60)
    ir[0] += 1.0
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-12
    return fftconvolve(sig, ir)[: len(sig)]

t_fft = bench(lambda: reverb_fft(x, rng), "reverb via fftconvolve")

print()
print(f"  reverb: {t_rev:.1f} ms -> {t_fft:.1f} ms = **{t_rev/t_fft:.0f}x lebih cepat**")
print()
# perkiraan waktu epoch
n = 13956
p_codec, p_noise, p_rev = 0.5, 0.5, 0.25
old = n * (p_codec*t_codec + p_noise*t_noise + p_rev*t_rev) / 1000 / 2   # 2 worker
new = n * (p_codec*t_codec + p_noise*t_noise + p_rev*t_fft) / 1000 / 2
print(f"  perkiraan CPU augmentasi per epoch (2 worker):")
print(f"    sekarang : {old/60:6.2f} menit")
print(f"    setelah  : {new/60:6.2f} menit")
