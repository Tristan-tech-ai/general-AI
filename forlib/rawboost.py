"""
RawBoost — augmentasi khusus anti-spoofing, di-port dari repo Nes2Net.

Sumber: Tak et al., "RawBoost: A Raw Data Boosting and Augmentation Method
applied to Automatic Speaker Verification Anti-Spoofing", ICASSP 2022.
Fork lokal: forks/Nes2Net_ASVspoof_ITW/RawBoost.py

Bedanya dengan augmentasi generik (noise putih/pink, reverb, codec): RawBoost
mensimulasikan degradasi yang benar-benar terjadi pada rantai transmisi dan
perekaman telepon/VoIP:

  1. LnL convolutive  — distorsi linear DAN non-linear (harmonik orde-2),
                        meniru kompresi/amplifier
  2. ISD additive     — noise impulsif yang bergantung sinyal (dropout paket)
  3. SSI additive     — noise stasioner berpita, meniru derau kanal

Ketiganya bekerja pada waveform mentah tanpa memerlukan korpus noise eksternal,
sehingga tidak ada risiko kebocoran korpus (masalah yang menjerat MUSAN/ESC-50/
FSD50K yang sama-sama bersumber Freesound).
"""
from __future__ import annotations

import copy

import numpy as np
from scipy import signal

# Parameter default mengikuti paper & repo resmi
P = dict(N_f=5, nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000,
         minCoeff=10, maxCoeff=100, minG=0, maxG=0,
         minBiasLinNonLin=5, maxBiasLinNonLin=20,
         P_isd=10, g_sd=2, SNRmin=10, SNRmax=40)


def _rand(a, b, integer, rng):
    """
    Dua koreksi terhadap implementasi asli:
      * NumPy 2.x melarang int() pada array 1-elemen -> kembalikan skalar Python.
      * Kode asli dapat memanggil dengan low > high (pada LnL, minG menjadi -5
        sementara maxG menjadi -20 karena bias non-linear dikurangkan dengan
        besaran berbeda). NumPy lama membiarkannya; NumPy 2.x melempar
        "high - low < 0". Batas diurutkan agar maksudnya terjaga: G di [-20, -5].
    """
    lo, hi = (a, b) if a <= b else (b, a)
    y = float(rng.uniform(lo, hi))
    return int(y) if integer else y


def _norm(x, always):
    m = np.amax(np.abs(x))
    if m <= 0:
        return x
    if always or m > 1:
        return x / m
    return x


def _notch(nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff,
           minG, maxG, fs, rng):
    b = 1
    for _ in range(nBands):
        fc = _rand(minF, maxF, 0, rng)
        bw = _rand(minBW, maxBW, 0, rng)
        c = _rand(minCoeff, maxCoeff, 1, rng)
        if c % 2 == 0:
            c += 1
        f1, f2 = fc - bw / 2, fc + bw / 2
        f1 = 1 / 1000 if f1 <= 0 else f1
        f2 = fs / 2 - 1 / 1000 if f2 >= fs / 2 else f2
        b = np.convolve(
            signal.firwin(c, [float(f1), float(f2)], window="hamming", fs=fs), b)
    G = _rand(minG, maxG, 0, rng)
    _, h = signal.freqz(b, 1, fs=fs)
    return pow(10, G / 20) * b / np.amax(np.abs(h))


def _fir(x, b):
    N = b.shape[0] + 1
    y = signal.lfilter(b, 1, np.pad(x, (0, N), "constant"))
    return y[int(N / 2): int(y.shape[0] - N / 2)]


def lnl_convolutive(x, fs, rng):
    """Distorsi konvolutif linear + non-linear."""
    y = np.zeros(x.shape[0])
    minG, maxG = P["minG"], P["maxG"]
    for i in range(P["N_f"]):
        if i == 1:
            minG -= P["minBiasLinNonLin"]
            maxG -= P["maxBiasLinNonLin"]
        b = _notch(P["nBands"], P["minF"], P["maxF"], P["minBW"], P["maxBW"],
                   P["minCoeff"], P["maxCoeff"], minG, maxG, fs, rng)
        y = y + _fir(np.power(x, i + 1), b)
    return _norm(y - np.mean(y), 0)


def isd_additive(x, rng):
    """Noise impulsif bergantung sinyal."""
    beta = _rand(0, P["P_isd"], 0, rng)
    y = copy.deepcopy(x)
    n = int(x.shape[0] * (beta / 100))
    if n <= 0:
        return y
    p = rng.permutation(x.shape[0])[:n]
    f_r = ((2 * rng.random(p.shape[0])) - 1) * ((2 * rng.random(p.shape[0])) - 1)
    y[p] = x[p] + P["g_sd"] * x[p] * f_r
    return _norm(y, 0)


def ssi_additive(x, fs, rng):
    """Noise stasioner berpita, independen sinyal."""
    noise = rng.normal(0, 1, x.shape[0])
    b = _notch(P["nBands"], P["minF"], P["maxF"], P["minBW"], P["maxBW"],
               P["minCoeff"], P["maxCoeff"], P["minG"], P["maxG"], fs, rng)
    noise = _norm(_fir(noise, b), 1)
    snr = _rand(P["SNRmin"], P["SNRmax"], 0, rng)
    noise = noise / np.linalg.norm(noise, 2) * np.linalg.norm(x, 2) / 10.0 ** (0.05 * snr)
    return x + noise


def rawboost(x, fs, algo, rng):
    """
    algo 1 = LnL · 2 = ISD · 3 = SSI
    algo 4 = seri (1->2->3)  <- dipakai repo Nes2Net
    algo 5 = seri (1->2)
    algo 6 = paralel (1+2)
    """
    if algo == 1:
        return lnl_convolutive(x, fs, rng)
    if algo == 2:
        return isd_additive(x, rng)
    if algo == 3:
        return ssi_additive(x, fs, rng)
    if algo == 4:
        return ssi_additive(isd_additive(lnl_convolutive(x, fs, rng), rng), fs, rng)
    if algo == 5:
        return isd_additive(lnl_convolutive(x, fs, rng), rng)
    if algo == 6:
        a = lnl_convolutive(x, fs, rng)
        b = isd_additive(x, rng)
        return _norm(a + b, 0)
    return x
