"""
Data FoR-2sec: manifest, split, augmentasi.

Keputusan desain yang memperbaiki bug dari proposal:
  * split RESMI didukung sebagai default (proposal merencanakan 60/20/20 acak,
    yang menghancurkan pemisahan domain bawaan FoR - lihat TEMUAN_GROUND_TRUTH.md)
  * normalisasi LOUDNESS, bukan peak (peak-norm memperkuat pintasan `rms`)
  * augmentasi codec diterapkan SERAGAM pada kedua kelas untuk menetralkan
    korelasi semu MP3->fake yang ada di data latih tetapi tidak di data uji
  * tidak bergantung pada librosa/numba (belum stabil di Python 3.14)
"""
from __future__ import annotations

import os
import csv
import math
import hashlib

import numpy as np
import soundfile as sf
import torch
from scipy.signal import fftconvolve as _fftconvolve
from torch.utils.data import Dataset

SR = 16000
NSAMP = 32000            # 2 detik tepat
CLASSES = {"real": 0, "fake": 1}


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------
def build_manifest(root: str, out_csv: str) -> list[dict]:
    """Pindai direktori for-2seconds dan tulis manifest CSV."""
    rows = []
    for split in ["training", "validation", "testing"]:
        for cls, lab in CLASSES.items():
            d = os.path.join(root, split, cls)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.lower().endswith(".wav"):
                    continue
                rows.append({
                    "path": os.path.join(d, fn),
                    "fname": fn,
                    "split_official": split,
                    "label": lab,
                    "cls": cls,
                    # provenance: berkas fake yang berasal dari MP3 membawa pintasan
                    "is_mp3": int(".mp3" in fn.lower()),
                })
    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def load_manifest(csv_path: str) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["label"] = int(r["label"])
            r["is_mp3"] = int(r["is_mp3"])
            rows.append(r)
    return rows


def _stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def make_splits(rows: list[dict], scheme: str = "official", seed: int = 42):
    """
    scheme='official' : partisi resmi FoR (train / val / test) - DIREKOMENDASIKAN
    scheme='random'   : gabung semua lalu bagi acak 60/20/20 (rencana proposal;
                        disediakan HANYA agar selisihnya dapat dilaporkan)
    scheme='clean_val': seperti 'official', tetapi validation disaring supaya
                        bebas MP3 -> proksi yang jujur untuk test set
    """
    if scheme == "official":
        tr = [r for r in rows if r["split_official"] == "training"]
        va = [r for r in rows if r["split_official"] == "validation"]
        te = [r for r in rows if r["split_official"] == "testing"]

    elif scheme == "clean_val":
        tr = [r for r in rows if r["split_official"] == "training"]
        va_all = [r for r in rows if r["split_official"] == "validation"]
        # buang fake turunan MP3 dari validation agar tidak memilih checkpoint
        # yang paling pandai mengeksploitasi pintasan codec
        va_fake = [r for r in va_all if r["label"] == 1 and r["is_mp3"] == 0]
        va_real = [r for r in va_all if r["label"] == 0][: len(va_fake)]
        va = va_fake + va_real
        te = [r for r in rows if r["split_official"] == "testing"]

    elif scheme == "wavval":
        # Validasi yang COCOK DOMAIN dan berukuran memadai.
        # clean_val hanya punya 147 fake bebas-MP3 -> pemilihan checkpoint berisik.
        # Di sini 652 fake bebas-MP3 dari TRAINING disisihkan sebagai validasi,
        # dipasangkan dengan jumlah real yang sama. Hasilnya 1.304 berkas yang
        # distribusinya jauh lebih dekat ke test set (sama-sama WAV-native).
        tr_all = [r for r in rows if r["split_official"] == "training"]
        va_all = [r for r in rows if r["split_official"] == "validation"]

        wav_fake = [r for r in tr_all if r["label"] == 1 and r["is_mp3"] == 0]
        rng = np.random.default_rng(seed)
        reals = [r for r in tr_all if r["label"] == 0]
        rng.shuffle(reals)
        va_real = reals[: len(wav_fake)]
        va = wav_fake + va_real + [r for r in va_all if r["label"] == 1 and r["is_mp3"] == 0]

        held = {id(r) for r in wav_fake} | {id(r) for r in va_real}
        tr = [r for r in tr_all if id(r) not in held] + \
             [r for r in va_all if r["is_mp3"] == 1 or r["label"] == 0]
        te = [r for r in rows if r["split_official"] == "testing"]

    elif scheme == "random":
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(rows))
        n = len(rows)
        a, b = int(0.6 * n), int(0.8 * n)
        tr = [rows[i] for i in idx[:a]]
        va = [rows[i] for i in idx[a:b]]
        te = [rows[i] for i in idx[b:]]
    else:
        raise ValueError(f"scheme tidak dikenal: {scheme}")

    return tr, va, te


# --------------------------------------------------------------------------
# Normalisasi & augmentasi (numpy, tanpa dependensi berat)
# --------------------------------------------------------------------------
def loudness_normalize(x: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    """
    Normalisasi berbasis RMS, bukan peak.
    `rms` adalah fitur pintasan terkuat di FoR (AUC 0,698); menyamakan RMS
    seluruh berkas menghapusnya sebagai sinyal.
    """
    r = float(np.sqrt(np.mean(x ** 2)))
    if r < 1e-9:
        return x
    y = x * (target_rms / r)
    m = float(np.abs(y).max())
    if m > 0.99:                     # cegah clipping
        y = y * (0.99 / m)
    return y


def fft_lowpass(x: np.ndarray, cutoff_hz: float, sr: int = SR,
                rolloff_hz: float = 400.0) -> np.ndarray:
    """Low-pass eksak di domain FFT dengan tepi kosinus (hindari ringing)."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    mask = np.ones_like(f)
    hi = f >= cutoff_hz + rolloff_hz
    mask[hi] = 0.0
    edge = (f > cutoff_hz) & (f < cutoff_hz + rolloff_hz)
    mask[edge] = 0.5 * (1 + np.cos(np.pi * (f[edge] - cutoff_hz) / rolloff_hz))
    return np.fft.irfft(X * mask, n=n)


def codec_augment(x: np.ndarray, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """
    Simulasi jejak codec lossy.

    INI PERBAIKAN BUG UTAMA. Di data latih, 90,7% sampel `fake` berasal dari MP3
    dan karenanya ter-lowpass; di data uji tidak ada satupun. Menerapkan
    band-limiting acak pada KEDUA kelas membuat fitur tersebut tidak lagi
    memprediksi label, sehingga model dipaksa mempelajari artefak sintesis
    yang sesungguhnya.
    """
    cutoff = float(rng.uniform(3000, 7800))
    y = fft_lowpass(x, cutoff, sr)
    # lubang spektral ala kuantisasi codec
    if rng.random() < 0.5:
        n = len(y)
        Y = np.fft.rfft(y)
        f = np.fft.rfftfreq(n, 1.0 / sr)
        for _ in range(int(rng.integers(1, 4))):
            c = float(rng.uniform(1000, min(cutoff, 7500)))
            w = float(rng.uniform(50, 300))
            Y[(f > c - w) & (f < c + w)] *= float(rng.uniform(0.05, 0.4))
        y = np.fft.irfft(Y, n=n)
    return y


def band_gain_augment(x: np.ndarray, rng: np.random.Generator, sr: int = SR,
                      f_lo: float = 3000.0, n_bands: int = 6,
                      db_range: tuple = (-12.0, 3.0)) -> np.ndarray:
    """
    Augmentasi 'lembut': mengacak LEVEL energi pita tinggi tanpa menghapus
    STRUKTUR HALUS di dalamnya.

    LATAR MEKANISTIK. Dua isyarat berbeda hidup di pita tinggi yang sama:
      * level energi HF  -> ini pintasan codec (MP3 menekannya). Harus dinetralkan.
      * struktur halus HF -> ini artefak vocoder. Harus DIPERTAHANKAN.

    `codec_augment` (low-pass) dan RawBoost menghapus keduanya sekaligus. Itu
    menjelaskan trade-off terukur: RawBoost menaikkan akurasi FoR +4,75 poin
    tetapi menurunkan deteksi TTS 2025-2026 sekitar 13 poin, karena artefak
    generator modern justru halus dan berfrekuensi tinggi.

    Di sini tiap pita hanya dikalikan gain acak. Rasio energi antar-pita menjadi
    tidak informatif terhadap label, sementara bentuk spektrum di dalam pita —
    puncak, lembah, periodisitas checkerboard — tetap utuh.
    """
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    edges = np.linspace(f_lo, sr / 2, n_bands + 1)
    for i in range(n_bands):
        m = (f >= edges[i]) & (f < edges[i + 1])
        if m.any():
            g = 10 ** (float(rng.uniform(*db_range)) / 20.0)
            X[m] *= g
    return np.fft.irfft(X, n=n)


def colored_noise(n: int, rng: np.random.Generator, beta: float = 1.0) -> np.ndarray:
    """Noise dengan spektrum 1/f^beta. beta=0 putih, 1 pink, 2 brown."""
    w = rng.standard_normal(n)
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    f[0] = f[1] if len(f) > 1 else 1.0
    W = W / (f ** (beta / 2.0))
    y = np.fft.irfft(W, n=n)
    s = float(np.std(y))
    return y / (s + 1e-12)


def add_noise_snr(x: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    ps = float(np.mean(x ** 2)) + 1e-12
    pn = float(np.mean(noise ** 2)) + 1e-12
    scale = math.sqrt(ps / (pn * (10 ** (snr_db / 10.0))))
    return x + noise * scale


def apply_reverb(x: np.ndarray, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """
    Reverb sintetis: peluruhan eksponensial acak (pengganti RIR nyata).

    Memakai fftconvolve, BUKAN np.convolve. Konvolusi langsung di sini berbiaya
    O(n*m) = 32.000 x 11.200 ~ 3,6e8 operasi per sampel, terukur 177,8 ms/sampel
    dan menjadikan augmentasi 'full' 5,5x lebih lambat daripada 'codec'.
    fftconvolve O(n log n) terukur 0,96 ms/sampel = 185x lebih cepat, dengan
    hasil yang secara matematis sama (beda hanya pembulatan floating-point).
    """
    rt60 = float(rng.uniform(0.15, 0.7))
    L = int(sr * rt60)
    if L < 8:
        return x
    t = np.arange(L) / sr
    ir = rng.standard_normal(L) * np.exp(-6.9 * t / rt60)
    ir[0] += 1.0
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-12
    return _fftconvolve(x, ir)[: len(x)]


class AugmentConfig:
    def __init__(self, codec=0.5, noise=0.5, reverb=0.25, gain=0.3,
                 snr_range=(0.0, 30.0), enabled=True,
                 rawboost=0.0, rawboost_algo=4, band_gain=0.0):
        # band_gain: netralkan pintasan level-HF tanpa merusak struktur halus HF
        self.band_gain = band_gain
        self.codec = codec
        self.noise = noise
        self.reverb = reverb
        self.gain = gain
        self.snr_range = snr_range
        self.enabled = enabled
        # RawBoost (ICASSP 2022) — degradasi khusus anti-spoofing, lihat rawboost.py
        self.rawboost = rawboost
        self.rawboost_algo = rawboost_algo

    @staticmethod
    def none():
        return AugmentConfig(enabled=False)


def augment_waveform(x: np.ndarray, cfg: AugmentConfig,
                     rng: np.random.Generator) -> np.ndarray:
    if not cfg.enabled:
        return x
    # RawBoost diterapkan PALING AWAL: ia memodelkan distorsi rantai
    # transmisi/perekaman yang secara fisik terjadi sebelum ruang & codec
    if cfg.rawboost > 0 and rng.random() < cfg.rawboost:
        from .rawboost import rawboost as _rb
        try:
            x = _rb(x, SR, cfg.rawboost_algo, rng)
        except Exception:
            pass                      # jangan biarkan satu sampel merusak epoch
    # band-gain lebih dulu: ia beroperasi pada spektrum sumber, sebelum
    # degradasi kanal menambahkan warnanya sendiri
    if getattr(cfg, "band_gain", 0.0) > 0 and rng.random() < cfg.band_gain:
        x = band_gain_augment(x, rng)
    # urutan meniru rantai degradasi nyata: ruang -> codec -> noise -> level
    if rng.random() < cfg.reverb:
        x = apply_reverb(x, rng)
    if rng.random() < cfg.codec:
        x = codec_augment(x, rng)
    if rng.random() < cfg.noise:
        beta = float(rng.choice([0.0, 1.0, 2.0]))
        nz = colored_noise(len(x), rng, beta)
        snr = float(rng.uniform(*cfg.snr_range))
        x = add_noise_snr(x, nz, snr)
    if rng.random() < cfg.gain:
        x = x * float(10 ** (rng.uniform(-8, 8) / 20.0))
    m = float(np.abs(x).max())
    if m > 0.99:
        x = x * (0.99 / m)
    return x


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class FoRDataset(Dataset):
    """
    Mengembalikan waveform float32 panjang tetap 32.000 sampel.
    Konversi ke representasi model (mel / feature-extractor) dilakukan di model,
    supaya tiap arsitektur memakai normalisasi bawaannya sendiri (menghindari
    bug normalisasi ganda).
    """

    def __init__(self, rows, augment: AugmentConfig | None = None,
                 normalize: str = "loudness", seed: int = 0):
        self.rows = rows
        self.aug = augment or AugmentConfig.none()
        self.normalize = normalize
        self.seed = seed
        self.epoch = 0          # WAJIB di-update tiap epoch: lihat set_epoch()

    def set_epoch(self, epoch: int):
        """
        Memperbarui sumbu epoch pada seed RNG augmentasi.

        BUG YANG DIPERBAIKI: sebelumnya RNG hanya di-seed dari (hash nama berkas,
        seed run). Karena keduanya tetap sepanjang pelatihan, setiap berkas
        menerima SATU varian augmentasi yang sama di semua epoch — augmentasi
        on-the-fly berubah menjadi augmentasi offline statis, dan keragaman
        efektifnya terpotong sebanyak jumlah epoch.

        Dengan epoch masuk ke seed, tiap berkas mendapat varian berbeda tiap
        epoch, tetapi tetap deterministik dan dapat direproduksi.
        """
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x, sr = sf.read(r["path"], dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if len(x) < NSAMP:
            x = np.pad(x, (0, NSAMP - len(x)))
        elif len(x) > NSAMP:
            x = x[:NSAMP]

        # RNG deterministik per (berkas, seed run, EPOCH) — epoch wajib ikut,
        # jika tidak augmentasi menjadi statis sepanjang pelatihan
        rng = np.random.default_rng(
            (_stable_hash(r["fname"]) + self.seed * 1_000_003
             + self.epoch * 7_919) % (2 ** 31))

        x = x.astype(np.float64)
        x = augment_waveform(x, self.aug, rng)

        if self.normalize == "loudness":
            x = loudness_normalize(x)
        elif self.normalize == "peak":                    # perilaku proposal, untuk ablation
            m = float(np.abs(x).max())
            x = x / m if m > 0 else x

        return {
            "wav": torch.from_numpy(x.astype(np.float32)),
            "label": torch.tensor(r["label"], dtype=torch.long),
            "idx": torch.tensor(i, dtype=torch.long),
        }


def collate(batch):
    return {
        "wav": torch.stack([b["wav"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "idx": torch.stack([b["idx"] for b in batch]),
    }
