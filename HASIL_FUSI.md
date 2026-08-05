# Varian Fusi: Mengatasi Dilusi Model Dominan

Rata-rata probabilitas polos membuat HuBERT (99,63%) turun ketika digabung dengan model lemah. Di sini diuji skema fusi alternatif.

Run HuBERT tersedia: 3 (seed [42, 1337, 2024])

## 1. Perilaku tiap seed HuBERT pada sampel yang salah di ensemble

| sampel | label | seed 1337 | seed 2024 | seed 42 | ensemble |
|---|---|---|---|---|---|
| 19 | **real** | ✅ 0.032 | ❌ 0.134 | ❌ 0.052 | ❌ 0.073 |
| 117 | **real** | ❌ 0.032 | ❌ 0.095 | ❌ 0.069 | ❌ 0.065 |
| 644 | **fake** | ❌ 0.028 | ❌ 0.089 | ❌ 0.034 | ❌ 0.050 |
| 1073 | **fake** | ❌ 0.020 | ❌ 0.041 | ❌ 0.031 | ❌ 0.031 |

Sampel yang salah di **SEMUA** seed: **3/4**

→ 1 sampel diperbaiki oleh sebagian seed. Menambah seed berpeluang membaliknya.

## 2. Perbandingan skema fusi (seluruh arsitektur)

| skema | akurasi | EER | AUC | salah |
|---|---|---|---|---|
| rata-rata probabilitas (baseline) | **98.53%** | 1.47% | 0.9982 | 16/1088 |
| rata-rata logit | **99.26%** | 0.74% | 0.9995 | 8/1088 |
| rata-rata peringkat | **99.26%** | 0.74% | 0.9992 | 8/1088 |
| median probabilitas | **99.26%** | 0.64% | 0.9994 | 8/1088 |
| maksimum probabilitas | **97.79%** | 2.21% | 0.9961 | 24/1088 |
| berbobot AUC^4 | **98.53%** | 1.29% | 0.9985 | 16/1088 |
| berbobot AUC^16 | **98.90%** | 1.01% | 0.9993 | 12/1088 |
| berbobot AUC^64 | **99.08%** | 1.10% | 0.9997 | 10/1088 |
| top-1 menurut AUC (hubert) | **99.63%** | 0.37% | 0.9998 | 4/1088 |
| top-2 menurut AUC (hubert, wavlm) | **99.63%** | 0.28% | 0.9999 | 4/1088 |
| top-3 menurut AUC (hubert, wavlm, cnn_asp) | **98.53%** | 1.47% | 0.9990 | 16/1088 |

**Terbaik: top-1 menurut AUC (hubert) → 99.63%, 4 salah dari 1088**
