# HuBERT Large — 8 Seed + Ensemble Kumulatif

Split resmi FoR, augmentasi codec per-epoch, batch 16, 10 epoch, ambang prior-matched.

## Per seed

| seed | akurasi | EER | AUC | salah |
|---|---|---|---|---|
| 2024 | 99.45% | 0.55% | 0.9998 | 6/1088 |
| 2718 | 99.08% | 0.92% | 0.9990 | 10/1088 |
| 7 | 98.90% | 1.10% | 0.9993 | 12/1088 |
| 99 | 97.79% | 2.21% | 0.9983 | 24/1088 |
| 555 | 97.61% | 2.39% | 0.9976 | 26/1088 |
| 42 | 97.52% | 2.48% | 0.9973 | 27/1088 |
| 31415 | 94.03% | 5.97% | 0.9898 | 65/1088 |
| 1337 | 93.93% | 6.07% | 0.9895 | 66/1088 |

**n=8 · rerata 97.29% ± 2.16 · EER 2.71% ± 2.16 · rentang 5.51 pp**

## Ensemble kumulatif (urut seed terbaik menurut akurasi)

| jumlah seed | akurasi | EER | AUC | salah |
|---|---|---|---|---|
| 1 | **99.45%** | 0.55% | 0.9998 | 6/1088 |
| 2 | **99.63%** | 0.37% | 0.9998 | 4/1088 |
| 3 | **99.63%** | 0.28% | 0.9999 | 4/1088 |
| 4 | **99.63%** | 0.37% | 0.9999 | 4/1088 |
| 5 | **99.82%** | 0.18% | 0.9999 | 2/1088 |
| 6 | **99.82%** | 0.18% | 0.9999 | 2/1088 |
| 7 | **99.82%** | 0.18% | 0.9999 | 2/1088 |
| 8 | **99.82%** | 0.18% | 0.9999 | 2/1088 |

**Seluruh 8 seed (tanpa seleksi apa pun — ini angka yang sah):** **99.82%**, EER 0.18%, AUC 0.9999, **2 salah dari 1088**

> Kolom kumulatif di atas mengurutkan seed menurut akurasi **test**, jadi puncaknya (99.82% pada 5 seed) adalah angka oracle — tidak sah dilaporkan sebagai hasil. Angka yang sah adalah baris 'seluruh 8 seed'.

## Sisa 2 error

| idx | label | skor ensemble | seed yang benar |
|---|---|---|---|
| 19 | real | 0.0545 | 1337, 7, 99 (3/8) |
| 1073 | fake | 0.0272 | 2718, 7 (2/8) |