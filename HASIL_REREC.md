
# Evaluasi Zero-Shot Lintas-Kondisi: for-2sec → for-rerec

Model dilatih pada **for-2sec** (klip bersih 2 detik), diuji tanpa penyesuaian apa pun pada **for-rerec** (816 berkas hasil pemutaran-ulang di ruangan dengan degradasi mirip telepon).

Kondisi ini **tidak pernah dipakai** untuk merancang augmentasi, memilih hyperparameter, atau memilih checkpoint.

Ambang: prior-matched (kelas seimbang 408/408).

| model | augmentasi latih | n seed | akurasi | EER | AUC |
|---|---|---|---|---|---|
| `ast` | codec | 3 | **79.00%** ±8.50 | 21.00% | 0.8788 |
| `cnn_asp` | codec | 3 | **70.06%** ±9.71 | 29.94% | 0.7624 |
| `cnn_asp` | full | 1 | **83.95%**  | 16.05% | 0.9148 |
| `cnn_asp` | none | 1 | **36.76%**  | 63.24% | 0.3137 |
| `cnnlstm` | codec | 3 | **63.52%** ±15.54 | 36.48% | 0.6846 |
| `hubert` | codec | 8 | **78.69%** ±9.75 | 21.28% | 0.8530 |
| `hubert` | full | 3 | **86.15%** ±3.55 | 13.89% | 0.9356 |
| `wav2vec2` | codec | 3 | **77.74%** ±3.11 | 22.30% | 0.8579 |
| `wavlm` | codec | 3 | **87.70%** ±2.95 | 12.21% | 0.9406 |
| `wavlm` | full | 3 | **97.30%** ±1.30 | 2.70% | 0.9965 |

## Uji kunci: apakah augmentasi codec menolong di kondisi yang tak pernah dilihat?

| model | tanpa augmentasi | + codec | Δ |
|---|---|---|---|
| `cnn_asp` | 36.76% | 70.06% | **+33.29 pp** |

Bila Δ positif, intervensi codec tergeneralisasi ke kondisi yang tidak dipakai merancangnya → tuduhan *test-set peeking* (serangan A1) gugur.