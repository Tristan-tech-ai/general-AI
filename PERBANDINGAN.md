# Perbandingan Arsitektur, split `official`, augmentasi `codec`

Test set: **1088** berkas. 1 berkas = 0.092 pp.

Ambang: prior-matched (transduktif, tanpa label test).

## 1. Rerata ± simpangan baku atas seed

| model | n seed | akurasi | EER | AUC | seed individual |
|---|---|---|---|---|---|
| `ast` | 3 | **86.43%** ±2.94 | 13.42% | 0.9418 | 89.34, 86.49, 83.46 |
| `cnn_asp` | 3 | **91.94%** ±3.50 | 8.06% | 0.9712 | 95.04, 92.65, 88.14 |
| `cnnlstm` | 3 | **83.52%** ±2.28 | 16.48% | 0.8976 | 84.93, 84.74, 80.88 |
| `hubert` | 4 | **97.56%** ±0.66 | 2.44% | 0.9974 | 98.16, 97.98, 97.43, 96.69 |
| `wav2vec2` | 3 | **90.75%** ±0.51 | 9.34% | 0.9689 | 91.08, 90.99, 90.17 |

Selisih peringkat 1 (`hubert`) dan 2 (`cnn_asp`): **5.62 pp**, sedangkan std gabungan **±2.52 pp**.
→ Selisih melampaui satu simpangan baku.

## 2. Uji McNemar berpasangan (seed terbaik per model)

Test set identik untuk semua model → data berpasangan. n01 = A benar & B salah; n10 = A salah & B benar.

| perbandingan | n01 | n10 | p mentah | p terkoreksi | signifikan? |
|---|---|---|---|---|---|
| ast vs hubert | 15 | 111 | 0 | 0 | ✅ ya |
| cnn_asp vs cnnlstm | 137 | 27 | 0 | 0 | ✅ ya |
| cnnlstm vs hubert | 10 | 154 | 0 | 0 | ✅ ya |
| hubert vs wav2vec2 | 87 | 10 | 1.199e-14 | 8.393e-14 | ✅ ya |
| ast vs cnn_asp | 44 | 106 | 6.338e-07 | 3.803e-06 | ✅ ya |
| cnnlstm vs wav2vec2 | 79 | 146 | 1.083e-05 | 5.413e-05 | ✅ ya |
| cnn_asp vs hubert | 15 | 49 | 3.707e-05 | 0.0001483 | ✅ ya |
| cnn_asp vs wav2vec2 | 91 | 48 | 0.0003675 | 0.001102 | ✅ ya |
| ast vs cnnlstm | 130 | 82 | 0.001247 | 0.002493 | ✅ ya |
| ast vs wav2vec2 | 88 | 107 | 0.1974 | 0.1974 | ❌ tidak |

Koreksi Holm-Bonferroni, α = 0,05.

## 3. Korelasi error antar model

φ rendah (< 0,5) → model gagal pada berkas berbeda → ensembling berguna.

| pasangan | φ | Jaccard error |
|---|---|---|
| ast vs cnn_asp | 0.058 | 0.062 |
| ast vs cnnlstm | 0.137 | 0.138 |
| ast vs hubert | 0.064 | 0.038 |
| ast vs wav2vec2 | -0.014 | 0.044 |
| cnn_asp vs cnnlstm | 0.223 | 0.141 |
| cnn_asp vs hubert | 0.126 | 0.072 |
| cnn_asp vs wav2vec2 | 0.018 | 0.041 |
| cnnlstm vs hubert | 0.134 | 0.057 |
| cnnlstm vs wav2vec2 | 0.030 | 0.074 |
| hubert vs wav2vec2 | 0.197 | 0.093 |

## 4. Ensemble (rata-rata skor)

| ensemble | anggota | akurasi | EER | AUC |
|---|---|---|---|---|
| `ast` (semua seed) | 3 | **89.71%** | 10.29% | 0.9651 |
| `cnn_asp` (semua seed) | 3 | **95.40%** | 4.87% | 0.9877 |
| `cnnlstm` (semua seed) | 3 | **86.03%** | 13.97% | 0.9304 |
| `hubert` (semua seed) | 4 | **98.71%** | 1.38% | 0.9993 |
| `wav2vec2` (semua seed) | 3 | **93.01%** | 6.99% | 0.9832 |
| **semua model** (seed terbaik) | 5 | **97.98%** | 1.93% | 0.9967 |
| **semua run** | 16 | **98.16%** | 1.84% | 0.9976 |
