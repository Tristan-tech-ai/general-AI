# Fusi Bertumpuk (Stacking) — Dilatih pada Validasi

23 run sebagai fitur. Meta-classifier **hanya melihat label validasi**; label test tidak dipakai untuk melatih maupun memilih apa pun.

## Batas atas oracle (bukan hasil yang dapat dicapai)

Sampel yang benar pada **setidaknya satu** run: **1088/1088** = **100.00%**

Angka ini memerlukan label test untuk memilih run per sampel, jadi **tidak dapat dicapai dalam praktik**. Fungsinya hanya menunjukkan bahwa sisa error bukan keterbatasan informasi kumpulan model.

## Hasil yang SAH

| metode | dilatih pada | akurasi test | EER | AUC | salah |
|---|---|---|---|---|---|
| rata-rata logit (tanpa pelatihan) | — | **99.26%** | 0.64% | 0.9993 | 8/1088 |
| run tunggal terbaik menurut val (`ast_official_codec_b32e10_s133`) | validasi | **86.49%** | 13.33% | 0.9409 | 147/1088 |
| stacking LogReg (C=1) | validasi | **99.63%** | 0.37% | 0.9996 | 4/1088 |
| stacking LogReg (C=0.1) | validasi | **99.45%** | 0.55% | 0.9995 | 6/1088 |
| stacking LogReg (C=0.01) | validasi | **99.45%** | 0.55% | 0.9994 | 6/1088 |
