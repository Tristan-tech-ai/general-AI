# Replikasi Metodologi Proposal Apa Adanya

Konfigurasi persis seperti tertulis di proposal: split acak 60/20/20, learning rate 0,001 seragam untuk semua model, 20 epoch tanpa early stopping, normalisasi peak amplitudo, augmentasi noise SNR 15 sampai 30 dB, batch 32, ambang keputusan 0,5.

| model | akurasi | presisi | recall | F1 | AUC | EER |
|---|---|---|---|---|---|---|
| wav2vec2 | **99.80%** | 99.66% | 99.94% | 99.80% | 1.0000 | 0.28% |

Metrik dihitung pada ambang 0,5 seperti yang tersirat di proposal. Jumlah berkas uji 3574 (split acak 20 persen).

## Confusion matrix

| model | TP | TN | FP | FN |
|---|---|---|---|---|
| wav2vec2 | 1777 | 1790 | 6 | 1 |

## Pembanding: metodologi yang diperbaiki

| model dan augmentasi | n seed | akurasi pada partisi resmi |
|---|---|---|
| wavlm + fullbgrb | 3 | **98.90%** |
| wavlm + fullbg | 3 | **98.65%** |
| wavlm + full | 3 | **98.62%** |
| nes2net + fullrb | 3 | **98.50%** |
| nes2net + fullbgrb | 3 | **97.46%** |

Perlu dicatat bahwa kedua kolom tidak setara. Replikasi proposal diuji pada split acak, sedangkan versi diperbaiki diuji pada partisi resmi yang memisahkan domain rekaman. Angka yang lebih tinggi pada split acak tidak berarti model lebih baik, melainkan bahwa tugasnya lebih mudah. Inilah inti temuan penelitian ini.
