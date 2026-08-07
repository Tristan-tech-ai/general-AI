# Replikasi Metodologi Proposal Apa Adanya

Seluruh angka di halaman ini berasal dari run setelah perbaikan bug encoder beku yang diuraikan di KOREKSI_REPLIKASI_PROPOSAL.md. Sebelum perbaikan itu, encoder tidak pernah menerima gradien meskipun proposal menetapkan satu learning rate untuk seluruh model.

Konfigurasi persis seperti tertulis di proposal: split acak 60/20/20, learning rate 0,001 seragam untuk semua model, 20 epoch tanpa early stopping, normalisasi peak amplitudo, augmentasi noise SNR 15 sampai 30 dB, batch 32, ambang keputusan 0,5.

| model | akurasi | presisi | recall | F1 | AUC | EER |
|---|---|---|---|---|---|---|
| cnnlstm_proposal | **99.83%** | 99.66% | 100.00% | 99.83% | 0.9999 | 0.03% |
| wav2vec2 | **99.78%** | 99.55% | 100.00% | 99.78% | 1.0000 | 0.08% |
| ast | **99.69%** | 99.44% | 99.94% | 99.69% | 1.0000 | 0.14% |
| hubert | **99.69%** | 99.44% | 99.94% | 99.69% | 0.9985 | 0.39% |
| wavlm | **99.52%** | 99.11% | 99.94% | 99.52% | 0.9979 | 0.31% |

Metrik dihitung pada ambang 0,5 seperti yang tersirat di proposal. Jumlah berkas uji 3574 (split acak 20 persen).

## Confusion matrix

| model | TP | TN | FP | FN |
|---|---|---|---|---|
| cnnlstm_proposal | 1778 | 1790 | 6 | 0 |
| wav2vec2 | 1778 | 1788 | 8 | 0 |
| ast | 1777 | 1786 | 10 | 1 |
| hubert | 1777 | 1786 | 10 | 1 |
| wavlm | 1777 | 1780 | 16 | 1 |

## Konfigurasi proposal yang sama, diuji pada partisi resmi

Perbedaan dengan tabel di atas hanya pada cara data dibagi. Seluruh setelan pelatihan identik. Angka dalam kurung adalah simpangan baku antar inisialisasi acak.

| model | n | akurasi @0,5 | akurasi @prior | AUC | EER |
|---|---|---|---|---|---|
| ast | 3 | 74.85% | **93.57% (1.75)** | 0.9824 | 6.50% |
| wavlm | 5 | 53.40% | **63.29% (4.19)** | 0.7096 | 36.73% |
| hubert | 5 | 51.47% | **51.64% (6.47)** | 0.5597 | 48.60% |

Selisih antara kedua tabel jauh lebih besar daripada selisih antar arsitektur di dalam masing-masing tabel. Pada split acak seluruh model berkerumun dalam rentang setengah poin persentase, sedangkan pada partisi resmi rentangnya puluhan poin. Pemeringkatan yang dihasilkan kedua protokol juga tidak sama, sehingga memilih model berdasarkan split acak dapat menghasilkan pilihan yang keliru.

## Pembanding: metodologi yang diperbaiki

| model dan augmentasi | n seed | akurasi pada partisi resmi |
|---|---|---|
| wavlm + fullbgrb | 3 | **98.90%** |
| wavlm + fullbg | 3 | **98.65%** |
| nes2net + fullrb | 3 | **98.50%** |
| wavlm + full | 5 | **98.36%** |
| nes2net + fullbgrb | 3 | **97.46%** |

Perlu dicatat bahwa kedua kolom tidak setara. Replikasi proposal diuji pada split acak, sedangkan versi diperbaiki diuji pada partisi resmi yang memisahkan domain rekaman. Angka yang lebih tinggi pada split acak tidak berarti model lebih baik, melainkan bahwa tugasnya lebih mudah. Inilah inti temuan penelitian ini.
