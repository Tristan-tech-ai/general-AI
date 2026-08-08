# Sapuan Parameter Augmentasi Band-Gain

Seluruh sapuan dijalankan pada WavLM Large berencoder beku dengan preset augmentasi penuh ditambah band-gain, pada partisi resmi dan ambang prior-matched. Satu parameter diubah pada satu waktu, dua yang lain dipertahankan pada nilai bawaannya, yaitu f_lo 3000 Hz, enam pita, dan redaman sampai 12 dB.

Konfigurasi ini dipilih karena simpangan bakunya terkecil di antara seluruh konfigurasi dalam penelitian ini. Menyapu parameter pada arsitektur dengan ragam besar akan menenggelamkan selisih antar parameter di dalam derau antar inisialisasi.

| f_lo (Hz) | jumlah pita | redaman maks (dB) | n | Akurasi | Selisih dari bawaan | AUC | EER |
|---|---|---|---|---|---|---|---|
| 2000 | 6 | 12 | 1 | 98.90 | +0.25 | 0.9994 | 0.92 |
| 3000 | 3 | 12 | 3 | 97.95 (0.98) | -0.70 | 0.9970 | 2.02 |
| 3000 (tanpa band-gain) | 6 | 0 | 5 | 98.36 (0.63) | -0.29 | 0.9985 | 1.65 |
| 3000 | 6 | 1 | 3 | 98.44 (0.69) | -0.21 | 0.9987 | 1.59 |
| 3000 | 6 | 3 | 5 | 98.22 (1.89) | -0.44 | 0.9976 | 1.73 |
| 3000 | 6 | 6 | 3 | 98.71 (0.64) | +0.06 | 0.9988 | 1.23 |
| 3000 (bawaan) | 6 | 12 | 3 | 98.65 (0.37) | +0.00 | 0.9991 | 1.29 |
| 3000 | 6 | 20 | 1 | 97.43 | -1.23 | 0.9978 | 2.48 |
| 3000 | 12 | 12 | 3 | 98.65 (1.25) | -0.00 | 0.9990 | 1.32 |
| 4000 | 6 | 12 | 1 | 98.71 | +0.06 | 0.9994 | 1.29 |

Nilai bawaan mencapai 98.65 persen. Kombinasi terbaik dalam sapuan ini adalah f_lo 2000 Hz dengan 6 pita dan redaman 12 dB, yaitu 98.90 persen atau +0.25 poin persentase.

## Pengujian terhadap dua acuan

Sebuah titik hanya berguna bila mengungguli konfigurasi bawaan dan titik tanpa band-gain sekaligus. Mengungguli bawaan saja tidak cukup, karena hal itu juga akan terjadi bila band-gain sebaiknya dilemahkan sampai hampir tidak ada.

| Titik | Acuan | n | Selisih | p mentah | p Holm | Bacaan |
|---|---|---|---|---|---|---|
| bawaan 12 dB | tanpa band-gain | 3/5 | +0.29 | 0.4479 | 1.0000 | **belum terbukti berbeda** |
| f_lo 3000, 6 pita, 1 dB | bawaan 12 dB | 3/3 | -0.21 | 0.6686 | 1.0000 | **belum terbukti berbeda** |
| f_lo 3000, 6 pita, 1 dB | tanpa band-gain | 3/5 | +0.07 | 0.8880 | 1.0000 | **belum terbukti berbeda** |
| f_lo 3000, 6 pita, 3 dB | bawaan 12 dB | 5/3 | -0.44 | 0.6405 | 1.0000 | **belum terbukti berbeda** |
| f_lo 3000, 6 pita, 3 dB | tanpa band-gain | 5/5 | -0.15 | 0.8753 | 1.0000 | **belum terbukti berbeda** |
| f_lo 3000, 6 pita, 6 dB | bawaan 12 dB | 3/3 | +0.06 | 0.8941 | 1.0000 | **belum terbukti berbeda** |
| f_lo 3000, 6 pita, 6 dB | tanpa band-gain | 3/5 | +0.35 | 0.4903 | 1.0000 | **belum terbukti berbeda** |
| f_lo 3000, 12 pita, 12 dB | bawaan 12 dB | 3/3 | -0.00 | 1.0000 | 1.0000 | **belum terbukti berbeda** |
| f_lo 3000, 12 pita, 12 dB | tanpa band-gain | 3/5 | +0.29 | 0.7383 | 1.0000 | **belum terbukti berbeda** |
| f_lo 3000, 3 pita, 12 dB | bawaan 12 dB | 3/3 | -0.70 | 0.3422 | 1.0000 | **belum terbukti berbeda** |
| f_lo 3000, 3 pita, 12 dB | tanpa band-gain | 3/5 | -0.42 | 0.5576 | 1.0000 | **belum terbukti berbeda** |

## Cakupan

Sebanyak 3 run bertag fullbg tidak masuk sapuan ini karena nama tag-nya di luar pola yang ditangani. Jumlahnya dicatat supaya pelewatan tidak berlangsung tanpa diketahui.

Tag yang dilewati: `wavlm_official_fullbgrb_b16e10_s1337`, `wavlm_official_fullbgrb_b16e10_s2024`, `wavlm_official_fullbgrb_b16e10_s42`.
