# Sapuan Parameter Augmentasi Band-Gain

Seluruh sapuan dijalankan pada WavLM Large berencoder beku dengan preset augmentasi penuh ditambah band-gain, pada partisi resmi dan ambang prior-matched. Satu parameter diubah pada satu waktu, dua yang lain dipertahankan pada nilai bawaannya, yaitu f_lo 3000 Hz, enam pita, dan redaman sampai 12 dB.

Konfigurasi ini dipilih karena simpangan bakunya terkecil di antara seluruh konfigurasi dalam penelitian ini. Menyapu parameter pada arsitektur dengan ragam besar akan menenggelamkan selisih antar parameter di dalam derau antar inisialisasi.

| f_lo (Hz) | jumlah pita | redaman maks (dB) | n | Akurasi | Selisih dari bawaan | AUC | EER |
|---|---|---|---|---|---|---|---|
| 2000 | 6 | 12 | 1 | 98.90 | +0.25 | 0.9994 | 0.92 |
| 3000 | 3 | 12 | 1 | 97.33 | -1.32 | 0.9965 | 2.67 |
| 3000 (tanpa band-gain) | 6 | 0 | 5 | 98.36 (0.63) | -0.29 | 0.9985 | 1.65 |
| 3000 | 6 | 1 | 1 | 99.08 | +0.43 | 0.9998 | 0.92 |
| 3000 | 6 | 3 | 1 | 99.45 | +0.80 | 0.9998 | 0.46 |
| 3000 | 6 | 6 | 2 | 98.53 (0.78) | -0.12 | 0.9986 | 1.47 |
| 3000 (bawaan) | 6 | 12 | 3 | 98.65 (0.37) | +0.00 | 0.9991 | 1.29 |
| 3000 | 6 | 20 | 1 | 97.43 | -1.23 | 0.9978 | 2.48 |
| 3000 | 12 | 12 | 1 | 97.24 | -1.41 | 0.9977 | 2.67 |
| 4000 | 6 | 12 | 1 | 98.71 | +0.06 | 0.9994 | 1.29 |

Nilai bawaan mencapai 98.65 persen. Kombinasi terbaik dalam sapuan ini adalah f_lo 3000 Hz dengan 6 pita dan redaman 3 dB, yaitu 99.45 persen atau +0.80 poin persentase.

## Pengujian terhadap dua acuan

Sebuah titik hanya berguna bila mengungguli konfigurasi bawaan dan titik tanpa band-gain sekaligus. Mengungguli bawaan saja tidak cukup, karena hal itu juga akan terjadi bila band-gain sebaiknya dilemahkan sampai hampir tidak ada.

| Titik | Acuan | n | Selisih | p mentah | p Holm | Bacaan |
|---|---|---|---|---|---|---|
| f_lo 3000, 6 pita, 6 dB | bawaan 12 dB | 2/3 | -0.12 | 0.8634 | 1.0000 | **belum terbukti berbeda** |
| f_lo 3000, 6 pita, 6 dB | tanpa band-gain | 2/5 | +0.17 | 0.8204 | 1.0000 | **belum terbukti berbeda** |

## Cakupan

Sebanyak 3 run bertag fullbg tidak masuk sapuan ini karena nama tag-nya di luar pola yang ditangani. Jumlahnya dicatat supaya pelewatan tidak berlangsung tanpa diketahui.

Tag yang dilewati: `wavlm_official_fullbgrb_b16e10_s1337`, `wavlm_official_fullbgrb_b16e10_s2024`, `wavlm_official_fullbgrb_b16e10_s42`.
