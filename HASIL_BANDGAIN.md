# Sapuan Parameter Augmentasi Band-Gain

Seluruh sapuan dijalankan pada WavLM Large berencoder beku dengan preset augmentasi penuh ditambah band-gain, pada partisi resmi dan ambang prior-matched. Satu parameter diubah pada satu waktu, dua yang lain dipertahankan pada nilai bawaannya, yaitu f_lo 3000 Hz, enam pita, dan redaman sampai 12 dB.

Konfigurasi ini dipilih karena simpangan bakunya terkecil di antara seluruh konfigurasi dalam penelitian ini. Menyapu parameter pada arsitektur dengan ragam besar akan menenggelamkan selisih antar parameter di dalam derau antar inisialisasi.

| f_lo (Hz) | jumlah pita | redaman maks (dB) | n | Akurasi | Selisih dari bawaan | AUC | EER |
|---|---|---|---|---|---|---|---|
| 2000 | 6 | 12 | 1 | 98.90 | +0.25 | 0.9994 | 0.92 |
| 3000 | 3 | 12 | 1 | 97.33 | -1.32 | 0.9965 | 2.67 |
| 3000 | 6 | 6 | 1 | 99.08 | +0.43 | 0.9996 | 0.92 |
| 3000 (bawaan) | 6 | 12 | 3 | 98.65 (0.37) | +0.00 | 0.9991 | 1.29 |
| 3000 | 6 | 20 | 1 | 97.43 | -1.23 | 0.9978 | 2.48 |
| 3000 | 12 | 12 | 1 | 97.24 | -1.41 | 0.9977 | 2.67 |
| 4000 | 6 | 12 | 1 | 98.71 | +0.06 | 0.9994 | 1.29 |

Nilai bawaan mencapai 98.65 persen. Kombinasi terbaik dalam sapuan ini adalah f_lo 3000 Hz dengan 6 pita dan redaman 6 dB, yaitu 99.08 persen atau +0.43 poin persentase.

Selisih itu belum dapat dinyatakan bermakna. Sapuan ini memakai satu inisialisasi acak per titik, sedangkan simpangan baku konfigurasi bawaannya sendiri 0.37 poin persentase pada 3 inisialisasi. Titik terbaik perlu diulang dengan beberapa inisialisasi sebelum dapat dibandingkan secara sah, dan itu berlaku juga bila selisihnya tampak besar.

## Cakupan

Sebanyak 3 run bertag fullbg tidak masuk sapuan ini karena nama tag-nya di luar pola yang ditangani. Jumlahnya dicatat supaya pelewatan tidak berlangsung tanpa diketahui.

Tag yang dilewati: `wavlm_official_fullbgrb_b16e10_s1337`, `wavlm_official_fullbgrb_b16e10_s2024`, `wavlm_official_fullbgrb_b16e10_s42`.
