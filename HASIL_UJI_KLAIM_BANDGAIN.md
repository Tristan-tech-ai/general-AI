# Apakah Klaim Generalisasi Band-Gain Bertahan?

Klaim mengenai band-gain dinyatakan pada dua sumbu, yaitu recall terhadap sistem text-to-speech generasi 2025 sampai 2026 dan recall terhadap sistem lama yang tidak dikompresi MP3. Angka yang dilaporkan berupa rerata atas tiga inisialisasi acak, namun selisihnya belum pernah diuji terhadap sebaran itu sendiri. Tabel berikut melakukan pengujian tersebut dengan uji t Welch dan koreksi Holm-Bonferroni.

| Arsitektur | Tambahan | Sumbu | n | Tanpa | Dengan | Selisih | p mentah | p Holm | Bacaan |
|---|---|---|---|---|---|---|---|---|---|
| nes2net | band-gain | recall TTS 2025-2026 | 3/3 | 94.97 (5.34) | 93.50 (4.77) | -1.47 | 0.7400 | 1.0000 | **belum terbukti berbeda** |
| nes2net | band-gain | recall TTS 2019 non-MP3 | 3/3 | 82.11 (22.95) | 92.15 (4.19) | +10.04 | 0.5297 | 1.0000 | **belum terbukti berbeda** |
| nes2net | RawBoost | recall TTS 2025-2026 | 3/3 | 94.97 (5.34) | 87.64 (6.41) | -7.33 | 0.2046 | 1.0000 | **belum terbukti berbeda** |
| nes2net | RawBoost | recall TTS 2019 non-MP3 | 3/3 | 82.11 (22.95) | 60.67 (28.44) | -21.44 | 0.3694 | 1.0000 | **belum terbukti berbeda** |
| nes2net | band-gain + RawBoost | recall TTS 2025-2026 | 3/3 | 94.97 (5.34) | 93.67 (3.47) | -1.31 | 0.7431 | 1.0000 | **belum terbukti berbeda** |
| nes2net | band-gain + RawBoost | recall TTS 2019 non-MP3 | 3/3 | 82.11 (22.95) | 85.04 (11.54) | +2.93 | 0.8564 | 1.0000 | **belum terbukti berbeda** |
| wavlm | band-gain | recall TTS 2025-2026 | 3/3 | 88.39 (7.29) | 92.56 (2.15) | +4.17 | 0.4298 | 1.0000 | **belum terbukti berbeda** |
| wavlm | band-gain | recall TTS 2019 non-MP3 | 3/3 | 97.48 (3.11) | 94.26 (9.37) | -3.22 | 0.6196 | 1.0000 | **belum terbukti berbeda** |
| wavlm | band-gain + RawBoost | recall TTS 2025-2026 | 3/3 | 88.39 (7.29) | 92.69 (4.68) | +4.31 | 0.4457 | 1.0000 | **belum terbukti berbeda** |
| wavlm | band-gain + RawBoost | recall TTS 2019 non-MP3 | 3/3 | 97.48 (3.11) | 99.33 (0.80) | +1.85 | 0.4127 | 1.0000 | **belum terbukti berbeda** |
| hubert | band-gain | recall TTS 2025-2026 | 3/3 | 85.67 (6.11) | 91.33 (4.04) | +5.67 | 0.2616 | 1.0000 | **belum terbukti berbeda** |
| hubert | band-gain | recall TTS 2019 non-MP3 | 3/3 | 29.22 (28.96) | 58.41 (5.23) | +29.19 | 0.2203 | 1.0000 | **belum terbukti berbeda** |

## Sebaran, bukan rerata

Rerata tidak dapat dibedakan, tetapi sebarannya berbeda secara konsisten. Tabel berikut membandingkan simpangan baku antar inisialisasi acak.

| Arsitektur | Sumbu | Simpangan tanpa tambahan | Simpangan dengan band-gain | Rasio |
|---|---|---|---|---|
| nes2net | recall TTS 2025-2026 | 5.34 | 4.77 | 1.1x lebih kecil |
| nes2net | recall TTS 2019 non-MP3 | 22.95 | 4.19 | 5.5x lebih kecil |
| wavlm | recall TTS 2025-2026 | 7.29 | 2.15 | 3.4x lebih kecil |
| wavlm | recall TTS 2019 non-MP3 | 3.11 | 9.37 | 3.0x lebih besar |
| hubert | recall TTS 2025-2026 | 6.11 | 4.04 | 1.5x lebih kecil |
| hubert | recall TTS 2019 non-MP3 | 28.96 | 5.23 | 5.5x lebih kecil |

## Bacaan

Tidak satu pun klaim bertahan. Seluruh selisih, termasuk yang besarannya belasan sampai puluhan poin persentase, berada di dalam ragam antar inisialisasi acak.

Penyebabnya terlihat langsung pada kolom simpangan baku. Recall pada sumbu-sumbu ini jauh lebih tidak stabil daripada akurasi Fake-or-Real. Sebagai contoh, Nes2Net tanpa band-gain menghasilkan recall 93,8 dan 96,9 dan 55,7 persen pada sistem lama non-MP3, sehingga selisih 10 poin persentase yang sempat dilaporkan sebagai keunggulan band-gain sebenarnya ditentukan hampir seluruhnya oleh satu inisialisasi yang buruk.

Hal yang sama berlaku bagi pembandingnya. Klaim bahwa RawBoost menurunkan generalisasi juga tidak bertahan, dengan selisih 7,33 dan 21,44 poin persentase yang keduanya berada di dalam ragam. Menguji band-gain sambil menerima klaim pembandingnya apa adanya akan menjadi pemilihan yang tidak sah, sehingga keduanya diuji dengan cara yang sama dan keduanya sama-sama tidak terbukti.

Satu pola tetap terlihat, yaitu pada sebarannya dan bukan pada reratanya. Pada lima dari enam perbandingan, band-gain menghasilkan simpangan baku yang lebih kecil, dalam dua kasus sekitar lima setengah kali lebih kecil. Pada satu perbandingan polanya terbalik. Pola ini dilaporkan sebagai pengamatan deskriptif dan tidak diuji secara formal, karena pengujian kesamaan ragam pada tiga inisialisasi memiliki daya yang bahkan lebih rendah daripada pengujian rerata.

Konsekuensinya, klaim mengenai keunggulan generalisasi band-gain harus ditarik sebagai temuan dan dinyatakan ulang sebagai pengamatan yang belum diuji.

Sumbu recall menuntut jumlah inisialisasi yang jauh lebih banyak daripada tiga. Dengan simpangan baku belasan poin persentase, mendeteksi selisih 10 poin secara meyakinkan membutuhkan puluhan inisialisasi, dan itu di luar anggaran komputasi penelitian ini. Keterbatasan tersebut dilaporkan apa adanya.
