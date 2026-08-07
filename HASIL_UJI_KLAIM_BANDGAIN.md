# Apakah Klaim Generalisasi Band-Gain Bertahan?

Klaim mengenai band-gain dinyatakan pada dua sumbu, yaitu recall terhadap sistem text-to-speech generasi 2025 sampai 2026 dan recall terhadap sistem lama yang tidak dikompresi MP3. Angka yang dilaporkan berupa rerata atas tiga inisialisasi acak, namun selisihnya belum pernah diuji terhadap sebaran itu sendiri. Tabel berikut melakukan pengujian tersebut dengan uji t Welch dan koreksi Holm-Bonferroni.

| Arsitektur | Sumbu | n | Tanpa band-gain | Dengan band-gain | Selisih | p mentah | p Holm | Bacaan |
|---|---|---|---|---|---|---|---|---|
| nes2net | recall TTS 2025-2026 | 3/3 | 94.97 (5.34) | 93.50 (4.77) | -1.47 | 0.7400 | 1.0000 | **belum terbukti berbeda** |
| nes2net | recall TTS 2019 non-MP3 | 3/3 | 82.11 (22.95) | 92.15 (4.19) | +10.04 | 0.5297 | 1.0000 | **belum terbukti berbeda** |
| wavlm | recall TTS 2025-2026 | 3/3 | 88.39 (7.29) | 92.56 (2.15) | +4.17 | 0.4298 | 1.0000 | **belum terbukti berbeda** |
| wavlm | recall TTS 2019 non-MP3 | 3/3 | 97.48 (3.11) | 94.26 (9.37) | -3.22 | 0.6196 | 1.0000 | **belum terbukti berbeda** |
| hubert | recall TTS 2025-2026 | 3/3 | 85.67 (6.11) | 91.33 (4.04) | +5.67 | 0.2616 | 1.0000 | **belum terbukti berbeda** |
| hubert | recall TTS 2019 non-MP3 | 3/3 | 29.22 (28.96) | 58.41 (5.23) | +29.19 | 0.2203 | 1.0000 | **belum terbukti berbeda** |

## Bacaan

Tidak satu pun klaim bertahan. Seluruh selisih, termasuk yang besarannya belasan sampai puluhan poin persentase, berada di dalam ragam antar inisialisasi acak.

Penyebabnya terlihat langsung pada kolom simpangan baku. Recall pada sumbu-sumbu ini jauh lebih tidak stabil daripada akurasi Fake-or-Real. Sebagai contoh, Nes2Net tanpa band-gain menghasilkan recall 93,8 dan 96,9 dan 55,7 persen pada sistem lama non-MP3, sehingga selisih 10 poin persentase yang sempat dilaporkan sebagai keunggulan band-gain sebenarnya ditentukan hampir seluruhnya oleh satu inisialisasi yang buruk.

Konsekuensinya, klaim mengenai keunggulan generalisasi band-gain harus ditarik sebagai temuan dan dinyatakan ulang sebagai pengamatan yang belum diuji. Yang masih dapat dinyatakan adalah bahwa band-gain tidak merusak, sedangkan RawBoost menurunkan recall pada kedua sumbu dengan besaran yang lebih konsisten. Perbandingan itu pun perlu diuji dengan cara yang sama sebelum dipakai.

Sumbu recall menuntut jumlah inisialisasi yang jauh lebih banyak daripada tiga. Dengan simpangan baku belasan poin persentase, mendeteksi selisih 10 poin secara meyakinkan membutuhkan puluhan inisialisasi, dan itu di luar anggaran komputasi penelitian ini. Keterbatasan tersebut dilaporkan apa adanya.
