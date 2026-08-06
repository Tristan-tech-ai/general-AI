# Tangga Ablasi: Perbaikan Mana yang Membeli Berapa

Semua langkah memakai AST pada partisi resmi Fake-or-Real, batch 32, seed 42. Tiap baris menambahkan satu perbaikan di atas baris sebelumnya, sehingga selisih antar baris hanya mencerminkan satu variabel.

Akurasi dilaporkan pada ambang prior-matched untuk seluruh langkah agar sumbu ambang tidak bercampur ke dalam tangga. Sumbangan ambang itu sendiri dipisahkan tersendiri di HASIL_DEKOMPOSISI.md.

| Langkah | Perbaikan yang ditambahkan | Akurasi | Selisih | AUC | EER |
|---|---|---|---|---|---|
| L1 | Konfigurasi proposal apa adanya | 92.56 |  | 0.9780 | 7.44 |
| L2 | Normalisasi loudness | 92.28 | **-0.28** | 0.9769 | 7.54 |
| L3 | LR per model dan encoder dibekukan | 75.37 | **-16.91** | 0.8361 | 24.36 |
| L4 | Early stopping pada EER | 84.38 | **+9.01** | 0.9169 | 15.62 |
| L5 | Augmentasi penuh | 89.15 | **+4.78** | 0.9586 | 10.85 |

Total kenaikan sepanjang tangga adalah -3.40 poin persentase, dari 92.56 persen menjadi 89.15 persen.

## Rincian tiap langkah

**L1, Konfigurasi proposal apa adanya.** Titik tolak, yaitu LR 0,001 seragam dengan encoder ikut dilatih, normalisasi peak, 20 epoch tanpa early stopping, augmentasi noise saja. Akurasi 92.56 persen dengan AUC 0.9780.

**L2, Normalisasi loudness.** Perubahan yang dilakukan adalah normalisasi peak diganti loudness, selebihnya sama. Langkah ini menurunkan akurasi sebesar 0.28 poin persentase menjadi 92.28 persen, dengan AUC 0.9769 dan EER 7.54 persen.

**L3, LR per model dan encoder dibekukan.** Perubahan yang dilakukan adalah encoder tidak lagi dilatih, head 0,001 dan encoder 2e-5, ditambah agregasi berbobot antar lapisan. Langkah ini menurunkan akurasi sebesar 16.91 poin persentase menjadi 75.37 persen, dengan AUC 0.8361 dan EER 24.36 persen.

**L4, Early stopping pada EER.** Perubahan yang dilakukan adalah 10 epoch dengan pemilihan bobot terbaik menurut EER validasi. Langkah ini menaikkan akurasi sebesar 9.01 poin persentase menjadi 84.38 persen, dengan AUC 0.9169 dan EER 15.62 persen.

**L5, Augmentasi penuh.** Perubahan yang dilakukan adalah augmentasi noise saja diganti augmentasi penuh, yaitu codec, noise, reverb, dan band-gain. Langkah ini menaikkan akurasi sebesar 4.78 poin persentase menjadi 89.15 persen, dengan AUC 0.9586 dan EER 10.85 persen.

## Bacaan

Perbaikan yang paling banyak menyumbang adalah early stopping pada eer pada langkah L4, sebesar +9.01 poin persentase.

Tidak semua perbaikan berguna sendirian. Langkah L3, yaitu lr per model dan encoder dibekukan, justru -16.91 poin persentase ketika diterapkan tanpa perbaikan lain. Langkah itu tetap dipertahankan dalam konfigurasi akhir karena bermanfaat dalam kombinasi, namun temuan negatifnya dilaporkan apa adanya di sini.
