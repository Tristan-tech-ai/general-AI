# Tangga Ablasi: Perbaikan Mana yang Membeli Berapa

Semua langkah memakai AST pada partisi resmi Fake-or-Real, batch 32, seed 42. Tiap baris menambahkan satu perbaikan di atas baris sebelumnya, sehingga selisih antar baris hanya mencerminkan satu variabel.

Akurasi dilaporkan pada ambang prior-matched untuk seluruh langkah agar sumbu ambang tidak bercampur ke dalam tangga. Sumbangan ambang itu sendiri dipisahkan tersendiri di HASIL_DEKOMPOSISI.md.

Selisih tiap langkah diuji terhadap langkah sebelumnya dengan uji t Welch. Seluruh tangga dijalankan pada AST, yaitu arsitektur dengan ragam antar inisialisasi terbesar di antara yang diuji, sehingga selisih yang kecil di sini menuntut kehati-hatian khusus.

| Langkah | Perbaikan yang ditambahkan | n | Akurasi | Selisih | p mentah | p Holm | AUC | EER |
|---|---|---|---|---|---|---|---|---|
| L1 | Konfigurasi proposal apa adanya | 3 | 93.57 (1.75) |  |  |  | 0.9824 | 6.50 |
| L2 | Normalisasi loudness | 2 | 90.53 (2.47) | **-3.03** | 0.293 | 0.401 | 0.9683 | 9.33 |
| L3 | LR per model dan encoder dibekukan | 2 | 76.10 (1.04) | **-14.43** | 0.046 | 0.137 | 0.8426 | 23.76 |
| L4 | Early stopping pada EER | 2 | 85.39 (1.43) | **+9.28** | 0.023 | 0.090 | 0.9277 | 14.61 |
| L5 | Augmentasi penuh | 3 | 89.15 (3.49) | **+3.77** | 0.201 | 0.401 | 0.9540 | 10.81 |

Total kenaikan sepanjang tangga adalah -4.41 poin persentase, dari 93.57 persen menjadi 89.15 persen.

## Rincian tiap langkah

**L1, Konfigurasi proposal apa adanya.** Titik tolak, yaitu LR 0,001 seragam dengan encoder ikut dilatih, normalisasi peak, 20 epoch tanpa early stopping, augmentasi noise saja. Akurasi 93.57 persen dengan AUC 0.9824.

**L2, Normalisasi loudness.** Perubahan yang dilakukan adalah normalisasi peak diganti loudness, selebihnya sama. Langkah ini menurunkan akurasi sebesar 3.03 poin persentase menjadi 90.53 persen, dengan AUC 0.9683 dan EER 9.33 persen.

**L3, LR per model dan encoder dibekukan.** Perubahan yang dilakukan adalah encoder tidak lagi dilatih, head 0,001 dan encoder 2e-5, ditambah agregasi berbobot antar lapisan. Langkah ini menurunkan akurasi sebesar 14.43 poin persentase menjadi 76.10 persen, dengan AUC 0.8426 dan EER 23.76 persen.

**L4, Early stopping pada EER.** Perubahan yang dilakukan adalah 10 epoch dengan pemilihan bobot terbaik menurut EER validasi. Langkah ini menaikkan akurasi sebesar 9.28 poin persentase menjadi 85.39 persen, dengan AUC 0.9277 dan EER 14.61 persen.

**L5, Augmentasi penuh.** Perubahan yang dilakukan adalah augmentasi noise saja diganti augmentasi penuh, yaitu codec, noise, reverb, dan band-gain. Langkah ini menaikkan akurasi sebesar 3.77 poin persentase menjadi 89.15 persen, dengan AUC 0.9540 dan EER 10.81 persen.

## Bacaan

Empat selisih diuji sekaligus, sehingga koreksi Holm-Bonferroni diterapkan dan keputusan diambil dari kolom p Holm.

Langkah yang berada di garis batas dan belum dapat dinyatakan mapan: L3 (-14.43 poin, p Holm 0.137); L4 (+9.28 poin, p Holm 0.090).

Langkah yang selisihnya belum terbukti berbeda dari nol: L2 (-3.03 poin, p Holm 0.401); L5 (+3.77 poin, p Holm 0.401).

Pola yang muncul cukup jelas. Dua langkah dengan selisih terbesar, yaitu pembekuan encoder dan early stopping, memiliki nilai p mentah di bawah 0,05 sedangkan dua langkah dengan selisih kecil tidak. Setelah koreksi untuk empat pengujian sekaligus, tidak ada satu pun yang bertahan di bawah ambang. Perlu diingat bahwa seluruh tangga ini dijalankan pada AST, yaitu arsitektur dengan ragam antar inisialisasi terbesar di antara yang diuji, sehingga daya ujinya paling rendah di sini dan bukan karena efeknya tidak ada.

Kesimpulan yang dapat dipertanggungjawabkan dari tangga ini karena itu terbatas. Arah tiap langkah konsisten dengan penjelasan mekanistik yang diajukan, tetapi besarannya belum dapat dipisahkan dari ragam pada ukuran sampel ini. Tangga ablasi lebih tepat dibaca sebagai peta kemungkinan sebab, bukan sebagai pengukuran sumbangan tiap perbaikan.
