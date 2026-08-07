# Tangga Ablasi: Perbaikan Mana yang Membeli Berapa

Semua langkah memakai AST pada partisi resmi Fake-or-Real, batch 32, seed 42. Tiap baris menambahkan satu perbaikan di atas baris sebelumnya, sehingga selisih antar baris hanya mencerminkan satu variabel.

Akurasi dilaporkan pada ambang prior-matched untuk seluruh langkah agar sumbu ambang tidak bercampur ke dalam tangga. Sumbangan ambang itu sendiri dipisahkan tersendiri di HASIL_DEKOMPOSISI.md.

Selisih tiap langkah diuji terhadap langkah sebelumnya dengan uji t Welch. Seluruh tangga dijalankan pada AST, yaitu arsitektur dengan ragam antar inisialisasi terbesar di antara yang diuji, sehingga selisih yang kecil di sini menuntut kehati-hatian khusus.

| Langkah | Perbaikan yang ditambahkan | n | Akurasi | Selisih | p mentah | p Holm | AUC | EER |
|---|---|---|---|---|---|---|---|---|
| L1 | Konfigurasi proposal apa adanya | 3 | 93.57 (1.75) |  |  |  | 0.9824 | 6.50 |
| L2 | Normalisasi loudness | 3 | 90.50 (1.75) | **-3.06** | 0.099 | 0.197 | 0.9688 | 9.41 |
| L3 | LR per model dan encoder dibekukan | 3 | 74.26 (3.27) | **-16.24** | 0.004 | 0.018 | 0.8256 | 25.64 |
| L4 | Early stopping pada EER | 3 | 87.93 (4.52) | **+13.66** | 0.016 | 0.048 | 0.9448 | 12.10 |
| L5 | Augmentasi penuh | 3 | 89.15 (3.49) | **+1.23** | 0.730 | 0.730 | 0.9540 | 10.81 |

Total kenaikan sepanjang tangga adalah -4.41 poin persentase, dari 93.57 persen menjadi 89.15 persen.

## Rincian tiap langkah

**L1, Konfigurasi proposal apa adanya.** Titik tolak, yaitu LR 0,001 seragam dengan encoder ikut dilatih, normalisasi peak, 20 epoch tanpa early stopping, augmentasi noise saja. Akurasi 93.57 persen dengan AUC 0.9824.

**L2, Normalisasi loudness.** Perubahan yang dilakukan adalah normalisasi peak diganti loudness, selebihnya sama. Langkah ini menurunkan akurasi sebesar 3.06 poin persentase menjadi 90.50 persen, dengan AUC 0.9688 dan EER 9.41 persen.

**L3, LR per model dan encoder dibekukan.** Perubahan yang dilakukan adalah encoder tidak lagi dilatih, head 0,001 dan encoder 2e-5, ditambah agregasi berbobot antar lapisan. Langkah ini menurunkan akurasi sebesar 16.24 poin persentase menjadi 74.26 persen, dengan AUC 0.8256 dan EER 25.64 persen.

**L4, Early stopping pada EER.** Perubahan yang dilakukan adalah 10 epoch dengan pemilihan bobot terbaik menurut EER validasi. Langkah ini menaikkan akurasi sebesar 13.66 poin persentase menjadi 87.93 persen, dengan AUC 0.9448 dan EER 12.10 persen.

**L5, Augmentasi penuh.** Perubahan yang dilakukan adalah augmentasi noise saja diganti augmentasi penuh, yaitu codec, noise, reverb, dan band-gain. Langkah ini menaikkan akurasi sebesar 1.23 poin persentase menjadi 89.15 persen, dengan AUC 0.9540 dan EER 10.81 persen.

## Bacaan

Empat selisih diuji sekaligus, sehingga koreksi Holm-Bonferroni diterapkan dan keputusan diambil dari kolom p Holm.

Langkah yang selisihnya melampaui ragam antar inisialisasi: L3 (-16.24 poin, p Holm 0.018); L4 (+13.66 poin, p Holm 0.048).

Langkah yang selisihnya belum terbukti berbeda dari nol: L2 (-3.06 poin, p Holm 0.197); L5 (+1.23 poin, p Holm 0.730).

Pola yang muncul cukup jelas. Dua langkah dengan selisih terbesar, yaitu pembekuan encoder dan early stopping, memiliki nilai p mentah di bawah 0,05 sedangkan dua langkah dengan selisih kecil tidak. Setelah koreksi untuk empat pengujian sekaligus, tidak ada satu pun yang bertahan di bawah ambang. Perlu diingat bahwa seluruh tangga ini dijalankan pada AST, yaitu arsitektur dengan ragam antar inisialisasi terbesar di antara yang diuji, sehingga daya ujinya paling rendah di sini dan bukan karena efeknya tidak ada.

Kesimpulan yang dapat dipertanggungjawabkan dari tangga ini karena itu terbatas. Arah tiap langkah konsisten dengan penjelasan mekanistik yang diajukan, tetapi besarannya belum dapat dipisahkan dari ragam pada ukuran sampel ini. Tangga ablasi lebih tepat dibaca sebagai peta kemungkinan sebab, bukan sebagai pengukuran sumbangan tiap perbaikan.
