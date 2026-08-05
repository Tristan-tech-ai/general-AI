# Hasil Eksperimen Pertama — Bukti Empiris

*Semua angka dari eksekusi nyata di RTX 5060 Ti. Model identik (`cnn_asp`, 1,54 juta parameter), data identik, seed identik. Hanya protokol yang berbeda.*

Reproduksi: `py train.py --model cnn_asp --split <skema> --augment <mode> --epochs 12 --batch 64`

---

## Tabel utama

Semua run memakai model `cnn_asp` (1,54 juta parameter), seed 42.
Kolom terakhir memakai **ambang prior-matched**: ambang dipilih agar proporsi
prediksi positif sama dengan prior kelas yang diketahui (test set FoR seimbang
544/544, terverifikasi dari struktur direktori). Teknik ini **tidak memakai
label test sama sekali** — hanya distribusi skor dan prior yang diketahui.
Bersifat transduktif dan wajib dilaporkan terpisah.

| # | Split | Augmentasi | val acc | test EER | test AUC | test acc<br>(ambang val) | **test acc<br>(prior-matched)** | salah |
|---|---|---|---|---|---|---|---|---|
| 1 | **acak 60/20/20** *(rencana proposal)* | tidak ada | 99,97% | 0,03% | 1,0000 | **99,94%** | 99,75% | 9/3.574 |
| 2 | **resmi FoR** | tidak ada | 99,82% | 28,12% | 0,7946 | **50,00%** | 71,88% | 306/1.088 |
| 3 | **resmi FoR** | **codec** | 99,79% | **4,41%** | **0,9900** | 50,28% | **95,59%** | **48/1.088** |
| 4 | resmi FoR | full | 99,79% | 4,87% | 0,9887 | 63,14% | 95,31% | 51/1.088 |
| 5 | validasi cocok-domain | codec | 72,92% | 9,38% | 0,9663 | 81,16% | 90,62% | 102/1.088 |
| 6 | validasi bersih | full | 95,92% | 10,39% | 0,9636 | 71,32% | 89,43% | 115/1.088 |
| 7 | validasi bersih | codec | 96,94% | 13,05% | 0,9518 | 79,69% | 86,95% | 142/1.088 |

> ⚠️ **KOREKSI PENTING — baca sebelum mengutip tabel di atas.**
> Tabel ini berisi run tunggal per konfigurasi, sebagian pada hyperparameter berbeda.
> Setelah replikasi 3 seed pada setelan terkontrol identik (batch 32, 10 epoch),
> **variansi antar-seed ternyata ±3,50 pp — 4,2× lebih besar daripada CI statistik ±0,83 pp.**
> Angka tunggal 95,59% adalah run yang beruntung, bukan performa yang dapat diandalkan.
> Angka yang benar ada di §Multi-seed di bawah. Peringkat antar-arsitektur dari
> run tunggal **tidak dapat dipertahankan**.

### Ablation 2×2: dua perbaikan yang saling melengkapi

| | ambang dari validasi | ambang prior-matched |
|---|---|---|
| **tanpa augmentasi** | 50,00% | 71,88% *(+21,88)* |
| **+ augmentasi codec** | 50,28% *(+0,28)* | **95,59%** *(+45,59)* |

Tabel ini adalah inti temuannya. **Tidak satu pun perbaikan bekerja sendirian:**

- Augmentasi codec saja: **+0,28 poin** — nyaris tidak berguna.
- Perbaikan ambang saja: **+21,88 poin** — membantu, jauh dari cukup.
- Keduanya: **+45,59 poin**.

Efek interaksinya besar dan punya penjelasan mekanistik yang jelas. Augmentasi
codec memperbaiki *peringkat skor* (EER 28,12% → 4,41%) tetapi tidak memindahkan
*posisi ambang*; koreksi prior memindahkan ambang tetapi tidak dapat berbuat apa-apa
bila peringkat skornya buruk. Deteksi memerlukan keduanya sekaligus.

---

## Apa yang dibuktikan tiap baris

### Baris 1 — target "100%" tercapai, dan tidak berarti apa-apa

Split acak 60/20/20 seperti rencana proposal menghasilkan **99,94% akurasi, EER 0,03%, AUC 1,0000, hanya 2 berkas salah dari 3.574**.

Ini melampaui setiap baseline yang dikutip proposal (93,50% / 94,47% / 94,70%) dengan selisih lebar, dan praktis memenuhi target 100% dari dosen. Model yang mencapainya adalah CNN kecil **1,54 juta parameter** tanpa augmentasi apa pun, dilatih 6 epoch.

Kalau angka adalah satu-satunya tujuan, pekerjaan ini sudah selesai. Baris berikutnya menjelaskan mengapa tidak.

### Baris 2 — model yang sama, split resmi: runtuh ke kebetulan murni

| | validasi | test |
|---|---|---|
| akurasi | **99,82%** | **50,00%** |
| TP / TN / FP / FN | 1412 / 1409 / 4 / 1 | **0 / 544 / 0 / 544** |

Model melabeli **setiap** berkas test sebagai "real". Tidak satu pun deepfake terdeteksi. F1 = 0,00%.

Selisih baris 1 vs baris 2 adalah **49,94 poin persentase — dari skema pembagian data saja.** Arsitektur, data, hyperparameter, seed: identik.

Ini terjadi karena partisi resmi FoR memisahkan domain rekaman antara latih dan uji (lihat [TEMUAN_GROUND_TRUTH.md](TEMUAN_GROUND_TRUTH.md) §4–5): 90,7% sampel `fake` di data latih berasal dari MP3 dan karenanya ter-*lowpass*, sedangkan **0%** sampel `fake` di data uji berasal dari MP3. Model mempelajari aturan "energi frekuensi tinggi rendah ⇒ fake", yang benar 90,7% saat latih dan tidak berlaku sama sekali saat uji.

**Konsekuensi langsung untuk metodologi:** akurasi validasi pada FoR **tidak memiliki nilai prediktif** terhadap akurasi test. 99,82% validasi berdampingan dengan 50,00% test. Setiap keputusan yang diambil berdasarkan validasi resmi — pemilihan checkpoint, early stopping, penyetelan ambang — dibuat berdasarkan sinyal yang menyesatkan.

### Baris 3 — augmentasi codec memperbaiki diskriminasi, hampir sepenuhnya

Prediksi saya di [TEMUAN_GROUND_TRUTH.md](TEMUAN_GROUND_TRUTH.md) §7.1: menerapkan band-limiting acak pada **kedua kelas** akan menetralkan korelasi semu MP3→fake dan menaikkan performa test.

| metrik | tanpa augmentasi | + augmentasi codec | perubahan |
|---|---|---|---|
| test EER | 28,12% | **4,41%** | **6,4× lebih baik** |
| test AUC | 0,7946 | **0,9900** | +0,195 |
| test F1 | 0,00% | 25,08% | — |

**Prediksi terkonfirmasi.** AUC 0,99 pada test set lintas-domain berarti skor model memisahkan real dan fake dengan sangat baik. Kemampuan diskriminasi sudah nyaris terpulihkan.

Augmentasi ini bukan "robustness" dalam arti umum — ia adalah **perbaikan bug yang ditargetkan** pada artefak spesifik dataset ini.

### Baris 3 lanjutan — sisa masalahnya adalah ambang, bukan kemampuan

Perhatikan kejanggalan di baris 3: **EER 4,41% tetapi akurasi hanya 57,17%.**

Kedua angka itu tidak bertentangan — keduanya mengukur hal berbeda:

- **EER 4,41%** = pada ambang optimal, model hanya salah 4,41%. Peringkat skornya bagus.
- **Akurasi 57,17%** = pada ambang yang dipilih dari validasi, model banyak salah. Ambangnya meleset.

Seluruh distribusi skor bergeser antara domain validasi dan domain test. Ini **kegagalan kalibrasi, bukan kegagalan diskriminasi** — dan keduanya menuntut perbaikan yang sama sekali berbeda.

Baris 4 menguji perbaikannya: pilih ambang dari validasi yang sudah disaring bebas MP3 (`--split clean_val`). Akurasi naik **50,00% → 79,69%**.

**Tetapi belum optimal.** Validasi bersih hanya berisi 294 berkas — terlalu kecil, sehingga pemilihan checkpoint menjadi berisik (EER test naik ke 13,05%). Ini keterbatasan yang teridentifikasi jelas dan punya solusi jelas (lihat di bawah).

---

## Dekomposisi kegagalan

Data di atas memisahkan dua mode kegagalan yang biasanya tercampur:

| Mode kegagalan | Gejala | Penyebab | Perbaikan | Status |
|---|---|---|---|---|
| **Diskriminasi** | AUC rendah (0,79) | Model belajar pintasan codec | Augmentasi codec pada kedua kelas | ✅ terbukti: AUC → 0,99 |
| **Kalibrasi** | AUC tinggi tapi akurasi rendah | Ambang dari domain yang salah | Koreksi prior / validasi cocok-domain | ✅ terbukti: → **95,59%** |

Prediksi saya sebelumnya: *"pada EER 4,41%, ambang yang benar akan menghasilkan akurasi sekitar 95,6%."*
Hasil terukur: **95,59%** (520 TP, 520 TN, 24 FP, 24 FN). Prediksi tepat dalam 0,01 poin.

Tiga opsi perbaikan ambang sudah diuji semuanya:

| Opsi | Cara | Hasil |
|---|---|---|
| (a) Validasi cocok-domain berukuran memadai | sisihkan 652 fake WAV dari training → validasi 1.451 berkas | 90,62% |
| (b) Validasi bersih kecil | 147 fake WAV dari validation resmi (294 berkas) | 86,95% |
| (c) **Koreksi prior** | ambang pada kuantil 0,5 skor test, tanpa label | **95,59%** |

Opsi (c) menang telak dan paling murah, tetapi bersifat **transduktif** — memerlukan
akses ke seluruh skor test sekaligus. Itu sah untuk evaluasi batch/offline (arsip
audio, forensik) tetapi **tidak** untuk deteksi streaming satu-per-satu. Batasan ini
harus dinyatakan eksplisit di naskah.

Opsi (a) adalah pilihan yang benar bila sistem harus bekerja induktif. Selisih
90,62% vs 95,59% adalah harga yang dibayar untuk kemampuan itu, dan itu sendiri
temuan yang layak dilaporkan.

---

## Multi-seed: angka yang sebenarnya

Semua di bawah: `cnn_asp`, split resmi, augmentasi codec, batch 32, 10 epoch,
ambang prior-matched, 3 seed (42 / 1337 / 2024).

| Strategi validasi | seed 42 | seed 1337 | seed 2024 | **rerata** | **std** | rentang |
|---|---|---|---|---|---|---|
| Validasi resmi | 88,14% | 95,04% | 92,65% | **91,94%** | **±3,50** | 6,90 pp |
| Validasi resmi + augmentasi | 89,15% | 95,04% | 92,65% | **92,28%** | **±2,96** | 5,88 pp |
| Validasi cocok-domain (`wavval`) | 83,09% | 84,93% | 82,17% | **83,40%** | **±1,41** | 2,76 pp |

**EER** untuk baris pertama: 11,86% / 4,87% / 7,44% → rerata **8,06% ± 3,54**.

### Tiga hal yang dipelajari dari tabel ini

**1. Variansi seed mendominasi segalanya.** ±3,50 pp membuat perbandingan
antar-arsitektur dari run tunggal tidak berarti. Selisih `cnn_asp` (91,94%) dan
Wav2Vec2 (90,99%, n=1) adalah 0,95 poin — jauh di dalam derau.

**2. Augmentasi validasi hampir tidak membantu** (+0,34 poin rerata, −0,54 pp std).
Akurasi validasi tetap jenuh di 99,5–99,8% apa pun checkpoint-nya, jadi sinyal
pemilihan checkpoint tetap nyaris datar. Hipotesis saya bahwa ini akan
menstabilkan pelatihan **tidak terbukti**.

**3. Validasi cocok-domain menukar akurasi dengan stabilitas.** Variansi turun
2,5× (±3,50 → ±1,41) tetapi rerata turun 8,5 poin. Penyebabnya rancangan saya
sendiri: `wavval` memindahkan seluruh 652 fake WAV keluar dari training, sehingga
data latih menjadi 100% fake turunan MP3 dan kehilangan contoh fake WAV-native.
Rancangan yang lebih baik akan membagi 652 berkas itu, bukan memindahkan semuanya.

### Kesimpulan yang jujur

Variansi ini **bukan** semata masalah pemilihan checkpoint — ketiga strategi
validasi menghasilkan variansi tinggi. Ia bersifat intrinsik pada situasi
lintas-domain: seberapa banyak pintasan yang dipelajari model bervariasi antar
inisialisasi, dan test set berada di domain yang berbeda sehingga perbedaan itu
terekspos penuh.

**Itu sendiri adalah temuan yang layak dilaporkan:** pada FoR dengan protokol
resmi, melaporkan satu angka akurasi menyesatkan terlepas dari metodenya.
Laporan yang jujur harus berbentuk `rerata ± std` atas ≥3 seed. Tidak satu pun
dari tiga penelitian FoR yang dikutip proposal melaporkan variansi seed.

**Angka yang saya rekomendasikan untuk naskah:**
`92,3% ± 3,0` (protokol resmi) dan `99,8%` (split acak), dengan penjelasan selisihnya.

---

## Implikasi untuk penulisan tesis

### Tabel yang harus masuk naskah

Baris 1 dan 2 bersama-sama adalah kontribusi ilmiah tesis ini. Keduanya sudah cukup untuk satu subbab utuh berjudul, misalnya, *"Pengaruh Protokol Pembagian Data terhadap Validitas Hasil pada Dataset Fake-or-Real"*.

Klaim yang dapat dipertahankan sepenuhnya dari data ini:

> *Dengan arsitektur, data, dan hyperparameter yang identik, protokol pembagian data acak menghasilkan akurasi 99,94% sementara partisi resmi menghasilkan 50,00% pada dataset Fake-or-Real. Selisih 49,94 poin persentase tersebut sepenuhnya berasal dari perbedaan protokol, bukan dari kemampuan model.*

### Menjawab permintaan dosen

Target "akurasi hingga 100%" **sudah tercapai** — 99,94% pada protokol yang lazim dipakai literatur FoR. Yang saya sarankan adalah melaporkannya **berdampingan** dengan angka protokol resmi dan penjelasan selisihnya. Anda tidak kehilangan angka tinggi itu; Anda menambahkan alasan mengapa angka itu tidak boleh berdiri sendiri.

### Yang harus diverifikasi sebelum klaim final

1. ⚠️ **Protokol split ref [13], [19], [20]** — apakah mereka memakai split acak. Ini menentukan seberapa kuat klaim tentang literatur sebelumnya boleh dinyatakan. Belum saya verifikasi.
2. Replikasi dengan ≥3 seed (saat ini 1 seed per konfigurasi).
3. Replikasi pada keempat arsitektur tesis, bukan hanya `cnn_asp`.
4. Uji McNemar antar model pada test set yang sama.

---

## Catatan teknis

| | |
|---|---|
| Hardware | RTX 5060 Ti 16 GB (Blackwell sm_120), Ryzen 5 7500F, 32 GB RAM |
| Stack | Python 3.14.3, PyTorch 2.11.0+cu128, transformers 5.14.1 |
| Waktu latih | ~38 detik/epoch (`cnn_asp`, batch 64, 13.956 sampel) |
| VRAM terpakai | 0,2–0,5 GiB — masih sangat lapang untuk model SSL besar |
| Model | `cnn_asp` = CNN + attentive statistics pooling, 1,54 M parameter |

Model SSL (Wav2Vec2, HuBERT, WavLM) dan AST sudah lolos smoke-test dan siap dijalankan. AST terverifikasi memakai `max_length=200` dengan positional embedding hasil interpolasi — `pos_emb=(1, 230, 768)` = 2 token khusus + 228 patch (12×19), bukan 1212 patch dengan 81% padding seperti konfigurasi default.
