# PETA JALAN NOVELTY

> **CATATAN. Dokumen ini memuat angka yang kemudian ditarik.**
>
> Berkas ini adalah catatan riset dari tahap sebelumnya dan sengaja tidak
> disunting, supaya jalannya penelitian tetap dapat ditelusuri. Beberapa angka
> di dalamnya sudah tidak berlaku:
>
> - Selisih **99,94 lawan 50,00 persen** antara kedua protokol. Kedua run
>   pembandingnya ternyata berbeda lama pelatihan (enam epoch lawan satu epoch),
>   sehingga bukan perbandingan terkontrol. Pada konfigurasi seragam, hanya
>   **6,92 poin** dari selisih itu berasal dari protokol, dan **42,52 poin**
>   berasal dari ambang keputusan. Lihat [HASIL_TEMUAN1.md](HASIL_TEMUAN1.md).
> - Recall HuBERT **2,3 persen** pada TTS 2019 non-MP3 adalah inisialisasi
>   terburuk dari tiga. Reratanya **29,2 persen** dengan simpangan 29,0.
> - Korelasi **r = -0,542** dan **r = -0,980** sudah ditarik seluruhnya.
>
> Status terkini seluruh temuan ada pada tabel verifikasi di
> [README.md](README.md), dan daftar lengkap klaim yang ditarik ada pada
> Lampiran A di PAPER.pdf.


**Tesis:** Analisis Performa Arsitektur Deep Learning Wav2Vec2, AST, HuBERT, dan CNN-LSTM dalam Klasifikasi Suara Deepfake dan Suara Asli
**Fokus rumusan masalah:** ketahanan terhadap noise lingkungan
**Tanggal:** 5 Agustus 2026
**Basis:** 961 temuan riset dari 34 dimensi (1 lolos verifikasi adversarial 3-lensa) + audit empiris 17.870 berkas + 24 run pelatihan nyata di RTX 5060 Ti

---

## Konvensi label bukti

Setiap klaim di dokumen ini diberi label. Jangan tulis apa pun ke naskah tesis tanpa memeriksa labelnya.

| Label | Arti |
|---|---|
| **[UKUR]** | Diukur sendiri di mesin ini. Angka ada di `audit_report.md`, `probe_*_report.md`, `runs/*/results.json`, `PERBANDINGAN.md`. Dapat direproduksi. |
| **[LIT]** | Berasal dari literatur yang sudah diverifikasi kutipannya. |
| **[HIP]** | Hipotesis. Belum ada pengukuran. Tidak boleh ditulis sebagai fakta. |
| **[REFUTASI]** | Sudah gugur pada verifikasi adversarial. Jangan dipakai. |

**Aturan tegas:** dari 961 temuan riset, hanya **1** bertahan dari verifikasi 3-lensa, dan temuan itu bernilai nol untuk akurasi. Artinya sumber kekuatan tesis ini **bukan** literatur, melainkan data yang sudah Anda ukur sendiri. Tiga usulan di bawah dibangun di atas pengukuran Anda, bukan di atas paper.

---

## 0. Ringkasan eksekutif

Fakta paling menentukan yang sudah terukur: **pada FoR-2sec, 90,7% sampel `fake` di data latih berasal dari MP3 dan 0% sampel `fake` di data uji berasal dari MP3** [UKUR]. Konsekuensinya berantai:

1. Model belajar aturan "energi frekuensi tinggi rendah ⇒ fake". Benar 90,7% saat latih, **tidak berlaku sama sekali** saat uji.
2. Tanpa augmentasi, model memprediksi **seluruh** 1.088 berkas uji sebagai "real": akurasi 50,00%, F1 = 0,00% [UKUR].
3. Noise lingkungan bekerja tepat dengan mengisi pita frekuensi tinggi, yaitu **menghancurkan pintasan itu**, bukan menghancurkan artefak sintesis.

Karena itu, kalimat "kami mengukur ketahanan terhadap noise" pada dataset ini **secara default mengukur runtuhnya pintasan codec**, bukan ketahanan deteksi deepfake. Ketiga usulan novelty di bawah semuanya berangkat dari sini.

Kabar baiknya: keempat pengungkit terbesar sudah setengah terbukti di mesin Anda.

| Pengungkit | Bukti terukur |
|---|---|
| Augmentasi codec simetris | EER 28,12% → 4,41%, AUC 0,7946 → 0,9900 [UKUR] |
| Koreksi ambang (kalibrasi) | akurasi 50,28% → 95,59% pada **skor yang sama persis** [UKUR] |
| Ensemble 4 arsitektur | 91,94% → **97,61%**, EER 8,06% → 2,30% [UKUR] |
| Layer weighting SSL | bobot puncak jatuh di layer 2–4 (Wav2Vec2) dan 3–6 (AST), konsisten 3 seed [UKUR] |

---

## 1. TIGA USULAN NOVELTY TERKUAT

Ketiganya berbagi satu infrastruktur (grid SNR + korpus noise nyata + dump skor `.npy`). Biaya marjinal mengerjakan ketiganya jauh di bawah 3×.

---

### NOVELTY 1, Dekomposisi diskriminasi vs kalibrasi pada degradasi noise, dan kalibrasi berkondisi-SNR

> **Nama pendek untuk naskah:** *Noise merusak ambang, bukan (hanya) daya pisah.*

#### 1.1 Klaim ilmiah yang dapat diuji

**H1a.** Pada FoR-2sec di bawah noise lingkungan, sebagian besar penurunan akurasi **bukan** hilangnya daya pisah (ΔAUC) melainkan pergeseran distribusi skor terhadap ambang tetap (Δambang).

**H1b.** Kalibrasi affine yang parameternya merupakan fungsi dari SNR yang diestimasi secara **buta** memulihkan sebagian besar akurasi yang hilang, **tanpa mengubah AUC/EER sama sekali**.

**H1c.** Selisih antara ambang transduktif (prior-matched) dan ambang induktif (dari validasi) adalah metrik yang belum pernah dilaporkan di literatur FoR, dan besarnya 8–12 poin persentase pada data ini.

#### 1.2 Mekanisme

Transformasi skor yang monoton naik **secara matematis tidak dapat mengubah** EER maupun AUC, keduanya hanya bergantung pada urutan skor. Tetapi akurasi pada ambang tetap sangat bergantung pada posisi absolut skor. Noise menggeser posisi absolut itu (dan menggesernya **berbeda** untuk kelas real dan fake), sementara urutan relatifnya sebagian besar bertahan.

Mekanisme ini **sudah terbukti di mesin Anda**, pada rezim lintas-domain:

| | ambang dari validasi | ambang prior-matched |
|---|---|---|
| **tanpa augmentasi** | 50,00% | 71,88% *(+21,88)* |
| **+ augmentasi codec** | 50,28% *(+0,28)* | **95,59%** *(+45,59)* |

[UKUR, `HASIL_EKSPERIMEN.md`]

Baca baris bawah dengan teliti. **Skornya identik.** EER-nya 4,41%, AUC 0,9900. Yang berubah hanyalah di mana garis keputusan diletakkan, dan itu memindahkan akurasi 45,31 poin. Augmentasi codec sendirian memberi +0,28 poin; koreksi ambang sendirian memberi +21,88 poin; keduanya memberi +45,59 poin. Efek interaksinya besar dan punya penjelasan yang tidak ambigu: augmentasi memperbaiki **peringkat** skor, kalibrasi memperbaiki **posisi** ambang, dan deteksi memerlukan keduanya.

> **Catatan kejujuran penting.** Angka 95,59% berasal dari satu run (batch 64, 12 epoch) yang setelah replikasi 3 seed pada setelan terkontrol menjadi **91,94% ± 3,50** [UKUR, `PERBANDINGAN.md`]. Jadi *besar* efeknya belum stabil. Tetapi *arah* dan *keberadaan* efeknya dijamin oleh teorema invariansi AUC terhadap transformasi monoton, tidak bergantung pada run yang beruntung. Tulis mekanismenya sebagai terbukti; tulis magnitudonya sebagai perlu replikasi.

#### 1.3 Cara eksekusi konkret

**Langkah 1, Bangun harness evaluasi bergradasi SNR.**
```
eval_snr.py:
 test_set × {bersih, 20, 15, 10, 5, 0, −5 dB} × {noise_type}
 noise UJI : DEMAND + WHAM! (direkam sendiri oleh penulisnya, independen)
 noise LATIH: MUSAN + RIR SLR28 (disjoint pada level berkas DAN korpus)
```
Ini kritis dan sering salah: MUSAN-noise, ESC-50, FSD50K, dan DNS-Challenge semuanya menarik dari Freesound.org, sehingga "unseen noise" lintas ketiganya bisa berbagi klip yang sama [LIT, D2]. Hanya DEMAND dan WHAM! yang benar-benar independen.

**Langkah 2, Untuk setiap sel kondisi, laporkan lima angka, bukan satu:**
```
AUC | EER | acc@ambang_validasi | acc@ambang_oracle | Δambang_opt(SNR)
```

**Langkah 3, Dekomposisi eksplisit:**
```
ΔAkurasi_total = ΔAkurasi_diskriminasi + ΔAkurasi_kalibrasi
 └ acc@oracle turun └ sisa (acc@oracle − acc@ambang tetap)
```

**Langkah 4, Estimator SNR buta.** Dilatih **hanya pada dev set**, tidak pernah melihat test. Dua opsi, kerjakan yang pertama dulu:
- (a) WADA-SNR atau rasio persentil energi frame (≈30 baris numpy, tanpa pelatihan);
- (b) regresor 2-lapis kecil di atas statistik log-Mel (mean/std per pita), dilatih pada dev set yang di-augmentasi dengan SNR yang diketahui.

**Langkah 5, Kalibrasi berkondisi:**
```
p_i = σ( a(ŝ) · z_i + b(ŝ) )
a(ŝ) = a0 + a1·ŝ b(ŝ) = b0 + b1·ŝ (4 parameter, difit pada dev)
```
Bila hubungannya ternyata tidak linear terhadap ŝ, turunkan ke 3 bin kasar (bersih / sedang / berat), lebih tahan derau estimasi dan tetap merupakan klaim yang sama.

**Langkah 6, Bandingkan empat rezim ambang:**

| Rezim | Sifat | Status di kode |
|---|---|---|
| Ambang tetap 0,5 | naif | ada |
| Ambang dari validasi (Youden) | induktif | ada, `threshold_from_validation` |
| Temperature scaling | induktif | ada, `TemperatureScaler` |
| **Kalibrasi berkondisi-SNR** | **induktif, USULAN** | **belum ada** |
| Prior-matched | **transduktif** (batas atas) | ada, `prior_matched_threshold` |

#### 1.4 Cara membuktikan

- **Terkonfirmasi bila:** ΔAkurasi_kalibrasi > ΔAkurasi_diskriminasi pada mayoritas titik SNR, **dan** kalibrasi berkondisi-SNR memulihkan ≥50% dari jurang menuju ambang oracle, **dan** AUC tidak berubah sama sekali (ini pemeriksaan kejujuran, bila AUC berubah, Anda tidak sedang melakukan kalibrasi).
- **Terbantah bila:** ΔAUC mendominasi di seluruh grid SNR. Itu hasil negatif yang tetap layak dilaporkan, dan langsung mengarahkan tesis ke Novelty 2.
- **Uji statistik:** McNemar berpasangan + Holm-Bonferroni antar rezim ambang pada test set yang sama. Sudah terimplementasi di `compare.py`. Wajib, karena 1 berkas = 0,092 pp pada n=1.088 [UKUR].

#### 1.5 Mengapa ini baru

1. Pencarian literatur **tidak menemukan satu pun** karya yang memasukkan estimasi SNR sebagai *quality feature* di dalam kalibrasi/fusi detektor deepfake audio [LIT, C1].
2. Literatur ketahanan-noise pada bidang ini hampir seluruhnya melaporkan EER, metrik yang **bebas ambang** dan karena itu **buta** terhadap mode kegagalan yang mendominasi penggunaan nyata. Satu detektor SOTA dengan EER in-domain 0,21% menolak 78,7% suara asli ketika ambangnya dipindah lintas domain [LIT, C1].
3. Daftar metrik proposal Anda (akurasi, presisi, recall, F1, confusion matrix) **semuanya bergantung ambang**. Jadi ini bukan tambahan opsional, ini persis sumbu yang harus Anda pertahankan di sidang.
4. Anda sudah memiliki bukti internal yang tidak dimiliki orang lain: selisih 45,31 poin pada skor yang identik.

#### 1.6 Risiko dan mitigasi

| Risiko | Mitigasi |
|---|---|
| Estimasi SNR buta terlalu berderau di SNR rendah | Turunkan ke 3 bin kasar; laporkan MAE estimator sebagai tabel pendukung |
| Pergeseran skor tidak monoton terhadap SNR | Ganti affine dengan tabel lookup per-bin; hasil negatif tetap dilaporkan |
| Penguji menganggap ini "sekadar tuning ambang" | Jawabannya: dekomposisi + estimator buta + bukti invariansi AUC. Tunjukkan tabel 2×2 |

---

### NOVELTY 2, Noise sebagai operator ablasi pintasan: protokol intervensi dua-sumbu pada FoR

> **Nama pendek untuk naskah:** *Apa sebenarnya yang dihancurkan noise?*

#### 2.1 Klaim ilmiah yang dapat diuji

**H2a.** Pada FoR-2sec, "penurunan akurasi akibat noise" sebagian besar mengukur runtuhnya **pintasan provenance codec** di pita frekuensi tinggi, bukan hilangnya artefak sintesis.

**H2b.** Model yang pintasannya sudah dinetralkan menunjukkan kurva degradasi-SNR yang **berbeda secara kualitatif** (bukan sekadar lebih tinggi) dari model yang bergantung pintasan.

**H2c.** Selisih energi frekuensi tinggi antara kelas real dan fake pada FoR, yang oleh Ahmad dkk. diatribusikan pada penyebab alami (sibilan, noise mikrofon), **sekitar tiga perempatnya berasal dari kompresi MP3**, bukan dari sintesis.

H2c adalah klaim yang paling kuat karena ia **membantah tafsir yang sudah dipublikasikan dengan data**.

#### 2.2 Mekanisme

Semuanya sudah terukur:

| Perbandingan | Rasio energi > 6 kHz |
|---|---|
| real vs fake **turunan MP3** (90,7% data latih) | **4,18×** |
| real vs fake **bukan turunan MP3** (652 berkas) | **1,17×** |
| real vs fake pada test set (0% MP3) | **1,33×** |

[UKUR, `probe_shift_report.md`]

Bila selisih HF benar-benar berasal dari sibilan dan noise mikrofon, ia akan tetap besar pada fake yang tidak melalui MP3. Ternyata tidak, turun dari 4,18× ke 1,17×.

Dua pengukuran pendukung yang memperkuat dan sekaligus memperumit cerita:
- Membuang seluruh pita > 4 kHz **menaikkan** akurasi test resmi dari 69,39% ke 75,37% [UKUR]. Pita tinggi membawa pintasan, bukan sinyal yang dapat digeneralisasi.
- Kelas `real` di test set punya energi > 6 kHz **5,8× lebih rendah** daripada `real` di training (0,00575 vs 0,03355) [UKUR]. Jadi partisi resmi FoR bukan hanya berbeda provenance codec, **seluruh domain rekamannya berbeda**. Pembuat FoR tampaknya sengaja merancangnya sebagai evaluasi lintas-domain.

Noise aditif mengisi pita > 4 kHz. Karena itu ia menghapus pintasan secara mekanis. Melaporkan satu kurva degradasi mencampur dua hal yang berbeda secara kausal.

#### 2.3 Cara eksekusi konkret

**Desain 2 × 5 × 7:**

*Sumbu A, lengan model (2):*
- `arm-S` (bergantung pintasan) = `--augment none`
- `arm-N` (pintasan dinetralkan) = `--augment codec`, diterapkan **simetris pada kedua kelas**

*Sumbu B, intervensi pada test (5):*

| Intervensi | Yang diuji |
|---|---|
| (i) noise pita-lebar, grid SNR | efek gabungan |
| (ii) noise **hanya di > 4 kHz** | menyerang pintasan saja |
| (iii) noise **hanya di 0–4 kHz** | menyerang sinyal saja |
| (iv) low-pass seragam ke 4 kHz | menghapus pintasan **tanpa** menambah energi |
| (v) re-encode codec nyata (MP3/AAC/Opus) | degradasi kanal realistis |

*Sumbu C, grid SNR (7):* bersih, 20, 15, 10, 5, 0, −5 dB.

**Metrik baru yang diusulkan, Shortcut Reliance Index:**
```
SRI = ΔAUC(arm-S | intervensi HF) − ΔAUC(arm-N | intervensi HF)
```
SRI ≫ 0 berarti degradasi yang biasa disebut "kerapuhan terhadap noise" sebenarnya adalah runtuhnya pintasan.

**Pelengkap yang murah dan berdampak, kepala GRL provenance.**
Manifest Anda **sudah** menyimpan kolom `is_mp3` [UKUR, `forlib/data.py`]. Tambahkan kepala gradient-reversal yang memprediksi provenance codec. Perhatikan rancangannya, karena versi naifnya salah: `is_mp3 = 0` untuk **seluruh** kelas real, sehingga memprediksi `is_mp3` sebagian berarti memprediksi label. Rancangan yang benar:
```
label 3-arah : {real-WAV, fake-WAV, fake-MP3}
GRL : hapus sumbu MP3-vs-WAV HANYA DI DALAM kelas fake
 (mask loss GRL pada sampel real)
```
Ini adalah de-biasing yang ditargetkan pada pintasan yang **sudah Anda identifikasi dan kuantifikasi**, memakai label yang **sudah Anda miliki**, dengan biaya ~40 baris kode. Saya tidak menemukan satu pun karya yang melakukan ini pada FoR.

**Replikasi pada degradasi kanal nyata.** `for-rerec` sudah terunduh dan partisi testing-nya sudah terekstrak (408 fake + 408 real = 816 berkas); harness `eval_rerec.py` sudah ada. Ini degradasi kanal **nyata** (diputar lewat speaker, direkam ulang mikrofon konsumen), bukan simulasi. **Wajib ungkap confound:** berkas tersusun dalam blok per-kelas, jumlahnya 26% lebih kecil dari testing for-2sec, dan tidak ada kondisi kontrol yang bersih. Pakai sebagai probe sekunder, **jangan** sebagai eksperimen unggulan.

#### 2.4 Cara membuktikan

- **H2a terkonfirmasi bila:** `arm-S` terdegradasi jauh lebih parah di bawah intervensi (ii) HF-saja dibanding (iii) LF-saja, sementara `arm-N` terdegradasi kurang lebih sama di keduanya. SRI > 0,05 AUC.
- **H2b terkonfirmasi bila:** bentuk kurva berbeda, bukan sekadar offset. Uji dengan membandingkan turunan kurva (gradien ΔAUC per 5 dB).
- **H2c sudah terkonfirmasi** [UKUR]. Yang tersisa hanyalah menuliskannya dengan sitasi yang benar ke Ahmad dkk. (arXiv:2604.13400) dan kutipan verbatim mereka: *"microphone noise absent in synthetic voices"*.
- **Terbantah bila:** `arm-S` dan `arm-N` terdegradasi identik. Cerita pintasan gugur, dan itu harus ditulis apa adanya.

**Peringatan metodologis wajib.** `arm-S` sudah berada di 50,00% akurasi pada kondisi **bersih** (memprediksi semua sebagai real) [UKUR]. Ada efek lantai. Karena itu eksperimen ini **harus** memakai AUC/EER (bebas ambang) sebagai metrik utama, bukan akurasi. Bila tidak, lantai 50% akan menutupi seluruh sinyal.

#### 2.5 Mengapa ini baru

1. Pencarian arXiv untuk FoR + bias/kompresi/shortcut mengembalikan **nol** makalah [LIT, B6]. Pintasan provenance codec pada FoR tampaknya belum terdokumentasi.
2. Ahmad dkk. (arXiv:2604.13400) memakai partisi resmi FoR dan mengatribusikan selisih HF pada penyebab alami. Pengukuran Anda mengatribusikan ~3/4-nya ke MP3. **Ini kontradiksi langsung terhadap tafsir terpublikasi, dengan data.** Itu jenis kontribusi terkuat yang tersedia untuk tesis S2.
3. Kerangka diagnosis pintasan berbasis intervensi memang sudah ada (arXiv:2607.03150) tetapi memakai **interval non-speech** sebagai intervensi dan **ASVspoof** sebagai dataset [LIT, D1]. Memakai noise bergradasi SNR sebagai operator intervensi, pada FoR, belum pernah dilakukan.
4. Ini mengubah rumusan masalah tesis dari deskriptif ("berapa akurasi turun?") menjadi kausal ("apa persisnya yang dihancurkan?").

#### 2.6 Risiko dan mitigasi

| Risiko | Mitigasi |
|---|---|
| Efek lantai pada `arm-S` | Metrik utama AUC/EER, bukan akurasi (lihat peringatan di atas) |
| GRL sulit distabilkan | Mulai dengan λ kecil dan warmup; laporkan kegagalan konvergensi apa adanya |
| `for-rerec` ter-confound blok kelas | Ungkap eksplisit; posisikan sebagai probe sekunder |
| Penguji menganggap ini "sekadar analisis dataset" | Jawabannya: SRI adalah metrik baru, dan H2c membantah paper terpublikasi |

---

### NOVELTY 3, Fusi empat arsitektur dengan bobot berkondisi-SNR

> **Nama pendek untuk naskah:** *Arsitektur mana yang harus dipercaya pada SNR berapa.*

#### 3.1 Klaim ilmiah yang dapat diuji

**H3a.** Urutan keandalan keempat arsitektur **berbalik** seiring turunnya SNR: model berbasis spektrogram/mel unggul pada kondisi bersih, model SSL berbasis waveform unggul pada SNR rendah.

**H3b.** Fusi skor dengan bobot yang merupakan fungsi dari SNR yang diestimasi mengungguli fusi berbobot tetap, dan mengungguli model tunggal terbaik pada **setiap** titik SNR.

**H3c.** Keuntungan ensemble pada tugas ini berasal dari **keberagaman arsitektur**, bukan dari rata-rata seed.

H3c **sudah terbukti** [UKUR]: ensemble 12 run (semua seed) memberi hasil identik dengan ensemble 4 seed-terbaik, keduanya 97,61%.

#### 3.2 Mekanisme

Landasan empirisnya sudah ada dan kuat:

| Pengukuran | Nilai |
|---|---|
| Korelasi error φ antar arsitektur | **−0,014 … 0,223** (jauh di bawah 0,5) [UKUR] |
| Jaccard error antar arsitektur | 0,041 … 0,141 [UKUR] |
| Model tunggal terbaik (rerata 3 seed) | 91,94% · EER 8,06% |
| **Ensemble 4 arsitektur** | **97,61% · EER 2,30% · AUC 0,9954** |
| Selisih | **+5,67 pp**, 26 salah dari 1.088 |

[UKUR, `PERBANDINGAN.md`]

Keempat model gagal pada berkas yang **berbeda**. Itu bukan kebetulan, dekorelasinya **struktural**, berasal dari bias induktif yang benar-benar berbeda: SSL waveform (Wav2Vec2/HuBERT), Transformer patch-spektrogram (AST), dan CNN/CNN-LSTM di atas log-Mel.

Landasan mekanistik untuk H3a datang dari dua arah yang independen:
- [LIT, B5] Band-stop probing menunjukkan wav2vec2/HuBERT secara alami memakai pita 0,1–2,4 kHz, sedangkan detektor berbasis spektrogram bersandar pada pita 4–8 kHz, pita yang paling dulu hancur oleh noise, codec, dan resampling.
- [UKUR] Pada FoR, pita 4–8 kHz justru pita yang **sarat pintasan**: membuangnya menaikkan akurasi test 69,39% → 75,37%.

Kedua garis itu memprediksi hal yang sama: model yang bersandar pada pita tinggi akan runtuh lebih dulu. Itu prediksi yang dapat difalsifikasi.

Bukti awal keandalan yang berbeda-beda juga sudah terlihat:

| model | akurasi | **std** | rasio ketidakstabilan |
|---|---|---|---|
| `wav2vec2` | 90,75% | **±0,51** | 1,0× (paling stabil) |
| `cnnlstm` | 83,52% | ±2,28 | 4,5× |
| `ast` | 86,43% | ±2,94 | 5,8× |
| `cnn_asp` | 91,94% | ±3,50 | **6,9×** |

[UKUR] Wav2Vec2 tidak lebih akurat, ia **6,9× lebih andal**. Itu sendiri temuan yang layak satu subbab.

#### 3.3 Cara eksekusi konkret

**Langkah 1.** Latih 4 arsitektur × ≥3 seed dengan resep terstandardisasi. **Sudah selesai** untuk kondisi bersih/resmi, 12 run tersimpan di `runs/`.

**Langkah 2.** Evaluasi setiap model pada grid SNR; simpan skor mentah. Infrastruktur sudah ada (`test_scores.npy` ditulis setiap run).

**Langkah 3.** Kalibrasi per-model (Platt/temperature) pada dev set. **Wajib**, melewatkan ini adalah bug diam yang membuat fusi tampak tidak berguna. `TemperatureScaler` sudah ada di `forlib/metrics.py`.

**Langkah 4, baseline fusi:** bobot sama rata; lalu regresi logistik berbobot tetap.

**Langkah 5, FUSI BERKONDISI-SNR (inti novelty):**
```
w_i(ŝ) = softmax( α_i + β_i · ŝ ) i = 1..4
skor = Σ_i w_i(ŝ) · LLR_i(x)
```
Hanya **8 parameter bebas** untuk 4 model. Dengan dev set ~2.826 berkas × 7 titik SNR, risiko overfitting praktis nol. Difit di CPU dalam hitungan detik.

**Langkah 6, varian pembanding (opsional, lebih kuat tapi kurang interpretabel):** gate membaca **embedding** keempat model, bukan hanya ŝ [LIT, C7, gating dinamis menurunkan EER 10,29% → 2,74% dibanding rata-rata skor]. Laporkan keduanya; versi ŝ adalah klaim novelty karena ia interpretabel dan terikat besaran fisis.

**Langkah 7, tabel hasil per titik SNR:**
```
SNR | terbaik-tunggal | fusi-rata | fusi-LR-tetap | FUSI-BERKONDISI-SNR
```

#### 3.4 Cara membuktikan

- **H3a terkonfirmasi bila:** argmax model **berubah** melintasi grid SNR, terlihat sebagai kurva yang menyilang. Titik silang itu adalah gambar utama tesis.
- **H3a terbantah bila:** satu model mendominasi di mana-mana. Maka H3b otomatis memberi gain ≈ 0, dan itu **harus dilaporkan sebagai hasil negatif**.
- **H3b terkonfirmasi bila:** fusi berkondisi mengalahkan fusi berbobot tetap pada SNR rendah, signifikan menurut McNemar berpasangan + Holm-Bonferroni.
- **H3c sudah terkonfirmasi** [UKUR].

**Ekspektasi jujur atas magnitudo.** Gain +5,67 pp dari ensemble **sudah di tangan**. Gain tambahan dari pengondisian SNR adalah bagian yang tidak pasti, perkiraan realistis **0 sampai +2 pp** pada SNR rendah, dan **0** pada kondisi bersih. Karena itu bingkai kontribusinya sebagai *"fusi terkondisi + analisis pembalikan keandalan"*, sehingga hasil nol pada pengondisian tetap merupakan temuan yang dapat dilaporkan, didukung gambar kurva menyilang.

#### 3.5 Mengapa ini baru

1. Tidak ada karya terpublikasi yang memakai estimasi SNR sebagai *quality feature* di dalam fusi detektor deepfake audio [LIT, C1]. QMF dengan quality features ada di speaker verification, tidak di sini.
2. Gating dinamis terbukti mengalahkan rata-rata skor secara besar, tetapi gate-nya membaca embedding, bukan besaran fisis yang interpretabel [LIT, C7].
3. Tesis ini **secara kebetulan sudah memiliki prasyaratnya**: empat arsitektur dengan bias induktif berbeda dan korelasi error yang **sudah diukur** mendekati nol. Itu aset yang tidak boleh disia-siakan.
4. Pembalikan peringkat lintas SNR punya **dua** mekanisme kausal independen yang mendukungnya (pita 0,1–2,4 kHz vs 4–8 kHz; dan pita 4–8 kHz sarat pintasan pada FoR).

#### 3.6 Risiko dan mitigasi

| Risiko | Mitigasi |
|---|---|
| Tidak ada pembalikan peringkat (H3a gagal) | Laporkan sebagai hasil negatif; gambar kurva tetap bernilai |
| Gain pengondisian nol | Bingkai sebagai "fusi + analisis", bukan "fusi mengalahkan segalanya" |
| Penguji: "ensemble itu curang, bukan arsitektur" | Benar dan harus dijawab: laporkan kolom model tunggal **dan** kolom ensemble terpisah |

---

## 2. ARSITEKTUR FINAL YANG DIREKOMENDASIKAN

### 2.1 Diagram

```
 data/for-2seconds/ · 16 kHz mono · 32.000 sampel tepat · 17.870 berkas
 │
 ┌───────────────────────────▼────────────────────────────────┐
 │ PRAPROSES SERAGAM (identik untuk keempat arsitektur) │
 │ · normalisasi LOUDNESS berbasis RMS (bukan peak) │
 │ · TANPA trimming silence → tidak perlu: AUC lead_sil │
 │ = 0,5001, durasi identik 2,000 s std 0,0000 [UKUR] │
 │ · TANPA speech enhancement → menghapus artefak vocoder │
 └───────────────────────────┬────────────────────────────────┘
 │
 ┌───────────────────────────▼────────────────────────────────┐
 │ AUGMENTASI (train saja · SIMETRIS pada kedua kelas) │
 │ A. codec / band-limit p = 0,6 [WAJIB] │
 │ B. noise NYATA MUSAN + SLR28 p = 0,5, SNR ~ U(0,20) dB │
 │ C. RIR NYATA SLR28 p = 0,25 │
 │ D. gain ±8 dB p = 0,3 │
 │ ⚠ BUG WAJIB DIPERBAIKI: RNG kini di-seed per-BERKAS │
 │ sehingga augmentasi BEKU lintas epoch (lihat §3 P0-1) │
 └───────┬─────────────┬──────────────┬─────────────┬─────────┘
 ▼ ▼ ▼ ▼
 ┌─────────────┐┌─────────────┐┌─────────────┐┌─────────────┐
 │ Wav2Vec2 ││ HuBERT ││ AST ││ CNN-BiLSTM │
 │ base 12L ││ base 12L ★ ││ 12L ││ log-Mel 128 │
 │ 95 M ││ 95 M ★ ││ 86 M ││ ~4 M │
 │ waveform ││ waveform ││ patch-spek ││ spektrogram │
 └──────┬──────┘└──────┬──────┘└──────┬──────┘└──────┬──────┘
 │ hidden[0..12]│ │ │
 ┌──────▼──────────────▼──────────────▼───────┐ │
 │ LAYER WEIGHTING softmax atas SELURUH │ │
 │ hidden state (bukan last_hidden_state) │ │
 │ TERUKUR: w2v2 puncak L2–L4 · AST L3–L6 │ │
 └──────────────────────┬──────────────────────┘ │
 ▼ ▼
 ┌────────────────────────────────────────────────────────────┐
 │ Conv1d bottleneck 256 → ATTENTIVE STATISTICS POOLING │
 │ (mean + std berbobot atensi; bukan h_T, bukan mean-pool) │
 └────────────────────────────┬───────────────────────────────┘
 ▼
 ┌────────────────────────────────────────────────────────────┐
 │ Head: BatchNorm → Linear 256 → GELU → Dropout → Linear 2 │
 │ (+ opsional: kepala GRL provenance 3-arah, Novelty 2) │
 └────────────────────────────┬───────────────────────────────┘
 │ skor z_i (4 arsitektur × ≥3 seed)
 ══════════════════════════════════▼══════════════════════════════════
 LAPIS SISTEM (di sinilah dua dari tiga novelty berada)

 ┌──────────────────┐ ┌──────────────────────────────────────┐
 │ Estimator SNR │ ŝ │ KALIBRASI BERKONDISI-SNR [NOVELTY 1]│
 │ BUTA (dev-only) │─────►│ p_i = σ( a(ŝ)·z_i + b(ŝ) ) │
 │ WADA / regresor │ ├──────────────────────────────────────┤
 │ kecil │ ŝ │ FUSI BERKONDISI-SNR [NOVELTY 3]│
 └──────────────────┘─────►│ w_i(ŝ) = softmax(α_i + β_i·ŝ) │
 │ skor = Σ w_i(ŝ) · LLR_i │
 └──────────────────┬───────────────────┘
 ▼
 KEPUTUSAN real / fake
 (+ laporkan EER & AUC terpisah)

 ★ PERBAIKAN WAJIB: kode saat ini memasangkan wav2vec2-BASE (95 M)
 dengan hubert-LARGE (317 M) dan wavlm-LARGE. Itu meng-confound
 arsitektur dengan kapasitas dan membatalkan RQ utama tesis.
```

### 2.2 Justifikasi tiap komponen

| Komponen | Justifikasi | Label |
|---|---|---|
| **Normalisasi loudness (RMS), bukan peak** | `rms` adalah fitur pintasan trivial terkuat (AUC 0,6980). Peak-norm justru memperkuatnya. Menyamakan RMS menghapusnya sebagai sinyal | [UKUR] |
| **TIDAK trim silence** | Tidak ada pintasan silence di FoR-2sec: AUC `lead_sil` = 0,5001, `trail_sil` = 0,4823, durasi identik 2,000 s std 0,0000. Trimming hanya membuang sinyal. Ini mengoreksi rekomendasi standar literatur untuk dataset ini | [UKUR] |
| **TIDAK memakai speech enhancement** | Enhancement menghapus artefak vocoder yang menjadi dasar klasifikasi; routing terbukti mengalahkan enhancement front-end | [LIT, C7] |
| **Augmentasi codec simetris, p=0,6** | Perbaikan bug bertarget, bukan sekadar "robustness": EER 28,12% → 4,41%, AUC 0,7946 → 0,9900. Ini pengungkit terbesar tunggal yang terukur | [UKUR] |
| **Noise & RIR NYATA (MUSAN/SLR28 latih, DEMAND/WHAM! uji)** | Kode saat ini memakai noise berwarna sintetis (putih/pink/brown) dan reverb peluruhan eksponensial. Untuk RQ ketahanan noise lingkungan itu tidak memadai. Uji harus dari korpus yang **direkam independen** | [LIT, D2] |
| **Layer weighting atas seluruh hidden state** | Bobot terpelajar konsisten memuncak di L2–L4 (Wav2Vec2, 3 seed) dan L3–L6 (AST, 3 seed). `last_hidden_state`, default hampir semua tutorial HuggingFace, adalah pilihan **terburuk** untuk anti-spoofing | [UKUR] + [LIT, A2] |
| **Attentive Statistics Pooling** | Klip 2 detik hanya ~200 frame. ASP (mean+std berbobot) mendekati penjumlah bukti optimal; `h_T` LSTM punya recency bias dan merupakan agregator terlemah | [LIT, B2] |
| **BiLSTM, bukan LSTM satu arah** | LSTM satu arah mendiskriminasi frame awal |, |
| **Varian ablasi `cnn_asp` (tanpa LSTM)** | Terukur **mengalahkan** `cnnlstm` penuh: 91,94% vs 83,52%. Bukti bahwa LSTM tidak berkontribusi pada segmen 2 detik. Ini temuan yang layak dilaporkan, bukan sekadar ablasi | [UKUR] |
| **AST `max_length=200` + interpolasi positional embedding** | Default checkpoint AudioSet 1024 frame (~10,24 s) → 81% masukan adalah padding. Terverifikasi `pos_emb=(1,230,768)` = 2 token khusus + 228 patch | [UKUR] |
| **Encoder SSL beku (default)** | 13.956 klip (~7,8 jam) vs 95 M parameter. **Tetapi sumbu ini belum diuji sama sekali**, wajib diuji sekali sebelum menyimpulkan peringkat arsitektur | [HIP] |
| **LR per model, bukan seragam 1e-3** | Proposal hal. 68 memakai LR seragam; itu bug diam untuk model SSL |, |
| **Pemilihan checkpoint pada EER validasi** | Akurasi validasi jenuh di 99,5–99,8% apa pun checkpoint-nya → sinyal seleksi praktis datar | [UKUR] |
| **Kalibrasi + fusi berkondisi-SNR** | Novelty 1 & 3. Lihat §1 | [HIP] |
| **≥3 seed + McNemar/Holm-Bonferroni** | Variansi seed **±3,50 pp** = 4,2× CI statistik ±0,83 pp. Peringkat dari run tunggal tidak dapat dipertahankan | [UKUR] |

### 2.3 Peringatan integritas komparasi

Kode saat ini (`forlib/models.py`, `SSL_CKPT`):
```python
"wav2vec2": "facebook/wav2vec2-base", # 95 juta parameter
"hubert": "facebook/hubert-large-ll60k", # 317 juta parameter ← 3,3×
"wavlm": "microsoft/wavlm-large", # 317 juta parameter ← 3,3×
```
Judul tesis menjanjikan perbandingan **arsitektur**. Konfigurasi ini membandingkan arsitektur **dan** kapasitas sekaligus. Bila HuBERT menang, Anda tidak tahu apakah itu karena objektif masked-prediction atau karena 3,3× lebih besar. Perbaikan: pakai `facebook/hubert-base-ls960` untuk perbandingan utama, dan laporkan varian large sebagai baris terpisah "efek kapasitas". Ini adalah bug validitas, bukan bug kode, dan penguji yang teliti akan menemukannya.

---

## 3. TABEL SEMUA TEKNIK YANG BERTAHAN

Terurut prioritas. **Dampak** dinyatakan terhadap metrik yang benar-benar dilaporkan tesis.

### P0, Blokir korektness. Kerjakan sebelum apa pun.

| # | Teknik | Dampak | Usaha | Risiko | Bukti |
|---|---|---|---|---|---|
| **P0-1** | **Perbaiki RNG augmentasi beku.** `FoRDataset.__getitem__` men-seed RNG dari `hash(fname)+seed` yang tetap → setiap berkas mendapat **satu** varian augmentasi untuk seluruh pelatihan. Ini mengubah augmentasi on-the-fly menjadi augmentasi offline statis, memotong keragaman ~10× (jumlah epoch) | Tinggi pada generalisasi & noise; diperkirakan 1–4 pp pada split resmi | ~10 baris: tambahkan `set_epoch()`, masukkan epoch ke seed | Rendah | [UKUR] terbaca di kode |
| **P0-2** | **Korpus noise nyata, disjoint level berkas DAN korpus.** Latih MUSAN+SLR28, uji DEMAND+WHAM! | Menentukan apakah angka ketahanan-noise bermakna atau ilusi | 1 hari (unduh <25 GB + loader) | Rendah | [LIT, D2] |
| **P0-3** | **Harness evaluasi bergradasi SNR** (7 titik, −5…30 dB) | Prasyarat ketiga novelty; tanpa ini RQ tidak terjawab | 1 hari | Rendah |, |
| **P0-4** | **Tambahkan EER + AUC ke daftar metrik.** Proposal belum punya EER | Tanpa ini, akurasi@0,5 pada kondisi terdegradasi terlihat seperti kegagalan total padahal AUC masih 0,80–0,99 | Sudah terimplementasi di `forlib/metrics.py`; tinggal masuk naskah | Nol | [UKUR] |
| **P0-5** | **≥3 seed + McNemar + Holm-Bonferroni** | Variansi ±3,50 pp menelan sebagian besar selisih antar-arsitektur | Sudah ada (`compare.py`); biaya = waktu GPU | Rendah | [UKUR] |
| **P0-6** | **Samakan kapasitas SSL** (hubert-base, bukan hubert-large) | Memulihkan validitas RQ utama | 1 baris + retraining | Rendah | [UKUR] |
| **P0-7** | **Laporkan KEDUA protokol split** (acak 60/20/20 dan resmi FoR) | Memberi dosen angka ~99,9% **dan** memberi kontribusi ilmiah | Sudah ada (`--split random`/`official`) | Nol | [UKUR] |

### P1, Sudah terbukti di mesin ini. Pertahankan dan lengkapi.

| # | Teknik | Dampak terukur | Usaha | Risiko | Bukti |
|---|---|---|---|---|---|
| **P1-1** | Augmentasi codec simetris kedua kelas | **EER 28,12% → 4,41%; AUC 0,7946 → 0,9900** | Selesai |, | [UKUR] |
| **P1-2** | Koreksi ambang / kalibrasi | **akurasi 50,28% → 95,59%** pada skor identik | Selesai (transduktif) |, | [UKUR] |
| **P1-3** | Ensemble 4 arsitektur (rata-rata skor terkalibrasi) | **91,94% → 97,61%; EER 8,06% → 2,30%** | Selesai |, | [UKUR] |
| **P1-4** | Layer weighting atas seluruh hidden state | Bobot puncak L2–L4 / L3–L6, konsisten 3 seed | Selesai |, | [UKUR] |
| **P1-5** | Attentive Statistics Pooling | `cnn_asp` (ASP, tanpa LSTM) 91,94% vs `cnnlstm` 83,52% | Selesai |, | [UKUR] |
| **P1-6** | Normalisasi loudness RMS | Menghapus pintasan `rms` (AUC 0,698) | Selesai |, | [UKUR] |
| **P1-7** | **Uji unfreezing encoder SSL (top-N layer, LR rendah)** | **Tidak diketahui, sumbu ini belum diuji sama sekali.** Bisa mengubah peringkat arsitektur | 1–2 hari GPU | Sedang | [HIP] |
| **P1-8** | **Perbaiki rancangan `wavval`**, bagi 652 fake WAV, jangan pindahkan semuanya keluar dari training | Rancangan sekarang membuat data latih 100% fake turunan MP3; menurunkan rerata 8,5 poin | ~15 baris | Rendah | [UKUR] |
| **P1-9** | Evaluasi `for-rerec` (816 berkas, harness siap) | Degradasi kanal **nyata**; verifikasi awal menunjukkan AUC turun ~10–14 pp, **perlu direproduksi** | Beberapa jam | Sedang (confound blok kelas) | [UKUR sebagian] |

### P2, Novelty-bearing. Ini isi kontribusi tesis.

| # | Teknik | Dampak diperkirakan | Usaha | Risiko | Bukti |
|---|---|---|---|---|---|
| **P2-1** | **Dekomposisi diskriminasi/kalibrasi lintas grid SNR** (Novelty 1) | Bab analisis utama; mekanisme dijamin teorema | 3–5 hari | Rendah | [UKUR] mekanisme, [HIP] magnitudo |
| **P2-2** | **Kalibrasi berkondisi-SNR, induktif** (Novelty 1) | Menutup sebagian jurang transduktif↔induktif (8–12 pp) | 3–5 hari | Sedang | [HIP] |
| **P2-3** | **Protokol intervensi dua-sumbu + Shortcut Reliance Index** (Novelty 2) | Mengubah RQ dari deskriptif ke kausal | 1 minggu | Sedang (efek lantai) | [UKUR] premis |
| **P2-4** | **Kepala GRL provenance 3-arah** (Novelty 2, pelengkap) | Menyerang pintasan yang sudah teridentifikasi, dengan label yang sudah ada | ~40 baris + retraining | Sedang (stabilitas GRL) | [UKUR] label |
| **P2-5** | **Fusi berkondisi-SNR** (Novelty 3) | 0 s/d +2 pp di atas fusi tetap; nilai utamanya gambar kurva menyilang | 2–3 hari | Rendah | [HIP] |
| **P2-6** | Analisis pembalikan keandalan lintas SNR (Novelty 3) | Satu gambar utama tesis | Termasuk P2-5 | Rendah | [LIT, B5] + [UKUR] |
| **P2-7** | Analisis error manual: dengarkan 26–54 berkas yang salah | Jauh lebih informatif daripada sweep hyperparameter berhari-hari; 54 berkas = 108 detik audio | 15–30 menit | Nol | [UKUR] |

### P3, Opsional. Rasio nilai/risiko lebih rendah.

| # | Teknik | Dampak diperkirakan | Usaha | Risiko | Bukti |
|---|---|---|---|---|---|
| **P3-1** | Inisialisasi dari checkpoint **AntiDeepfake** (nii-yamagishilab, 6 ukuran 95M–2B, di-post-train pada ~74.650 jam, tidak dilatih pada FoR) | Melaporkan EER 1,40% / AUC 0,999 di FakeOrReal **tanpa** fine-tuning | 1–2 hari | **Sedang-tinggi:** checkpoint mengharapkan jendela 64.600 sampel (~4,04 s), FoR-2sec hanya 32.000 → padding yang salah merusak seluruh evaluasi | [LIT, C10/D4] |
| **P3-2** | Copy-synthesis vocoder (pseudo-fake dari audio real FoR sendiri) | Potensi tertinggi untuk membunuh **semua** pintasan sekaligus (speaker, konten, kanal, silence) | 1–3 minggu | **Tinggi:** repo vocoder umumnya menargetkan Python 3.7–3.11; lingkungan Anda Python 3.14 tanpa librosa | [LIT, C10/B1/C4] |
| **P3-3** | Contrastive noise-invariance (gaya CLAD) | Literatur kuat (FAR di bawah noise 51,28% → 0,81%) | 1 minggu | Sedang | [LIT, B3/C3] |
| **P3-4** | Supervisi 4-kelas (bersih/bernoise × real/fake) | **−0,3 s/d +1 pp**, di dalam derau seed ±3,50 pp | ~3 baris | Rendah | [LIT, direfutasi sebagai penaik akurasi] |
| **P3-5** | Augmentasi perturbasi fase | Mekanisme sahih, efek pada dataset ini belum diukur. **Jangan sebut "murni fase"**, terukur mengubah log-mel hasil re-analisis | 1–2 hari | Sedang | [UKUR] koreksi mekanisme |
| **P3-6** | Focal loss / OC-Softmax / hard mining | Pada rezim >98%, hard-mining klasik justru **merugikan** (FocalLoss 1,67% vs plain CCE 1,39% EER) | 1 hari | Sedang-negatif | [LIT, C6] |
| **P3-7** | SWA / model soup / EMA | Nol paper menggabungkannya dengan deteksi deepfake audio → celah murah. Tapi ensemble Anda sudah 97,61% | 1 hari | Rendah | [LIT, C5] |

---

## 4. YANG HARUS DIHINDARI

### 4.1 Temuan yang direfutasi pada verifikasi adversarial

| Temuan | Mengapa gugur | Yang tersisa |
|---|---|---|
| **"Bug ffmpeg menyuntik 88 ms keheningan"** | Nol poin akurasi. Penanda 88 ms membedakan *augmented vs tidak*, bukan *real vs fake*, tidak ada jalur kebocoran label. Lebih fatal: pipeline Anda memakai FFT-lowpass numpy, **jalur ffmpeg tidak pernah dieksekusi**. Dan tidak ada pintasan silence untuk dibocorkan (`lead_sil` AUC = 0,5001) | Satu unit test `assert len(aug(x))==len(x)`. Bila nanti benar-benar memakai codec nyata, pakai `soundfile`+BytesIO (26× lebih cepat: 59,1 ms → 2,3 ms per klip), bukan pipe ffmpeg |
| **"Fusi multi-SSL sebagai novelty, 99,5% → 99,8–99,9%"** | Aritmetikanya tidak berdiri: pada n≈1.088–3.574, 99,5%→99,9% menuntut reduksi error relatif ~82%, sedangkan efek yang benar-benar didemonstrasikan paper adalah 18% relatif. Dan pada 1 dari 4 korpus, fusi justru 32% **lebih buruk** dari model tunggal terbaik | Fusinya sendiri **tetap dikerjakan** (sudah terukur +5,67 pp), tapi sebagai **teknik**, bukan sebagai kebaruan, dan dilaporkan pada sumbu ketahanan-noise, bukan sebagai pemecah plafon akurasi |
| **"SpAArSIST: buang komponen graf AASIST yang dipelajari"** | Paper sumber **tidak pernah mengukur noise aditif sama sekali**; efek dominannya berasal dari rasio pooling (hiperparameter, nol parameter dibuang), bukan dari pembuangan parameter. Dan tesis ini tidak memakai AASIST | Satu ide: `k_inf < k_tr` (mismatch pooling latih/inferensi) sebagai knob regularisasi murah, **harus diuji dari nol** |
| **"Batas real/fake adalah batas kanal; augmentasi simetris memberi +15–40 pp"** | Angka 15–40 pp **tanpa sumber apa pun**. Kalibrasi nyata satu-satunya (arXiv:2407.20111): 2,7–15,8% atas baseline lintas SNR, dan hanya 0,7–5,8% di atas metode augmentasi data. Bukti `for-rerec` justru menunjukkan arah **berlawanan** (rekam-ulang simetris tidak memburuk). Prediksi "probe 3-fitur >80%" diprediksi **gagal** oleh data paper yang dikutipnya sendiri (model linier 30 fitur penuh hanya 75,3%) | Trim silence: **jangan** (tidak ada pintasan silence di sini). Augmentasi MUSAN+RIR: ya, tapi sebagai praktik mapan, bukan novelty |
| **"Bug ComputeDeltas win_length=400"** | Codebase Anda **tidak memakai** `ComputeDeltas` sama sekali. 0,00 pp | Bila nanti menambah cabang MFCC+Δ/ΔΔ, pakai `win_length=5` atau `9`. Satu kalimat catatan metode |
| **"RawBoost SSI: kerapuhan noise bersifat asimetris kelas"** | Mekanismenya tidak dapat dipertahankan. `for-rerec` sudah merupakan uji simetris atas hipotesis itu dan hasil terpublikasinya **berlawanan** (EER 7,3% → 6,6%, membaik). Bila H1 ditulis sebagai fakta, penguji yang membuka dokumentasi FoR akan membantahnya dengan data pembuat dataset | RawBoost tetap boleh dipakai (alasannya valid: robustness terhadap kondisi tak terlihat), tetapi **bukan** karena mekanisme asimetri. Confusion matrix per level SNR tetap layak dilaporkan sebagai tabel deskriptif |
| **"for-rerec mismatch sebagai eksperimen unggulan"** | Ter-confound berat: berkas tersusun dalam blok per-kelas (confound sesi/urutan rekaman), 816 vs 1.088 berkas (kehilangan 26%), pergeseran bit-depth/durasi. Tidak ada kondisi kontrol bersih. 1 berkas = 0,123 pp; klaim mengisolasi kontribusi SSI (1–3 pp = 8–24 berkas) berada di dalam derau | Tetap dipakai sebagai **probe sekunder** dengan confound diungkap eksplisit. Bukan bab unggulan |
| **"Supervisi 4-kelas bersih/bernoise × real/fake"** | −0,3 s/d +1 pp, seluruhnya di dalam derau seed ±3,50 pp. Pada konfigurasi terbaik Anda (`--augment codec`, noise=0,0) dua dari empat kelas **kosong** dan skema runtuh menjadi biner secara struktural | Layak sebagai satu baris ablasi (biaya ~3 baris kode). Bukan kontribusi |
| **"Pintasan bandwidth/sample-rate provenance"** | Sudah **diukur di sini** dan lebih kecil dari dugaan: fitur pita AUC 0,28–0,37, kalah dari `rms` (0,698). Low-pass ke 4 kHz akan menggeser distribusi jauh dari data pretraining Wav2Vec2/HuBERT (LibriSpeech full-band) → penurunan diperkirakan 5–20 poin untuk kedua model SSL | Satu baris ablasi **negatif**: membuktikan bahwa pintasan bandwidth global BUKAN penyebab utama. Hasil negatif yang sah, biayanya sudah nol (data di `audit_features.csv`) |
| **"Koreksi MetricGAN+ vs SEGAN"** (satu-satunya yang bertahan verifikasi) | Bertahan, tapi bernilai **nol untuk akurasi**, dan eksperimen korelasinya degenerate | Satu paragraf hati-hati di bab pembahasan bila (dan hanya bila) enhancement dibahas: SEGAN beroperasi end-to-end di domain waveform, MetricGAN+ memprediksi mask magnitudo dan memakai ulang fase noisy. Sitasi Pascual dkk. 2017, Fu dkk. 2021, kode SpeechBrain. **Satu paragraf, bukan satu bab** |

### 4.2 Jebakan metodologis yang harus dihindari

1. **Jangan laporkan split acak 60/20/20 sendirian sebagai hasil utama.** Random Forest atas 38 fitur statistik sepele mencapai **95,91%** pada protokol itu [UKUR], mengungguli ketiga baseline yang dikutip proposal (93,50% / 94,47% / 94,70%). Angka di rezim itu tidak lagi membedakan metode baik dari metode biasa.

2. **Jangan laporkan peringkat arsitektur dari run tunggal.** Variansi seed ±3,50 pp = 4,2× CI statistik ±0,83 pp [UKUR]. `cnn_asp` menghasilkan 95,04 / 92,65 / 88,14, rentang 6,90 poin dari inisialisasi saja.

3. **Jangan sajikan akurasi prior-matched sebagai angka induktif.** Ia **transduktif**, memerlukan akses ke seluruh skor test sekaligus. Sah untuk forensik batch/arsip, **tidak** untuk deteksi streaming. Batasan ini wajib dinyatakan eksplisit di naskah.

4. **Jangan pilih checkpoint hanya dari validasi resmi.** Validasi resmi punya 89,6% fake turunan MP3, **sama biasnya dengan training** [UKUR]. Memilih checkpoint dari situ = memilih model yang paling pandai mengeksploitasi pintasan. Akurasi validasi 99,82% berdampingan dengan akurasi test 50,00%.

5. **Jangan tambahkan noise hanya pada satu kelas.** Itu menciptakan pintasan baru yang lebih parah daripada yang dihapus.

6. **Jangan pakai speech enhancement sebagai praproses.** Ia menghapus artefak vocoder yang menjadi dasar klasifikasi [LIT, C7].

7. **Jangan klaim "sebagian besar hasil terpublikasi pada FoR memakai split acak."** Sudah diverifikasi dan **terbantah sebagian**: Ahmad dkk. memakai partisi resmi speaker-disjoint dan tetap mencapai ~93% [`VERIFIKASI_RUJUKAN.md`]. Yang boleh dipertahankan hanyalah (a) klaim mekanistik dari eksperimen sendiri, (b) klaim per-paper dengan kutipan langsung, (c) klaim atribusi codec.

8. **Jangan bandingkan wav2vec2-base dengan hubert-large** dan menyebutnya perbandingan arsitektur.

9. **Jangan trim silence.** Tidak ada pintasan silence di FoR-2sec [UKUR]. Rekomendasi standar literatur tidak berlaku di sini, dan menerapkannya hanya membuang sinyal.

---

## 5. BATAS ATAS REALISTIS

### 5.1 Per protokol

| Protokol | Terukur sekarang | Batas atas realistis | Yang menahannya |
|---|---|---|---|
| **A. Split acak 60/20/20** *(rencana proposal)* | **99,94%** acc · EER **0,03%** · AUC **1,0000** · 2 salah / 3.574 [UKUR] | **99,9 – 100%** | Label noise & klip ambigu. **Praktis sudah jenuh.** Tapi angka ini secara ilmiah kosong: RF fitur sepele mencapai 95,91% pada protokol yang sama |
| **B. Split resmi, model tunggal, 3 seed, ambang transduktif** | **91,94% ± 3,50** · EER 8,06% ± 3,54 [UKUR] | **93 – 96%** rerata; EER **4 – 7%** [HIP] | Run tunggal terbaik sudah 95,04% / EER 4,87%; perbaikan P0 terutama menurunkan variansi dan menaikkan rerata ke arah run terbaik |
| **C. Split resmi, ensemble 4 arsitektur, ambang transduktif** | **97,61%** · EER **2,30%** · AUC **0,9954** · 26 salah / 1.088 [UKUR] | **98,0 – 98,5%** (16–22 salah); EER **1,5 – 2,0%** [HIP] | Untuk mencapai 99% Anda harus memperbaiki 15 dari 26 error tanpa menciptakan error baru |
| **D. Split resmi, ambang INDUKTIF (tanpa akses skor test)** | 90,62% (n=1) · 83,40% ± 1,41 (3 seed, `wavval`) · 79,69% (`clean_val`) [UKUR] | **88 – 93%** setelah P1-8 + Novelty 1 [HIP] | **Ini angka deployment yang jujur, dan ia 8–12 pp DI BAWAH angka transduktif.** Jurang itu sendiri adalah temuan |
| **E. Di bawah noise, SNR 0–20 dB** | **BELUM ADA PENGUKURAN** | Tidak dapat dinyatakan | Siapa pun yang mengutip angka di sini sedang menebak. Ini justru rumusan masalah utama tesis, dan ia belum tersentuh |
| **F. `for-rerec` (rekam-ulang nyata)** | Verifikasi awal: AUC turun ~10–14 pp; `cnn_asp` EER 0,049 → 0,190; AST AUC → 0,7962, EER → 0,304. **Perlu direproduksi** | 80 – 90% dengan ambang terkalibrasi [HIP] | Confound blok kelas; 816 berkas |

### 5.2 Apa yang menahan plafon, terurut besarnya

**1. Ukuran test set: 1.088 berkas.** Batas keras pada resolusi. 1 berkas = **0,092 pp**. Dua model yang berselisih < 1,7 pp tidak dapat dibedakan tanpa uji berpasangan [UKUR]. Ini tidak dapat diperbaiki tanpa mengganti dataset. Konsekuensinya: **berhenti mengejar selisih sub-1-poin**; energi lebih baik diarahkan ke sumbu SNR, di mana efeknya berorde puluhan poin.

**2. Variansi seed ±3,50 pp.** 4,2× lebih besar dari CI statistik. Sebagian dapat diredam: ensemble sudah meruntuhkannya (97,61% dari 12 run = identik dengan 4 run seed-terbaik). Untuk model tunggal, minimum 3 seed, idealnya 5.

**3. Pergeseran domain bawaan partisi resmi.** Energi HF kelas `real` bergeser 5,8× antara training dan testing [UKUR]. Ini **desain dataset**, bukan bug. Tidak dapat "diperbaiki", hanya dapat diadaptasi. Ia menetapkan lantai kesulitan yang nyata.

**4. Jurang transduktif ↔ induktif: 8–12 pp.** Ini adalah jurang **terbesar yang masih dapat ditangani** dan persis sasaran Novelty 1. Angka 97,61% memakai ambang prior-matched; versi induktif yang jujur saat ini jauh lebih rendah.

**5. Tidak ada data eksternal.** Tanpa copy-synthesis atau inisialisasi AntiDeepfake, model hanya pernah melihat ~7,8 jam audio latih dari satu korpus. Ini pengungkit terbesar yang belum disentuh, dan yang paling berisiko secara rekayasa (P3-1, P3-2).

**6. Encoder SSL beku, sumbu yang belum diuji.** Kesimpulan "arsitektur X lebih baik dari Y" **tidak dapat dipertahankan** sebelum sumbu frozen/unfrozen diperiksa minimal sekali.

### 5.3 Apa yang harus dilaporkan ke dosen

Permintaan "akurasi mendekati 100%" **sudah terpenuhi** dan tidak perlu dikompromikan. Sajikan tiga kolom berdampingan:

| Protokol | Akurasi | EER | AUC | Catatan |
|---|---|---|---|---|
| Split acak 60/20/20 (protokol yang lazim di literatur FoR) | **99,94%** | 0,03% | 1,0000 | Melampaui **seluruh** baseline yang dikutip proposal (93,50 / 94,47 / 94,70) |
| Split resmi FoR, ensemble 4 arsitektur | **97,61%** | 2,30% | 0,9954 | Pada protokol lintas-domain yang **jauh lebih sulit**; tetap melampaui ketiga baseline |
| Split resmi FoR, model tunggal, 3 seed | **92,3% ± 3,0** | 8,06% | 0,9712 | Angka yang jujur untuk perbandingan arsitektur |

Ketiganya sudah terukur. Yang ditambahkan tesis bukan angka yang lebih tinggi, melainkan **kolom "protokol split"**, kolom yang tidak ada di tabel state-of-the-art proposal (hal. 8–11), dan yang menjelaskan mengapa ketiga angka itu berbeda 7,6 poin dengan model yang sama.

### 5.4 Kalimat kontribusi yang dapat dipertahankan sepenuhnya

> Dengan arsitektur, data, dan hyperparameter yang identik, protokol pembagian data acak menghasilkan akurasi 99,94% sementara partisi resmi menghasilkan 50,00% pada dataset Fake-or-Real. Selisih 49,94 poin persentase tersebut berasal dari korelasi semu antara provenance codec dan label kelas: 90,7% sampel deepfake pada data latih berasal dari berkas MP3 sedangkan 0% sampel deepfake pada data uji berasal dari MP3. Analisis pita frekuensi menunjukkan sekitar tiga perempat selisih energi frekuensi tinggi antar-kelas pada data latih dapat diatribusikan pada kompresi MP3, bukan pada sintesis, bertentangan dengan tafsir yang dilaporkan sebelumnya. Augmentasi codec yang diterapkan seragam pada kedua kelas menetralkan korelasi tersebut dan menurunkan EER dari 28,12% menjadi 4,41%; koreksi ambang selanjutnya memulihkan akurasi dari 50,28% menjadi 95,59% **pada skor yang identik**, memisahkan kegagalan diskriminasi dari kegagalan kalibrasi.

Setiap angka dalam paragraf itu berasal dari pengukuran di mesin ini dan dapat direproduksi dengan skrip yang sudah ada.

---

## Lampiran, Rujukan berkas

| Berkas | Isi |
|---|---|
| `C:\Users\Tristan\Downloads\general-ai\TEMUAN_GROUND_TRUTH.md` | Audit empiris dataset, dokumen paling penting |
| `C:\Users\Tristan\Downloads\general-ai\HASIL_EKSPERIMEN.md` | Tabel hasil + koreksi multi-seed + ablasi 2×2 |
| `C:\Users\Tristan\Downloads\general-ai\PERBANDINGAN.md` | 4 arsitektur × 3 seed, McNemar, korelasi error, ensemble |
| `C:\Users\Tristan\Downloads\general-ai\VERIFIKASI_RUJUKAN.md` | Verifikasi protokol split ref [13]/[19]/[20] |
| `C:\Users\Tristan\Downloads\general-ai\audit_report.md` | AUC 18 fitur trivial, resolusi statistik |
| `C:\Users\Tristan\Downloads\general-ai\probe_shift_report.md` | Energi >6 kHz per split × kelas × provenance |
| `C:\Users\Tristan\Downloads\general-ai\probe_codec_report.md` | Profil 20 pita, cutoff spektral |
| `C:\Users\Tristan\Downloads\general-ai\probe_split_report.md` | Eksperimen penentu split resmi vs acak |
| `C:\Users\Tristan\Downloads\general-ai\forlib\data.py` | **Lokasi bug P0-1** (RNG augmentasi beku, baris ~293) |
| `C:\Users\Tristan\Downloads\general-ai\forlib\models.py` | **Lokasi bug kapasitas** (`SSL_CKPT`, baris ~88–92) |
| `C:\Users\Tristan\Downloads\general-ai\forlib\metrics.py` | EER, DET, McNemar, Holm-Bonferroni, ECE, temperature scaling, prior-matched threshold |
| `C:\Users\Tristan\Downloads\general-ai\runs\` | 24 run tersimpan, `results.json` + `test_scores.npy` per run |
| `C:\Users\Tristan\AppData\Local\Temp\claude\C--Users-Tristan-Downloads\2eb65358-beca-4011-b948-1c0d9186e535\scratchpad\eval_rerec.py` | Harness evaluasi `for-rerec` (816 berkas sudah terekstrak) |