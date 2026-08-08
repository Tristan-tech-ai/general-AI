# Temuan Ground Truth: Audit Empiris Dataset FoR-2sec

*Semua angka di dokumen ini berasal dari eksekusi nyata pada 17.870 berkas audio yang diunduh dari sumber resmi York University, bukan dari literatur atau perkiraan. Skrip yang menghasilkannya ada di folder ini dan dapat dijalankan ulang.*

**Tanggal audit:** 4 Agustus 2026
**Sumber dataset:** `https://bil.eecs.yorku.ca/share/for-2sec.tar.gz` (1.000 MB, resmi APTLY Lab)
**Skrip:** [audit.py](audit.py) · [probe_codec.py](probe_codec.py) · [probe_shift.py](probe_shift.py) · [probe_split.py](probe_split.py)
**Laporan mentah:** [audit_report.md](audit_report.md) · [probe_codec_report.md](probe_codec_report.md) · [probe_shift_report.md](probe_shift_report.md) · [probe_split_report.md](probe_split_report.md)

---

## Koreksi asumsi saya sebelumnya

Sebelum masuk temuan: pada dokumen [ANALISIS_DAN_RENCANA.md](ANALISIS_DAN_RENCANA.md) saya mencatat asumsi **A1** bahwa angka ~98% berasal dari eksperimen aktual yang sudah Anda jalankan. **Asumsi itu salah**, Anda menyampaikan bahwa belum ada kode maupun log, dan tugasnya adalah membangun penelitian ini dari nol. Seluruh dokumen tetap berlaku sebagai rencana, tetapi baris "baseline sekarang ~98%" harus dibaca sebagai *target*, bukan kondisi saat ini.

Temuan di bawah ini juga membatalkan sebagian kekhawatiran saya di ANALISIS_KRITIS.md (kebocoran duplikat: **tidak ada**) dan sekaligus menemukan masalah yang jauh lebih besar dan lebih menarik yang tidak saya duga sama sekali.

---

## 1. Struktur resmi dataset, proposal menyebut angka yang keliru

```
data/for-2seconds/
├── training/ real 6.978 | fake 6.978 = 13.956 (78,1%)
├── validation/ real 1.413 | fake 1.413 = 2.826 (15,8%)
└── testing/ real 544 | fake 544 = 1.088 ( 6,1%)
 TOTAL = 17.870
```

✅ Proposal hal. 55 menyatakan split **60% / 20% / 20%**. Split resmi FoR sebenarnya **78,1% / 15,8% / 6,1%**. Untuk mencapai 60/20/20, seluruh data harus digabung lalu dibagi ulang secara acak, dan §5 di bawah menunjukkan mengapa itu adalah kesalahan yang menghancurkan validitas seluruh penelitian.

**Format sepenuhnya seragam** (ini kabar baik):

| Properti | Nilai |
|---|---|
| Sampling rate | 16.000 Hz, **100% berkas**, tanpa kecuali |
| Kanal | Mono, 100% berkas |
| Durasi | **Tepat 2,000 detik**, std = 0,0000 s |
| Berkas gagal dibaca | **0** |

Karena durasi identik hingga presisi penuh, **tidak ada pintasan berbasis durasi** (AUC `dur` = 0,5000, persis kebetulan). Ini menutup salah satu kecurigaan utama saya.

---

## 2. Tidak ada kebocoran duplikat, kekhawatiran T0-1 saya keliru

| Uji | Hasil |
|---|---|
| Grup berkas byte-identik | **0** |
| Duplikat melintasi split | **0** |
| Duplikat dengan label bertentangan | **0** |

✅ Partisi resmi FoR bersih dari duplikasi eksak. Rencana audit kebocoran (T0-1) tetap perlu dijalankan untuk *near-duplicate* berbasis pembicara, tetapi kekhawatiran terbesar sudah terbantah.

---

## 3. Pintasan trivial ada, tapi tidak dominan

Klasifikasi hanya dari 18 fitur global (RMS, peak, DC, silence, ZCR, energi pita, centroid, rolloff, **tanpa informasi fonetik sama sekali**), dilatih pada `training` resmi, diuji pada `testing` resmi:

| Model | Akurasi test |
|---|---|
| Logistic Regression | 53,12% |
| **Random Forest** | **77,48%** |

AUC fitur tunggal terkuat: `rms` = 0,6980. Tidak ada satu pun fitur trivial yang melampaui AUC 0,70.

**Tafsir:** ~77% tugas ini dapat diselesaikan dari statistik sinyal global saja. Signifikan, tetapi tidak katastrofik, masih ada ~21 poin yang menuntut pemodelan sesungguhnya. Yang menarik justru **mengapa** angka 77% itu muncul, dan itu dijelaskan di §4.

---

## 4. Temuan besar #1: kelas `fake` berasal dari MP3, kelas `real` tidak

Terlihat pertama kali dari pola nama berkas:

```
real : file1000.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav
fake : file10005.mp3.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav
 ^^^^
```

Distribusi provenance:

| split | kelas | total | berasal MP3 | persen |
|---|---|---|---|---|
| training | real | 6.978 | 0 | **0,0%** |
| training | fake | 6.978 | 6.326 | **90,7%** |
| validation | real | 1.413 | 0 | **0,0%** |
| validation | fake | 1.413 | 1.266 | **89,6%** |
| testing | real | 544 | 0 | **0,0%** |
| **testing** | **fake** | **544** | **0** | **0,0%** ← |

Konsekuensi akustiknya terukur. Energi relatif di pita > 6 kHz:

| split | kelas | provenance | n | energi > 6 kHz |
|---|---|---|---|---|
| training | real | WAV | 6.978 | 0,03355 |
| training | fake | **MP3** | 6.326 | **0,00802** ← 4,18× lebih rendah |
| training | fake | WAV | 652 | 0,02872 ← hanya 1,17× lebih rendah |
| testing | real | WAV | 544 | 0,00575 |
| testing | fake | WAV | 544 | 0,00431 ← hanya 1,33× lebih rendah |

**Baca tabel itu baik-baik.** Selama pelatihan, sampel `fake` punya energi frekuensi tinggi **4,18× lebih rendah** daripada `real`, tetapi itu terjadi karena 90,7% di antaranya melewati kompresi MP3, bukan karena disintesis. Bukti langsungnya: 652 sampel `fake` yang **tidak** melalui MP3 hanya berselisih **1,17×**, praktis tidak berbeda.

Model apa pun yang dilatih pada partisi ini akan mempelajari aturan **"energi HF rendah ⇒ fake"**. Aturan itu benar 90,7% dari waktu di data latih, dan **sama sekali tidak berlaku di test set**, di mana tidak ada satu pun `fake` berasal dari MP3.

Ini adalah *spurious correlation* buku teks, terdokumentasi dalam data, dan dapat diverifikasi ulang siapa pun dalam dua menit.

---

## 5. Temuan besar #2, dan ini yang paling penting

Test set resmi FoR bukan sekadar berbeda pada provenance codec. **Seluruh domain rekamannya berbeda:**

```
energi >6 kHz, kelas REAL:
 training : 0,03355
 testing : 0,00575 → 5,8× lebih rendah
```

Kelas `real` di test set punya energi pita tinggi **5,8× lebih rendah** daripada kelas `real` di training. Test set berasal dari sumber rekaman yang berbeda. Pembuat FoR tampaknya **sengaja** merancang partisi resmi sebagai **evaluasi lintas-domain**.

Itu desain dataset yang bagus. Tapi berarti: siapa pun yang membagi ulang FoR secara acak **menghancurkan pemisahan domain tersebut** dan mengukur sesuatu yang sama sekali berbeda.

### Eksperimen penentu

Classifier identik (Random Forest 400 pohon), fitur identik (38 fitur spektral + statistik global), **tanpa deep learning, tanpa fonetik, tanpa fase**. Satu-satunya yang berubah adalah skema pembagian data:

| Skema split | n latih | n uji | **Akurasi uji** |
|---|---|---|---|
| **Resmi FoR** (training+validation → testing) | 16.782 | 1.088 | **79,23%** |
| **Acak 60/20/20** ← *rencana proposal, hal. 55* | 10.722 | 3.574 | **95,91%** |
| Acak, hanya dalam training resmi (kontrol) | 11.164 | 2.792 | **96,92%** |

**Selisih: +16,69 poin persentase, dari skema split saja.** Model tidak menjadi lebih pintar sedikit pun.

### Mengapa ini penting sekali

Bandingkan angka 95,91%, yang diperoleh dari **fitur spektral sepele di bawah split acak**, dengan seluruh baseline deep learning yang dikutip proposal pada dataset FoR:

| Sumber (dikutip proposal) | Metode | Akurasi |
|---|---|---|
| Ref [13] | SVM + MFCC + PCA | 93,50% |
| Ref [19] MFAAN | CNN multi-feature fusion | 94,47% |
| Ref [20] | Hybrid CNN-LSTM | 94,70% |
| **Audit ini** | **Random Forest, 38 fitur trivial, split acak** | **95,91%** |

**Random Forest atas 38 angka statistik mengungguli ketiganya.**

Ini bukan berarti ketiga penelitian itu tidak kompeten. Artinya: **pada FoR dengan split acak, tugasnya menjadi sedemikian mudah sehingga akurasi ~95% tidak lagi membedakan metode yang baik dari yang biasa.** Angka-angka itu sebagian besar mengukur kebocoran domain, bukan kemampuan mendeteksi deepfake.

> ⚠️ **Batas klaim.** Saya tidak dapat memastikan protokol split yang dipakai ref [13], [19], [20] karena proposal tidak mencantumkannya dan saya belum membaca ketiga paper aslinya. Yang terbukti di sini adalah **mekanismenya**: split acak pada FoR menaikkan akurasi ~17 poin tanpa perbaikan model. Bahwa angka mereka jatuh persis di rentang yang dihasilkan mekanisme itu adalah **indikasi kuat, bukan bukti**. Memverifikasinya (membaca protokol ketiga paper) adalah pekerjaan satu jam dan **wajib** dilakukan sebelum klaim ini masuk naskah tesis.

---

## 6. Resolusi statistik: test set hanya 1.088 berkas

| Akurasi | Jumlah berkas salah | CI 95% |
|---|---|---|
| 95,0% | 54 | ±1,30 pp |
| 97,0% | 33 | ±1,01 pp |
| 98,0% | 22 | ±0,83 pp |
| 99,0% | 11 | ±0,59 pp |
| 99,5% | 5 | ±0,42 pp |
| 100% | 0 |, |

**Satu berkas = 0,092 poin persentase.**

Dua konsekuensi langsung:

1. **Dua model yang berselisih < 1,7 pp tidak dapat dibedakan secara statistik** tanpa uji berpasangan. Karena keempat model dievaluasi pada berkas yang sama persis, **uji McNemar wajib**, membandingkan interval kepercayaan independen akan selalu menghasilkan "tidak signifikan" dan membuat kesimpulan tesis tidak dapat dipertahankan.

2. **Analisis error menjadi sangat murah.** Pada 95%, hanya ada 54 berkas salah, 108 detik audio. Anda bisa mendengarkan **semuanya** dalam 15 menit dan mengetahui persis apa yang gagal. Ini jauh lebih informatif daripada hyperparameter sweep berhari-hari.

---

## 7. Apa artinya untuk tesis ini

### Yang berubah

Proposal berangkat dari premis: *"FoR relatif mudah; baseline 94%; kita kejar >97%."*
Data menunjukkan premis sebenarnya: **"FoR dengan protokol resmi adalah benchmark lintas-domain yang sulit; angka 94–99% di literatur kemungkinan besar berasal dari split acak."**

### Pilihan yang harus diambil, dan rekomendasi saya

| Opsi | Split | Perkiraan akurasi | Nilai ilmiah |
|---|---|---|---|
| **A** | Acak 60/20/20 (rencana proposal) | **98–99,5%** | Rendah, mengukur kebocoran domain |
| **B** | Resmi FoR | **85–95%** | Tinggi, jujur & sulit |
| **C** | **Laporkan KEDUANYA + kuantifikasi selisihnya** | keduanya | **Tertinggi** |

**Rekomendasi tegas: Opsi C.**

Alasannya praktis sekaligus ilmiah. Dosen meminta akurasi setinggi mungkin, idealnya 100%, **Opsi C tetap memberikan angka itu** (kolom split acak akan mencapai 98–99,5% dan melampaui setiap baseline yang dikutip proposal). Tetapi ia menyertainya dengan kolom kedua yang jujur, plus penjelasan mengapa keduanya berbeda. Anda tidak kehilangan apa pun dan memperoleh kontribusi ilmiah yang sesungguhnya.

Kalimat kontribusi tesis menjadi:

> *"Penelitian ini menunjukkan bahwa akurasi yang dilaporkan pada dataset Fake-or-Real sangat bergantung pada protokol pembagian data. Dengan model dan fitur yang identik, split acak menghasilkan akurasi 16,7 poin lebih tinggi daripada partisi resmi, karena partisi resmi memisahkan domain rekaman antara data latih dan data uji. Penelitian ini melaporkan kedua protokol dan mengukur kontribusi masing-masing sumber artefak."*

Itu klaim yang dapat diuji, dapat direproduksi, penting bagi bidangnya, dan **sepenuhnya berbasis data yang sudah kita ukur**, bukan spekulasi.

### Konsekuensi teknis langsung

1. **Augmentasi codec bukan lagi sekadar "robustness", ia adalah perbaikan bug.**
 Karena 90,7% `fake` latih adalah turunan MP3 dan 0% `fake` uji, menerapkan kompresi MP3 secara **seragam pada kedua kelas** saat pelatihan akan menetralkan fitur semu tersebut. Ini diperkirakan **menaikkan** akurasi pada test set resmi, bukan sekadar membuat model lebih tahan banting. Ini prediksi yang tajam dan mudah diuji.

2. **Normalisasi loudness menjadi wajib.** `rms` adalah fitur trivial terkuat (AUC 0,698). Normalisasi ke −23 LUFS menghapusnya.

3. **Rencana augmentasi noise perlu ditinjau ulang.** Menambah noise akan menutupi pita tinggi, persis wilayah tempat pintasan MP3 berada. Sebagian "ketahanan noise" yang akan Anda ukur sebenarnya adalah "pintasan yang tertutup noise". Efek keduanya harus dipisahkan dalam ablation.

4. **Metrik seleksi model harus EER pada validation resmi.** Tetapi perhatikan: validation resmi punya 89,6% fake dari MP3, **sama biasnya dengan training**. Jadi validation resmi **bukan proksi yang baik** untuk performa test. Perlu dibuat *validation set kedua* yang bebas MP3 (dari 147 fake WAV di validation + subset real) untuk model selection yang jujur.

 Poin ini penting dan halus: memilih checkpoint berdasarkan validation resmi akan memilih model yang paling baik mengeksploitasi pintasan MP3.

---

## 8. Status verifikasi klaim arsitektur saya

Audit ini juga menguji sebagian klaim di [ARSITEKTUR.md](ARSITEKTUR.md). Hasilnya campuran, dan saya catat apa adanya:

| Klaim saya | Status | Catatan |
|---|---|---|
| Artefak hidup di frekuensi tinggi | ⚠️ **Sebagian keliru** | Perbedaan HF pada FoR didominasi **codec**, bukan sintesis. Pada pasangan yang provenance-nya seimbang (fake WAV vs real WAV) selisihnya hanya 1,17–1,33× |
| Membuang pita > 4 kHz merusak deteksi | ❌ **Terbantah** | Justru **menaikkan** akurasi test (69,4% → 75,4%). Pita tinggi membawa pintasan, bukan sinyal yang dapat digeneralisasi |
| Pintasan durasi/silence ada | ❌ **Terbantah** | Durasi identik 2,000 s; AUC lead-silence 0,5001 |
| Kebocoran duplikat | ❌ **Terbantah** | Nol duplikat |
| Perbedaan LF juga membawa sinyal | ✅ **Didukung** | Pita 0–4 kHz saja mencapai 75,4%; rasio 500–750 Hz = 1,39× |
| Test set terlalu kecil untuk membedakan model | ✅ **Terkonfirmasi & terkuantifikasi** | 1.088 berkas; 1 berkas = 0,092 pp |

Klaim tentang **fase** (§2.3 ARSITEKTUR.md) belum diuji, itu memerlukan eksperimen pengacakan fase yang sudah dirancang di sana, dan tetap menjadi hipotesis yang layak dikejar.

**Pelajaran metodologis:** dua dari enam klaim domain saya terbantah oleh data dalam waktu 20 menit pengukuran. Ini justru alasan mengapa Tier 0 ada di rencana. Jangan percayai analisis mana pun, termasuk analisis saya, sebelum diukur pada data Anda sendiri.

---

## 9. Langkah berikutnya

| # | Aksi | Status |
|---|---|---|
| 1 | Unduh & verifikasi dataset resmi | ✅ selesai |
| 2 | Audit integritas & pintasan | ✅ selesai |
| 3 | Kuantifikasi artefak codec & pergeseran domain | ✅ selesai |
| 4 | Eksperimen split resmi vs acak | ✅ selesai |
| 5 | Bangun pipeline pelatihan bebas-bug (4 model) | ⏳ berikutnya |
| 6 | Baseline jujur pada kedua protokol split | ⏳ |
| 7 | Ablation augmentasi codec (uji prediksi §7.1) | ⏳ |
| 8 | Evaluasi noise unseen + `for-rerec` | ⏳ (`for-rerec` sudah diunduh) |
| 9 | Verifikasi protokol split ref [13]/[19]/[20] | ⏳ **wajib sebelum klaim §5** |

---

*Seluruh angka dapat direproduksi dengan menjalankan keempat skrip probe di folder ini. Dataset `for-2sec` (17.870 berkas) dan `for-rerec` sudah tersedia lokal.*
