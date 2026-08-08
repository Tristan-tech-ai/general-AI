# Analisis Proposal Tesis & Rencana Peningkatan Akurasi

**Judul tesis:** *Analisis Performa Arsitektur Deep Learning Wav2Vec2, AST, HuBERT, dan CNN-LSTM dalam Klasifikasi Suara Deepfake dan Suara Asli*
**Penulis:** Gusti Ayu Putu Kesari Purnama Yani (242011009), S2 Sistem Informasi, ITB STIKOM Bali
**Pembimbing:** Dr. Dandy Pramana Hostiadi, S.Kom., M.T. / Dr. Gede Angga Pradipta, S.T., M.Eng.
**Status dokumen sumber:** Proposal tesis, 89 halaman, lulus ujian proposal 16 Maret 2026
**Dokumen ini:** analisis + rencana teknis peningkatan akurasi, disusun 4 Agustus 2026

---

> **Dokumen ini punya pendamping: [ANALISIS_KRITIS.md](ANALISIS_KRITIS.md).**
> Dokumen yang sedang Anda baca menjawab pertanyaan yang diajukan, *"bagaimana menaikkan akurasi"*, dalam bentuk rencana teknis yang bisa langsung dieksekusi.
> Dokumen pendamping menjawab pertanyaan yang lebih menentukan: kesalahan konseptual di BAB IV, dugaan bug implementasi AST, batas statistik yang mungkin membuat perbandingan empat model tidak dapat difalsifikasi, dan tiga reframing yang menurut saya lebih berharga daripada +1,5 poin akurasi. **Kalau waktu Anda terbatas, baca dokumen itu lebih dulu**, di dalamnya ada rencana satu minggu yang memprioritaskan keduanya.

> **Status epistemik.** Saya membaca PROPOSAL_TESIS.pdf secara lengkap, tetapi **tidak** punya akses ke dataset FoR, kode, notebook, atau log training Anda. Karena itu: klaim tentang isi proposal dapat saya buktikan (disertai nomor halaman), sedangkan semua **estimasi dampak numerik** ("+0,5 s/d +2 poin") adalah perkiraan berbasis pola umum literatur audio anti-spoofing, **urutan prioritas, bukan janji**. Daftar lengkap hal yang tidak saya ketahui ada di [ANALISIS_KRITIS.md §6](ANALISIS_KRITIS.md#6-apa-yang-saya-tidak-tahu).

## 0. Ringkasan Eksekutif (baca ini dulu)

Tiga hal terpenting dari analisis:

1. **Angka "~98%" tidak ada di dalam proposal.** Proposal hanya menyebut *target* ("harapan mencapai akurasi lebih dari 97%", hal. 72) dan *ambang minimum* ("apabila akurasi tidak melebihi capaian penelitian sebelumnya sebesar 94%, maka harus dilakukan penyesuaian", hal. 71). Semua angka akurasi lain dalam dokumen adalah milik penelitian orang lain (state of the art). Jadi ~98% diasumsikan berasal dari hasil eksperimen aktual yang sudah dijalankan setelah proposal disetujui, bukan dari dokumen ini. Lihat [§1 Asumsi](#1-asumsi-yang-diambil).

2. **Ada satu bug konfigurasi yang hampir pasti sedang menahan akurasi Anda: `learning_rate = 0,001` diterapkan seragam ke keempat model** (hal. 68). Untuk Wav2Vec2, HuBERT Large, dan AST, yang semuanya adalah Transformer pra-latih, LR 1e-3 adalah 20–100× terlalu besar dan menyebabkan *catastrophic forgetting*: bobot pra-latih rusak di ratusan langkah pertama. Memperbaikinya adalah perubahan satu baris dengan dampak terbesar dalam seluruh dokumen ini. Lihat [T1-1](#t1-1-perbaiki-learning-rate-per-model-bug-terbesar).

3. **Pada angka ~98% di FoR, prioritas nomor satu bukan lagi menaikkan akurasi, tapi membuktikan bahwa 98% itu nyata.** Dataset Fake-or-Real punya bias struktural yang terkenal (durasi/silence, level loudness, sumber rekaman berbeda antar kelas). Model bisa mencapai 98–99% dengan "menghafal" artefak dataset, lalu jatuh ke 60–70% pada data lain. Karena tesis Anda *justru* mengklaim ketahanan terhadap noise, klaim itu akan menjadi target utama penguji. Lihat [T0](#tier-0--audit-validitas-wajib-sebelum-optimasi-apa-pun).

**Ekspektasi realistis hasil akhir** (dengan asumsi audit T0 lolos):

| Skenario uji | Baseline sekarang (asumsi) | Target setelah rencana ini |
|---|---|---|
| FoR-2sec bersih | ~98% | **99,0–99,6%** (EER < 1%) |
| FoR-2sec + noise seen (SNR 15–30 dB) | belum dilaporkan | **98–99%** |
| FoR-2sec + noise unseen (SNR 0–10 dB) | belum diuji | **93–97%** (ini nilai jual tesis) |
| for-rerec (rekam ulang) | belum diuji | **85–93%** |
| Cross-dataset (ASVspoof 2019 LA / In-the-Wild) | belum diuji | **EER 8–20%**, turun tajam, dan itu wajar & harus dilaporkan |

Ruang naik pada FoR bersih hanya ~1,5 poin (98 → 99,5). **Kontribusi ilmiah yang jauh lebih besar ada di kolom noise unseen dan cross-dataset**, yang sekarang belum dievaluasi sama sekali. Rencana di bawah dirancang untuk mengejar keduanya.

---

## 1. Asumsi yang Diambil

Proposal belum memuat hasil eksperimen, jadi beberapa hal harus diasumsikan. Semua asumsi dicatat eksplisit agar bisa dikoreksi.

| # | Asumsi | Dasar | Dampak bila salah |
|---|---|---|---|
| A1 | Angka ~98% adalah hasil eksperimen aktual pasca-proposal, kemungkinan besar akurasi salah satu model (paling mungkin Wav2Vec2/HuBERT/AST) pada test set FoR-2sec **bersih** | Tidak ada di dokumen; target proposal ">97%" | Jika 98% itu akurasi pada kondisi ber-noise, target akhir bisa dinaikkan lebih agresif |
| A2 | Splitting 60/20/20 dilakukan dengan **random split** atas gabungan data, bukan memakai folder `training/validation/testing` bawaan FoR | Proposal hal. 55 menyebut proporsi 60/20/20, sedangkan rilis resmi FoR sudah memiliki partisi sendiri dengan proporsi berbeda | Jika ternyata random split → ada risiko kebocoran serius (lihat T0-1) |
| A3 | Split **tidak** speaker-disjoint (tidak ada kontrol agar pembicara yang sama tak muncul di train & test) | Tidak disebutkan di proposal | Bila benar, 98% sebagian berasal dari hafalan identitas pembicara |
| A4 | Optimizer yang dipakai adalah Adam/AdamW (proposal hanya menulis "optimizer" tanpa nama), loss = cross-entropy | Hal. 68 menyebut "penggunaan optimizer" tanpa spesifikasi | Rendah, rekomendasi tetap AdamW |
| A5 | Wav2Vec2 = varian **Base** (12 layer, dinyatakan hal. 61), HuBERT = **Large** (24 layer, hal. 65), AST = varian standar 12 layer pra-latih AudioSet (hal. 63) | Dinyatakan di proposal |, |
| A6 | Augmentasi noise DNC (SNR 15–30 dB) hanya diterapkan ke **data latih**, dan test set ber-noise dibuat dengan sumber noise yang sama | Hal. 57 & 55 | Jika sumber noise train = test → hasil "tahan noise" adalah hasil ber-noise *seen*, bukan bukti generalisasi |
| A7 | Komputasi = Google Colab (batasan penelitian butir 2), artinya GPU T4/L4/A100 dengan batas sesi | Hal. 6 | Membatasi opsi model besar & jumlah run; rencana disesuaikan |
| A8 | Seluruh audio dipotong/dipad ke 2 detik @16 kHz mono, sesuai for-2sec | Hal. 55–56 |, |

> **Tindakan:** verifikasi A2, A3, dan A6 lebih dulu, ketiganya menentukan apakah 98% valid.

---

## 2. Ringkasan Isi Proposal

### 2.1 Kerangka penelitian

| Aspek | Isi proposal |
|---|---|
| **Masalah** | Rekaman suara dunia nyata jarang bersih dari noise; detektor deepfake harus tetap andal pada noise lingkungan, kompresi, dan rekam-ulang |
| **Rumusan masalah** | Bagaimana kinerja Wav2Vec2, AST, HuBERT Large, dan CNN-LSTM dalam klasifikasi suara asli vs deepfake **pada kondisi bergangguan noise** |
| **Tujuan** | Mengevaluasi & membandingkan performa keempat arsitektur dalam kondisi ber-noise, dengan protokol pengujian seragam |
| **Kebaruan yang diklaim** | Perbandingan 4 paradigma representasi audio pada protokol identik + skenario noise bertingkat, penelitian sebelumnya memakai dataset/protokol berbeda sehingga tak bisa dibandingkan langsung |
| **Batasan** | Google Colab; hanya FoR varian for-2sec; metrik = akurasi, presisi, recall, F1, ROC, AUC, confusion matrix |
| **Jadwal** | Maret–Agustus 2026 (pelatihan Mar–Apr, pelatihan model pra-latih Apr–Jun, evaluasi Apr–Jun, penulisan Jul, revisi Ags) |

### 2.2 Dataset

- **Fake-or-Real (FoR)**, APTLY Lab, York University. >195.000 ujaran, real (rekaman manusia) vs fake (berbagai sistem TTS).
- Varian tersedia: `for-original`, `for-norm`, `for-2sec`, `for-rerec`.
- **Yang dipakai: `for-2sec`**, potongan 2 detik, sudah dinormalisasi & seimbang antar kelas.
- Split: **60% train / 20% validation / 20% test**, konsisten di semua model.
- Test set juga dipakai ulang dalam skenario tambahan dengan penambahan noise.

### 2.3 Preprocessing (hal. 56–58)

1. **Pembacaan audio** → konversi ke mono, resample ke **16 kHz** (2 detik = 32.000 sampel).
2. **Normalisasi amplitudo** → peak normalization (skala terhadap nilai maksimum absolut per berkas).
3. **Augmentasi** (hanya data latih) → penambahan noise latar non-ucapan dari **DNC: Dataset for Noise Classification** (4.377 rekaman; *mechanic* 1.545, *melodic* 1.427, *quiet* 1.405). Noise di-mono-kan, di-resample 16 kHz, dipotong/di-loop ke 2 detik, lalu dicampur pada **SNR 15–30 dB**. Label kelas dipertahankan.
4. **Konversi log-Mel spectrogram** → hanya untuk AST dan CNN-LSTM. Wav2Vec2 & HuBERT memakai waveform langsung.

### 2.4 Arsitektur (hal. 60–67)

| Model | Input | Alur | Paradigma yang diwakili |
|---|---|---|---|
| **Wav2Vec 2.0 Base** | Waveform mentah | CNN feature encoder → quantization (product quantization + Gumbel-Softmax) → masking laten → **12 layer** Transformer encoder (multi-head self-attention) → head klasifikasi | SSL waveform berbasis *contrastive learning* |
| **AST** | Log-Mel spectrogram | Patch embedding → token + [CLS] + positional embedding → **12 layer** Transformer encoder → representasi [CLS] → head klasifikasi | Transformer murni pada domain waktu–frekuensi, tanpa konvolusi |
| **HuBERT Large** | Waveform mentah | Jalur 1: Acoustic Unit Discovery → pseudo-label. Jalur 2: **7 layer Conv1D** encoder → masking → **24 layer** Transformer → prediksi unit akustik pada segmen ter-mask | SSL waveform berbasis *masked-unit prediction* |
| **CNN-LSTM** | Log-Mel spectrogram | Konvolusi + aktivasi + pooling → feature map disusun sebagai urutan waktu → LSTM (forget/input/output gate) → hidden state akhir → softmax | Hybrid spectral-temporal konvensional (baseline) |

### 2.5 Konfigurasi pelatihan (hal. 68), **seragam untuk keempat model**

| Parameter | Nilai di proposal |
|---|---|
| Batch size | 32 |
| Learning rate | **0,001** |
| Epoch | 20 |
| Activation | GeLU |
| Regularisasi | Dropout (nilai tidak disebutkan) |
| Optimizer | Tidak disebutkan namanya |
| Scheduler / warmup | Tidak ada |
| Early stopping | Tidak ada |
| Mixed precision | Tidak ada |
| Class weight / loss khusus | Tidak ada |

> Alasan yang diberikan: keseragaman agar perbedaan hasil murni mencerminkan arsitektur. **Niatnya benar secara ilmiah, tapi implementasinya justru merusak validitas perbandingan**, lihat [§3 G-2](#g-2-lr-seragam-adalah-perbandingan-yang-tidak-adil).

### 2.6 Evaluasi (hal. 70–72)

Akurasi, presisi, recall, F1 (scikit-learn `classification_report`), kurva ROC, AUC (trapezoidal rule), confusion matrix. **Tidak ada EER, tidak ada t-DCF, tidak ada kalibrasi, tidak ada threshold tuning, tidak ada cross-validation.**

### 2.7 Semua angka akurasi dalam proposal, dan di mana "98%" seharusnya muncul

| Sumber | Metode | Dataset | Angka |
|---|---|---|---|
| Ref [13] (rujukan utama) | SVM + MFCC/spectral centroid/roll-off/ZCR + PCA | FoR | **93,50%** |
| Ref [19] MFAAN | CNN multi-feature fusion (MFCC+LFCC+Chroma-STFT) | FoR | **94,47%** |
| Ref [20] Hybrid CNN-LSTM | CNN-LSTM (MFCC + spectrogram) | FoR | **94,7%** (CNN saja 87,3%; LSTM saja 82,7%) |
| Ref [14] Continuous Learning AST | AST + augmentasi + XGBoost | ASVspoof2019 / FakeAVCeleb / In-the-Wild | EER 4,06% / **92,6%** akurasi, AUC 99,9% / EER In-the-Wild 31,14% → 8,01% |
| Ref [15] CNN-Transformer | Hybrid + SpecAugment + pitch/speed | ASVspoof 2019 LA | **91,47%** |
| Ref [16] AST fine-tuned | MIT/ast, MattyB95 | HSAD | **~97%** |
| Ref [17] | MFCC + CNN/LSTM/VGG-16 | 200.052 sampel | **93%** (VGG-16) |
| Ref [21] XMUspeech | HuBERT feature extractor | ASVspoof 5 | EER 20,45% (≈ **79,6%**) |
| Ref [22] Zero-Shot to Zero-Lies | Wav2Vec2-Base fine-tune | BanglaFake | **65,28%** |
| Ref [18] | Wav2Vec2 + attentive stat pooling + layer weighting | ADD 2022 | EER 21,71% / 16,59% |
| **Proposal ini, ambang minimum** |, | FoR-2sec | **> 94%** (hal. 71) |
| **Proposal ini, target** |, | FoR-2sec | **> 97%** (hal. 72) |

**Konfirmasi: angka ~98% tidak muncul di dokumen mana pun dalam proposal.** Nilai tertinggi yang disebut adalah target ">97%" dan referensi eksternal "~97%" pada dataset HSAD. Sesuai asumsi A1, ~98% diperlakukan sebagai hasil eksperimen aktual Anda.

**Implikasi penting untuk penulisan tesis:** 98% pada FoR **sudah melampaui semua baseline FoR di tabel state of the art** (93,50 / 94,47 / 94,7). Artinya syarat kelulusan metodologis (hal. 71: harus > 94%) sudah terpenuhi. Kontribusi tambahan sekarang harus digeser dari "akurasi lebih tinggi" ke **"akurasi tinggi yang terbukti tidak semu"**, yaitu robustness dan generalisasi.

---

## 3. Diagnosis: Celah yang Membatasi (atau Memalsukan) Akurasi

Diurutkan berdasarkan keparahan. Kode G = *gap*.

| Kode | Celah | Bukti di proposal | Keparahan |
|---|---|---|---|
| **G-1** | Learning rate 1e-3 untuk model Transformer pra-latih | hal. 68 | 🔴 Kritis |
| **G-2** | LR/optimizer/epoch seragam → perbandingan antar-arsitektur tidak adil | hal. 68 | 🔴 Kritis (validitas ilmiah) |
| **G-3** | Split 60/20/20 kemungkinan random & tidak speaker-disjoint → risiko kebocoran | hal. 55 | 🔴 Kritis (validitas hasil) |
| **G-4** | Noise train dan noise test dari sumber yang sama (DNC) → klaim "tahan noise" hanya berlaku *seen noise* | hal. 55, 57 | 🔴 Kritis (validitas klaim inti) |
| **G-5** | Tidak ada EER, metrik standar de facto di bidang anti-spoofing | hal. 70–72 | 🟠 Tinggi |
| **G-6** | Tidak ada evaluasi cross-dataset | seluruh BAB IV | 🟠 Tinggi |
| **G-7** | Rentang SNR uji terlalu sempit & terlalu mudah (15–30 dB) | hal. 57 | 🟠 Tinggi |
| **G-8** | Tidak ada early stopping / model selection; 20 epoch tetap | hal. 68–69 | 🟠 Tinggi |
| **G-9** | Peak amplitude normalization dapat memperkuat artefak level & menghapus sinyal yang berguna | hal. 56–57 | 🟡 Sedang |
| **G-10** | Augmentasi hanya noise injection, tidak ada SpecAugment, RIR, codec, MixUp, pitch/speed | hal. 57 | 🟡 Sedang |
| **G-11** | Head klasifikasi = mean/[CLS] pooling sederhana; tidak ada attentive pooling atau layer-weighting | hal. 61–67 | 🟡 Sedang |
| **G-12** | Tidak ada ensembling meski 4 model sudah dilatih (aset gratis yang tidak dipakai) | hal. 67 | 🟡 Sedang |
| **G-13** | Tidak ada threshold tuning / kalibrasi probabilitas | hal. 70–72 | 🟡 Sedang |
| **G-14** | Tidak ada cross-validation atau multi-seed → tidak ada estimasi variansi | hal. 69 | 🟡 Sedang |
| **G-15** | Tidak ada mixed precision / gradient accumulation → boros kuota Colab | hal. 68 | 🟢 Rendah |
| **G-16** | `for-rerec` tersedia tapi tidak dipakai, padahal ini persis skenario "rekam ulang" yang disebut di latar belakang | hal. 4, 55 | 🟡 Sedang (peluang terlewat) |

### G-2: LR seragam adalah perbandingan yang tidak adil

Proposal beralasan konfigurasi seragam menjaga "kesetaraan kondisi eksperimen". Secara metodologis ini keliru. Setiap arsitektur punya *rezim optimasi* berbeda:

- CNN-LSTM dilatih **dari nol** → LR 1e-3 memang wajar.
- Wav2Vec2 / HuBERT / AST **sudah pra-latih** → LR 1e-3 menghancurkan bobot pra-latih.

Hasilnya: yang Anda ukur bukan "arsitektur mana yang lebih baik", tapi "arsitektur mana yang paling toleran terhadap LR 1e-3". Perbandingan yang adil adalah **tuning LR per model dengan budget pencarian yang sama** (mis. masing-masing 6 percobaan), lalu bandingkan performa terbaik masing-masing. Ini adalah praktik standar dan harus ditulis eksplisit di metodologi revisi.

---

## 4. Rencana Peningkatan Akurasi, Bertingkat

Setiap item punya format: **Masalah → Aksi konkret → Dampak → Risiko/biaya**.
Kode: T0 = audit wajib, T1 = high-impact/low-effort, T2 = high-impact/medium-effort, T3 = eksperimental/high-effort.

---

## TIER 0, Audit Validitas (WAJIB sebelum optimasi apa pun)

> Jika Anda melewatkan tier ini dan langsung optimasi, Anda berisiko menghabiskan 3 bulan menaikkan angka yang ternyata tidak sah. Total waktu tier ini: **2–3 hari**.

### T0-1: Audit kebocoran data (data leakage)

**Masalah:** FoR mengandung banyak potongan yang berasal dari ujaran/kalimat/pembicara yang sama. Random split 60/20/20 dapat menempatkan potongan dari rekaman induk yang sama di train dan test sekaligus → akurasi test membengkak secara artifisial.

**Aksi:**

```python
# 1. Cek duplikat & near-duplicate berbasis hash audio
import hashlib, soundfile as sf, numpy as np, collections
def audio_md5(p):
 x, sr = sf.read(p); return hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest()
h = collections.defaultdict(list)
for p in all_files: h[audio_md5(p)].append(p)
dupes = {k: v for k, v in h.items() if len(v) > 1}
print("duplikat eksak:", len(dupes))

# 2. Cek near-duplicate lintas split via embedding cosine similarity
# (pakai model speaker-verification pra-latih, mis. speechbrain ECAPA-TDNN)
# Flag semua pasangan train-test dengan cos_sim > 0.95

# 3. Cek overlap nama file / prefix pembicara antara train dan test
```

**Aksi korektif:** ganti ke **partisi resmi FoR** (`for-2seconds/training`, `/validation`, `/testing`) yang memang dirancang bebas overlap. Jika tetap ingin 60/20/20, lakukan **group split** dengan `GroupShuffleSplit` di mana `group` = ID pembicara / ID rekaman induk.

**Dampak:** Bukan menaikkan akurasi, kemungkinan besar **menurunkannya** (perkiraan 98% → 94–97%). Tapi ini menyelamatkan tesis dari pertanyaan penguji yang mematikan.
**Risiko:** Angka turun dan harus dijelaskan. Justru ini bahan pembahasan yang bagus.

### T0-2: Uji artefak non-akustik (shortcut detection)

**Masalah:** Beberapa dataset deepfake bisa diklasifikasikan >90% hanya dari durasi silence di awal, energi rata-rata, atau panjang file, tanpa melihat konten suara sama sekali. Bila FoR-2sec Anda begitu, 98% tidak bermakna.

**Aksi, jalankan 4 tes "kontrol negatif" ini:**

| Tes | Cara | Interpretasi |
|---|---|---|
| **Fitur trivial** | Latih Logistic Regression / Random Forest hanya dari 5 fitur: durasi non-silence, RMS energy, peak amplitude, jumlah leading zeros, DC offset | Akurasi > 75% ⇒ ada shortcut serius |
| **Shuffle waktu** | Acak urutan frame spectrogram, latih ulang | Akurasi masih tinggi ⇒ model tak memakai info temporal |
| **Low-pass 4 kHz** | Buang seluruh konten > 4 kHz, latih ulang | Akurasi anjlok drastis ⇒ model bergantung artefak pita tinggi TTS (mungkin sah, tapi rapuh terhadap kompresi) |
| **Label acak** | Acak label train, latih penuh | Akurasi test harus ≈ 50%. Jika > 60% ⇒ ada kebocoran struktural |

**Aksi korektif bila ada shortcut:** tambahkan normalisasi yang menetralkan artefak tersebut, trim silence konsisten (`librosa.effects.trim(top_db=30)`), **ganti peak-normalization dengan loudness normalization ke -23 LUFS** (`pyloudnorm`), dan pad/crop ke panjang tetap dengan cara identik untuk kedua kelas.

**Dampak:** Validitas. Juga menjadi satu subbab analisis yang kuat di tesis.
**Risiko:** Rendah, murah, wajib.

### T0-3: Definisi ulang protokol noise (memisahkan *seen* vs *unseen*)

**Masalah (G-4):** Sekarang noise latih dan noise uji sama-sama dari DNC. Model bisa menghafal 4.377 rekaman noise itu.

**Aksi, split sumber noise secara disjoint:**

| Himpunan | Sumber noise | SNR | Fungsi |
|---|---|---|---|
| Train | DNC subset A (70% file, tiap kategori) | 5–30 dB (diperluas dari 15–30) | Belajar |
| Val | DNC subset B (30% file) | 5–30 dB | Model selection |
| **Test-seen** | DNC subset B | 15, 20, 25, 30 dB | Perbandingan dengan protokol lama |
| **Test-unseen** | **MUSAN** (noise/music/babble) + **DEMAND** + **ESC-50** | **0, 5, 10, 15, 20 dB** | **Klaim inti tesis** |
| **Test-channel** | Kompresi MP3 64k/32k, µ-law, band-limit 8 kHz |, | Ketahanan kanal |
| **Test-reverb** | RIR dari **OpenSLR RIR/Noises (SLR28)** | RT60 0,2–0,8 s | Far-field |

**Dampak:** Mengubah klaim dari "tahan noise" (lemah) menjadi "tahan noise yang belum pernah dilihat" (kuat, layak publikasi).
**Risiko:** Angka pada noise unseen akan lebih rendah, itu **hasil**, bukan kegagalan.

---

## TIER 1, High Impact / Low Effort (kerjakan minggu 1–2)

### T1-1: Perbaiki learning rate per model (bug terbesar)

**Masalah (G-1):** LR 1e-3 untuk Wav2Vec2/HuBERT/AST menghancurkan bobot pra-latih.

**Aksi, konfigurasi pengganti:**

| Model | LR encoder | LR head | Optimizer | Warmup | Scheduler | Epoch | Batch (efektif) |
|---|---|---|---|---|---|---|---|
| Wav2Vec2 Base | **1e-5** | 1e-4 | AdamW (wd 0,01) | 10% steps | cosine decay | 10–15 | 32 |
| HuBERT Large | **5e-6** | 5e-5 | AdamW (wd 0,01) | 10% steps | cosine decay | 8–12 | 16 (+grad accum ×2) |
| AST | **2e-5** | 1e-4 | AdamW (wd 0,01) | 10% steps | cosine decay | 10–15 | 32 |
| CNN-LSTM | **1e-3** (tetap) |, | AdamW |, | ReduceLROnPlateau | 40–60 | 64 |

```python
from transformers import get_cosine_schedule_with_warmup
optim = torch.optim.AdamW([
 {"params": model.encoder.parameters(), "lr": 1e-5},
 {"params": model.head.parameters(), "lr": 1e-4},
], weight_decay=0.01)
sched = get_cosine_schedule_with_warmup(optim, int(0.1*total_steps), total_steps)
```

**Dampak:** **+1 sampai +8 poin akurasi** pada model SSL, tergantung seberapa rusak run sekarang. Ini item dengan rasio dampak/usaha tertinggi di seluruh dokumen.
**Risiko:** Nyaris nol. Biaya: satu run ulang per model.

### T1-2: Early stopping + model selection berbasis validasi

**Masalah (G-8):** 20 epoch tetap tanpa penyimpanan checkpoint terbaik → yang dievaluasi adalah model epoch terakhir, yang sering sudah overfit.

**Aksi:**
```python
# Monitor: val_EER (bukan val_accuracy), lebih sensitif di rezim akurasi tinggi
# patience = 5, restore_best_weights = True, save_top_k = 1
```
Simpan checkpoint berdasarkan **EER validasi terendah**, bukan akurasi tertinggi. Pada akurasi 98%, val_accuracy sudah jenuh dan tidak lagi membedakan checkpoint; EER masih punya resolusi.

**Dampak:** +0,3–1,5 poin, dan menghemat waktu GPU.
**Risiko:** Nol.

### T1-3: Tambahkan EER, min-tDCF, dan kurva DET

**Masalah (G-5):** Bidang audio anti-spoofing memakai EER sebagai metrik utama. Tesis tanpa EER akan terlihat tidak mengikuti literatur, dan tabel state of the art Anda sendiri sudah penuh angka EER (ref [14], [18], [21]), sehingga hasil Anda tidak bisa dibandingkan langsung dengan rujukan sendiri.

**Aksi:**
```python
import numpy as np
from sklearn.metrics import roc_curve

def compute_eer(y_true, y_score):
 fpr, tpr, thr = roc_curve(y_true, y_score)
 fnr = 1 - tpr
 i = np.nanargmin(np.abs(fnr - fpr))
 eer = (fpr[i] + fnr[i]) / 2
 return eer, thr[i] # thr[i] = threshold EER

# Laporkan juga: FRR@FAR=1%, FAR@FRR=1%, dan kurva DET (skala probit)
```
Tambahkan **min t-DCF** bila ingin sejajar dengan protokol ASVspoof.

**Dampak:** Tidak menaikkan akurasi, tapi **menaikkan kualitas dan kredibilitas tesis secara signifikan**, dan menurut aturan sendiri (hal. 71) hasil harus dibandingkan dengan penelitian sebelumnya, yang mustahil tanpa EER.
**Risiko:** Nol. Effort: ~2 jam.

### T1-4: Threshold tuning (jangan pakai 0,5)

**Masalah (G-13):** Threshold default 0,5 hampir tidak pernah optimal, terutama setelah augmentasi mengubah distribusi skor.

**Aksi:**
1. Ambil threshold dari **validation set** (bukan test) dengan tiga kriteria: (a) maksimum Youden's J = TPR−FPR, (b) titik EER, (c) maksimum F1 pada kurva Precision-Recall.
2. Terapkan threshold terkunci itu ke test set.
3. Laporkan akurasi pada threshold 0,5 **dan** threshold teroptimasi, perbedaannya menarik untuk dibahas.

**Dampak:** +0,2–1,0 poin akurasi, gratis.
**Risiko:** Harus disiplin: threshold **wajib** dipilih dari validasi. Memilih dari test = kebocoran.

### T1-5: Label smoothing + dropout yang tepat

**Aksi:**
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.05) # 0.05–0.1
# Dropout: klasifier head 0.2–0.3; SpecAugment sudah bertindak sebagai dropout input
# Untuk HuBERT/Wav2Vec2: aktifkan layerdrop=0.05 (bawaan HF config)
```
Label smoothing juga **memperbaiki kalibrasi**, bonus untuk T2-6.

**Dampak:** +0,1–0,5 poin, kalibrasi lebih baik, overconfidence berkurang.
**Risiko:** Terlalu besar (>0,15) justru menurunkan akurasi. Mulai dari 0,05.

### T1-6: Mixed precision + gradient checkpointing

**Aksi:**
```python
scaler = torch.amp.GradScaler()
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
 loss = criterion(model(x), y)
model.gradient_checkpointing_enable() # untuk HuBERT Large di Colab
```

**Dampak:** ~2× lebih cepat, memori ~40% lebih hemat → HuBERT Large muat di T4, dan **jumlah eksperimen yang bisa dijalankan berlipat**. Ini yang membuat sisa rencana feasible dalam batas Colab.
**Risiko:** Rendah. Gunakan bfloat16 bila GPU mendukung (A100/L4); fp16 pada T4 kadang butuh loss scaling hati-hati.

### T1-7: Multi-seed (minimal 3 seed per konfigurasi)

**Masalah (G-14):** Satu run tidak memberi tahu apakah selisih 98,2% vs 97,9% itu nyata atau derau.

**Aksi:** jalankan tiap konfigurasi final dengan seed {42, 1337, 2024}, laporkan **mean ± std**. Untuk perbandingan antar model, gunakan **McNemar's test** (data berpasangan pada test set yang sama) atau paired t-test antar seed.

**Dampak:** Validitas statistik. Tanpa ini, klaim "model X lebih baik dari Y" tidak terbukti.
**Risiko:** 3× waktu komputasi, dikompensasi oleh T1-6.

---

## TIER 2, High Impact / Medium Effort (minggu 3–8)

### T2-1: Augmentasi lengkap (naik dari 1 teknik menjadi 7)

**Masalah (G-10):** Sekarang hanya noise injection. Literatur di tabel state of the art Anda sendiri (ref [14], [15], [18]) semuanya memakai augmentasi majemuk.

**Aksi, pipeline augmentasi bertingkat, diterapkan on-the-fly pada data latih:**

| # | Teknik | Parameter rekomendasi | Prob. | Tujuan |
|---|---|---|---|---|
| 1 | **Additive noise** (perluas yang ada) | MUSAN + DNC + ESC-50, SNR **0–30 dB** (uniform) | 0,5 | Noise lingkungan |
| 2 | **RIR / reverberation** | OpenSLR SLR28, RT60 0,1–1,0 s | 0,3 | Far-field, ruang |
| 3 | **SpecAugment** | 2 freq mask (F=15 dari 128 mel), 2 time mask (T=20 frame) | 0,5 | Regularisasi domain T-F |
| 4 | **Pitch shift** | ±2 semitone | 0,2 | Variasi pembicara |
| 5 | **Speed / tempo perturb** | faktor {0,9; 1,0; 1,1} | 0,3 | Variasi kecepatan bicara |
| 6 | **Codec / band-limit** | MP3 32–128 kbps, µ-law, low-pass 4/8 kHz | 0,3 | Ketahanan kanal & telepon |
| 7 | **Gain / clipping** | ±10 dB gain, clipping ringan 0,5% | 0,2 | Variasi level rekaman |

```python
from audiomentations import (Compose, AddBackgroundNoise, ApplyImpulseResponse,
 PitchShift, TimeStretch, Mp3Compression, Gain)
augment = Compose([
 AddBackgroundNoise(sounds_path="musan/noise", min_snr_db=0, max_snr_db=30, p=0.5),
 ApplyImpulseResponse(ir_path="RIRS_NOISES/simulated_rirs", p=0.3),
 PitchShift(min_semitones=-2, max_semitones=2, p=0.2),
 TimeStretch(min_rate=0.9, max_rate=1.1, leave_length_unchanged=True, p=0.3),
 Mp3Compression(min_bitrate=32, max_bitrate=128, p=0.3),
 Gain(min_gain_db=-10, max_gain_db=10, p=0.2),
])
# SpecAugment diterapkan setelah konversi log-Mel (untuk AST & CNN-LSTM)
# Untuk Wav2Vec2/HuBERT: gunakan mask_time_prob=0.05, mask_feature_prob=0.004 bawaan HF
```

⚠️ **Peringatan penting untuk tugas anti-spoofing:** pitch shift dan time stretch dilakukan lewat resampling/phase-vocoder yang **menambah artefak sintetis sendiri**. Ada risiko nyata ini "mengajari" model bahwa artefak pemrosesan = fake, sehingga real yang teraugmentasi salah dilabeli. **Mitigasi wajib: terapkan augmentasi #4 dan #5 dengan probabilitas sama persis pada kedua kelas**, dan jalankan ablation untuk memastikan keduanya tidak menurunkan performa. Jika ablation menunjukkan penurunan, buang #4 dan #5, #1, #2, #3, #6 adalah yang paling aman dan paling berdampak.

**Dampak:** Pada FoR bersih **+0,2–0,8 poin**; pada noise unseen **+5 sampai +15 poin**. Ini item terbesar untuk klaim inti tesis.
**Risiko:** Sedang, lihat peringatan di atas. Wajib ablation per teknik.

### T2-2: MixUp / manifold MixUp

**Aksi:**
```python
lam = np.random.beta(0.2, 0.2) # alpha kecil untuk audio biner
x = lam * x_a + (1 - lam) * x_b
loss = lam * crit(out, y_a) + (1 - lam) * crit(out, y_b)
```
Untuk model SSL, terapkan mixup di level **embedding** (manifold mixup) alih-alih waveform, mencampur waveform real+fake secara fisik menghasilkan sinyal yang secara akustik aneh.

**Dampak:** +0,2–0,7 poin, kalibrasi lebih baik, robustness naik.
**Risiko:** Pada tugas deteksi artefak, mixup waveform bisa kontraproduktif (mencampur artefak). Uji dulu di level embedding.

### T2-3: Attentive statistical pooling + layer-wise feature weighting

**Masalah (G-11):** Mean pooling / [CLS] membuang informasi. Referensi [18] di proposal Anda sendiri memakai persis teknik ini.

**Aksi, ganti head klasifikasi:**

```python
class AttentiveStatsPool(nn.Module):
 """Menghasilkan [mean; std] berbobot attention, 2x dim, jauh lebih informatif."""
 def __init__(self, d, bottleneck=128):
 super().__init__()
 self.att = nn.Sequential(
 nn.Conv1d(d, bottleneck, 1), nn.ReLU(), nn.BatchNorm1d(bottleneck),
 nn.Tanh(), nn.Conv1d(bottleneck, d, 1), nn.Softmax(dim=2))
 def forward(self, x): # x: (B, D, T)
 w = self.att(x)
 mu = (x * w).sum(2)
 sg = torch.sqrt(((x**2) * w).sum(2) - mu**2 + 1e-9)
 return torch.cat([mu, sg], dim=1)

class LayerWeighting(nn.Module):
 """Bobot belajar untuk tiap layer Transformer, layer tengah sering paling
 informatif untuk deteksi artefak, bukan layer terakhir."""
 def __init__(self, n_layers):
 super().__init__(); self.w = nn.Parameter(torch.zeros(n_layers))
 def forward(self, hs): # hs: tuple of (B, T, D)
 a = torch.softmax(self.w, 0)
 return sum(a[i] * h for i, h in enumerate(hs))
```

Pipeline baru: `SSL encoder (output_hidden_states=True) → LayerWeighting → AttentiveStatsPool → BN → Linear(2D→256) → ReLU → Dropout(0.2) → Linear(256→2)`

**Bonus analitik:** plot bobot layer yang dipelajari. Jika layer 6–9 dari HuBERT Large mendapat bobot terbesar, itu temuan yang bisa dibahas, layer awal menangkap artefak akustik lokal, layer akhir terlalu fonetik/semantik.

**Dampak:** **+0,5–2,0 poin**, terutama pada kondisi ber-noise. Salah satu item paling efisien.
**Risiko:** Rendah. Menambah ~1M parameter.

### T2-4: Unfreezing bertahap + LLRD (Layer-wise Learning Rate Decay)

**Aksi, jadwal tiga fase untuk Wav2Vec2/HuBERT/AST:**

| Fase | Epoch | Yang dilatih | LR encoder |
|---|---|---|---|
| 1 | 1–2 | **Hanya head** (encoder dibekukan penuh) | 0 |
| 2 | 3–5 | Head + **6 layer teratas** | 1e-5 |
| 3 | 6–12 | Seluruh encoder (kecuali CNN feature extractor, tetap beku) | 5e-6 dengan LLRD |

```python
def llrd_param_groups(model, base_lr=1e-5, decay=0.9):
 """Layer bawah dapat LR lebih kecil, melindungi fitur akustik umum."""
 groups, layers = [], model.encoder.layers
 n = len(layers)
 for i, layer in enumerate(layers):
 groups.append({"params": layer.parameters(),
 "lr": base_lr * (decay ** (n - 1 - i))})
 groups.append({"params": model.head.parameters(), "lr": base_lr * 10})
 return groups
```

**Catatan penting:** **selalu bekukan CNN feature extractor** pada Wav2Vec2/HuBERT (`model.freeze_feature_encoder()`). Fine-tuning bagian ini hampir selalu merusak dan merupakan penyebab umum kegagalan.

**Dampak:** +0,3–1,5 poin, dan **konvergensi jauh lebih stabil**, mengurangi variansi antar seed.
**Risiko:** Menambah kompleksitas kode. Effort ~1 hari.

### T2-5: Ensemble 4 model (aset gratis yang belum dipakai)

**Masalah (G-12):** Anda melatih 4 model dengan 4 paradigma representasi yang berbeda. Itu kondisi ideal untuk ensembling, error mereka kemungkinan besar tidak berkorelasi.

**Aksi, tiga level, kerjakan berurutan:**

**(a) Score fusion, 30 menit kerja, tanpa training ulang:**
```python
# Wajib: kalibrasi skor tiap model dulu (T2-6), baru fusi
p_ens = (w1*p_w2v + w2*p_ast + w3*p_hubert + w4*p_cnnlstm)
# Bobot dicari lewat grid/Nelder-Mead pada VALIDATION set, bukan test
# Alternatif yang sering lebih baik: rata-rata logit, atau rank averaging
```

**(b) Stacking, 2 jam:** latih Logistic Regression atau LightGBM di atas vektor `[p1, p2, p3, p4]` (atau embedding gabungan) menggunakan prediksi **out-of-fold** dari validation set.

**(c) Feature-level fusion, 1–2 hari:** gabungkan embedding pooled dari Wav2Vec2 + HuBERT + AST (concat → 3×D), lalu latih head bersama. Lebih kuat tapi lebih berat.

**Analisis pendukung yang harus dilaporkan:** matriks korelasi error antar model + diagram Venn sampel yang salah diklasifikasi. Jika korelasi rendah (<0,6), ensemble akan sangat efektif, dan ini temuan bagus untuk pembahasan.

**Dampak:** **+0,5–1,5 poin** di atas model terbaik pada data bersih; **+2–6 poin** pada kondisi ber-noise (karena model berbeda gagal pada noise berbeda). Cara paling andal menembus 99%.
**Risiko:** Rendah untuk (a). Perlu dijelaskan bahwa ensemble adalah *kontribusi tambahan*, bukan mengaburkan perbandingan per-arsitektur yang menjadi tujuan utama tesis, laporkan keduanya secara terpisah.

### T2-6: Kalibrasi probabilitas

**Masalah:** Model deep learning yang overfit menghasilkan probabilitas overconfident (0,999 padahal salah). Ini merusak score fusion di T2-5 dan membuat threshold tuning tidak stabil.

**Aksi:**
```python
# Temperature scaling, 1 parameter, dioptimasi pada validation set
class TempScale(nn.Module):
 def __init__(self): super().__init__(); self.T = nn.Parameter(torch.ones(1)*1.5)
 def forward(self, logits): return logits / self.T
# Optimasi T dengan LBFGS meminimalkan NLL pada validation set
```
Laporkan **ECE (Expected Calibration Error)** dan **reliability diagram** sebelum vs sesudah. Bandingkan juga dengan Platt scaling dan isotonic regression.

**Dampak:** Tidak langsung menaikkan akurasi, tapi **prasyarat agar T2-5 dan T1-4 bekerja optimal**. Menambah satu subbab evaluasi yang jarang ada di tesis S2 → nilai plus.
**Risiko:** Nol.

### T2-7: Front-end denoising & robustness terhadap noise

Ini fokus utama tesis, jadi perlu perlakuan tersendiri. **Uji tiga strategi dan bandingkan**, hasil perbandingannya sendiri adalah kontribusi.

| Strategi | Cara | Ekspektasi |
|---|---|---|
| **S1: Tanpa denoising, augmentasi agresif** | Model belajar langsung dari audio ber-noise (T2-1) | **Biasanya menang.** Denoising menghapus artefak halus yang justru menjadi sinyal deteksi deepfake |
| **S2: Denoising enhancement** | Pre-process dengan **DeepFilterNet3** / **Demucs (htdemucs)** / spectral gating (`noisereduce`) sebelum masuk model | Sering **menurunkan** akurasi pada audio bersih karena enhancement memperkenalkan artefaknya sendiri |
| **S3: Hybrid dual-branch** | Dua cabang paralel: (a) audio asli, (b) audio ter-denoise. Concat embedding → head bersama | Paling menjanjikan; model belajar sendiri kapan mempercayai cabang mana |

**Fitur robust tambahan yang layak diuji** (untuk cabang CNN-LSTM khususnya):
- **LFCC** (Linear Frequency Cepstral Coefficients), di literatur anti-spoofing LFCC **konsisten mengungguli MFCC** karena artefak TTS banyak berada di frekuensi tinggi yang justru dikompresi oleh skala Mel. Referensi [19] (MFAAN) di proposal Anda memakai LFCC.
- **CQCC** (Constant-Q Cepstral Coefficients), baseline klasik ASVspoof.
- **Per-Channel Energy Normalization (PCEN)** sebagai pengganti log, dirancang khusus untuk ketahanan terhadap noise dan variasi kanal.
- **CMVN** (Cepstral Mean & Variance Normalization) pada level utterance, menghilangkan bias kanal.

```python
# PCEN, pengganti log-Mel yang jauh lebih tahan noise
import librosa
S = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128, power=1)
pcen = librosa.pcen(S * (2**31), sr=16000, gain=0.98, bias=2, power=0.5, time_constant=0.4)
```

**Dampak:** S1 + LFCC/PCEN diperkirakan **+3 sampai +10 poin pada noise unseen**. S3 berpotensi lebih tinggi lagi.
**Risiko:** Sedang. S2 berpotensi menurunkan performa, **tapi itu tetap temuan yang layak dilaporkan** dan sesuai judul tesis ("analisis performa").

### T2-8: Class weighting / focal loss

**Catatan:** FoR-2sec sudah seimbang antar kelas, jadi ini **prioritas rendah untuk balancing**. Namun focal loss tetap berguna untuk alasan lain: pada akurasi 98%, sebagian besar sampel sudah mudah, dan gradiennya membanjiri sinyal dari 2% sampel sulit.

```python
class FocalLoss(nn.Module):
 def __init__(self, gamma=2.0, alpha=None):
 super().__init__(); self.g, self.a = gamma, alpha
 def forward(self, logits, y):
 ce = F.cross_entropy(logits, y, weight=self.a, reduction='none')
 pt = torch.exp(-ce)
 return ((1 - pt) ** self.g * ce).mean()
```
Gunakan `gamma=1.0–2.0`. Alternatif yang lebih tepat sasaran: **hard example mining**, kumpulkan sampel yang salah diklasifikasi di epoch N, oversampling ×3 di epoch N+1.

**Dampak:** +0,1–0,6 poin. Terutama menolong pada test set ber-noise (di mana sampel sulit lebih banyak).
**Risiko:** Rendah. Bisa membuat pelatihan lebih berisik; pantau val_EER.

### T2-9: Cross-validation

**Aksi:** **Stratified Group 5-Fold** (group = ID pembicara). Bukan K-Fold biasa, group wajib, agar konsisten dengan T0-1.

Laporkan `mean ± std` lintas fold. Bila komputasi Colab terbatas, jalankan CV hanya untuk model terbaik dan CNN-LSTM (yang murah), dan gunakan single-split untuk sisanya, **catat batasan ini secara jujur di tesis**.

**Dampak:** Validitas + estimasi variansi. Bisa juga menaikkan hasil akhir via ensembling 5 fold (+0,2–0,5 poin).
**Risiko:** 5× biaya komputasi. Ini item pertama yang dikorbankan jika waktu mepet.

---

## TIER 3, High Effort / Eksperimental (opsional, minggu 9+)

### T3-1: Ganti / tambah dataset untuk generalisasi

**Masalah (G-6):** Satu dataset = satu distribusi. Model yang 98% di FoR bisa 60% di tempat lain, dan itu tidak akan pernah diketahui tanpa uji lintas dataset.

**Aksi, dua mode:**

**Mode A, evaluasi cross-dataset (WAJIB, murah):** latih di FoR, uji tanpa fine-tuning pada:

| Dataset | Karakter | Ukuran | Kegunaan |
|---|---|---|---|
| **ASVspoof 2019 LA** | 6 sistem TTS/VC (eval: 13 sistem unseen) | ~121k eval | Standar emas; punya baseline EER untuk dibandingkan |
| **ASVspoof 2021 DF** | + kompresi codec | ~600k | Uji ketahanan kanal |
| **WaveFake** | 6 vocoder neural (MelGAN, HiFi-GAN, dll.) | ~118k | Deepfake generasi baru (FoR relatif tua, TTS-nya bukan neural vocoder modern) |
| **In-the-Wild** | Audio selebriti dari internet | 31,8 jam | Kondisi paling realistis; baseline EER-nya tinggi (~30%) |

**Mode B, pelatihan multi-dataset (opsional, mahal):** latih pada FoR + WaveFake + ASVspoof 2019 LA gabungan, uji pada In-the-Wild. Ini menaikkan generalisasi drastis tetapi **mengubah ruang lingkup tesis** (batasan penelitian butir 3 mengunci ke FoR). Bila diambil, revisi batasan penelitian dulu dan konsultasikan ke pembimbing.

**Rekomendasi:** ambil **Mode A saja**. Cukup satu subbab "Uji Generalisasi Lintas Dataset", tidak melanggar batasan penelitian (karena pelatihan tetap di FoR), dan memberi nilai tambah besar. Perkiraan biaya: 1–2 hari.

**Dampak:** Menurunkan angka yang dilaporkan (itu tujuannya, kejujuran), tapi **menaikkan kualitas tesis secara drastis**. Ini pembeda antara tesis S2 yang baik dan yang biasa.
**Risiko:** Penguji melihat EER 25% di In-the-Wild. Framing yang benar: *"model mencapai 99% pada domain terlatih namun EER 25% lintas domain, konsisten dengan temuan [14] yang melaporkan 31,14%, menunjukkan generalisasi lintas domain masih menjadi masalah terbuka di bidang ini."* Ini justru menunjukkan kematangan ilmiah.

### T3-2: Hyperparameter search sistematis

**Aksi:** Optuna dengan TPE sampler + median pruner.

```python
import optuna
def objective(trial):
 cfg = dict(
 lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True),
 wd = trial.suggest_float("wd", 1e-4, 1e-1, log=True),
 dropout = trial.suggest_float("dropout", 0.0, 0.4),
 smoothing = trial.suggest_float("ls", 0.0, 0.15),
 warmup = trial.suggest_float("warmup", 0.0, 0.2),
 pool = trial.suggest_categorical("pool", ["mean", "attentive", "cls"]),
 )
 return train_and_eval(cfg)["val_eer"] # minimalkan EER
study = optuna.create_study(direction="minimize",
 pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))
study.optimize(objective, n_trials=30)
```

**Kompromi untuk Colab:** 30 trial × 4 model tidak realistis. Lakukan search penuh untuk **satu model saja** (yang paling menjanjikan, kemungkinan Wav2Vec2 atau AST), lalu transfer hyperparameter terbaik ke model lain dengan penyesuaian LR proporsional. Catat sebagai keterbatasan.

**Dampak:** +0,3–1,2 poin.
**Risiko:** Sangat boros komputasi. Item kedua yang dikorbankan bila waktu mepet.

### T3-3: Model yang lebih kuat / SSL yang lebih cocok

Bila ingin mendorong batas atas lebih jauh:
- **WavLM Large**, dirancang dengan denoising masked prediction; **konsisten mengungguli HuBERT untuk tugas anti-spoofing dan pada audio ber-noise**. Ini pengganti/tambahan paling logis mengingat fokus noise tesis Anda.
- **Wav2Vec2-XLS-R (300M)**, multilingual, generalisasi lebih baik.
- **AASIST / RawNet2**, arsitektur yang dirancang khusus untuk anti-spoofing (graph attention pada raw waveform); sering menjadi baseline SOTA di ASVspoof.

**Dampak:** WavLM Large bisa memberi **+1–3 poin pada kondisi ber-noise** dibanding HuBERT Large.
**Risiko:** Menambah model kelima = menambah ruang lingkup. Alternatif aman: **ganti** HuBERT Large → WavLM Large, dengan justifikasi eksplisit bahwa WavLM dirancang untuk kondisi ber-noise sehingga lebih relevan dengan rumusan masalah. Diskusikan dengan pembimbing.

### T3-4: Self-distillation & SWA

- **Stochastic Weight Averaging**, rata-ratakan bobot dari 5 epoch terakhir. Hampir gratis, +0,1–0,4 poin, generalisasi lebih baik.
- **Self-distillation**, latih model student dengan soft label dari ensemble T2-5. Menghasilkan **satu** model dengan performa mendekati ensemble → berguna bila ada argumen efisiensi deployment.

---

## 5. Tabel Master Prioritas

Urutan eksekusi dari atas ke bawah. Δ = estimasi perubahan akurasi pada FoR bersih; Δ-noise = pada test set noise unseen.

| Prio | Kode | Aksi | Effort | Δ bersih | Δ noise unseen | Risiko |
|---|---|---|---|---|---|---|
| 1 | T0-1 | Audit kebocoran data + group split | 1 hari | **−1 s/d −4** ⚠️ |, | Angka turun, tapi jadi valid |
| 2 | T0-2 | Uji shortcut / kontrol negatif | 0,5 hari | 0 |, | Nol |
| 3 | T0-3 | Protokol noise seen vs unseen | 1 hari | 0 | (baseline baru) | Nol |
| 4 | **T1-1** | **Perbaiki LR per model** | **2 jam** | **+1 s/d +8** | +2 s/d +8 | **Nyaris nol** |
| 5 | T1-6 | Mixed precision + grad checkpoint | 1 jam | 0 | 0 | Rendah |
| 6 | T1-2 | Early stopping + best-checkpoint on EER | 1 jam | +0,3 s/d +1,5 | +0,5 s/d +2 | Nol |
| 7 | T1-3 | Tambah EER / DET / min-tDCF | 2 jam | 0 | 0 | Nol |
| 8 | T1-4 | Threshold tuning dari validasi | 1 jam | +0,2 s/d +1,0 | +1 s/d +3 | Nol (disiplin) |
| 9 | T1-5 | Label smoothing + dropout | 1 jam | +0,1 s/d +0,5 | +0,3 s/d +1 | Rendah |
| 10 | **T2-1** | **Augmentasi lengkap (7 teknik)** | **3 hari** | +0,2 s/d +0,8 | **+5 s/d +15** | Sedang ⚠️ |
| 11 | **T2-3** | **Attentive pooling + layer weighting** | **1 hari** | **+0,5 s/d +2,0** | +1 s/d +3 | Rendah |
| 12 | T2-4 | Unfreezing bertahap + LLRD | 1 hari | +0,3 s/d +1,5 | +0,5 s/d +2 | Rendah |
| 13 | T2-6 | Kalibrasi (temperature scaling) | 3 jam | 0 (enabler) | 0 (enabler) | Nol |
| 14 | **T2-5** | **Ensemble 4 model** | **1–2 hari** | **+0,5 s/d +1,5** | **+2 s/d +6** | Rendah |
| 15 | T2-7 | Front-end denoising + LFCC/PCEN | 3 hari | −0,5 s/d +0,5 | **+3 s/d +10** | Sedang |
| 16 | T1-7 | Multi-seed (3×) | 3× compute | 0 | 0 | Nol |
| 17 | T2-2 | MixUp (level embedding) | 0,5 hari | +0,2 s/d +0,7 | +0,5 s/d +2 | Sedang |
| 18 | T2-8 | Focal loss / hard mining | 0,5 hari | +0,1 s/d +0,6 | +0,5 s/d +2 | Rendah |
| 19 | **T3-1** | **Cross-dataset eval (Mode A)** | **2 hari** | 0 | 0 | Angka rendah, nilai tinggi |
| 20 | T2-9 | Stratified Group 5-Fold CV | 5× compute | +0,2 s/d +0,5 | +0,3 s/d +1 | Mahal |
| 21 | T3-2 | Optuna hyperparameter search | 3–5 hari | +0,3 s/d +1,2 | +0,5 s/d +2 | Sangat mahal |
| 22 | T3-3 | WavLM Large (ganti/tambah) | 2 hari | +0,3 s/d +1,0 | +1 s/d +3 | Ubah ruang lingkup |
| 23 | T3-4 | SWA + self-distillation | 1 hari | +0,1 s/d +0,4 | +0,2 s/d +1 | Rendah |

**Jalur minimum bila waktu sangat terbatas (± 2 minggu):** T0-1 → T0-3 → T1-1 → T1-2 → T1-3 → T2-3 → T2-5. Tujuh item ini saja sudah memberi sebagian besar keuntungan.

⚠️ **Catatan penting soal penjumlahan:** kolom Δ **tidak boleh dijumlahkan**. Efeknya sangat tumpang tindih dan mengalami *diminishing returns* tajam di atas 98%. Naik dari 98% ke 99,5% berarti memotong error rate separuh, itu sudah hasil yang sangat baik. Jangan menjanjikan 99,9% di proposal revisi.

---

## 6. Roadmap Eksperimen

Kode E = eksperimen. Setiap eksperimen menghasilkan satu baris di tabel hasil tesis.

### Fase A, Validasi & Baseline Bersih (Minggu 1–2)

| ID | Eksperimen | Output |
|---|---|---|
| **E0** | Audit kebocoran + shortcut (T0-1, T0-2) | Laporan audit; keputusan skema split final |
| **E1** | Reproduksi baseline dengan split baru, konfigurasi proposal apa adanya (LR 1e-3, 20 epoch) | **Baseline jujur**, angka pembanding untuk semua perbaikan |
| **E2** | + LR per model, AdamW, warmup+cosine, early stopping, AMP (T1-1, T1-2, T1-6) | Baseline terkoreksi, kemungkinan lompatan terbesar |
| **E3** | + EER/DET/threshold tuning/label smoothing (T1-3, T1-4, T1-5) | Tabel metrik lengkap 4 model |

**Milestone Fase A:** 4 model terlatih dengan benar, metrik lengkap, kebocoran terverifikasi bersih.

### Fase B, Robustness terhadap Noise (Minggu 3–6), *inti tesis*

| ID | Eksperimen | Output |
|---|---|---|
| **E4** | Matriks evaluasi noise: 4 model × {bersih, seen 15/20/25/30 dB, unseen 0/5/10/15/20 dB} | **Tabel & grafik utama tesis**, akurasi/EER vs SNR |
| **E5** | Ablation augmentasi: tambahkan teknik satu per satu (T2-1), catat kontribusi tiap teknik | Tabel ablation, bagian metodologi yang kuat |
| **E6** | Perbandingan front-end: log-Mel vs LFCC vs CQCC vs PCEN (T2-7) | Rekomendasi fitur untuk kondisi ber-noise |
| **E7** | Strategi denoising S1 vs S2 vs S3 (T2-7) | Jawaban: apakah denoising membantu deteksi deepfake? |
| **E8** | + Reverb (RIR) dan uji pada `for-rerec` (T0-3, G-16) | Skenario rekam-ulang, dijanjikan di latar belakang tapi belum ada di metodologi |

**Milestone Fase B:** Kurva akurasi-vs-SNR untuk 4 model, plus rekomendasi front-end yang tervalidasi.

### Fase C, Optimasi Arsitektur & Fusion (Minggu 7–10)

| ID | Eksperimen | Output |
|---|---|---|
| **E9** | Attentive statistical pooling + layer-wise weighting (T2-3) | Δ per model + plot bobot layer |
| **E10** | Unfreezing bertahap + LLRD (T2-4) | Kurva pelatihan yang lebih stabil |
| **E11** | Kalibrasi + reliability diagram + ECE (T2-6) | Subbab kalibrasi |
| **E12** | Ensemble: score fusion → stacking → feature fusion (T2-5) | **Hasil terbaik keseluruhan** + matriks korelasi error |
| **E13** | Multi-seed 3× untuk konfigurasi final + uji McNemar (T1-7) | mean ± std + signifikansi statistik |

**Milestone Fase C:** Angka akhir untuk setiap model + ensemble, dengan interval kepercayaan.

### Fase D, Generalisasi & Analisis (Minggu 11–13)

| ID | Eksperimen | Output |
|---|---|---|
| **E14** | Cross-dataset: FoR→ASVspoof2019LA, FoR→WaveFake, FoR→In-the-Wild (T3-1) | Subbab generalisasi, pembeda kualitas tesis |
| **E15** | Analisis error: 50 sampel salah teratas per model, dengarkan & kategorikan | Analisis kualitatif |
| **E16** | (Opsional) Optuna pada model terbaik (T3-2) / WavLM (T3-3) | Hasil puncak |

**Milestone Fase D:** Semua tabel & gambar siap; tinggal penulisan.

### Pemetaan ke jadwal proposal (Tabel 5.1, hal. 73)

| Bulan | Jadwal asli | Isi baru |
|---|---|---|
| Maret 2026 | Proposal, dataset, pra-proses | ✔ Selesai |
| April 2026 | Pelatihan model + evaluasi | **Fase A (E0–E3)** |
| Mei 2026 | Pelatihan pra-latih + evaluasi | **Fase B (E4–E8)** |
| Juni 2026 | Pelatihan pra-latih + evaluasi | **Fase C (E9–E13)** |
| Juli 2026 | Analisis & penulisan hasil | **Fase D (E14–E16)** + penulisan |
| Agustus 2026 | Revisi & laporan akhir | Revisi |

Roadmap ini muat dalam jadwal asli tanpa perlu perpanjangan.

---

## 7. Kriteria Sukses & Target Metrik

Ganti kriteria proposal (">97%", hal. 72) dengan kriteria bertingkat berikut, jauh lebih dapat dipertahankan di sidang:

| Kondisi uji | Metrik | Target minimum | Target ideal |
|---|---|---|---|
| FoR-2sec bersih | Akurasi / EER | ≥ 97% / ≤ 2,5% | ≥ 99% / ≤ 1,0% |
| FoR-2sec + noise **seen** (15–30 dB) | Akurasi / EER | ≥ 95% / ≤ 4% | ≥ 98% / ≤ 1,5% |
| FoR-2sec + noise **unseen** (10–20 dB) | Akurasi / EER | ≥ 92% / ≤ 7% | ≥ 96% / ≤ 3% |
| FoR-2sec + noise **unseen berat** (0–5 dB) | Akurasi / EER | ≥ 82% / ≤ 16% | ≥ 90% / ≤ 9% |
| Reverb (RT60 0,2–0,8 s) | Akurasi | ≥ 88% | ≥ 94% |
| `for-rerec` | Akurasi | ≥ 80% | ≥ 90% |
| Cross-dataset (ASVspoof 2019 LA) | EER | ≤ 25% | ≤ 12% |
| Cross-dataset (In-the-Wild) | EER | ≤ 35% | ≤ 20% |
| Kalibrasi | ECE | ≤ 0,05 | ≤ 0,02 |
| Stabilitas | std akurasi lintas 3 seed | ≤ 0,5 poin | ≤ 0,2 poin |

**Definisi ulang "sukses" untuk tesis ini:** bukan angka tunggal tertinggi, melainkan **degradasi yang landai** dari kondisi bersih ke kondisi terburuk. Model yang turun 99% → 94% pada noise unseen lebih berharga daripada model yang turun 99,5% → 71%. Jadikan **selisih (degradasi)** sebagai metrik pembanding utama antar arsitektur, itulah jawaban langsung atas rumusan masalah Anda.

---

## 8. Template Tabel Hasil untuk Tesis

### Tabel utama, Performa 4 arsitektur pada berbagai kondisi

| Model | Kondisi | Akurasi | Presisi | Recall | F1 | AUC | **EER** | Δ dari bersih |
|---|---|---|---|---|---|---|---|---|
| Wav2Vec2 Base | Bersih | | | | | | |, |
| Wav2Vec2 Base | Noise seen 20 dB | | | | | | | |
| Wav2Vec2 Base | Noise unseen 10 dB | | | | | | | |
| Wav2Vec2 Base | Noise unseen 0 dB | | | | | | | |
| AST | … (idem 4 baris) | | | | | | | |
| HuBERT Large | … | | | | | | | |
| CNN-LSTM | … | | | | | | | |
| **Ensemble** | … | | | | | | | |

### Tabel ablation augmentasi (model terbaik)

| Konfigurasi | Akurasi bersih | Akurasi noise unseen 10 dB | Δ |
|---|---|---|---|
| Tanpa augmentasi | | | |
| + Noise (DNC, 15–30 dB), *baseline proposal* | | | |
| + Noise diperluas (MUSAN, 0–30 dB) | | | |
| + SpecAugment | | | |
| + RIR / reverb | | | |
| + Codec / band-limit | | | |
| + Pitch & speed | | | |
| **Semua** | | | |

### Tabel perbandingan dengan state of the art

| Penelitian | Metode | Dataset | Akurasi | EER |
|---|---|---|---|---|
| Ref [13] | SVM + MFCC + PCA | FoR | 93,50% |, |
| Ref [19] MFAAN | CNN multi-feature | FoR | 94,47% |, |
| Ref [20] | CNN-LSTM | FoR | 94,70% |, |
| **Penelitian ini** | Wav2Vec2 fine-tuned | FoR-2sec | **…** | **…** |
| **Penelitian ini** | Ensemble 4 model | FoR-2sec | **…** | **…** |

---

## 9. Risiko & Mitigasi

| Risiko | Kemungkinan | Dampak | Mitigasi |
|---|---|---|---|
| Audit T0 mengungkap kebocoran → akurasi turun dari 98% ke ~93% | Sedang | Tinggi (psikologis & naratif) | Framing: "koreksi metodologis meningkatkan validitas". Angka setelah T1+T2 akan naik lagi melewati 98%. Penguji **jauh** lebih menghargai kejujuran ini daripada 98% yang rapuh |
| Kuota GPU Colab habis | Tinggi | Sedang | T1-6 (AMP) wajib; simpan checkpoint ke Drive tiap epoch; jalankan HuBERT Large terakhir; pertimbangkan Colab Pro atau Kaggle (30 jam GPU/minggu gratis) |
| Augmentasi pitch/speed justru menurunkan akurasi | Sedang | Sedang | Ablation per teknik (E5); buang yang merugikan; terapkan seragam pada kedua kelas |
| Denoising (S2) menurunkan akurasi | **Tinggi** | Rendah | Ini **hasil yang valid dan menarik**, laporkan sebagai temuan, bukan kegagalan |
| Ensemble dianggap "mengaburkan" tujuan perbandingan arsitektur | Rendah | Sedang | Pisahkan pelaporan: hasil per-arsitektur (tujuan utama) dan hasil ensemble (kontribusi tambahan) |
| Cross-dataset EER buruk (>30%) | **Tinggi** | Rendah | Sudah diantisipasi: ref [14] di proposal Anda sendiri melaporkan 31,14% baseline In-the-Wild. Kutip itu sebagai konteks |
| Waktu tidak cukup untuk semua eksperimen | Sedang | Sedang | Ikuti urutan tabel master; korbankan dari bawah (T2-9, T3-2, T3-3 lebih dulu) |
| Perubahan (WavLM, dataset tambahan) melanggar batasan penelitian | Sedang | Sedang | Konsultasikan revisi batasan penelitian ke pembimbing **sebelum** eksekusi |

---

## 10. Rekomendasi Revisi untuk Dokumen Tesis

Perubahan yang sebaiknya masuk ke naskah tesis (bukan hanya ke kode):

1. **BAB IV §4.1**, ganti "60/20/20" dengan protokol split resmi FoR **atau** `StratifiedGroupKFold` yang speaker-disjoint, disertai justifikasi anti-kebocoran.
2. **BAB IV §4.2**, ganti peak normalization menjadi loudness normalization (−23 LUFS) + trim silence konsisten; perluas subbab augmentasi menjadi 7 teknik dengan tabel parameter.
3. **BAB IV §4.5**, hapus klaim "konfigurasi seragam"; ganti dengan **"budget tuning seragam"** (tiap model mendapat jumlah percobaan hyperparameter yang sama), dan sertakan tabel LR per model.
4. **BAB IV §4.5**, tambahkan: optimizer (AdamW), weight decay, scheduler, warmup, early stopping, mixed precision, jumlah seed.
5. **BAB IV §4.8**, tambahkan EER, min-tDCF, kurva DET, ECE/reliability diagram, threshold tuning, dan uji signifikansi (McNemar).
6. **BAB IV, subbab baru §4.9 "Rancangan Skenario Pengujian Robustness"**, matriks kondisi uji (bersih / seen / unseen / reverb / codec / rerec) dengan sumber noise yang eksplisit disjoint. **Ini bagian terpenting yang hilang dari proposal**, rumusan masalah menyebut noise sebagai fokus, tapi metodologi hanya mengalokasikan satu paragraf untuknya.
7. **BAB IV, subbab baru §4.10 "Ensemble & Fusion"**.
8. **BAB IV, subbab baru §4.11 "Uji Generalisasi Lintas Dataset"**.
9. **BAB I §1.4 Batasan**, bila mengadopsi T3-1/T3-3, perbarui butir 1 dan 3.
10. **Kriteria sukses (hal. 71–72)**, ganti ">94%" dan ">97%" dengan tabel kriteria bertingkat di [§7](#7-kriteria-sukses--target-metrik).
11. **Perbaikan minor:** "Phyton" → "Python" (Daftar Isi, §3.2, Gambar 3.1); "deppfake" → "deepfake" (hal. 71); "hybdrid" → "hybrid" (hal. 5); penomoran subbab §4.4 Modeling memakai anak-subbab 4.3.1–4.3.4 (seharusnya 4.4.1–4.4.4).
12. **Koreksi rumus (hal. 72)**, `Sensitivity` ditulis `TP/(TP+FP)` dan `Specificity` ditulis `TP/(TP+FN)`. Keduanya keliru. Yang benar: **Sensitivity (TPR) = TP/(TP+FN)**, **Specificity (TNR) = TN/(TN+FP)**, dan **FPR = 1 − Specificity = FP/(FP+TN)**. Karena sumbu ROC didefinisikan dari kedua nilai ini, kesalahan tersebut harus diperbaiki agar definisi ROC di naskah konsisten.
13. **Koreksi keterangan confusion matrix (hal. 71)**, FP dideskripsikan "sebenarnya benar dan diprediksi salah" dan FN "sebenarnya salah dan diprediksi benar". Tertukar. Yang benar: **FP = sebenarnya negatif, diprediksi positif**; **FN = sebenarnya positif, diprediksi negatif**. (Isi Tabel 4.1 sendiri sudah benar; hanya keterangannya yang salah, dan keterangan ini berulang di dua tempat.)

---

## 11. Lampiran A, Konfigurasi Rekomendasi Lengkap

```python
# ===================== KONFIGURASI FINAL YANG DIREKOMENDASIKAN =====================
CONFIG = {
 "data": {
 "dataset": "for-2seconds",
 "split": "official", # ATAU StratifiedGroupKFold(groups=speaker_id)
 "sample_rate": 16000,
 "duration": 2.0,
 "normalize": "loudness_-23LUFS", # ganti dari peak normalization
 "trim_silence": {"top_db": 30}, # konsisten untuk kedua kelas
 },
 "augment_train": {
 "noise": {"sources": ["MUSAN", "DNC_subsetA", "ESC-50"],
 "snr_db": [0, 30], "p": 0.5},
 "rir": {"source": "OpenSLR-SLR28", "rt60": [0.1, 1.0], "p": 0.3},
 "specaugment": {"freq_mask": (2, 15), "time_mask": (2, 20), "p": 0.5},
 "codec": {"mp3_kbps": [32, 128], "mulaw": True, "p": 0.3},
 "gain": {"db": [-10, 10], "p": 0.2},
 "pitch": {"semitones": [-2, 2], "p": 0.2}, # uji ablation dulu
 "speed": {"rate": [0.9, 1.1], "p": 0.3}, # uji ablation dulu
 "mixup": {"alpha": 0.2, "level": "embedding", "p": 0.3},
 },
 "models": {
 "wav2vec2": {"ckpt": "facebook/wav2vec2-base",
 "lr": 1e-5, "head_lr": 1e-4, "epochs": 12, "bs": 32,
 "freeze_feature_encoder": True, "layerdrop": 0.05},
 "ast": {"ckpt": "MIT/ast-finetuned-audioset-10-10-0.4593",
 "lr": 2e-5, "head_lr": 1e-4, "epochs": 12, "bs": 32},
 "hubert": {"ckpt": "facebook/hubert-large-ll60k",
 "lr": 5e-6, "head_lr": 5e-5, "epochs": 10, "bs": 16,
 "grad_accum": 2, "freeze_feature_encoder": True,
 "gradient_checkpointing": True},
 "cnn_lstm": {"lr": 1e-3, "epochs": 50, "bs": 64,
 "conv": [32, 64, 128], "lstm_hidden": 256, "lstm_layers": 2,
 "bidirectional": True},
 },
 "head": {
 "pooling": "attentive_stats", # ganti dari mean / CLS
 "layer_weighting": True, # hanya untuk model SSL
 "hidden": 256, "dropout": 0.2,
 },
 "optim": {
 "name": "AdamW", "weight_decay": 0.01,
 "scheduler": "cosine", "warmup_ratio": 0.1,
 "label_smoothing": 0.05,
 "grad_clip": 1.0,
 "amp_dtype": "bfloat16",
 },
 "train": {
 "early_stopping": {"monitor": "val_eer", "mode": "min", "patience": 5},
 "checkpoint": {"monitor": "val_eer", "mode": "min", "save_top_k": 1},
 "seeds": [42, 1337, 2024],
 },
 "eval": {
 "metrics": ["accuracy", "precision", "recall", "f1", "auc",
 "eer", "min_tdcf", "ece"],
 "threshold": "from_validation_youden_J",
 "calibration": "temperature_scaling",
 "test_conditions": {
 "clean": {},
 "noise_seen": {"src": "DNC_subsetB", "snr_db": [15, 20, 25, 30]},
 "noise_unseen": {"src": ["MUSAN", "DEMAND"], "snr_db": [0, 5, 10, 15, 20]},
 "reverb": {"src": "SLR28", "rt60": [0.2, 0.5, 0.8]},
 "codec": {"mp3_kbps": [32, 64], "mulaw": True},
 "rerec": {"src": "for-rerec"},
 },
 "cross_dataset": ["ASVspoof2019LA_eval", "WaveFake", "InTheWild"],
 },
}
```

## 12. Lampiran B, Sumber Daya

| Kebutuhan | Sumber | Catatan |
|---|---|---|
| Noise (train+unseen) | MUSAN (openslr.org/17), DEMAND (Zenodo), ESC-50 (GitHub karoldvl) | MUSAN paling standar di literatur |
| RIR | OpenSLR SLR28 (RIRS_NOISES) | Simulated + real RIR |
| Dataset uji silang | ASVspoof 2019/2021 (datashare.ed.ac.uk), WaveFake (Zenodo), In-the-Wild (deepfake-total.com) | Semuanya gratis untuk penelitian |
| Library augmentasi | `audiomentations`, `torch-audiomentations`, `torchaudio.transforms` | `torch-audiomentations` mendukung GPU (lebih cepat) |
| Denoising | `DeepFilterNet` (pip), `demucs`, `noisereduce` | DeepFilterNet3 ringan & real-time |
| Metrik anti-spoofing | Kode resmi ASVspoof (EER + min-tDCF), `sklearn.metrics` | |
| Loudness | `pyloudnorm` | Untuk normalisasi −23 LUFS |
| HPO | `optuna` | Median pruner hemat compute |
| Speaker embedding (audit) | `speechbrain/spkrec-ecapa-voxceleb` | Untuk deteksi near-duplicate di T0-1 |

---

## Penutup

Tiga kalimat inti:

1. **Perbaiki learning rate dulu** (T1-1), itu perubahan satu baris dengan dampak terbesar, dan hampir pasti bug yang sedang menahan model SSL Anda.
2. **Buktikan 98% itu nyata sebelum mengejar 99%** (Tier 0), kebocoran data adalah pertanyaan pertama yang akan diajukan penguji, dan pada dataset FoR pertanyaan itu sangat beralasan.
3. **Kontribusi terbesar tesis ini ada di kolom yang belum Anda ukur**, noise unseen, reverb, `for-rerec`, dan cross-dataset. Naik dari 98% ke 99,5% di FoR bersih bernilai kecil; menunjukkan model mana yang bertahan di SNR 0 dB menjawab rumusan masalah Anda secara langsung.

---

*Dokumen ini disusun dari pembacaan lengkap PROPOSAL_TESIS.pdf (89 halaman). Semua rujukan halaman mengacu pada nomor halaman cetak di dalam PDF. Estimasi dampak bersifat perkiraan berdasarkan pola yang lazim di literatur audio anti-spoofing dan harus divalidasi secara empiris melalui ablation.*
