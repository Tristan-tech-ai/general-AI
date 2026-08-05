# Analisis Mendalam: Arsitektur Terbaik untuk Deteksi Deepfake Audio

*Analisis dari prinsip pertama untuk tesis: klasifikasi suara asli vs deepfake, FoR-2sec, 2 detik @ 16 kHz, fokus ketahanan noise.*

---

## Ringkasan jawaban

**Arsitektur terbaik untuk tugas ini adalah: SSL front-end berbasis waveform + agregasi berbobot antar-layer + attentive statistics pooling.** Dalam batasan empat model tesis, itu berarti **Wav2Vec2** (bukan HuBERT Large, bukan AST) — tetapi dengan tiga modifikasi yang bukan opsional.

Alasannya bukan "karena Transformer lebih modern". Alasannya ada tiga, dan semuanya bisa diturunkan dari fisika sinyal:

1. **Log-Mel spectrogram membuang fase.** Artefak vocoder sebagian besar adalah artefak fase. AST dan CNN-LSTM secara arsitektural buta terhadap bukti paling diskriminatif dalam tugas ini.
2. **Skala Mel memampatkan frekuensi tinggi 12,4× lebih rapat daripada frekuensi rendah** — tepat di pita tempat artefak sintesis berada.
3. **Pada 2 detik, tidak ada dinamika temporal yang layak dimodelkan.** LSTM kehilangan alasan keberadaannya; yang dibutuhkan adalah *agregasi bukti*, bukan *pemodelan urutan*.

Sisa dokumen ini membuktikan ketiganya, lalu membangun arsitektur yang mengikuti dari sana.

---

## 1. Pertanyaan yang benar: apa yang sebenarnya sedang dideteksi?

Hampir semua kesalahan desain dalam deteksi deepfake audio berasal dari salah merumuskan tugas.

**Rumusan yang salah:** *"Bedakan suara manusia dari suara mesin."*
Ini menggiring ke arsitektur yang memodelkan **wicara** — fonem, prosodi, identitas pembicara. Itulah yang dioptimasi oleh Wav2Vec2 dan HuBERT dalam pra-pelatihannya, dan itu **bukan** yang kita butuhkan.

**Rumusan yang benar:** *"Deteksi jejak proses sintesis pada sinyal."*

Deepfake TTS modern menghasilkan wicara yang secara linguistik dan prosodik sangat meyakinkan. Yang tidak bisa disembunyikan adalah jejak *proses rekonstruksi sinyal*. Sumbernya spesifik dan dapat diidentifikasi:

| Sumber artefak | Mekanisme | Di mana muncul di sinyal |
|---|---|---|
| **Rekonstruksi fase** | Vocoder memprediksi magnitudo lalu merekonstruksi fase (Griffin-Lim) atau menghasilkan waveform langsung dengan konsistensi fase tak sempurna | **Hubungan fase antar-bin STFT**; koherensi antar-harmonik |
| **Checkerboard artifact** | Transposed convolution pada vocoder GAN menghasilkan periodisitas buatan | Puncak sempit di spektrum pada frekuensi kelipatan stride |
| **Spectral rolloff tak wajar** | Vocoder dilatih pada target band-limited; energi di atas batas itu "diarang" | **Pita 6–8 kHz**, sering terlalu halus atau terlalu tajam terpotong |
| **Hilangnya mikro-variasi** | Model generatif menghasilkan sinyal "terlalu bersih" — jitter/shimmer glotal, noise aspirasi, dan resonansi ruang hilang | Struktur halus beramplitudo rendah di seluruh spektrum |
| **Digital silence** | TTS menghasilkan keheningan sempurna; mikrofon selalu punya *noise floor* | Segmen dengan amplitudo persis nol |
| **Kuantisasi & pemulusan** | Prediksi mel yang di-oversmooth oleh loss L1/L2 | Kontras spektral lebih rendah dari rekaman asli |

**Konsekuensi desain yang langsung mengikuti:** dua dari empat sumber artefak teratas hidup di **fase** dan **frekuensi tinggi**. Setiap keputusan representasi yang mengorbankan keduanya adalah kerugian informasi yang tidak bisa dipulihkan oleh arsitektur secanggih apa pun di hilirnya.

Ini prinsip yang menentukan seluruh sisa analisis: **arsitektur tidak bisa memulihkan informasi yang sudah dibuang oleh representasi.**

---

## 2. Analisis representasi: apa yang dibuang masing-masing

Ini bagian terpenting dari dokumen ini.

### 2.1 Ketersediaan informasi per representasi

| Representasi | Fase | Resolusi frekuensi | Dipakai oleh | Kerugian informasi |
|---|---|---|---|---|
| **Raw waveform** 32.000 sampel | ✅ **Utuh** | Penuh (Nyquist 8 kHz) | Wav2Vec2, HuBERT | Nol — ini sinyal aslinya |
| **Linear spectrogram** \|STFT\| | ❌ **Dibuang** | Seragam | — | Fase |
| **Log-Mel** 128 bin | ❌ **Dibuang** | **Terwarp** — HF dimampatkan | **AST, CNN-LSTM** | Fase **+** resolusi HF |
| **LFCC** | ❌ Dibuang | Seragam | (usulan) | Fase saja |
| **CQCC** | ❌ Dibuang | Log-frekuensi | (usulan) | Fase saja |

### 2.2 Berapa besar sebenarnya pemampatan Mel? (hitungan, bukan intuisi)

Skala Mel: `m = 2595 · log₁₀(1 + f/700)`

Untuk audio 16 kHz, rentang penuh 0–8000 Hz:
```
mel(8000) = 2595 · log₁₀(1 + 8000/700) = 2595 · log₁₀(12,4286) = 2839,9 mel
```
Dengan 128 bin mel: setiap bin = 2839,9 / 128 = **22,19 mel**.

Lebar bin dalam Hz = `22,19 · df/dm`, di mana `df/dm = 0,62115 · 10^(m/2595)`:

| Posisi | Frekuensi | Lebar 1 bin mel |
|---|---|---|
| Bin terendah | ~0 Hz | **13,8 Hz** |
| Bin tengah | ~1,4 kHz | ~44 Hz |
| Bin tertinggi | ~8 kHz | **171,3 Hz** |

**Rasio: 171,3 / 13,8 = 12,4×.**

Artinya: di pita 6–8 kHz — tempat artefak *spectral rolloff* dan *checkerboard* paling terlihat — log-Mel memberikan resolusi **12,4 kali lebih kasar** daripada di pita rendah. Struktur spektral halus selebar 50 Hz pada 7 kHz, yang merupakan tanda vocoder yang jelas, **jatuh seluruhnya di dalam satu bin mel dan menjadi tak terlihat.**

Skala Mel dirancang untuk meniru persepsi pendengaran manusia. Itu tepat untuk pengenalan wicara. Untuk deteksi deepfake, itu justru **membuang tepat bagian yang manusia tidak bisa dengar — dan karena itu tepat bagian yang tidak dioptimasi oleh perancang TTS untuk terdengar meyakinkan.**

Inilah alasan teknis, bukan sekadar empiris, mengapa **LFCC secara konsisten mengungguli MFCC** dalam literatur anti-spoofing. LFCC memakai filter bank berjarak **linear** — resolusi seragam di seluruh pita.

### 2.3 Fase: bukti yang hilang total

Log-Mel dihitung sebagai `log(Mel(|STFT|²))`. Operasi `|·|` membuang seluruh informasi fase secara permanen.

Mengapa ini fatal untuk deteksi deepfake:

- Vocoder berbasis Griffin-Lim (umum pada TTS era 2017–2019, **yang persis mengisi dataset FoR**) merekonstruksi fase secara iteratif dari magnitudo. Hasilnya punya **inkonsistensi fase** yang terukur — ini adalah tanda pengenal paling kuat yang tersedia.
- Vocoder neural (WaveNet, WaveRNN, HiFi-GAN) menghasilkan waveform langsung, tetapi koherensi fase antar-harmonik tetap berbeda dari produksi wicara manusia yang digerakkan sumber glotal tunggal.
- Fitur berbasis fase (Modified Group Delay, Relative Phase Shift, All-Pole Group Delay) adalah keluarga fitur yang mapan dalam literatur anti-spoofing, dan sering mengungguli fitur magnitudo pada serangan tertentu.

**Kesimpulan yang tidak bisa dihindari:** AST dan CNN-LSTM, sebagaimana dirancang di proposal (hal. 58, 62, 66 — keduanya menerima log-Mel), **secara arsitektural tidak akan pernah bisa mengakses kelas bukti terkuat dalam tugas ini.** Bukan karena kapasitasnya kurang, tapi karena datanya sudah tidak ada saat sampai ke model.

Ini bukan sekadar kritik. Ini adalah **hipotesis penelitian yang dapat diuji dan layak menjadi temuan inti tesis:**

> *Model berbasis waveform mengungguli model berbasis magnitudo-spektrogram pada deteksi deepfake, dan selisihnya berasal dari akses terhadap informasi fase — bukan dari kapasitas model.*

Cara menguji (elegan, murah, dan sangat meyakinkan di sidang): jalankan Wav2Vec2 dua kali — sekali pada waveform asli, sekali pada waveform yang **fasenya diacak lalu direkonstruksi** (`istft(|STFT| · e^{jφ_acak})`). Selisih akurasi antara keduanya adalah **pengukuran langsung berapa banyak informasi diskriminatif yang terkandung dalam fase.** Saya belum pernah melihat eksperimen ini di literatur deteksi deepfake berbahasa Indonesia, dan hasilnya akan langsung menjelaskan seluruh peringkat model Anda.

---

## 3. Kendala 2 detik: mengapa ini mengubah segalanya

✅ Proposal mengunci `for-2sec` (hal. 6, 55). Konsekuensi kuantitatifnya jarang dipikirkan, padahal menentukan.

### 3.1 Aritmetika

```
2 detik @ 16 kHz                              = 32.000 sampel

Encoder CNN Wav2Vec2/HuBERT:
  7 lapis conv1d, stride (5,2,2,2,2,2,2)      = total stride 320
  Panjang keluaran = 32.000 / 320             = 100 frame laten
  Receptive field per frame = 400 sampel      = 25 ms

AST (log-Mel 128 mel, hop 10 ms):
  Panjang keluaran                            = ~200 frame
  Patch 16×16, stride 10                      = 12 × 19 = 228 patch
  (checkpoint AudioSet default: 1212 patch — lihat §6.2)
```

### 3.2 Konsekuensi 1: LSTM kehilangan alasan keberadaannya

✅ Proposal (hal. 66–67) menempatkan LSTM untuk "menangkap hubungan temporal jangka pendek maupun jangka panjang".

**Tetapi artefak vocoder bersifat kuasi-stasioner.** Jika sebuah rekaman dihasilkan HiFi-GAN, jejak HiFi-GAN ada di **setiap** frame — bukan pada transisi antar-frame. Tidak ada "narasi temporal" yang perlu diikuti.

Formalnya: misalkan tiap frame *t* memberikan bukti log-likelihood-ratio `ℓ_t` yang lemah namun kurang-lebih independen. Penduga optimal untuk keseluruhan ujaran adalah **penjumlahan/perata-rataan**:

```
ℓ_total = Σ ℓ_t        →   SNR keputusan tumbuh ∝ √N
```

Dengan N = 100 frame, agregasi yang benar memberi peningkatan rasio sinyal-terhadap-derau sebesar **10×** dibanding keputusan satu frame. **Yang dibutuhkan adalah penjumlah bukti yang baik, bukan pemodel urutan.**

Sekarang bandingkan dengan apa yang dilakukan LSTM. ✅ Proposal hal. 67 & rumus (38) hal. 45: klasifikasi dilakukan dari **hidden state akhir** `h_t`. LSTM memiliki *recency bias* struktural — informasi dari frame ke-1 harus melewati 100 gerbang forget untuk mencapai keputusan, dan sebagian besar teredam. Jadi LSTM bukan sekadar tidak membantu; **LSTM adalah agregator yang lebih buruk daripada rata-rata sederhana** untuk sinyal stasioner.

**Prediksi yang dapat diuji, dan menurut saya paling layak masuk tesis:**

> CNN + attentive statistics pooling akan **menyamai atau mengungguli** CNN-LSTM pada tugas ini, dengan parameter lebih sedikit. Kontribusi LSTM ≈ nol.

Ini ablation tiga baris kode yang menghasilkan temuan arsitektural sejati — jauh lebih bernilai daripada satu baris lagi di tabel benchmark. Dan jika ternyata LSTM *membantu*, itu justru temuan yang lebih menarik lagi (berarti ada struktur temporal yang tidak saya perkirakan).

### 3.3 Konsekuensi 2: HuBERT Large adalah salah-pasang kapasitas

| Model | Parameter | Frame yang diproses | Parameter per frame |
|---|---|---|---|
| HuBERT Large | 317 juta | 100 | 3,17 juta |
| Wav2Vec2 Base | 95 juta | 100 | 0,95 juta |

Dengan asumsi FoR-2sec training ≈ 26.000 klip berlabel biner, konten informasi label seluruh training set adalah **≈ 26.000 bit ≈ 3,2 kilobyte**. Melatih 317 juta parameter dari sinyal supervisi sebesar 3,2 KB hanya masuk akal bila **hampir seluruh model dibekukan** dan transfer learning menanggung beban.

Ini bukan argumen bahwa HuBERT Large buruk — melainkan bahwa **fine-tuning penuh HuBERT Large pada tugas ini secara matematis tidak dapat dibenarkan.** Resep yang benar: bekukan encoder, latih agregator + head yang kecil. Kebetulan, itu juga persis resep yang dipakai sistem ASVspoof papan atas.

📚 Prediksi (keyakinan sedang-tinggi): **HuBERT Large tidak akan mengungguli Wav2Vec2 Base secara sepadan dengan 3,3× ukurannya.** Jika selisihnya < 0,5 poin, itu temuan yang layak dilaporkan — *"kapasitas model bukan faktor pembatas pada segmen pendek."*

### 3.4 Konsekuensi 3: domain pra-pelatihan mengalahkan arsitektur

Ini menurut saya adalah wawasan paling berharga dari seluruh analisis, dan tesis Anda berada di posisi sempurna untuk mengujinya.

| Model | Korpus pra-pelatihan | Karakter akustik | Prediksi ketahanan noise |
|---|---|---|---|
| Wav2Vec2 Base | LibriSpeech 960 jam | Audiobook, bersih, mikrofon dekat | Rendah |
| HuBERT Large | LibriLight 60.000 jam | Audiobook, bersih | Rendah |
| **AST** | **AudioSet ~5.000 jam** | **Video YouTube — berisik, gaung, beragam, dunia nyata** | **Tinggi** |
| CNN-LSTM | — (dari nol) | Distribusi latih Anda sendiri | Tinggi pada noise *seen* |

**Hipotesis:** untuk **audio bersih**, model waveform (Wav2Vec2/HuBERT) menang karena akses fase. Untuk **noise berat yang belum pernah dilihat**, AST mungkin menang meski representasinya lebih miskin — karena AudioSet mengajarinya cara memisahkan sinyal dari gangguan dunia nyata, sementara LibriLight tidak pernah menunjukkan noise sama sekali kepada Wav2Vec2/HuBERT.

Jika terbukti, kalimat kesimpulan tesis Anda menjadi:

> *"Ketahanan terhadap noise pada deteksi deepfake audio lebih ditentukan oleh distribusi domain pra-pelatihan daripada oleh arsitektur model."*

Itu pernyataan ilmiah sejati — dapat digeneralisasi, dapat difalsifikasi, dan berguna bagi peneliti lain. Bandingkan dengan *"Wav2Vec2 mencapai 98,3%, AST 97,9%"*, yang tidak memberi tahu siapa pun apa pun. Keempat model Anda kebetulan menjangkau empat rezim pra-pelatihan berbeda — itu **keunggulan desain yang belum disadari proposal**, dan mengubahnya dari benchmark menjadi eksperimen terkontrol.

---

## 4. Evaluasi keempat arsitektur dari prinsip pertama

### 4.1 Kartu skor

Skala: ✅ kuat · ⚠️ terbatas · ❌ tidak mampu

| Kriteria | Wav2Vec2 Base | HuBERT Large | AST | CNN-LSTM |
|---|---|---|---|---|
| Akses informasi **fase** | ✅ | ✅ | ❌ | ❌ |
| Resolusi **frekuensi tinggi** | ✅ | ✅ | ⚠️ Mel | ⚠️ Mel |
| Kesesuaian kapasitas untuk 100 frame | ✅ | ❌ berlebihan | ⚠️ | ✅ |
| Kualitas **agregasi bukti** (bawaan) | ⚠️ mean pool | ⚠️ mean pool | ⚠️ [CLS] | ❌ hidden akhir |
| Ketahanan noise dari **pra-pelatihan** | ⚠️ bersih | ⚠️ bersih | ✅ AudioSet | ⚠️ bergantung augmentasi |
| Kecocokan panjang input | ✅ fleksibel | ✅ fleksibel | ❌ butuh perbaikan (§6.2) | ✅ |
| Biaya latih di RTX 5060 Ti 16 GB | ✅ ringan | ⚠️ berat | ✅ sedang | ✅ sangat ringan |
| **Potensi setelah diperbaiki** | **✅ Tertinggi** | ✅ Tinggi | ⚠️ Sedang | ⚠️ Baseline |

### 4.2 Wav2Vec2 Base — **pemenang dalam batasan tesis**

**Kelebihan struktural.** Beroperasi pada waveform mentah → fase utuh, resolusi frekuensi penuh. Encoder CNN 7 lapis dengan receptive field 25 ms adalah skala yang tepat untuk artefak vocoder. Kapasitas 95M sesuai untuk 100 frame. Ringan — muat nyaman di 16 GB dengan batch besar.

**Kelemahan bawaan, dan cara memperbaikinya.** Pra-pelatihan pada LibriSpeech mengoptimasi representasi **fonetik**, dan layer akhir adalah yang paling fonetik — justru paling tidak berguna untuk deteksi artefak.

📚 Keyakinan sedang-tinggi: layer **awal-menengah (3–9)** membawa informasi akustik tingkat rendah yang kita butuhkan; layer akhir (10–12) sudah terlalu abstrak. Implementasi standar `Wav2Vec2ForSequenceClassification` hanya memakai **layer terakhir**. Itu berarti membuang informasi paling relevan.

**Perbaikan: agregasi berbobot antar-layer yang dipelajari** (§5.2). Perubahannya kecil, dan menurut saya ini modifikasi tunggal paling bernilai untuk model SSL pada tugas ini.

### 4.3 HuBERT Large — kuat tapi salah ukuran

Representasi pra-latihnya lebih kaya (60.000 jam vs 960 jam) dan 24 layer memberi ruang lebih besar untuk *layer selection* — bisa jadi ada layer yang sangat baik untuk artefak.

Tetapi 317M parameter untuk 100 frame adalah pemborosan, biayanya 3,3× Wav2Vec2, dan pra-pelatihannya sama-sama pada audiobook bersih.

**Resep yang benar:** bekukan encoder sepenuhnya, ekstrak seluruh 25 hidden state, latih hanya agregator + head (~2M parameter). Ini murah, stabil, dan hampir pasti lebih baik daripada fine-tuning penuh pada data sebesar ini.

**Alternatif yang lebih baik bila diizinkan pembimbing:** **WavLM Large**. Arsitekturnya identik dengan HuBERT Large, tetapi pra-pelatihannya menambahkan **simulasi ucapan tumpang-tindih dan denoising masked prediction** — WavLM secara harfiah dilatih untuk memulihkan wicara bersih dari campuran yang berisik. Untuk tesis yang rumusan masalahnya adalah ketahanan noise (hal. 6), WavLM adalah pilihan yang jauh lebih sesuai, dan penggantian ini bisa dibenarkan dalam satu paragraf metodologi. Kode identik — cukup ganti string checkpoint.

### 4.4 AST — tertahan oleh representasi, bukan oleh arsitekturnya

Arsitektur AST bagus: self-attention penuh pada bidang waktu–frekuensi menangkap korelasi jauh yang tidak bisa ditangkap CNN. Pra-pelatihan AudioSet memberi ketahanan noise dunia nyata yang tidak dimiliki model lain.

Tapi AST **memakan log-Mel**, sehingga mewarisi kedua kerugian di §2. Ia menganalisis representasi yang sudah kehilangan bukti terkuat.

**Tiga perbaikan, urut dampaknya:**
1. **Perbaiki `max_length`** menjadi ~200 frame + interpolasi positional embedding (§6.2). Tanpa ini, ~81% masukan adalah padding. Ini bug, bukan pilihan desain.
2. **Naikkan resolusi frekuensi:** ganti 128 mel → **80 LFCC linear** atau **160 mel**. Mengurangi pemampatan HF.
3. **Suntikkan kembali fase:** tambahkan kanal kedua berisi **Modified Group Delay** atau **turunan fase antar-frame**, lalu susun sebagai input 2-kanal. Ini mengembalikan informasi yang dibuang `|·|`, dan mengubah AST dari model yang buta-fase menjadi model yang melihat keduanya.

Perbaikan #3 secara teknis paling menarik: ia menguji apakah kesenjangan waveform-vs-spektrogram memang berasal dari fase. Kalau AST + fase menyusul Wav2Vec2, hipotesis §2.3 terkonfirmasi dari arah yang berlawanan.

### 4.5 CNN-LSTM — baseline yang layak, dengan komponen yang salah

Perannya sebagai baseline konvensional itu sah dan penting untuk tesis. Tapi dua komponennya bermasalah:

- **LSTM tidak berkontribusi** untuk sinyal stasioner sepanjang 2 detik (§3.2), dan sebagai agregator lebih buruk daripada rata-rata.
- **Klasifikasi dari hidden state akhir** (✅ rumus 38, hal. 45) adalah pilihan agregasi terlemah yang tersedia.

**Perbaikan dengan menjaga identitas arsitektur:** pertahankan CNN + LSTM (agar tetap sesuai judul tesis), tetapi ganti agregasi dari `h_T` menjadi **attentive statistics pooling atas seluruh keluaran LSTM** — dan gunakan **BiLSTM** agar frame awal tidak terdiskriminasi. Perubahan kecil, dan menurut prediksi saya bernilai beberapa poin pada kondisi ber-noise.

Lalu tambahkan ablation `CNN + ASP tanpa LSTM` sebagai varian ketiga. Tiga varian ini menghasilkan tabel yang menjawab pertanyaan arsitektural nyata.

---

## 5. Arsitektur yang saya rekomendasikan

### 5.1 Prinsip desain

1. **Waveform di depan** — jangan buang fase sebelum model melihatnya.
2. **Encoder dibekukan, agregator dilatih** — 26k sampel tidak bisa melatih 95M+ parameter dengan jujur.
3. **Ambil dari semua layer, jangan hanya yang terakhir** — biarkan model memilih.
4. **Agregasi = statistik berbobot atensi**, bukan hidden state akhir, bukan rata-rata polos.
5. **Head kecil** — dua lapis linear cukup. Kapasitas ada di encoder pra-latih.
6. **Loss yang sadar sifat open-set** — kelas "fake" tidak berbatas.

### 5.2 Rancangan lengkap

```
                       Waveform 2 dtk @ 16 kHz  (32.000 sampel)
                                    │
                        ┌───────────┴───────────┐
                        │  Normalisasi          │  zero-mean unit-var
                        │  (feature extractor)  │  — BUKAN peak norm
                        └───────────┬───────────┘
                                    │
              ╔═════════════════════▼══════════════════════╗
              ║  SSL Encoder — DIBEKUKAN                   ║
              ║  wav2vec2-base / wavlm-large               ║
              ║  output_hidden_states=True                 ║
              ╚═════════════════════╤══════════════════════╝
                                    │  L+1 tensor (B, 100, D)
                                    │  L = 12 (base) atau 24 (large)
              ┌─────────────────────▼──────────────────────┐
              │  Layer-Weighted Aggregation  ◄── DILATIH   │
              │  w = softmax(α)        α ∈ ℝ^(L+1)         │
              │  H = Σ wᵢ · Hᵢ                              │
              │  → (B, 100, D)      [+13 s/d 25 parameter] │
              └─────────────────────┬──────────────────────┘
                                    │
              ┌─────────────────────▼──────────────────────┐
              │  Conv1d bottleneck  D → 256, k=5, BN, GELU │
              │  (menurunkan dimensi + konteks lokal)      │
              └─────────────────────┬──────────────────────┘
                                    │  (B, 256, 100)
              ┌─────────────────────▼──────────────────────┐
              │  Attentive Statistics Pooling              │
              │  a = softmax(conv(tanh(conv(h))))          │
              │  μ = Σ aₜ·hₜ                                │
              │  σ = √(Σ aₜ·hₜ² − μ²)                       │
              │  out = [μ ; σ]        → (B, 512)           │
              └─────────────────────┬──────────────────────┘
                                    │
              ┌─────────────────────▼──────────────────────┐
              │  BatchNorm → Linear 512→256 → GELU         │
              │  → Dropout 0,2 → Linear 256→2              │
              └─────────────────────┬──────────────────────┘
                                    │
                    CE + label smoothing 0,05   (baseline)
                    ATAU  OC-Softmax            (open-set)

Parameter yang dilatih ≈ 1,2 juta  (vs 95 juta bila fine-tune penuh)
```

### 5.3 Mengapa setiap komponen ada

| Komponen | Alasan | Perkiraan dampak 📚 |
|---|---|---|
| Encoder dibekukan | 26k sampel tidak bisa melatih 95M parameter; mencegah *catastrophic forgetting*; latih 5× lebih cepat | Stabilitas; +variansi turun |
| **Layer weighting** | Artefak ada di layer awal-menengah; layer akhir terlalu fonetik. Biaya: 13 parameter | **+1 s/d +3 poin** |
| Conv1d bottleneck | Konteks lokal lintas frame + turunkan dimensi sebelum pooling | +0,2 s/d +0,8 |
| **Attentive stats pooling** | Agregator bukti mendekati-optimal untuk sinyal stasioner (§3.2); std menangkap *konsistensi* artefak | **+0,5 s/d +2 poin** |
| Label smoothing | Kalibrasi lebih baik; mencegah overconfidence | +0,1 s/d +0,5 |
| OC-Softmax (opsi) | Kelas fake tidak berbatas → open-set | +sedikit in-domain, **+besar lintas dataset** |

### 5.4 Kalau boleh keluar dari batasan tesis: SOTA sebenarnya

📚 Untuk kelengkapan — arsitektur terbaik yang diketahui saat ini untuk anti-spoofing audio:

**`wav2vec2-XLS-R-300M` (atau WavLM-Large) dibekukan + layer weighting → back-end AASIST.**

AASIST menggantikan attentive pooling dengan **graph attention network** yang memodelkan node spektral dan temporal secara eksplisit, plus mekanisme *max-graph operation* yang menggabungkan keduanya. Ini merepresentasikan hubungan "artefak di pita X berkorelasi dengan artefak pada waktu Y" yang tidak bisa ditangkap pooling biasa.

Untuk tesis ini saya **tidak merekomendasikannya sebagai model utama** — ia keluar dari judul dan menambah risiko. Tetapi menambahkannya sebagai **satu baris pembanding "SOTA reference"** akan sangat memperkuat bab pembahasan: ia menunjukkan seberapa jauh keempat model tesis dari batas atas yang diketahui. Itu konteks yang dihargai penguji, dan biayanya rendah karena implementasinya tersedia publik.

---

## 6. Bug yang harus diperbaiki sebelum angka apa pun bermakna

Karena Anda membangun dari nol, ini bukan lagi "temuan audit" melainkan **spesifikasi implementasi**. Semua ini adalah kesalahan yang gagal secara diam-diam — tidak ada pesan error, hanya akurasi yang lebih rendah tanpa penjelasan.

### 6.1 Learning rate per model — bukan seragam

✅ Proposal hal. 68 menetapkan LR 0,001 untuk keempat model. Untuk tiga model pra-latih ini merusak.

| Model | LR encoder | LR head | Catatan |
|---|---|---|---|
| Wav2Vec2 Base | 1e-5 (atau 0 bila beku) | 1e-3 | |
| HuBERT/WavLM Large | 5e-6 (atau 0 bila beku) | 1e-3 | |
| AST | 2e-5 | 1e-3 | |
| CNN-LSTM | — | 1e-3 | Dari nol; 1e-3 memang benar |

### 6.2 AST: `max_length` dan positional embedding

Checkpoint AudioSet default: `max_length=1024` (≈10,24 dtk). Audio Anda 2 detik ≈ 200 frame. Tanpa perbaikan, **~81% masukan adalah padding**.

```python
cfg = ASTConfig.from_pretrained(CKPT, max_length=200, num_labels=2)
model = ASTForAudioClassification.from_pretrained(CKPT, config=cfg,
                                                  ignore_mismatched_sizes=True)
```
⚠️ `ignore_mismatched_sizes=True` **menginisialisasi ulang positional embedding secara acak**. Lebih baik interpolasi dari bobot terlatih (1212 → 228 patch) dengan `F.interpolate` — menyelamatkan informasi posisi yang sudah dipelajari.

### 6.3 Normalisasi ganda

✅ Proposal hal. 56–57 menerapkan **peak normalization**. Tetapi setiap feature extractor HuggingFace melakukan normalisasinya sendiri:

| Model | Normalisasi yang diharapkan |
|---|---|
| Wav2Vec2 Base | `do_normalize=False` (checkpoint base) |
| HuBERT/WavLM Large | `do_normalize=True` — zero-mean unit-var |
| AST | mean −4,2677 / std 4,5690 pada log-Mel |

Peak normalization di hulu mengubah statistik yang diasumsikan. **Gunakan loudness normalization (−23 LUFS) untuk konsistensi level antar-berkas, lalu serahkan normalisasi statistik sepenuhnya kepada feature extractor.**

### 6.4 Bekukan CNN feature extractor

Untuk Wav2Vec2/HuBERT/WavLM, **selalu** `model.freeze_feature_encoder()`. Melatih 7 lapis conv di depan hampir selalu merusak dan merupakan penyebab kegagalan fine-tuning SSL yang paling umum.

### 6.5 Split harus speaker-disjoint

Random split akan menempatkan potongan dari rekaman induk yang sama di train dan test. Gunakan partisi resmi FoR, atau `StratifiedGroupKFold` dengan `groups = speaker_id`.

### 6.6 Model selection pada EER, bukan akurasi

Pada rezim 98%, akurasi validasi sudah jenuh dan tidak lagi membedakan checkpoint. EER masih punya resolusi penuh.

---

## 7. Soal target 100%

Saya harus jujur di sini, karena ini menyangkut apa yang akan Anda pertahankan di sidang.

**100% pada FoR-2sec kemungkinan besar dapat dicapai.** Itulah masalahnya.

FoR dibangun dengan cara mengumpulkan wicara asli dari korpus terbuka dan wicara palsu dari sistem TTS. Kedua kelas karenanya berbeda **bukan hanya pada keaslian**, tetapi juga pada rantai perekaman, kondisi ruang, mikrofon, dan pipeline pemrosesan. Sebuah model yang cukup besar akan menemukan perbedaan-perbedaan itu — dan itu **lebih mudah** dipelajari daripada artefak vocoder yang sesungguhnya.

Jadi akurasi 100% memiliki dua tafsir, dan keduanya tidak bisa dibedakan dari angkanya saja:

- **Tafsir A:** model benar-benar menguasai deteksi artefak sintesis.
- **Tafsir B:** model menemukan pintasan spesifik-dataset dan akan gagal total pada data lain.

📚 Berdasarkan pola yang sangat konsisten di literatur — ✅ termasuk tiga rujukan dalam proposal Anda sendiri (ref [14]: EER 4,06% in-domain → **31,14%** In-the-Wild; ref [22]: **65,28%** lintas bahasa; ref [21]: EER **20,45%** lintas serangan) — Tafsir B jauh lebih mungkin.

**Yang membedakan keduanya bukan angka akurasi, melainkan bukti pendukungnya.** Karena itu saya sarankan Anda menyampaikan hal berikut kepada dosen:

> *Target 100% akan dikejar, tetapi disertai bukti bahwa angka itu bukan artefak dataset — melalui audit kebocoran, uji kontrol negatif, evaluasi pada noise yang belum pernah dilihat, dan pengujian lintas dataset.*

Ini **bukan** menurunkan target. Ini menaikkan standar buktinya. Hasil akhir yang saya perkirakan realistis:

| Kondisi | Perkiraan 📚 |
|---|---|
| FoR-2sec bersih, split resmi | **99,0 – 99,8%** |
| FoR-2sec, noise *seen* 15–30 dB | 98 – 99,5% |
| FoR-2sec, noise *unseen* 0–10 dB | 90 – 97% |
| Lintas dataset (In-the-Wild) | EER 15 – 30% |

Angka 99%+ pada baris pertama **sudah melampaui setiap baseline FoR dalam tabel state of the art proposal** (93,50 / 94,47 / 94,7) ✅. Sisa 1% hampir pasti terdiri dari berkas rusak dan label yang salah — bukan kekurangan model. Itu sebabnya §F3 di [ANALISIS_KRITIS.md](ANALISIS_KRITIS.md) menyarankan mendengarkan setiap berkas yang salah: pada 99%, jumlahnya cukup sedikit untuk diperiksa satu per satu, dan itu satu-satunya cara mengetahui apakah 100% secara teoretis masih mungkin.

---

## 8. Ringkasan keputusan arsitektur

| Keputusan | Pilihan | Alasan inti |
|---|---|---|
| Representasi utama | **Raw waveform** | Fase + resolusi HF utuh (§2) |
| Model terbaik dalam batasan tesis | **Wav2Vec2 Base** + layer weighting + ASP | Akses fase, kapasitas pas, biaya rendah (§4.2) |
| Penggantian yang direkomendasikan | HuBERT Large → **WavLM Large** | Pra-pelatihan denoising, selaras rumusan masalah (§4.3) |
| Strategi fine-tuning | **Encoder beku**, latih agregator + head | 26k sampel vs 95M+ parameter (§3.3) |
| Agregasi | **Attentive statistics pooling** | Penjumlah bukti optimal untuk sinyal stasioner (§3.2) |
| Pemilihan layer | **Berbobot & dipelajari, semua layer** | Artefak di layer awal-menengah (§4.2) |
| Perbaikan AST | `max_length=200` + interpolasi pos-emb + LFCC/fase | Menghapus 81% padding; kembalikan bukti (§4.4, §6.2) |
| Perbaikan CNN-LSTM | BiLSTM + ASP, plus ablation tanpa LSTM | Hidden state akhir = agregator terlemah (§4.5) |
| Loss | CE + label smoothing; **OC-Softmax** untuk lintas dataset | Kelas fake bersifat open-set (§5.1) |
| Metrik seleksi model | **EER**, bukan akurasi | Akurasi jenuh di atas 98% (§6.6) |

---

*Dokumen ini melengkapi [ANALISIS_DAN_RENCANA.md](ANALISIS_DAN_RENCANA.md) (rencana eksekusi) dan [ANALISIS_KRITIS.md](ANALISIS_KRITIS.md) (audit validitas). Klaim berlabel ✅ dapat diverifikasi pada PROPOSAL_TESIS.pdf dengan nomor halaman yang disebut; klaim berlabel 📚 adalah pengetahuan domain dengan tingkat keyakinan yang dicantumkan dan harus divalidasi secara empiris pada data Anda.*
