# RENCANA EKSPERIMEN E0–E9 — Ketahanan Noise & Akurasi Maksimum pada FoR-2sec
### Disusun terhadap kondisi nyata repo `C:\Users\Tristan\Downloads\general-ai` (24 run tersimpan, dataset lokal, GPU terverifikasi)

---

## 0. Titik berangkat faktual (bukan asumsi)

Saya membaca kode dan seluruh `results.json` yang ada. Ini kondisi sebenarnya:

| Fakta terukur | Nilai | Sumber |
|---|---|---|
| Dataset lokal | `data/for-2seconds/` 17.870 wav, 16 kHz mono 2,000 s | `TEMUAN_GROUND_TRUTH.md` |
| Split resmi | 13.956 / 2.826 / **1.088** | idem |
| Ensemble 4 arsitektur, split resmi | **97,61%**, EER 2,30%, 26 salah dari 1.088 | `PERBANDINGAN.md` |
| Model tunggal terbaik (rerata 3 seed) | `cnn_asp` 91,94% ±3,50 | idem |
| **HuBERT — model ke-4 tesis** | **belum pernah dijalankan sama sekali** | `runs/` tidak berisi `hubert_*` |
| Ketahanan noise | **belum diukur sama sekali** dengan noise nyata | tidak ada run `--augment noise` yang dievaluasi per-SNR |

**Konsekuensi perencanaan:** akurasi in-domain sudah hampir jenuh (97,61% ensemble; 99,75% split acak). Sisa 26 berkas salah = 2,39 pp. Kenaikan akurasi yang berarti **tidak akan datang dari tuning** — hanya dari (a) front-end yang jauh lebih kuat, (b) pelatihan multi-kondisi, (c) fusi. Sementara rumusan masalah tesis (ketahanan noise) **belum punya satu pun angka**. Rencana ini menyelesaikan keduanya.

### Waktu per-epoch TERUKUR di RTX 5060 Ti Anda (b=32, 13.956 klip, aug=codec, 6 worker)

| Model | s/epoch | param dilatih | catatan |
|---|---|---|---|
| `cnn_asp` | **6,9** | 1,54 M | |
| `cnnlstm` | **10,8** | 6,79 M | |
| `wav2vec2-base` (beku) | **25,9** | 1,18 M | |
| `ast` (beku) | **44,5** | 1,18 M | fbank per-sampel, tidak di-batch |
| `cnn_asp` + `--augment full` (b=64) | **200,8** | 1,54 M | **41× lebih lambat** dari clean (4,9 s) |

Baris terakhir adalah blocker terbesar rencana ini dan diagnosisnya pasti — lihat E0.2.

---

## 1. Urutan eksperimen

### **E0 — Perbaikan infrastruktur + baseline regresi** *(wajib, semua bergantung padanya)*

**Tujuan:** menutup 3 bug diam, menutup gap HuBERT, dan menaikkan seed dari 3 → 5 supaya perbandingan arsitektur bermakna (std terukur ±3,50 pp; dengan n=3 CI rerata ±4,3 pp, dengan n=5 ±3,1 pp).

#### E0.1 — Bug RNG augmentasi konstan antar-epoch

`forlib/data.py:293`:
```python
rng = np.random.default_rng((_stable_hash(r["fname"]) + self.seed) % (2 ** 31))
```
Seed hanya bergantung nama berkas + seed run. **Setiap berkas menerima augmentasi yang persis sama di setiap epoch.** Augmentasi Anda bukan stokastik — ia transformasi tetap satu kali. Efektif Anda melatih pada 13.956 sampel teraugmentasi statis, bukan 13.956 × 10 epoch varian.

Perbaikan: tambahkan `epoch` ke state dataset dan masukkan ke seed.
```python
seed_i = (_stable_hash(r["fname"]) ^ (self.seed * 0x9E3779B1) ^ (self.epoch * 0x85EBCA6B)) & 0x7FFFFFFF
```
plus `ds_tr.set_epoch(ep)` di awal tiap epoch training loop.

**Kriteria sukses:** dua panggilan `ds[i]` pada epoch berbeda menghasilkan tensor berbeda (assert `not torch.allclose`), dan pada epoch sama menghasilkan tensor identik (reproduksibilitas dipertahankan).

#### E0.2 — Bug performa: reverb `np.convolve` (blocker)

`apply_reverb` melakukan `np.convolve(x, ir)` dengan `len(x)=32.000`, `len(ir)` sampai `0,7×16.000 = 11.200` → **3,6×10⁸ MAC per klip dalam numpy murni**, ±86 ms/klip. Dengan 6 worker itu ~70 klip/s → 200 s/epoch.

Perbaikan: `scipy.signal.fftconvolve` (atau FFT di torch, di GPU, per-batch). Biaya turun ke ~1 ms/klip.

**Kriteria sukses terukur:** `cnn_asp --augment full --batch 64` turun dari **200,8 s/epoch** ke **< 12 s/epoch**. Bila tidak tercapai, profil ulang dengan `torch.utils.data` timing — jangan lanjut ke E2/E3 sebelum lolos, karena grid noise memerlukan ~500 epoch-ekuivalen.

#### E0.3 — Bug `gain` = no-op total

```python
x = augment_waveform(x, self.aug, rng)   # di dalamnya: x *= 10**(U(-8,8)/20)
if self.normalize == "loudness":
    x = loudness_normalize(x)            # set RMS ke 0,05 → gain DIBATALKAN
```
Augmentasi gain (p=0,3) tidak berpengaruh apa pun. Ablation "dengan/tanpa gain" akan menghasilkan selisih 0,00 dan Anda akan menyimpulkan hal yang salah. Pilih satu: buang `gain` dari config, atau pindahkan normalisasi loudness ke **sebelum** augmentasi dan matikan renormalisasi sesudahnya.

#### E0.4 — Bug encoder beku tetap dalam mode `train()`

`train.py` memanggil `model.train()`; ini merambat ke `self.encoder` walau semua parameternya `requires_grad=False`. Untuk `Wav2Vec2Model` HF itu berarti, di setiap forward saat training:
- `config.apply_spec_augment=True` → **masking waktu acak** pada feature vectors (`mask_time_prob=0,05`),
- `config.layerdrop` > 0 → **layer di-skip acak**, sehingga `output_hidden_states` yang dipakai `LayerWeighting` punya jumlah/isi yang berubah-ubah tiap step,
- dropout internal aktif.

Saat evaluasi semuanya mati. Jadi `LayerWeighting` dilatih di atas distribusi hidden state yang berbeda dari yang dipakai saat inferensi. Ini **tidak melempar error apa pun**.

Perbaikan:
```python
if self.frozen:
    self.encoder.eval()                       # panggil ulang setiap kali setelah model.train()
self.encoder.config.layerdrop = 0.0
self.encoder.config.apply_spec_augment = False
```
plus override `train()` di `SSLClassifier` agar encoder tetap `eval()` saat beku.

**Kriteria sukses:** dua forward berturut pada input identik dalam mode `train()` menghasilkan output identik untuk bagian encoder (assert allclose pada `out.hidden_states[-1]`).

#### E0.5 — Menutup gap HuBERT + menyamakan skala model

Saat ini `SSL_CKPT["hubert"] = "facebook/hubert-large-ll60k"` (317 M) sementara `wav2vec2` = `facebook/wav2vec2-base` (95 M). Membandingkan keduanya sebagai "arsitektur" adalah **membandingkan kapasitas**, bukan arsitektur.

Konfigurasi tetap:
```
Perbandingan utama tesis (adil, base-scale):
  wav2vec2 = facebook/wav2vec2-base
  hubert   = facebook/hubert-base-ls960     <-- TAMBAHKAN
  ast      = MIT/ast-finetuned-audioset-10-10-0.4593  (max_length=200)
  cnnlstm  = CNN-BiLSTM + ASP, n_mels=128

Ablation skala (dilaporkan terpisah, jangan dicampur ke tabel utama):
  hubert-large-ll60k, wavlm-large, wav2vec2-xls-r-300m
```

#### E0.6 — Run baseline regresi

```
untuk model in {wav2vec2, hubert, ast, cnnlstm, cnn_asp}:
  untuk seed in {42, 1337, 2024, 7, 20260805}:
    py scripts/06_train.py --model $model --split official --augment codec \
       --epochs 15 --batch 32 --patience 6 --seed $seed
```

**Estimasi waktu GPU (setelah E0.2 diperbaiki):**

| Model | s/ep (est.) | ×15 ep | ×5 seed |
|---|---|---|---|
| `cnn_asp` | 7 | 1,8 mnt | 9 mnt |
| `cnnlstm` | 11 | 2,8 mnt | 14 mnt |
| `wav2vec2-base` | 26 | 6,5 mnt | 33 mnt |
| `hubert-base` | ~26 *(estimasi — ukuran identik w2v2-base)* | 6,5 mnt | 33 mnt |
| `ast` | 45 | 11 mnt | 56 mnt |
| **Total E0.6** | | | **≈ 2,5 jam** |

**VRAM:** terukur 0,2–0,5 GiB untuk `cnn_asp`. Estimasi w2v2/hubert-base beku b=32: 1,5–2,5 GiB; AST b=32: 2–3 GiB. **Anda memakai < 20% dari 16 GB.** Naikkan batch ke 64 pada model kecil; bottleneck Anda CPU, bukan GPU.

**Kriteria sukses E0 (gate):**
1. `cnn_asp/official/codec` 5 seed masuk rentang **91,94% ± 2×3,50** = [84,9 ; 98,9] — kalau di luar, ada regresi dari perbaikan bug (kemungkinan besar E0.4 mengubah hasil w2v2/AST; **itu diharapkan dan harus dilaporkan sebagai koreksi**, bukan dianggap kegagalan).
2. `hubert-base` menghasilkan hasil pertama kalinya (≥ 5 run).
3. Throughput `--augment full` ≥ 1.100 klip/s.
4. Skor test punya **≥ 900 nilai unik dari 1.088** (deteksi kuantisasi bf16 — lihat Jebakan #8).

---

### **E1 — Harness evaluasi noise deterministik** *(fondasi rumusan masalah)*

**Tujuan:** membuat test-set berderau yang **dibuat sekali, disimpan ke disk, di-hash, dan tidak pernah berubah**. Semua model dievaluasi pada berkas byte-identik. Ini yang membuat McNemar sah antar model dan antar kondisi.

**Protokol korpus-disjoin (ini kontribusi metodologis, bukan detail teknis):**

| Peran | Korpus | Alasan |
|---|---|---|
| Noise **pelatihan** (E3) | MUSAN `noise/free-sound` + `noise/sound-bible` + `music` | besar, beragam |
| RIR **pelatihan** | `RIRS_NOISES/simulated_rirs` | sintetis, terpisah dari real |
| Noise **evaluasi** | **DEMAND** (17 lingkungan nyata) | korpus berbeda total → benar-benar *unseen* |
| RIR **evaluasi** | `RIRS_NOISES/real_rirs_isotropic_noises` | real, disjoin dari simulated |
| Babble **evaluasi** | dari **DEMAND** `PCAFETER`/`SCAFE`/`SPSQUARE` | **jangan** dibuat dari klip real FoR (lihat Jebakan #13) |

Hampir semua paper ketahanan noise memakai korpus yang sama untuk latih dan uji dan menyebutnya "unseen noise". Anda tidak. Nyatakan itu eksplisit di Bab 3.

**Grid evaluasi (dibekukan):**

```
Kondisi = clean  ∪  {noise_type} × {SNR}
noise_type (DEMAND, 16 kHz): STRAFFIC, TCAR, PCAFETER, OOFFICE, NPARK   (5)
SNR (dB)                   : -5, 0, 5, 10, 20                          (5)
reverb                     : RT60 rendah / sedang / tinggi (real RIR)   (3)
→ 1 + 25 + 3 = 29 kondisi × 1.088 berkas = 31.552 wav ≈ 2,0 GB
```

**Aturan level:** hitung SNR terhadap **active speech level ITU-T P.56**, bukan RMS seluruh klip. FoR-2sec sudah membuang silence, jadi selisihnya kecil, tapi harus dinyatakan agar angka SNR Anda dapat dibandingkan orang lain. Simpan level aktif terukur ke manifest.

**Waktu:** CPU murni, `fftconvolve`, ~5 ms/berkas → **< 10 menit**. GPU: nol.

**Kriteria sukses:**
1. `sha256` manifest kondisi stabil pada dua kali pembuatan dengan seed sama.
2. SNR terukur ulang dari (clean, noisy) pasangan menyimpang < 0,2 dB dari target.
3. Tidak ada clipping (`|x|max ≤ 0,99`) pada 100% berkas — kalau ada, catat dan renormalisasi *sebelum* mixing, bukan sesudah.

---

### **E2 — Kurva degradasi baseline + kontrol pintasan** *(jawaban langsung rumusan masalah, bagian 1)*

**Tujuan:** untuk keempat model tesis yang dilatih **bersih** (aug=codec saja), ukur akurasi/EER/AUC di 29 kondisi × 5 seed.

**Konfigurasi:** inferensi saja, checkpoint dari E0.6, batch 64.

**Waktu GPU:** 29 kondisi × 1.088 berkas × 5 model × 5 seed = 789 evaluasi. Inferensi 1.088 berkas: `cnn_asp` ~1 s, `w2v2/hubert-base` ~3 s, `ast` ~5 s, ditambah overhead loading model ~10 s per run. Amortisasi dengan memuat model **sekali** lalu iterasi 29 kondisi → **≈ 1,5–2,5 jam total**.

#### E2b — Kontrol pintasan (murah, orisinal, dan wajib)

`TEMUAN_GROUND_TRUTH.md` §7.3 memperingatkan: noise menutupi pita tinggi — persis tempat pintasan codec berada. Sebagian "penurunan akibat noise" yang akan Anda ukur sebenarnya adalah "pintasan yang tertutup noise", dan sebagian bahkan bisa **menaikkan** akurasi.

Kontrolnya sederhana: jalankan **Random Forest atas 38 fitur sepele** (`probe_split.py` yang sudah ada) pada setiap kondisi berderau.

| Pola yang terlihat | Tafsir |
|---|---|
| RF datar, DL turun | penurunan itu **nyata** — noise merusak artefak sintesis |
| RF turun sejalan DL | sebagian penurunan adalah pintasan yang hilang, bukan ketahanan |
| RF naik, DL naik | noise menyamarkan *domain shift* → efek yang berlawanan intuisi, dan **temuan yang layak dilaporkan** |

**Waktu:** CPU, ~15 menit untuk seluruh grid.

**Kriteria sukses E2:**
1. Kurva EER **monoton naik** saat SNR turun untuk minimal 3 dari 4 model. Bila tidak monoton, curigai bug mixing sebelum menafsirkan.
2. Selisih antar-model pada tiap SNR diuji **McNemar berpasangan + Holm** (fungsi sudah ada di `forlib/metrics.py`).
3. Semua angka dilaporkan dengan **ambang induktif** (dari validasi), bukan prior-matched. Prior-matched boleh dilaporkan sebagai kolom kedua dan **harus** diberi label transduktif.

---

### **E3 — Multi-condition training (MCT) + RawBoost** *(pengungkit ketahanan utama)*

**Tujuan:** ini intervensi yang benar-benar menaikkan ketahanan. Latih ulang keempat model dengan augmentasi noise nyata dari MUSAN + RIR simulated, **seragam pada kedua kelas**.

**Konfigurasi persis:**
```
--augment mct
  reverb  p=0.30  RIR: RIRS_NOISES/simulated_rirs, fftconvolve
  codec   p=0.60  (dipertahankan — memperbaiki pintasan MP3, sudah terbukti)
  noise   p=0.70  MUSAN, SNR ~ U(-5, 25) dB, tipe ~ {noise, music, babble-MUSAN}
  rawboost p=0.50 seri 1+2+3 (linear+non-linear conv, impulsif, stasioner)
  gain    DIBUANG (no-op, lihat E0.3)
--epochs 25  --patience 8  --batch 32
```

**Validasi harus ikut ter-noise.** Kalau validasi bersih sementara latih berderau, pemilihan checkpoint memilih model yang paling pandai pada kondisi bersih. Pakai `--augment-val` dengan seed berbeda (`seed+777`, sudah ada di `train.py`) **dan** noise dari subset MUSAN yang disjoin dari subset latih.

**Waktu GPU (est., setelah E0.2):**

| Model | s/ep MCT | ×25 ep | ×5 seed |
|---|---|---|---|
| `cnnlstm` | ~15 | 6 mnt | 31 mnt |
| `wav2vec2-base` | ~32 | 13 mnt | 67 mnt |
| `hubert-base` | ~32 | 13 mnt | 67 mnt |
| `ast` | ~52 | 22 mnt | 108 mnt |
| `cnn_asp` | ~11 | 4,6 mnt | 23 mnt |
| **Total E3** | | | **≈ 5 jam** |

Lalu ulangi E2 di atas checkpoint MCT: **+2 jam**.

**Kriteria sukses (falsifiable, tidak mengandaikan hasil):**
1. Pada SNR 0 dB, EER model MCT **lebih rendah** daripada model bersih untuk ≥ 3 dari 4 model, dengan McNemar terkoreksi p < 0,05.
2. Pada kondisi **clean**, penurunan akurasi model MCT vs model bersih **≤ 1,5 pp** (biaya ketahanan harus kecil; kalau lebih besar, turunkan p noise).
3. Ablation terpisah RawBoost on/off dilaporkan — jangan gabungkan dua intervensi dalam satu angka.

---

### **E4 — Front-end SSL kelas SOTA + unfreezing bertahap** *(pengungkit akurasi utama)*

**Tujuan:** ini satu-satunya jalur realistis menuju "mendekati 100%" pada protokol resmi. Encoder yang beku dengan kepala 1,18 M parameter tidak akan mengalahkan `cnn_asp` 1,54 M. Yang mengubah keadaan di literatur anti-spoofing adalah **XLS-R-300M yang di-fine-tune + back-end graph-attention**, bukan encoder beku.

**Tangga eksperimen (jalankan berurutan, berhenti bila E4.x gagal gate):**

| Tahap | Konfigurasi | b | VRAM est. | s/ep est. |
|---|---|---|---|---|
| E4.1 | `wav2vec2-base`, unfreeze 4 lapis atas, enc_lr 1e-5 | 32 | 4–6 GB | ~55 |
| E4.2 | `wav2vec2-base`, unfreeze penuh, enc_lr 5e-6 | 32 | 7–9 GB | ~75 |
| E4.3 | `wavlm-base-plus`, unfreeze penuh | 32 | 7–9 GB | ~80 |
| E4.4 | `wav2vec2-xls-r-300m` beku + kepala | 32 | 4–6 GB | ~110 |
| E4.5 | **`xls-r-300m` unfreeze 12 lapis atas**, enc_lr 3e-6, grad-ckpt | 16 | 9–12 GB | ~300 |
| E4.6 | `xls-r-300m` unfreeze penuh, grad-ckpt, accum 2 | 8×2 | 11–14 GB | ~550 |
| E4.7 | E4.5 + back-end **AASIST** (graph attention) menggantikan ASP | 16 | 10–13 GB | ~330 |

*Semua s/ep dan VRAM di tabel ini adalah **estimasi** yang harus diukur langsung — jalankan 1 epoch dulu sebelum meluncurkan 5 seed. Klip Anda hanya 2 detik (T=99 frame setelah CNN feature extractor), sehingga biaya attention nyaris nol dan biaya didominasi CNN extractor atas 32.000 sampel; ini membuat XLS-R jauh lebih murah di sini daripada pada klip 4–10 detik yang lazim di literatur.*

**Catatan wajib:** saat unfreeze, `freeze_feature_encoder()` **tetap** dipanggil (7 lapis conv depan jangan pernah dilatih) — kode Anda sudah benar di sini.

**Waktu GPU E4:** eksplorasi 1-seed untuk E4.1–E4.7 ≈ **4–6 jam**. Lalu 5 seed untuk 1–2 konfigurasi pemenang: E4.5 = 300 s × 15 ep × 5 = **6,3 jam**; E4.7 serupa. **Total realistis E4: 15–22 jam GPU.**

**Kriteria sukses:**
1. Konfigurasi pemenang mengalahkan `cnn_asp` (91,94% ±3,50) pada split resmi dengan McNemar terkoreksi p < 0,05 **atas ≥ 3 pasangan seed**, bukan pasangan seed terbaik.
2. Std antar-seed ≤ ±2,0 pp (kalau lebih besar, model terlalu tidak stabil untuk diklaim sebagai perbaikan).
3. Pada grid noise E2, konfigurasi pemenang tidak lebih buruk daripada baseline pada SNR ≥ 10 dB.

---

### **E5 — Uji terkontrol konsistensi magnitudo–fase** *(kontribusi ilmiah orisinal)*

**Ini realisasi teknik yang lolos verifikasi, dirancang ulang supaya benar-benar menguji mekanisme, bukan sekadar mengkorelasikan PESQ dengan EER (yang sudah dilakukan paper sumber).**

**Ide inti.** Jangan bandingkan enhancer sebagai kotak hitam. Bangun lengan yang **magnitudonya identik dan hanya fasenya berbeda**. Itu isolasi kausal sempurna dan tidak memerlukan SEGAN sama sekali.

**Lima lengan atas kondisi berderau yang sama:**

| Lengan | Magnitudo | Fase | Konsistensi |
|---|---|---|---|
| A0 `noisy` | noisy | noisy | konsisten (sinyal nyata) |
| A1 `mask+noisyphase` | masker Wiener/MetricGAN+ | **fase noisy** | **tidak konsisten** |
| A2 `mask+oraclephase` | **masker identik A1** | fase clean (oracle) | konsisten |
| A3 `mask+griffinlim` | **masker identik A1** | Griffin-Lim 32 iter | mendekati konsisten |
| A4 `waveform-enh` | — | — | konsisten (domain waktu) |

A1/A2/A3 **berbagi magnitudo yang persis sama**. Selisih di antaranya **hanya** fase. Ini yang tidak dilakukan siapa pun di literatur deteksi deepfake.

**Ukuran inkonsistensi (Le Roux dkk., *consistent Wiener filtering*):**
```
C(S) = 20·log10( ‖S − STFT(iSTFT(S))‖_F / ‖S‖_F )   [dB]
```
Skalar tunggal per berkas, murah, well-defined, dan **skalanya invarian**. Hitung untuk setiap berkas di setiap lengan.

**Prediksi diferensial yang diuji:**

| Kelompok model | Representasi | Prediksi |
|---|---|---|
| **Phase-blind**: AST (kaldi fbank), CNN-LSTM (MelSpectrogram `power=2.0`) | magnitudo saja | relatif kebal antar A1/A2/A3 |
| **Phase-aware**: Wav2Vec2, HuBERT (raw waveform) | gelombang penuh | degradasi **lebih besar** pada A1 |
| *(opsional, titik ketiga)* **APGDF/MGD** dari `probe_apgdf.py` | **fase** di domain spektral | degradasi **terbesar** pada A1 |

Perhatikan kehalusan yang memperkuat desain: model phase-blind **tidak sepenuhnya kebal**, karena mereka melakukan STFT ulang dengan window/hop sendiri atas `iSTFT(M, P_noisy)`, sehingga yang mereka lihat adalah proyeksi konsisten dari `M`, bukan `M`. Jadi prediksinya bukan "nol vs sesuatu" melainkan **gradien yang terurut**, dan urutannya dapat diuji.

**Analisis:** regresi ΔEER (relatif A0) terhadap `C(S)` rata-rata per lengan, **terpisah untuk kelompok phase-blind dan phase-aware**. Uji interaksi `C × kelompok`. Bila koefisien interaksi signifikan → mekanisme terkonfirmasi secara terukur.

**Ruang lingkup (supaya biayanya jujur "rendah"):** batasi ke 2 tipe noise (1 stasioner `STRAFFIC`, 1 non-stasioner `PCAFETER`) × 3 SNR (0, 10, 20) = **6 kondisi** × 5 lengan × 1.088 = 32.640 wav ≈ 2,1 GB.

**Waktu:** A1–A3 adalah DSP murni (numpy/torch), ~10 menit CPU. A4 (enhancer waveform di GPU) 30–90 menit. Evaluasi 5 model × 30 kombinasi ≈ 1 jam GPU. **Total E5 ≈ 3 jam.**

**Kriteria sukses:**
1. `C(S)` untuk A1 **jelas lebih buruk** (lebih besar dalam dB) daripada A0/A2/A4 — kalau tidak, implementasi masker/iSTFT Anda salah dan seluruh eksperimen batal.
2. Koefisien interaksi `C × kelompok` punya p < 0,05 **atau** dilaporkan sebagai hasil null yang jujur. Hasil null tetap layak naskah karena desainnya terkontrol.
3. **Wajib diverifikasi sendiri sebelum ditulis:** baca `speechbrain/lobes/models/MetricGAN.py` dan konfirmasi bahwa generator memprediksi masker magnitudo dan fase noisy dipakai ulang untuk iSTFT. Sitasi kode, bukan ingatan. Baca juga Pascual dkk. (Interspeech 2017) untuk konfirmasi SEGAN beroperasi di domain gelombang. Satu paragraf Bab Pembahasan, bukan satu bab.

---

### **E6 — Generalisasi lintas-korpus** *(uji kejujuran terkeras)*

**Tujuan:** angka 97,61% Anda diperoleh pada 1.088 berkas dari korpus yang sama. Tanpa E6, sidang akan menanyakan "apakah ini berlaku di luar FoR?" dan Anda tidak punya jawaban.

| Target | Isi | Status lokal |
|---|---|---|
| `for-rerec` | skenario rekam-ulang (mikrofon + ruangan nyata) | **sudah diunduh**, `for-rerec.tar.gz` 1.558 MB, belum diekstrak |
| **In-the-Wild** | 20,8 j bonafide + 17,2 j spoof, 58 tokoh publik | perlu unduh |
| ASVspoof 2021 DF *(opsional)* | evaluasi deepfake skala besar | perlu unduh, besar |

`for-rerec` adalah **tes ketahanan lingkungan yang paling relevan dan gratis** — degradasinya nyata, bukan disintesis. Jalankan ini lebih dulu.

**Waktu:** `for-rerec` inferensi 5 model × 5 seed ≈ **1 jam**. In-the-Wild perlu chunking ke 2 s (menghasilkan ~68.000 chunk) → **2–4 jam**.

**Kriteria sukses:** angka apa pun dapat diterima — ini eksperimen deskriptif. Yang **tidak** dapat diterima adalah tidak melaporkannya. Laporkan EER lintas-korpus berdampingan dengan EER in-domain.

---

### **E7 — Fusi, kalibrasi, dan angka headline**

**Tujuan:** memaksimalkan angka yang diminta dosen tanpa mengorbankan kejujuran.

Bukti yang sudah ada: φ error antar-arsitektur 0,058–0,223 (jauh di bawah 0,5) → fusi berguna, dan **sudah terbukti +5,67 pp**. Ensemble 12 run = ensemble 4 seed-terbaik (97,61% keduanya) → keuntungan berasal dari **keberagaman arsitektur**, bukan rata-rata seed. Maka: tambahkan keberagaman *representasi*, bukan seed.

Anggota fusi yang diusulkan:
```
1. cnnlstm      (mel magnitudo)          — phase-blind
2. ast          (kaldi fbank)            — phase-blind, patch transformer
3. wav2vec2     (raw waveform)           — phase-aware
4. hubert       (raw waveform)           — phase-aware
5. xls-r + AASIST dari E4                — phase-aware, kapasitas besar
6. APGDF/MGD-LR dari probe_apgdf.py      — fase eksplisit, biaya ~nol
```
Bobot fusi **dipelajari dari validasi**, jangan dari test. Laporkan tiga varian: rata-rata sederhana, logistic-regression-on-validation, dan rata-rata setelah temperature scaling per model.

**Waktu:** menit (skor sudah tersimpan di `test_scores.npy`).

**Kriteria sukses:**
1. Fusi 6-anggota mengalahkan fusi 4-anggota (97,61%) dengan McNemar terkoreksi p < 0,05. Ingat: 1 berkas = 0,092 pp; untuk signifikan Anda perlu memperbaiki **≥ 10 berkas bersih**.
2. ECE setelah kalibrasi < 0,05.
3. Kurva fusi di seluruh grid noise E2 dilaporkan, bukan hanya pada kondisi bersih.

---

### **E8 — Analisis error terarah** *(rasio nilai/biaya tertinggi di seluruh rencana)*

Pada 97,61% hanya ada **26 berkas salah = 52 detik audio**. Dengarkan **semuanya** (15 menit kerja). Kategorikan: durasi efektif, pembicara, tipe TTS, level, artefak. Ini akan memberi tahu Anda persis apa yang menghalangi 100% — jauh lebih informatif daripada sweep berhari-hari.

Lalu ulangi pada kondisi 0 dB: berkas mana yang bertahan dan mana yang runtuh.

**Waktu:** ~2 jam manusia, 0 jam GPU. **Hasil: 1 tabel + 1 gambar orisinal untuk naskah.**

---

### **E9 — Reproduksi & pembekuan hasil**

Regenerasi seluruh tabel/gambar dari `runs/**/results.json` + `test_scores.npy` lewat satu skrip (`scripts/09_report.py`). Tidak ada angka di naskah yang diketik tangan.

**Kriteria sukses:** `py scripts/09_report.py --check` memverifikasi setiap angka di naskah cocok dengan artefak run, dan gagal keras bila tidak.

---

## 2. Ringkasan anggaran GPU

| Eks. | Isi | GPU | Wall-clock realistis |
|---|---|---|---|
| E0 | perbaikan bug + 25 run baseline | 2,5 j | 2 hari (mayoritas debugging) |
| E1 | bangun 29 kondisi eval | ~0 (CPU 10 mnt) | 0,5 hari |
| E2 | grid degradasi + kontrol pintasan | 2,5 j | 1 hari |
| E3 | MCT + RawBoost, 25 run + re-eval | 7 j | 2 hari |
| E4 | tangga SSL sampai XLS-R+AASIST | **15–22 j** | **5–8 hari** |
| E5 | lengan fase terkontrol + konsistensi | 3 j | 2 hari |
| E6 | for-rerec + In-the-Wild | 3–5 j | 2 hari |
| E7 | fusi + kalibrasi | 0,2 j | 0,5 hari |
| E8 | analisis error manual | 0 | 0,5 hari |
| E9 | pembekuan reproduksi | 0,2 j | 0,5 hari |
| **TOTAL** | | **≈ 34–43 jam GPU** | **≈ 4–5 minggu paruh waktu** |

GPU bukan batasan Anda. **Batasan Anda adalah 6 core CPU untuk augmentasi dan waktu debugging.** Rencanakan sesuai itu.

---

## 3. Dataset & korpus noise yang harus diunduh

### Sudah lokal — jangan unduh ulang

| Berkas | Ukuran | Status |
|---|---|---|
| `for-2sec.tar.gz` | 1.000 MB | ✅ terekstrak, 17.870 wav |
| `for-rerec.tar.gz` | 1.558 MB | ✅ terunduh, **belum diekstrak** (E6) |

Sumber resmi yang berfungsi (Google Drive asli kena kuota): `https://bil.eecs.yorku.ca/share/for-2sec.tar.gz`. Tersedia juga `for-norm` (5.945 MB) dan `for-original` (7.926 MB).

### Harus diunduh — terverifikasi di sesi ini

| Korpus | URL | Ukuran | Lisensi | Dipakai untuk |
|---|---|---|---|---|
| **MUSAN** | `https://openslr.trmal.net/resources/17/musan.tar.gz` | **11 GB** | CC BY 4.0 | noise **pelatihan** E3 |
| **RIRS_NOISES** | `https://openslr.trmal.net/resources/28/rirs_noises.zip` | **1,3 GB** | Apache 2.0 | RIR latih (simulated) + eval (real) |
| **DEMAND** | `https://zenodo.org/records/1227121` | 16 kHz: **±78–130 MB per lingkungan**, 17 lingkungan (`SCAFE` hanya tersedia 48 kHz) → total 16 kHz ≈ **1,6 GB** | CC (cek di Zenodo) | noise **evaluasi** E1/E2/E5 |
| **In-the-Wild** | `https://huggingface.co/datasets/mueller91/In-The-Wild` | 20,8 j bonafide + 17,2 j spoof, 58 tokoh publik — *ukuran GB tidak dinyatakan di halaman resmi; cek sebelum unduh* | Apache 2.0 | generalisasi E6 |

Mirror openslr: `openslr.elda.org` (EU), `openslr.magicdatatech.com` (CN).

Unduh 16 kHz DEMAND saja — Anda tidak butuh 48 kHz dan itu menghemat ~3,5 GB.

### Opsional — URL **belum saya verifikasi di sesi ini**, konfirmasi dulu

| Korpus | Perkiraan lokasi | Catatan |
|---|---|---|
| ASVspoof 2019 LA | `datashare.ed.ac.uk` (Univ. Edinburgh) | untuk melatih ulang XLS-R+AASIST lalu transfer ke FoR |
| ASVspoof 2021 DF eval | Zenodo | besar (puluhan GB); hanya bila E6 mau diperluas |
| WHAM! noise | `wham.whisper.ai` | alternatif DEMAND; tidak diperlukan bila DEMAND sudah dipakai |

**Total disk yang harus disediakan:** ~14 GB unduhan + ~4 GB set eval yang dibangkitkan + ~10 GB checkpoint (XLS-R 300M × 5 seed ≈ 6 GB). Anda punya **439 GB bebas**. Aman.

---

## 4. Repo yang layak di-fork, dan apa yang diambil

| Repo | Yang diambil | Yang **jangan** diambil |
|---|---|---|
| **`TakHemlata/SSL_Anti-spoofing`** *(terverifikasi: XLSR-300M front-end, `RawBoost.py` ada, dilaporkan EER 0,82% ASVspoof21-LA / 2,85% DF)* | **`RawBoost.py`** (salin utuh — augmentasi terbaik untuk tugas ini, tanpa data tambahan), pola wiring XLSR→back-end, LR/optimizer untuk fine-tune SSL | `main_SSL_*.py` (terikat protokol ASVspoof); jangan pakai loader datanya |
| **`TakHemlata/RawBoost-antispoofing`** *(URL belum diverifikasi sesi ini)* | referensi seri 1/2/3 RawBoost bila versi di repo atas kurang lengkap | — |
| **`clovaai/aasist`** *(URL belum diverifikasi sesi ini)* | **back-end graph-attention** (`models/AASIST.py`) untuk menggantikan `AttentiveStatsPool` di E4.7 | front-end SincConv-nya (Anda sudah punya SSL); jangan pakai config `d_args` mentah |
| **`speechbrain/speechbrain`** | `lobes/models/MetricGAN.py` — **baca untuk memverifikasi klaim masker-magnitudo**; wrapper inferensi `speechbrain/metricgan-plus-voicebank` | jangan jadikan dependensi pelatihan; pakai sebagai alat DSP saja |
| **`facebookresearch/denoiser`** *(URL belum diverifikasi)* | enhancer **domain gelombang** untuk lengan A4 di E5 — pengganti SEGAN yang terawat dan mudah dijalankan | — |
| **`santi-pdp/segan_pytorch`** *(URL belum diverifikasi)* | hanya bila Anda benar-benar ingin SEGAN asli; **saya sarankan lewati** — checkpoint sulit didapat dan A4 sudah terpenuhi oleh denoiser | — |
| **`asteroid-team/torch-audiomentations`** | `ApplyImpulseResponse`, `AddBackgroundNoise` — versi **GPU-batched**, langsung menyelesaikan bottleneck E0.2 | jangan campur dengan augmentasi numpy Anda; pilih satu jalur |
| **`huggingface/transformers`** | `Wav2Vec2Model`, `HubertModel`, `WavLMModel`, `ASTModel` — sudah Anda pakai | — |

**Strategi fork yang saya sarankan:** jangan fork apa pun sebagai basis. Repo Anda sendiri sudah lebih bersih daripada ketiga repo anti-spoofing itu (Anda punya manifest, split terkontrol, McNemar, Holm, temperature scaling — mereka tidak). **Salin file, bukan arsitektur repo.** Yang benar-benar wajib disalin cuma dua: `RawBoost.py` dan `AASIST.py`.

---

## 5. Struktur kode yang disarankan

Repo Anda sekarang: `forlib/{data,models,metrics}.py` + 10 skrip di root. Itu sudah mulai pecah (`data.py` menampung manifest + split + 5 augmentasi + Dataset). Struktur target:

```
forlib/
  __init__.py
  config.py              # dataclass ExpConfig; SATU sumber kebenaran untuk seed/path/hparam
  manifest.py            # build_manifest, load_manifest, assert_provenance()   <- dari data.py
  splits.py              # official / random / clean_val / wavval (+ wavval yang diperbaiki)
  audio.py               # sf.read, resample, loudness_normalize, active_speech_level_p56()
  noisebank.py           # indeks MUSAN/DEMAND/RIR + pemisahan korpus latih vs uji
  mixing.py              # add_noise_snr(), apply_rir() via fftconvolve, guard clipping
  augment/
    __init__.py          # AugmentConfig, build_augmenter()
    cpu.py               # jalur numpy LAMA — dipertahankan verbatim untuk reproduksi run lama
    gpu.py               # jalur torch batched (perbaikan E0.2) — dipanggil di training loop
    rawboost.py          # disalin dari TakHemlata/SSL_Anti-spoofing
    codec.py             # fft_lowpass + spectral holes (dari data.py)
  datasets.py            # FoRDataset (dgn set_epoch!), PrebuiltEvalDataset (baca wav grid E1)
  models/
    __init__.py          # build_model(), DEFAULT_LR
    blocks.py            # AttentiveStatsPool, LayerWeighting, Head
    ssl.py               # SSLClassifier + unfreeze_top_k(), eval-mode-when-frozen fix
    ast.py               # ASTClassifier + _fbank BATCHED + subtract-mean fix
    cnnlstm.py           # CNNLSTMClassifier, cnn_asp, cnnlstm_proposal
    aasist.py            # back-end graph attention (disalin)
    groupdelay.py        # APGDF/MGD front-end (diangkat dari probe_apgdf.py)
  enhance/
    __init__.py
    masks.py             # estimasi masker (Wiener / MetricGAN+ mask)
    arms.py              # A0..A4 — magnitudo identik, fase berbeda
    consistency.py       # C(S) = ‖S - STFT(iSTFT(S))‖/‖S‖ dalam dB
    wrappers.py          # SpeechBrain MetricGAN+, denoiser (domain gelombang)
  eval/
    metrics.py           # (sudah ada) + bootstrap_ci(), det_curve()
    stats.py             # mcnemar, holm_bonferroni, seed_variance_report()
    calibrate.py         # TemperatureScaler, prior_matched (WAJIB tandai transduktif)
    fusion.py            # rata-rata, LR-on-val, bobot dari validasi
scripts/
  00_check_env.py            # sudah ada
  01_build_manifest.py       # + assert 6.326 fake-MP3 di training
  02_build_noise_index.py    # indeks MUSAN/DEMAND/RIR, tulis JSON + hash
  03_make_eval_grid.py       # E1: tulis 29 kondisi × 1.088 wav + manifest + sha256
  04_run_enhance_arms.py     # E5: A0..A4
  05_measure_consistency.py  # E5: C(S) per berkas per lengan
  06_train.py                # eks-train.py, sekarang memakai ExpConfig
  07_eval_grid.py            # E2: muat model SEKALI, iterasi 29 kondisi
  08_shortcut_control.py     # E2b: RandomForest fitur sepele per kondisi
  09_fuse.py                 # E7
  10_report.py               # E9: regenerasi SEMUA tabel/gambar; --check
runs/            # tidak berubah: <tag>/{best.pt, results.json, test_scores.npy}
evalsets/        # BARU: evalsets/<kondisi>/<cls>/*.wav + manifest.csv + SHA256SUMS
```

**Dua aturan yang membuat struktur ini berguna dan bukan sekadar rapi:**

1. **`config.py` sebagai satu-satunya sumber hyperparameter.** Tag run diturunkan dari hash config, bukan dirakit dari string argparse. `train.py` sekarang membangun tag dari 4 field (`model_split_augment_bXeY_sZ`) — begitu Anda menambah `--rawboost` atau `--unfreeze-top`, dua run berbeda akan menimpa direktori yang sama dan Anda kehilangan hasil tanpa peringatan. Sudah hampir terjadi: `cnn_asp_clean_val_codec_s42` (b=64,e=12) dan `..._b32e10_s42` hanya terpisah karena Anda menambal tag secara manual.

2. **Jalur augmentasi lama disimpan verbatim di `augment/cpu.py`.** Ke-24 run yang sudah ada diproduksi olehnya. Kalau Anda menimpanya, angka 91,94% ±3,50 di naskah tidak lagi dapat direproduksi dan Anda harus menjalankan ulang semuanya.

---

## 6. Jebakan implementasi yang gagal secara diam-diam

Semuanya konkret, dan sepuluh yang pertama **ada di kode Anda sekarang** atau pasti muncul dari langkah berikutnya.

**#1 — RNG augmentasi tidak bergantung epoch** *(`forlib/data.py:293`)*
Seed = `hash(fname) + self.seed`. Augmentasi identik di setiap epoch. **Gejala: tidak ada.** Loss turun normal, val acc 99,8%. Efeknya: keragaman augmentasi = N, bukan N×epoch, sehingga seluruh eksperimen MCT Anda akan meremehkan manfaat augmentasi. Deteksi: `assert not torch.allclose(ds[0]['wav'], ds_epoch2[0]['wav'])`.

**#2 — `gain` dibatalkan oleh `loudness_normalize`**
Urutan `augment → normalize` membuat augmentasi gain (p=0,3) menjadi no-op eksak. Ablation gain akan menghasilkan Δ = 0,00 pp dan Anda akan menulis "augmentasi level tidak membantu" — kesimpulan yang salah dari kode yang tidak pernah berjalan.

**#3 — `np.convolve` untuk reverb: 41× lebih lambat, terukur**
200,8 s/epoch vs 4,9 s/epoch. Bukan bug kebenaran, tapi ia akan membuat E2+E3+E5 memakan minggu alih-alih hari, dan Anda akan tergoda memotong ruang lingkup. `scipy.signal.fftconvolve`.

**#4 — Encoder beku tetap di mode `train()`** *(paling berbahaya)*
`model.train()` mengaktifkan, di dalam encoder HF yang "beku": SpecAugment masking (`apply_spec_augment=True`, `mask_time_prob=0,05`), `layerdrop`, dan dropout. Hidden state yang masuk `LayerWeighting` menjadi stokastik saat latih dan deterministik saat eval. **Tidak ada error, tidak ada warning.** Ini kandidat kuat penjelas mengapa AST (86,43% ±2,94) dan `cnnlstm` tertinggal. Perbaiki dengan override `train()` yang memaksa `encoder.eval()` saat `self.frozen`.

**#5 — AST: `_fbank` tidak mengurangi mean gelombang**
`ASTFeatureExtractor` HF melakukan `waveform = waveform - waveform.mean()` sebelum `kaldi.fbank`; `models.py:_fbank` tidak. Statistik normalisasi `(x - fe_mean)/(fe_std*2)` yang Anda pakai berasal dari checkpoint AudioSet dan mengasumsikan preprocessing itu. Ketidakcocokan kecil, diam, dan menggeser seluruh distribusi input AST.

**#6 — AST `freeze=True` secara default, tanpa disengaja**
`train.py:141` menerapkan `kw["freeze"] = not args.unfreeze` ke `ast` juga (terkonfirmasi: `trainM = 1,18 M`). Membekukan encoder SSL punya justifikasi (rasio data/parameter). Membekukan AST — model **supervised** AudioSet — adalah keputusan yang sangat berbeda dan tidak terdokumentasi di mana pun. Jalankan `ast --unfreeze` di E0 sebelum menyimpulkan apa pun tentang AST.

**#7 — Provenance `is_mp3` diturunkan dari nama berkas**
Bila arsip diekstrak ulang oleh tool yang membersihkan nama, `is_mp3` menjadi 0 untuk semua baris dan **seluruh analisis codec runtuh menjadi noise tanpa satu pun error**. Tambahkan di `01_build_manifest.py`:
```python
assert sum(r["is_mp3"] for r in rows if r["split_official"]=="training" and r["label"]==1) == 6326
assert sum(r["is_mp3"] for r in rows if r["split_official"]=="testing") == 0
```

**#8 — bf16 mengkuantisasi skor → ROC bertangga → EER bias**
Mantissa bf16 = 8 bit. Logit dihitung di bf16, lalu di-`.float()`. Bila jumlah skor unik jauh di bawah 1.088, kurva ROC bertangga dan EER (yang mencari titik potong) menjadi bias dan tidak stabil antar-seed — kandidat penjelas sebagian dari std ±3,50 pp Anda. Cek: `len(np.unique(pt))`. Perbaikan: jalankan **kepala klasifikasi** (bottleneck+ASP+head) di fp32; encoder boleh tetap bf16.

**#9 — Worker DataLoader di Windows (spawn) tidak menyeed ulang RNG global numpy**
Begitu Anda mengganti `default_rng(hash)` dengan RNG per-worker, Windows memakai `spawn`, dan `np.random` global tidak otomatis berbeda antar-worker. Semua worker dapat menghasilkan stream augmentasi identik. Pakai `worker_init_fn` yang membaca `torch.initial_seed()`, jangan `np.random.seed(os.getpid())`.

**#10 — SNR dihitung atas seluruh klip, bukan level wicara aktif**
FoR-2sec sudah membuang silence sehingga selisihnya kecil di sini, tetapi begitu Anda ke `for-rerec` dan In-the-Wild (yang punya silence), SNR "0 dB" Anda bisa meleset 3–6 dB dari SNR yang dimaksud orang lain. Pakai ITU-T P.56 active level, dan **catat level aktif terukur ke manifest** supaya dapat diaudit.

**#11 — Enhancer MetricGAN+ SpeechBrain dilatih pada VoiceBank-DEMAND**
Bila Anda mengevaluasi lengan enhancer pada noise **DEMAND**, enhancer itu sudah pernah melihat rekaman noise tersebut. Lengan enhancement Anda akan tampak lebih kuat daripada semestinya dan mekanisme yang Anda simpulkan salah. **Untuk E5 khusus, pakai noise dari MUSAN (bukan DEMAND)**, atau nyatakan tumpang tindih ini eksplisit. Ini persisnya jenis kesalahan yang membuat kesimpulan kausal E5 tidak dapat dipertahankan di sidang.

**#12 — Babble dari korpus real FoR**
Bila Anda membangun babble dengan menumpuk klip `real` FoR, Anda menyuntikkan wicara bonafide ke dalam noise, dan kelas `fake` yang diberi babble menjadi campuran fake+bonafide. Detektor dapat "benar" karena alasan yang salah. Ambil babble dari DEMAND (`PCAFETER`, `SCAFE`, `SPSQUARE`) atau MUSAN speech — **tidak pernah** dari FoR.

**#13 — Kebocoran berkas noise antara latih dan uji**
Split file-level pada satu korpus tidak cukup: MUSAN `free-sound` berisi banyak potongan dari rekaman sumber yang sama. Karena itu rencana ini memakai **korpus-disjoin** (MUSAN latih / DEMAND uji), bukan file-disjoin.

**#14 — Ambang prior-matched dipakai untuk kurva ketahanan**
Prior-matched bersifat **transduktif** — ia membaca seluruh distribusi skor test. Bila dipakai di grid noise, sebagian "ketahanan" yang Anda ukur sebenarnya adalah akses Anda ke distribusi test pada kondisi itu. Kurva utama harus induktif (ambang dari validasi ter-noise); prior-matched boleh sebagai kolom kedua berlabel jelas.

**#15 — Pemilihan checkpoint pada validasi bersih untuk model MCT**
Validasi resmi 89,6% fake turunan MP3 dan bersih dari noise. Memilih checkpoint di sana untuk model yang dilatih multi-kondisi memilih model yang terbaik pada kondisi yang bukan target. Validasi harus ter-noise dengan subset MUSAN yang disjoin.

**#16 — `drop_last` pada loader evaluasi**
Saat ini `drop_last=sh` sehingga hanya training yang membuang. Aman, tetapi 1.088 habis dibagi 32 dan **tidak** habis dibagi 48/64. Begitu seseorang menaikkan batch eval dan menyalin `drop_last=True`, Anda diam-diam mengevaluasi 1.056 berkas dan setiap angka bergeser ~0,3 pp tanpa jejak.

**#17 — `wavval` memindahkan 100% fake-WAV keluar dari training**
Sudah Anda identifikasi sendiri (`HASIL_EKSPERIMEN.md` §Multi-seed poin 3): data latih menjadi 100% fake turunan MP3 dan kehilangan seluruh contoh fake WAV-native, sehingga rerata turun 8,5 poin. Perbaiki dengan membagi 652 berkas itu (mis. 50/50), jangan memindahkan semuanya — ini pekerjaan 10 baris dan mengembalikan sebagian besar 8,5 poin tersebut.

**#18 — `LayerWeighting` menjumlahkan 13–25 hidden state dalam bf16**
Penjumlahan berbobot atas 25 tensor bf16 (hubert-large/XLS-R) kehilangan presisi. Cast ke fp32 di dalam `LayerWeighting.forward`. Murah, dan menghilangkan satu sumber variansi seed.

**#19 — Enhancer mengubah panjang sinyal**
Model domain-gelombang (denoiser/SEGAN) sering mengembalikan panjang berbeda karena padding internal. Bila Anda memotong/menambal ke 32.000 tanpa memeriksa, Anda menggeser fase relatif dan ukuran konsistensi `C(S)` Anda mengukur pergeseran itu, bukan enhancer. Assert `len(out) == len(in)` di `enhance/wrappers.py`.

**#20 — Membandingkan `C(S)` antar lengan dengan window STFT yang berbeda**
`C(S)` hanya sebanding bila window/hop/n_fft **identik** di semua lengan dan sama dengan yang dipakai enhancer. Bekukan satu konfigurasi STFT (`n_fft=512, hop=128, hann`) di `enhance/consistency.py` dan pakai konstanta itu di mana-mana.

---

## 7. Urutan eksekusi minggu pertama (konkret)

```
Hari 1  E0.1–E0.4  perbaiki 4 bug; smoke test; ukur ulang s/epoch
Hari 2  E0.5–E0.6  tambah hubert-base; 25 run baseline 5 seed        (2,5 j GPU)
Hari 3  E1         unduh MUSAN+RIRS+DEMAND; bangun 29 kondisi + hash (0 j GPU)
Hari 4  E2 + E2b   grid degradasi + kontrol RandomForest             (2,5 j GPU)
Hari 5  E3         MCT + RawBoost, 25 run                            (5 j GPU)
Hari 6  E3-eval    ulangi grid di atas checkpoint MCT                (2 j GPU)
Hari 7  E8         dengarkan 26 + ~60 berkas salah; tulis kategorinya (0 j GPU)
```

Setelah hari 7 Anda sudah punya **jawaban lengkap untuk rumusan masalah ketahanan noise** dengan angka nyata — sesuatu yang saat ini nol. E4 (XLS-R) berjalan paralel di latar belakang mulai hari 5 karena ia yang paling lama.

---

## 8. Catatan jujur tentang "mendekati 100%"

Yang **dapat** dijanjikan dari rencana ini:
- Split acak 60/20/20: **99,75–99,94% sudah tercapai** (terukur). Angka untuk dosen sudah ada.
- Split resmi: ensemble **97,61% sudah tercapai** (terukur, 26 salah dari 1.088).
- E4 + E7 memberi jalur teknis yang masuk akal untuk menekan 26 error itu. Berapa yang tersisa **tidak dapat saya prediksi tanpa mengarang**.

Yang **tidak boleh** dijanjikan: bahwa ketahanan noise akan "gratis". Setiap paper multi-condition training membayar sedikit akurasi bersih untuk ketahanan. Kriteria sukses E3 poin 2 (≤ 1,5 pp) adalah anggaran eksplisit untuk itu — nyatakan di naskah sebagai trade-off terukur, bukan sembunyikan.

Dan satu koreksi yang harus masuk Bab Pembahasan tanpa dibesar-besarkan (satu paragraf, bukan satu bab): **SEGAN beroperasi end-to-end di domain gelombang; MetricGAN+ memprediksi masker magnitudo dan memakai ulang fase noisy untuk iSTFT.** Sitasi: Pascual dkk. Interspeech 2017; Fu dkk. Interspeech 2021 (arXiv:2104.03538); dan kode `speechbrain/lobes/models/MetricGAN.py` yang **wajib Anda baca sendiri** sebelum menuliskannya.

---

**Sources:**
- [MUSAN — OpenSLR 17](https://www.openslr.org/17/)
- [RIR and Noise Database — OpenSLR 28](https://www.openslr.org/28/)
- [DEMAND noise database — Zenodo 1227121](https://zenodo.org/records/1227121)
- [In-the-Wild audio deepfake dataset](https://deepfake-total.com/in_the_wild) · [HF mirror](https://huggingface.co/datasets/mueller91/In-The-Wild)
- [TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)