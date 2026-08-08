# Verifikasi Protokol Split Penelitian Sebelumnya

*Menuntaskan butir wajib #7 di [LANJUTAN.md](LANJUTAN.md). Saya sebelumnya menandai
klaim tentang literatur sebagai **belum terverifikasi** dan melarangnya masuk naskah.
Berikut hasil verifikasinya, dan sebagian membantah hipotesis saya sendiri.*

---

## Ringkasan: hipotesis saya benar untuk satu paper, salah untuk yang lain

| Rujukan | Protokol split | Status hipotesis saya |
|---|---|---|
| Ref [20] Hybrid CNN-LSTM (94,7%) | **Acak 80/20** atas subset 15.000 | ✅ **Terkonfirmasi** |
| Ahmad dkk., FoR baselines (SVM ~93%) | **Split resmi, speaker-disjoint** | ❌ **Terbantah** |
| Ref [19] MFAAN (94,47%) | Tidak dinyatakan di abstrak | ⚠️ Belum terverifikasi |

**Kesimpulan yang harus ditulis di naskah:** praktik di literatur FoR **beragam**, sebagian memakai split acak, sebagian memakai partisi resmi. Klaim menyeluruh bahwa
"hasil terpublikasi pada FoR mengukur kebocoran domain" **tidak dapat dipertahankan**
dan harus dicabut. Yang dapat dipertahankan adalah klaim per-paper dan klaim
mekanistik.

---

## 1. Ref [20], Hybrid CNN-LSTM, 94,7% ✅ hipotesis terkonfirmasi

*"Hybrid CNN-LSTM Architectures for Deepfake Audio Detection Using MFCC and
Spectrogram Analysis", American Journal of Mathematical and Computer Modelling,
2025.* [sciencepg.com/article/10.11648/j.ajmcm.20251003.12](https://www.sciencepg.com/article/10.11648/j.ajmcm.20251003.12)

Kutipan langsung dari paper:

> *"A balanced subset consisting of 15,000 audio samples (7,500 real and 7,500
> synthetic) was extracted for computational efficiency. The data was divided with
> 80% allocated for training purposes and 20% designated for testing procedures."*

| | |
|---|---|
| Varian FoR | **tidak dinyatakan** |
| Split | **acak 80/20** atas subset, bukan partisi resmi |
| Ukuran | 12.000 latih / 3.000 uji |
| Hasil | Akurasi 94,7% · AUC 97,3% · Presisi 93,2% · Recall 95,8% · F1 94,5% |

**Relevansi dengan eksperimen kita.** Eksperimen [probe_split.py](probe_split.py)
menunjukkan Random Forest atas 38 fitur spektral sepele mencapai **95,91%** pada
split acak dan **79,23%** pada partisi resmi. Angka 94,7% dari CNN-LSTM di atas
diperoleh pada rezim split acak dan **berada di bawah** baseline fitur-sepele kita
pada rezim yang sama.

⚠️ Protokolnya tidak identik (mereka: subset 15.000, 80/20, varian tak diketahui;
kita: 17.870 berkas for-2sec, 60/20/20). Jadi perbandingan bersifat **indikatif atas
mekanisme**, bukan head-to-head. Ini harus dinyatakan sebagai catatan kaki.

---

## 2. Ahmad, Ahmed & Imtiaz, FoR baselines ❌ hipotesis saya terbantah

*"Classical Machine Learning Baselines for Deepfake Audio Detection on the
Fake-or-Real Dataset", Clarkson University.* [arXiv:2604.13400](https://arxiv.org/abs/2604.13400)

Kutipan langsung:

> *"For each sampling-rate condition, we use the dataset's **predefined training and
> test sets**. Speaker identities do not overlap between splits, preventing speaker
> leakage."*

> *"...which provide **speaker-disjoint training and test partitions** to prevent
> speaker-specific memorization."*

| | |
|---|---|
| Varian | for-2sec (dinyatakan 44,1 kHz) dan for-rerec (16 kHz) |
| Split | **partisi resmi, speaker-disjoint**, eksplisit |
| Ukuran | ~31.138 klip; 24.913 latih / 6.225 uji |
| Hasil terbaik | **RBF SVM ~93% akurasi, ~7% EER**; model linier ~75% |

**Ini membantah hipotesis saya.** Mereka memakai partisi resmi dan tetap mencapai
~93% dengan SVM klasik. Jadi tidak benar bahwa angka tinggi pada FoR *selalu*
berasal dari split acak.

### Dua ketidaksesuaian yang belum terselesaikan

**(a) Sampling rate.** Mereka menyebut for-2sec sebagai **44,1 kHz**. Audit kita atas
`for-2sec.tar.gz` resmi dari York University mengukur **16.000 Hz pada 100% dari
17.870 berkas**, tanpa kecuali. Kemungkinan mereka memakai rilis/mirror berbeda
(mis. Kaggle) atau memotong sendiri dari `for-original`.

**(b) Ukuran partisi.** Mereka: 24.913 latih / 6.225 uji (~31.138 total).
Kita: 13.956 / 2.826 / 1.088 (17.870 total). **Test set mereka 5,7× lebih besar.**

Karena partisi yang mereka pakai berbeda dari yang kita pakai, angka 93% mereka
dan angka 92% kita **tidak dapat dibandingkan langsung**. Ini harus dinyatakan.

### Yang justru menguatkan temuan kita

Paper mereka melaporkan pengamatan berikut:

> *"Real speech tends to have a **higher centroid**, indicating more high-frequency
> energy (e.g., **sibilants and microphone noise**) than synthetic speech."*

Ini **independen mengonfirmasi pengukuran kita**: audit kita menemukan centroid real
853,82 Hz vs fake 584,24 Hz, dan energi >6 kHz real 3,2× lebih tinggi.

**Tetapi penafsirannya berbeda, dan di sinilah kontribusi kita.** Mereka
mengatribusikan selisih itu pada penyebab *alami*, sibilan dan noise mikrofon pada
wicara manusia. Audit kita menunjukkan penyebabnya sebagian besar **artefak
provenance codec**:

| Perbandingan | Rasio energi >6 kHz |
|---|---|
| real vs fake **turunan MP3** (90,7% data latih) | **4,18×** |
| real vs fake **bukan turunan MP3** (652 berkas) | **1,17×** |
| real vs fake pada test set (0% MP3) | **1,33×** |

Bila selisih HF benar-benar berasal dari sibilan dan noise mikrofon, ia akan tetap
besar pada fake yang tidak melalui MP3. Ternyata tidak, turun dari 4,18× ke 1,17×.
**Sekitar tiga perempat selisih HF pada data latih dapat diatribusikan ke MP3, bukan
ke sintesis.**

Ini pernyataan yang dapat difalsifikasi, didukung data, dan **berbeda dari tafsir yang
sudah dipublikasikan**, kandidat kontribusi ilmiah yang jauh lebih kuat daripada
sekadar membandingkan empat arsitektur.

---

## 3. Ref [19] MFAAN, belum terverifikasi ⚠️

*"MFAAN: Unveiling Audio Deepfakes with a Multi-Feature Authenticity Network",
IEEE 2023.* [arXiv:2311.03509](https://arxiv.org/abs/2311.03509)

Abstrak hanya menyebut akurasi 94,47% (FoR) dan 98,93% (In-the-Wild). Varian FoR
dan protokol split **tidak dinyatakan di abstrak**. Perlu membaca PDF lengkap.

**Status: jangan kutip protokolnya sampai diverifikasi.**

---

## 4. Konsekuensi untuk naskah tesis

### Yang harus DICABUT

Rumusan yang saya usulkan sebelumnya di [TEMUAN_GROUND_TRUTH.md](TEMUAN_GROUND_TRUTH.md) §5, *"indikasi kuat bahwa sebagian besar hasil terpublikasi pada FoR memakai split acak"*, **tidak dapat dipertahankan.** Minimal satu paper terverifikasi memakai partisi resmi.

### Yang dapat dipertahankan sepenuhnya

1. **Klaim mekanistik** (murni dari eksperimen kita, tidak bergantung pada paper lain):
 > Dengan model, fitur, dan data identik, split acak menghasilkan akurasi 16,69 poin
 > lebih tinggi daripada partisi resmi pada FoR-2sec.

2. **Klaim per-paper** untuk ref [20], dengan kutipan langsung.

3. **Klaim atribusi codec**, yang menantang tafsir Ahmad dkk. dengan data.

4. **Klaim pelaporan variansi**: tidak satu pun dari ketiga paper melaporkan
 simpangan baku atas beberapa seed, padahal eksperimen kita menemukan variansi
 antar-seed ±3,50 pp, cukup besar untuk menelan sebagian besar selisih yang
 dilaporkan antar-metode.

### Tabel perbandingan yang jujur untuk naskah

| Penelitian | Varian FoR | Protokol split | n uji | Akurasi |
|---|---|---|---|---|
| Ahmad dkk. (SVM RBF) | for-2sec 44,1 kHz | **resmi, speaker-disjoint** | 6.225 | ~93% (EER ~7%) |
| Ref [20] (CNN-LSTM) | tidak dinyatakan | **acak 80/20** | 3.000 | 94,7% |
| Ref [19] MFAAN | tidak dinyatakan | ⚠️ belum diverifikasi | ? | 94,47% |
| **Penelitian ini**, split acak | for-2sec 16 kHz | acak 60/20/20 | 3.574 | **99,75%** |
| **Penelitian ini**, split resmi | for-2sec 16 kHz | **resmi** | 1.088 | **92,3% ± 3,0** |
| **Penelitian ini**, ensemble, split resmi | for-2sec 16 kHz | **resmi** | 1.088 | **97,61%** |

Kolom "Protokol split" adalah kolom terpenting dalam tabel ini, dan tidak ada di
tabel state-of-the-art proposal (hal. 8–11). Menambahkannya saja sudah merupakan
kontribusi metodologis.
