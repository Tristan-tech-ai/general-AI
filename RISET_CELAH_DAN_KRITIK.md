# Ujian Kritis: Celah Ilmiah, Kontribusi Utama, dan Titik Rapuh

**Untuk:** Tesis S2 Sistem Informasi — "Analisis Performa Arsitektur Deep Learning Wav2Vec2, AST, HuBERT, dan CNN-LSTM dalam Klasifikasi Suara Deepfake dan Suara Asli"
**Peran penulis dokumen ini:** penguji yang mencari celah, bukan pembimbing yang menyemangati
**Basis bukti:** hasil riset literatur yang diberikan **plus** verifikasi langsung ke repo kerja di `C:\Users\Tristan\Downloads\general-ai` (`TEMUAN_GROUND_TRUTH.md`, `HASIL_EKSPERIMEN.md`, `PERBANDINGAN.md`, `VERIFIKASI_RUJUKAN.md`, `audit_aucs.json`, `runs/`)

---

## 0. Posisi jujur: di mana tesis ini berdiri sekarang

Sebelum bicara celah, saya nyatakan dulu apa yang **sudah** Anda miliki, karena ini mengubah seluruh kalkulus. Dari 34 dimensi riset literatur, hampir semuanya direfutasi dengan alasan yang sama: *dampaknya nol pada dataset ini*. Yang tersisa dan bernilai ternyata bukan dari literatur sama sekali — melainkan dari empat pengukuran di mesin Anda sendiri:

| Fakta terukur | Angka | Sumber |
|---|---|---|
| Split acak 60/20/20 (rencana proposal) | **99,94%**, EER 0,03%, AUC 1,0000 | `HASIL_EKSPERIMEN.md` baris 1 |
| Split resmi FoR, model & seed identik | **50,00%**, EER 28,12%, AUC 0,7946 | baris 2 |
| Penyebab: 90,7% fake latih turunan MP3, **0%** fake uji | 4,18× vs 1,17× rasio energi >6 kHz | `TEMUAN_GROUND_TRUTH.md` §4 |
| Augmentasi codec simetris | EER 28,12% → **4,41%**, AUC → 0,9900 | baris 3 |
| Ambang prior-matched di atasnya | akurasi → **95,59%** (1 seed) / **91,94% ± 3,50** (3 seed) | §Multi-seed |
| Ensemble 4 arsitektur, protokol resmi | **97,61%**, EER 2,30%, AUC 0,9954, 26/1.088 salah | `PERBANDINGAN.md` §4 |
| Variansi antar-seed | **±3,50 pp** — 4,2× lebih besar dari CI binomial ±0,83 pp | §Multi-seed |
| Pintasan silence (yang terkenal itu) | **AUC 0,5001** — tidak ada | `audit_aucs.json` |

**Kesimpulan posisi:** target dosen sudah tercapai dua kali (99,94% pada protokol lazim, 97,61% pada protokol yang jauh lebih sulit). Masalah tesis ini **bukan lagi akurasi**. Masalahnya adalah: apakah ada kontribusi ilmiah yang tahan dibedah, dan apakah angka-angka itu tahan diserang. Sisa dokumen ini menjawab itu.

Satu peringatan struktural di depan: **`cnn_asp` — model dengan performa terbaik dalam hasil Anda — tidak ada di judul tesis, dan HuBERT yang ada di judul belum punya satu pun run** (`runs/` berisi `cnn_asp_*`, `wav2vec2_*`, `ast_*`, `cnnlstm_*`; nol `hubert_*`). Sementara CNN-LSTM yang ada di judul justru terburuk secara signifikan (83,52% ± 2,28, kalah McNemar terhadap ketiganya). Ini bukan detail administratif; ini masalah yang akan dibuka penguji di lima menit pertama. Diselesaikan di §3.

---

## 1. Celah ilmiah yang MASIH terbuka per 2026 — yang muat di satu GPU

Saya membuang semua celah yang butuh skala. Tujuh yang tersisa di bawah ini semuanya bisa dikerjakan dengan RTX 5060 Ti tunggal, sebagian bahkan tanpa GPU sama sekali karena hanya mengolah skor yang sudah tersimpan di `runs/*/`.

### Celah 1 — Bidang ini tidak pernah memisahkan kegagalan DISKRIMINASI dari kegagalan KALIBRASI

Ini celah terbesar, terbukti, dan paling murah.

Deteksi deepfake audio melaporkan dua jenis angka: EER/min-DCF (bebas-ambang, artinya **terkalibrasi-oracle** — ambangnya dipilih *setelah* melihat label uji) atau akurasi/F1 pada ambang 0,5 (tidak terkalibrasi sama sekali). Tidak satu pun memisahkan "model kehilangan kemampuan membedakan" dari "model masih bisa membedakan tapi ambangnya bergeser".

Bukti bahwa ini bukan kepedulian akademis: baris 3 tabel Anda menunjukkan **AUC 0,9900 berdampingan dengan akurasi 50,28%**. Model itu hampir sempurna dalam memeringkat, dan sama sekali tidak berguna dalam memutuskan. Ablation 2×2 Anda mengukurnya dengan bersih:

|  | ambang dari validasi | ambang prior-matched |
|---|---|---|
| tanpa augmentasi | 50,00% | 71,88% *(+21,88)* |
| + augmentasi codec | 50,28% *(+0,28)* | **95,59%** *(+45,59)* |

Augmentasi codec sendirian: **+0,28 poin**. Efek interaksinya raksasa. Ini adalah temuan yang tidak bisa dihasilkan oleh kerangka pelaporan yang dipakai bidang ini.

Perangkatnya sudah ada di bidang tetangga dan tinggal diimpor: **Cllr aktual vs min-Cllr** (Brümmer & du Preez, 2006), yang mendekomposisi kerugian total menjadi *discrimination loss* (min-Cllr) + *calibration loss* (selisihnya), plus plot APE. Speaker verification memakainya sejak dua dekade lalu; ASVspoof mengadopsi t-DCF tapi yang dilaporkan hampir selalu **min**-DCF — oracle lagi. Mengukur *calibration loss* per kondisi pada deteksi deepfake adalah lahan kosong.

**Biaya: nol.** `forlib/metrics.py` sudah punya EER, ECE, temperature scaling, prior-matched threshold. Skornya sudah tersimpan. Ini pekerjaan satu sore.

### Celah 2 — Tidak ada metode KALIBRASI INDUKTIF untuk deteksi deepfake lintas domain

Konsekuensi langsung dari Celah 1, dan di sinilah kebaruan sesungguhnya berada.

Angka 95,59% Anda diperoleh dengan ambang prior-matched — yang **transduktif**: butuh seluruh skor test set sekaligus, dan mengasumsikan prior 50/50 diketahui. Sah untuk forensik arsip, tidak sah untuk deteksi streaming. Alternatif induktif yang sudah Anda uji (validasi cocok-domain) memberi 83,40% ± 1,41. **Selisih ~8,5 poin itu adalah harga kemampuan induktif — dan tidak ada satu pun paper deteksi deepfake yang menyebut harga ini ada.**

Yang belum dicoba siapa pun di bidang ini, dan semuanya murah:

- **AS-norm / S-norm / T-norm** — normalisasi skor terhadap kohort imposter, standar di speaker verification sejak 2000-an, **belum pernah saya lihat diterapkan pada skor countermeasure deepfake**. Ini impor lintas-bidang yang sah dan mudah dipertahankan.
- **Kalibrasi terkondisi**: estimator kondisi buta (SNR, bandwidth efektif, deteksi cutoff codec) → memilih/menggeser ambang. Dilatih hanya pada data latih.
- **Quantile mapping skor** pada jendela bergulir — versi *online* dari prior-matching, yang mengembalikan sebagian keuntungan transduktif tanpa butuh seluruh batch.
- **Sensitivitas terhadap salah-spesifikasi prior**: kurva akurasi vs prior yang diasumsikan (0,1 … 0,9). Belum ada.

Prediksi jujur saya: kalibrasi induktif yang baik memulihkan **sebagian besar, bukan seluruh** selisih 83,40 → 91,94. Perkiraan realistis mendarat di 88–92% dengan std jauh lebih kecil dari ±3,50. Itu angka yang bagus **dan** merupakan kontribusi metodologis nyata.

### Celah 3 — Audit pintasan belum punya protokol baku, dan pintasan yang terkenal ternyata bukan yang dominan

Pintasan *silence* sudah terkenal (Müller dkk. 2022: EER 3,6% → 15,5%; Grommelt dkk. 2024). Semua orang mengasumsikan itu pintasan utama. **Pengukuran Anda menunjukkan pada FoR-2sec ia tidak ada sama sekali: `lead_sil` AUC = 0,5001, `trail_sil` 0,4823, `dur` 0,5000.** Yang dominan justru pintasan yang tidak ada di literatur: **provenance codec**.

Celahnya: bidang ini tidak punya **baterai probe nuisance yang portabel dan baku** — satu set deskriptor bebas-label (durasi, silence awal/akhir, loudness, offset DC, fraksi clipping, energi per pita, kemiringan HF, rolloff, estimasi bandwidth efektif, jejak cutoff codec) yang dilaporkan sebagai AUC untuk *setiap* korpus baru, plus audit provenance dari metadata/nama berkas. Anda sudah membangunnya (`audit.py`, `audit_aucs.json`) tanpa menyadari itu produk yang bisa dipublikasikan.

Nilai tambahannya: hasil **negatif** ("pintasan silence tidak ada di FoR-2sec") sama berharganya dengan hasil positif, karena membantah asumsi default bidang. Syaratnya satu, dan penting — lihat §5, klaim rapuh F6.

### Celah 4 — "Ketahanan noise" hampir selalu diuji pada kondisi TERCOCOKKAN

Ini menyentuh rumusan masalah Anda langsung. Mayoritas paper ketahanan-noise melatih dengan MUSAN dan menguji dengan MUSAN. Yang mereka ukur sebagian besar adalah *matched-condition training*, bukan generalisasi.

Yang terbuka dan murah:
- **Generalisasi lintas-keluarga-noise**: latih dengan keluarga A (mis. derau berwarna sintetis), uji dengan keluarga B (MUSAN babble / DEMAND traffic). Hampir tidak ada yang melaporkan ini.
- **Interaksi noise × pintasan** — dan ini prediksi tajam yang khas untuk dataset Anda. `TEMUAN_GROUND_TRUTH.md` §7.3 sudah menandainya: menambahkan noise mengisi pita >6 kHz, yaitu **persis tempat pintasan MP3 berada**. Artinya sebagian dari "ketahanan noise" yang akan Anda ukur sebenarnya adalah "pintasan yang tertutup noise". Memisahkannya menuntut desain faktorial 2×2: {pintasan utuh, pintasan dinetralkan} × {bersih, bernoise}. Desain itu orisinal, murah, dan langsung menjawab rumusan masalah.
- **Asimetri FN/FP per level SNR.** Metrik ini sudah ada di proposal (confusion matrix), biayanya nol, dan belum ada yang melaporkannya pada FoR.

### Celah 5 — Variansi seed tidak pernah dilaporkan, padahal ia menelan seluruh selisih antar-metode

Anda mengukur **±3,50 pp** antar-inisialisasi. Selisih antar-baseline yang dikutip proposal adalah 93,50 / 94,47 / 94,70 — **rentang 1,2 poin, sepertiga dari derau seed Anda.** Tidak satu pun dari ketiga paper melaporkan simpangan baku.

Ini bukan sekadar keluhan. Ini celah metodologis yang bisa diisi dengan hasil yang sudah Anda punya: **analisis daya statistik untuk benchmark FoR** — berapa seed dan berapa berkas uji yang dibutuhkan untuk mengklaim perbaikan 1 poin. Dengan n = 1.088 (1 berkas = 0,092 pp) dan σ ≈ 3,5 pp, *minimum detectable effect* pada 3 seed berada di kisaran 4–6 pp. Artinya: **hampir seluruh literatur perbandingan metode pada FoR melaporkan selisih yang berada di bawah ambang deteksinya sendiri.** Pernyataan itu tajam, dapat dihitung, dan tidak butuh GPU.

### Celah 6 — Sumber keberagaman ensemble tidak pernah diukur secara kausal

Anda mengukur φ error 0,058–0,223 antar keempat model — sangat rendah, dan ensemble melompat ke 97,61%. Semua orang mengatribusikan ini ke "arsitektur berbeda". Tapi keempat model Anda berbeda pada **tiga sumbu sekaligus**: representasi masukan (waveform / mel-patch / MFCC), korpus pra-pelatihan (60k jam / AudioSet / nol), dan induktif bias arsitektur. Faktorial 2×2 (representasi × pra-pelatihan) akan menjawab yang mana yang menghasilkan dekorelasi. Murah, dan hasilnya berguna praktis: kalau ternyata representasi yang dominan, orang bisa membangun ensemble murah tanpa empat backbone besar.

### Celah 7 — Protokol mismatch for-2sec → for-rerec belum pernah dipublikasikan

Terverifikasi dalam riset Anda: Ahmad dkk. mengevaluasi for-rerec **dalam kondisi tercocokkan** (latih dan uji sama-sama for-rerec) dan menyimpulkan performa tidak turun. Itu bukan uji ketahanan. Protokol mismatch — latih for-2sec, uji for-rerec tanpa fine-tuning — tidak saya temukan dipublikasikan. `for-rerec.tar.gz` (1.558 MB) sudah ada di disk Anda, belum diekstrak.

Peringatan yang harus menyertainya ada di §3 (serangan A9): korpus ini terkonfound blok-sesi/kelas dan hanya ~26% pasangannya tidak dapat dipetakan. Jadikan probe sekunder dengan confound diungkap, bukan eksperimen unggulan.

---

## 2. Pertanyaan penelitian yang paling layak jadi kontribusi utama

### Rekomendasi

> **RQ Utama:** Pada deteksi deepfake audio lintas-domain, berapa proporsi degradasi performa yang berasal dari **kehilangan daya pisah (discrimination loss)** versus dari **kehilangan kalibrasi (calibration loss)**, dan dapatkah komponen kalibrasi dipulihkan secara **induktif** tanpa melatih ulang model?

Dengan tiga sub-RQ:

1. **RQ1 (diagnosis).** Nuisance mana pada FoR-2sec yang berkorelasi dengan label, dan berapa besar kontribusinya terhadap akurasi yang dilaporkan? *(Sudah terjawab: provenance codec, +16,69 pp dari skema split saja; pintasan silence AUC 0,5001 — absen.)*
2. **RQ2 (dekomposisi).** Bagaimana Cllr aktual, min-Cllr, dan calibration loss berperilaku pada keempat arsitektur di sepanjang sumbu gangguan: bersih → SNR 30…−5 dB → rekam-ulang? *(Belum dikerjakan; biaya ≈ nol untuk kondisi yang sudah ada.)*
3. **RQ3 (remediasi).** Seberapa besar bagian keunggulan transduktif (95,6%) yang dapat direbut kembali secara induktif oleh normalisasi skor / kalibrasi terkondisi? *(Belum dikerjakan; ini kebaruan sesungguhnya.)*

### Mengapa ini, bukan yang lain

**Pertama, ia sudah 70% terbukti.** Ablation 2×2 Anda *adalah* temuannya. Anda tidak sedang bertaruh pada hipotesis yang mungkin gagal; Anda sedang memformalkan sesuatu yang sudah terukur. Untuk tesis dengan tenggat, ini alasan yang menentukan.

**Kedua, ia menyelamatkan judul.** RQ ini agnostik terhadap arsitektur — dekomposisinya dihitung untuk Wav2Vec2, AST, HuBERT, dan CNN-LSTM secara identik. Judul tetap valid, keempat model tetap jadi objek, tapi tesisnya berhenti menjadi "saya membandingkan empat model" (generik, mudah diserang, sudah ada ratusan) dan menjadi "saya membangun kerangka diagnostik dan menerapkannya pada empat paradigma representasi".

**Ketiga, ia menjawab rumusan masalah noise secara langsung dan lebih dalam.** Pertanyaan lama: "model mana paling tahan noise?" Pertanyaan baru: "ketika noise datang, apakah model kehilangan kemampuan **membedakan**, atau hanya kehilangan kemampuan **memutuskan**?" Bukti awal dari data Anda sendiri kuat menunjukkan yang kedua dominan — pada for-rerec AUC bertahan 0,80–0,89 sementara F1 runtuh ke 0,048–0,369. Kalau pola itu bertahan pada sumbu SNR, Anda memiliki temuan yang mengubah rekomendasi praktis: **jangan latih ulang, kalibrasi ulang.**

**Keempat, ia memberi dosen angkanya tanpa berbohong.** Naskah melaporkan tiga kolom: split acak 99,94% (protokol lazim literatur), split resmi ensemble 97,61% (protokol lintas-domain), dan induktif ~90% (protokol paling ketat). Angka tertinggi tetap ada di abstrak, disertai alasan mengapa ia tidak boleh berdiri sendiri.

**Kelima, biayanya nyaris nol.** RQ2 seluruhnya post-hoc atas `runs/*/test_scores.npy`. RQ3 adalah beberapa ratus baris kode di CPU. Anggaran GPU tersisa dipakai untuk kewajiban (HuBERT, tambahan seed, sweep SNR), bukan untuk mengejar kontribusi.

### Yang harus DITOLAK sebagai kontribusi utama

- **"Perbandingan empat arsitektur."** Data Anda sendiri membantahnya: selisih peringkat 1 dan 2 adalah 1,19 pp sementara std gabungan ±2,50 pp. Anda **tidak dapat** mengurutkan keempatnya. Menjadikan ini kontribusi utama berarti menjadikan hasil null sebagai hasil utama.
- **"Ketahanan noise keempat arsitektur"** sebagai deskripsi murni. Terlalu generik, sudah dilakukan, dan tanpa dekomposisi kalibrasi ia hanya menghasilkan empat kurva menurun.
- **"Augmentasi codec menaikkan akurasi."** Ini perbaikan bug spesifik-dataset. Layak satu subbab, bukan kontribusi utama — dan rawan serangan A1 di bawah.

---

## 3. Apa yang akan diserang penguji — dan cara bertahan

Diurutkan menurut daya rusak.

### A1. "Augmentasi codec Anda dipilih SETELAH Anda mengaudit test set. Itu test-set peeking."

**Ini serangan paling berbahaya**, dan sebagian benar secara kronologis. Urutan di dokumen Anda: audit menemukan 0% MP3 di test → augmentasi codec diterapkan → akurasi melompat.

**Pertahanan (harus ditulis eksplisit di metodologi, bukan diimprovisasi saat sidang):** intervensi ini **dapat diturunkan sepenuhnya dari data latih saja**. Yang melanggar asumsi adalah fakta bahwa di *training*, label berkorelasi dengan provenance: 90,7% fake berasal MP3, 0% real berasal MP3. Itu adalah pelanggaran independensi label ⟂ nuisance yang terlihat **tanpa menyentuh test set sama sekali**. Netralisasi dilakukan atas dasar itu. Rumuskan begitu di naskah, dan tunjukkan tabel provenance khusus training di bab metodologi sebelum tabel test muncul.

**Penguatan yang wajib dikerjakan:** validasi intervensi pada kondisi ketiga yang tidak pernah dipakai untuk merancangnya (for-rerec, atau slice korpus eksternal). Kalau augmentasi codec juga menolong di sana, tuduhan peeking gugur.

### A2. "cnn_asp tidak ada di judul, HuBERT ada di judul tapi tidak ada hasilnya."

Fatal jika tidak diantisipasi. `runs/` tidak berisi satu pun run HuBERT; `forlib/models.py` mendukungnya tapi belum dijalankan. Sementara model terbaik Anda (`cnn_asp`, 91,94%) bukan salah satu dari empat yang dijanjikan, dan CNN-LSTM yang dijanjikan adalah yang terburuk (83,52%, kalah signifikan terhadap ketiganya).

**Pertahanan:** tidak ada. Ini harus **dikerjakan**, bukan dibela. Jalankan HuBERT ×3 seed (biaya beberapa jam). Lalu putuskan satu dari dua: (a) deklarasikan `cnn_asp` secara eksplisit sebagai *baseline CNN tanpa rekurensi* dan jelaskan di metodologi mengapa ia disertakan — ia justru memperkuat cerita karena membuktikan bahwa keunggulan bukan datang dari pra-pelatihan besar; atau (b) keluarkan dari tabel utama dan pindahkan ke lampiran. Opsi (a) lebih jujur dan lebih menarik, tapi hanya jika dinyatakan di depan.

### A3. "Anda mengubah split dari proposal (60/20/20 → resmi). Itu mengubah objek penelitian."

**Pertahanan:** kuat, dan justru menjadi kontribusi. Laporkan **keduanya**, dan tunjukkan pengukuran +16,69 pp (RandomForest, fitur & model identik) sebagai alasan. Nyatakan bahwa proposal hal. 55 menyebut 60/20/20 sementara partisi resmi FoR adalah 78,1/15,8/6,1 — jadi 60/20/20 **menuntut penggabungan ulang seluruh data**, yang menghancurkan pemisahan domain yang sengaja dirancang pembuat dataset. Kalimat ini mengubah "penyimpangan dari proposal" menjadi "koreksi terhadap proposal berdasarkan audit".

### A4. "99,94% Anda tidak berarti apa-apa."

**Pertahanan:** setuju di depan, sebelum penguji sempat menyerang. Bingkai baris itu sebagai **demonstrasi kebocoran**, bukan pencapaian: "CNN 1,54 juta parameter tanpa augmentasi, 6 epoch, mencapai 99,94% — dan justru itu masalahnya." Menyerahkan angka itu sebagai bukti *melawan* protokolnya sendiri adalah gerakan yang mematikan senjata penguji sekaligus memenuhi permintaan dosen.

### A5. "Dengan n=1.088 dan σ=±3,50 pp, bagaimana Anda bisa mengklaim apa pun?"

**Pertahanan:** Anda sudah punya jawabannya di `PERBANDINGAN.md` — dan ini akan mengesankan penguji jika disajikan proaktif. Laporkan: (i) rerata ± std atas ≥3 seed, (ii) McNemar berpasangan + koreksi Holm-Bonferroni, (iii) *minimum detectable effect* yang eksplisit, (iv) ketegangan yang Anda temukan sendiri antara "agregat seed tidak dapat dibedakan" dan "McNemar signifikan pada seed terbaik", beserta penjelasan mengapa keduanya benar untuk pertanyaan berbeda. Bagian §2 Temuan 2 `PERBANDINGAN.md` sudah menulisnya dengan benar — pindahkan langsung ke naskah.

**Yang harus diperbaiki dulu:** naikkan ke 5 seed minimal untuk dua model teratas (lihat F1 di §5).

### A6. "Perbandingan empat arsitektur ini tidak adil — SSL dipretrain 60.000 jam, CNN-LSTM dari nol."

**Pertahanan:** akui dan bingkai ulang. Encoder SSL Anda **dibekukan**, jadi yang dibandingkan adalah *kualitas representasi sebagai ekstraktor fitur di bawah anggaran fine-tuning yang setara* — bukan kapasitas arsitektur. Laporkan jumlah parameter, parameter yang dilatih, dan korpus pra-pelatihan dalam satu tabel. Lalu tunjukkan temuan yang justru muncul dari ketidakadilan itu: **pra-pelatihan SSL tidak menaikkan akurasi puncak, tetapi menurunkan variansi 6,9×** (wav2vec2 ±0,51 vs cnn_asp ±3,50). Itu jawaban yang lebih menarik daripada peringkat akurasi.

**Kelemahan yang harus ditutup:** eksperimen *unfreezing* belum pernah dijalankan (item #3 di `LANJUTAN.md`). Penguji akan bertanya "bagaimana kalau di-fine-tune?" dan Anda harus punya angkanya.

### A7. "Ambang prior-matched Anda memakai skor test dan prior yang diketahui. Itu curang."

**Pertahanan:** sudah tertulis benar di `HASIL_EKSPERIMEN.md` — pertahankan struktur itu. Nyatakan sebagai **transduktif**, sah untuk forensik batch/arsip, tidak sah untuk streaming, dan **laporkan angka induktif berdampingan**. Jangan pernah menaruh 95,59% sendirian di abstrak.

**Lubang yang belum ditutup:** prior-matching mengasumsikan prior uji 50/50. Itu informasi tentang test set. **Tambahkan kurva sensitivitas salah-spesifikasi prior** (asumsi prior 0,1–0,9 → akurasi). Kalau akurasi runtuh ketika prior salah 20%, penguji berhak menyebut metode itu rapuh, dan lebih baik Anda yang menemukannya lebih dulu.

### A8. "Bukti pintasan codec Anda berasal dari NAMA BERKAS, bukan dari sinyal."

**Pertahanan yang ada:** pengukuran akustik mendukung (4,18× untuk fake turunan MP3 vs 1,17× untuk 652 fake WAV-native).

**Pertahanan yang jauh lebih kuat, dan wajib dikerjakan — ini figur terbaik yang bisa Anda hasilkan:** intervensi kausal. Ambil berkas **real** dari test set, kompres ke MP3 lalu dekode, dan berikan ke model yang dilatih **tanpa** augmentasi. Kalau model membalik prediksinya menjadi "fake", Anda telah membuktikan kausalitas, bukan korelasi. Biaya: sekitar 20 menit. Nilai: mengubah bab temuan dari "kami mengamati" menjadi "kami membuktikan".

### A9. "for-rerec terkonfound blok kelas/sesi."

**Pertahanan:** jangan jadikan eksperimen unggulan. Sajikan sebagai probe sekunder, ungkap confound-nya sendiri (urutan blok, pergeseran durasi/bit-depth, ~26% pasangan tak terpetakan), gunakan analisis berpasangan via korelasi envelope untuk subset yang terpetakan, dan gunakan McNemar berpasangan. Penguji yang menemukan confound yang **sudah Anda tulis sendiri** akan berhenti menyerang.

### A10. "Rumusan masalah Anda ketahanan noise, tapi temuan Anda tentang codec dan kalibrasi."

**Pertahanan:** codec/bandwidth **adalah** gangguan kanal, satu keluarga dengan derau lingkungan dan reverb — semuanya nuisance non-forensik yang mengubah distribusi masukan tanpa mengubah label. Perluas rumusan masalah menjadi "ketahanan terhadap gangguan kanal dan lingkungan", lalu **kerjakan sweep SNR-nya** (item #6, belum selesai) dengan dekomposisi Cllr yang sama. Tanpa sweep SNR yang benar-benar dijalankan, serangan ini mengenai sasaran.

### A11. "Proposal Anda tidak menyebut EER, sekarang EER ada di mana-mana."

**Pertahanan:** metrik proposal (akurasi, presisi, recall, F1, ROC, AUC, confusion matrix) tetap dilaporkan **seluruhnya**. EER, Cllr, dan min-Cllr ditambahkan sebagai pelengkap dengan alasan yang eksplisit: pada test set lintas-domain, akurasi@0,5 tidak dapat ditafsirkan (baris 2: akurasi 50,00% dengan AUC 0,7946 dan F1 = 0,00%). Menambah tidak sama dengan mengganti.

---

## 4. Ide berpotensi tingkat DISERTASI

Tiga kandidat, diurutkan menurut rasio (besarnya klaim) × (peluang berhasil).

### D1 — "Generalisasi lintas-domain pada deteksi deepfake adalah masalah KALIBRASI, bukan masalah REPRESENTASI"

**Klaimnya:** bidang ini menghabiskan lima tahun mencari representasi yang lebih baik (SSL front-end, graph attention back-end, augmentasi raw-waveform) untuk mengatasi kegagalan lintas-domain. Hipotesis disertasi: sebagian besar kegagalan itu bukan kehilangan informasi, melainkan **pergeseran fungsi keputusan**. Konsekuensinya membalik prioritas riset — dan membalik rekomendasi praktis dari "latih ulang" menjadi "kalibrasi ulang", yang berbeda tiga orde besar dalam biaya.

**Mengapa ini besar:** ia mengubah *apa yang dilaporkan* bidang ini. Kalau benar, min-EER dan min-DCF — dua metrik yang mendefinisikan seluruh papan peringkat ASVspoof — secara sistematis menyembunyikan mode kegagalan yang dominan di dunia nyata, karena keduanya oracle-terkalibrasi. Pernyataan itu setara dengan mengatakan bahwa papan peringkat mengukur hal yang salah.

**Bukti awal yang sudah Anda punya, dan ini luar biasa kuat untuk sebuah titik awal:**
- Augmentasi codec sendirian: **+0,28 pp**. Perbaikan ambang sendirian: **+21,88 pp**. Keduanya: **+45,59 pp**.
- AUC 0,9900 bersanding dengan akurasi 50,28%.
- Pada for-rerec, AUC bertahan 0,80–0,89 sementara F1 runtuh ke 0,048–0,369.

**Program disertasinya:** (i) dekomposisi Cllr/min-Cllr lintas 4–6 korpus (FoR, ASVspoof 2019/2021, In-the-Wild, MLAAD, WaveFake) × 4–6 backbone — semuanya post-hoc atas skor, muat di satu GPU; (ii) menunjukkan bahwa calibration loss mendominasi discrimination loss di sebagian besar pasangan lintas-domain; (iii) melatih dengan *proper scoring rule* (Cllr) alih-alih cross-entropy dan mengukur apakah kalibrasi ikut tergeneralisasi; (iv) kepala kalibrasi terkondisi yang digerakkan estimator kondisi buta; (v) mengusulkan pelaporan actual-Cllr sebagai metrik wajib.

**Risiko:** hipotesis bisa gagal pada korpus di mana pergeseran memang menghancurkan daya pisah. Tapi kegagalan itu sendiri informatif dan tetap menghasilkan disertasi — Anda akan memetakan *kapan* masing-masing dominan.

**Yang bisa dimasukkan ke tesis S2 sekarang:** butir (i) dan sebagian (iv), pada satu korpus. Itu sudah cukup untuk tesis, dan meninggalkan sisanya sebagai jalur S3 yang jelas.

### D2 — "Anggaran pintasan" sebagai prediktor transfer

**Klaimnya:** untuk sepasang korpus (A, B), selisih *nuisance probe battery* memprediksi celah transfer A→B **tanpa melatih model apa pun**. Kalau berhasil, orang bisa menyaring pasangan dataset dalam hitungan detik alih-alih hitungan hari GPU, dan celah transfer berhenti menjadi kejutan dan menjadi kuantitas terukur.

**Bukti awal:** audit Anda memprediksi keruntuhan sebelum model dilatih — energi >6 kHz real turun 5,8× dari training ke testing, provenance MP3 90,7% → 0%. Model lalu runtuh persis seperti diprediksi (50,00%, memprediksi semuanya "real"). Itu satu kasus prediksi berhasil. Sebuah disertasi mengubah satu anekdot menjadi hukum yang terkalibrasi.

**Mengapa besar:** ia bersifat **prediktif**, bukan deskriptif. Bidang ini penuh laporan "model X tidak transfer ke dataset Y". Belum ada yang bisa mengatakan *sebelumnya* seberapa buruk, dan mengapa.

**Risiko:** butuh banyak korpus (murah per korpus, tapi banyak izin/unduhan), dan hubungannya bisa saja terlalu berisik untuk dikalibrasi.

### D3 — Konsistensi STFT sebagai invarian forensik pemersatu

**Klaimnya:** satu skalar bebas-model — inkonsistensi STFT, ‖STFT(iSTFT(X)) − X‖ / ‖X‖ — menjelaskan empat fenomena terpisah sekaligus: (a) mengapa ucapan tervocoder terdeteksi; (b) mengapa enhancement berbasis mask magnitudo (MetricGAN+) menghancurkan countermeasure sementara enhancement domain-waveform (SEGAN) tidak; (c) mengapa augmentasi fase bekerja; (d) mengapa proses rekam-ulang menggeser kedua kelas secara asimetris.

**Mengapa besar:** menyatukan empat pengamatan terpisah di bawah satu besaran fisis yang dapat dihitung dalam milidetik adalah bentuk kontribusi yang bertahan lama. Ia juga memberi kriteria pemilihan pra-pemrosesan yang *dapat dihitung sebelumnya* untuk forensik: pilih enhancer dengan inkonsistensi rendah, bukan dengan PESQ tinggi.

**Kekuatan bukti saat ini:** paling lemah dari ketiganya. Pengukuran fase yang Anda punya (relL2 0,378 pada π, 2,72 dB pada frame aktif setelah RMS-matching) mengonfirmasi mekanismenya tapi belum menghubungkannya ke EER. Paper sumber sudah melakukan korelasi per-arm dengan PESQ/SRMR (R² 0,90–0,97), jadi menukar variabel adalah inkremental **kecuali** konsistensi terbukti *lebih* prediktif daripada PESQ. Itu uji tunggal yang menentukan, dan murah.

**Verdict saya:** D1 adalah kernel disertasi terbaik — klaim terbesar, bukti awal terkuat, biaya termurah. D2 paling ambisius. D3 paling elegan tapi paling rapuh.

---

## 5. Peringatan jujur: klaim paling rapuh yang WAJIB Anda verifikasi ulang sendiri

Diurutkan menurut kerusakan bila salah.

### F1. σ = ±3,50 pp berasal dari **3 seed** — dan estimasi itu sendiri sangat tidak pasti

Dengan n=3 (2 derajat kebebasan), interval kepercayaan 95% untuk simpangan baku membentang kira-kira **0,5× hingga 3,2×** dari estimasi. Artinya σ sesungguhnya bisa di mana saja antara ~1,8 dan ~11 pp. **Setiap** kesimpulan "dapat dibedakan" dan "tidak dapat dibedakan" di `PERBANDINGAN.md` bergantung pada angka ini.
→ **Aksi:** minimal 5 seed, idealnya 8–10 untuk dua model teratas. Biaya ~20 menit/run. Ini prioritas nomor satu.

### F2. Ensemble 97,61% dibangun dari **"seed terbaik per model"** — itu seleksi pada test set

Ini angka yang akan masuk abstrak Anda, dan ia terkontaminasi. Memilih seed terbaik masing-masing model *berdasarkan performa test* lalu mengensemblekannya adalah bentuk peeking yang halus tapi nyata.
→ **Aksi:** bangun ulang ensemble dengan protokol yang telah ditetapkan di muka (mis. selalu seed 42, atau rerata skor seluruh seed), dan laporkan ensemble sebagai rerata ± std atas beberapa triplet seed. Catatan Anda bahwa ensemble 12-run memberi 97,61% identik adalah tanda menggembirakan, tapi bukan pengganti protokol yang bersih.

### F3. `lead_sil` AUC = 0,5001 kemungkinan besar **artefak konstruksi for-2sec**, bukan sifat FoR

Perhatikan pola nama berkas yang Anda dokumentasikan sendiri:

```
file1000.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav
                                    ^^^^^^^
```

Ada **tahap `_silence`** dalam pipeline pembuatan dataset. Artinya silence sudah dibuang saat varian ini dibangun. Melaporkan "pintasan silence tidak ada di FoR" sebagai hasil negatif yang mengejutkan adalah kesalahan yang akan langsung dilihat penguji yang teliti — jawabannya trivial dan tertulis di nama berkas.
→ **Aksi:** ukur ulang `lead_sil` pada `for-norm` dan `for-original` (keduanya tersedia di server York: 5.945 MB dan 7.926 MB). Kalau pintasan silence muncul di sana dan hilang di for-2sec, klaim Anda menjadi jauh lebih menarik dan **benar**: *"pra-pemrosesan for-2sec menghilangkan pintasan silence, tetapi tidak menghilangkan pintasan provenance codec."* Itu pernyataan yang tajam. Versi sekarang tidak dapat dipertahankan.

### F4. Bukti provenance MP3 bersifat **korelasional dan berbasis nama berkas**

Rantai penalaran Anda: nama berkas → distribusi provenance → energi HF → performa model. Setiap mata rantai masuk akal, tapi tidak satu pun intervensi.
→ **Aksi:** jalankan intervensi kausal (serangan A8): MP3-kan berkas real dari test set, ukur pembalikan prediksi pada model non-augmentasi. Tambahkan juga verifikasi tingkat-sinyal (deteksi cutoff MDCT) yang independen dari nama berkas.

### F5. AUC probe di `audit_aucs.json` — belum jelas dihitung pada partisi mana

Beberapa nilai **terbalik**: `hf_slope` 0,323, `centroid` 0,342, `e_4_6k` 0,359, `rolloff95` 0,370. AUC di bawah 0,5 berarti prediktor kuat dengan arah terbalik (0,323 → 0,677 bila dibalik). Kalau ini dihitung pada gabungan seluruh dataset, angkanya mencampur dua domain yang berbeda dan berisiko paradoks Simpson: arah di training bisa berlawanan dengan arah di testing, dan agregatnya tidak berarti apa-apa.
→ **Aksi:** hitung ulang **per partisi** (training / validation / testing terpisah) dan laporkan ketiganya. Pembalikan arah antara training dan testing, jika benar-benar ada, justru adalah bukti pintasan yang paling elegan yang bisa Anda tunjukkan — jauh lebih kuat daripada tabel rasio energi.

### F6. Ketidaksesuaian dengan Ahmad dkk. **belum terselesaikan** dan bisa membatalkan seluruh perbandingan

Mereka: for-2sec **44,1 kHz**, ~31.138 klip, 24.913 latih / 6.225 uji. Anda: **16.000 Hz pada 100% dari 17.870 berkas**, 13.956 / 2.826 / 1.088. Test set mereka **5,7× lebih besar**. Ini bukan selisih kecil — ini indikasi bahwa mereka memakai korpus yang berbeda (kemungkinan `for-original` atau mirror Kaggle).
→ **Aksi:** selesaikan sebelum angka mereka dikutip sebagai pembanding. Kalau tidak terselesaikan, nyatakan di naskah bahwa keduanya **tidak dapat dibandingkan langsung**, dan jangan taruh keduanya dalam satu kolom akurasi.

### F7. Setiap ID arXiv dalam rentang 2509–2607 harus Anda buka sendiri

Riset hulu mengutip belasan makalah dengan ID sangat baru (2509.12003, 2512.13744, 2601.06560, 2603.07935, 2603.14767, 2604.13400, 2606.11674, 2607.03150). Satu di antaranya sudah terbukti mengandung pernyataan yang bertentangan dengan pengukuran Anda (klaim 44,1 kHz). Beberapa angka yang dikutip darinya saling bertabrakan (17.870 vs 31.138 berkas untuk korpus yang sama).
→ **Aturan yang saya sarankan tanpa pengecualian: jangan kutip makalah yang belum Anda buka sendiri dan catat nomor tabel/halamannya.** Ini bukan kehati-hatian berlebihan; satu sitasi yang tidak ada akan meruntuhkan kredibilitas seluruh bab tinjauan pustaka dalam satu pertanyaan.

### F8. Klaim mekanisme MetricGAN+ vs SEGAN — **terbalik** di riset hulu

Riset hulu menyatakan MetricGAN+ "melakukan re-sintesis seperti-vocoder" dan SEGAN "menjaga mikro-struktur fase". Faktanya kebalikannya: SEGAN beroperasi end-to-end di domain waveform (jadi ia yang meresintesis setiap sampel), sedangkan MetricGAN+ memprediksi mask magnitudo dan **memakai ulang fase noisy**.
→ **Aksi:** kalau kalimat apa pun tentang enhancement masuk naskah, verifikasi langsung ke `speechbrain/lobes/models/MetricGAN.py` dan paper Pascual dkk. (Interspeech 2017). Jangan mengandalkan ringkasan.

### F9. Klaim retraksi yang sudah benar — **jangan dihidupkan lagi**

`VERIFIKASI_RUJUKAN.md` sudah mencabut klaim *"sebagian besar hasil terpublikasi pada FoR memakai split acak"* karena Ahmad dkk. terverifikasi memakai partisi resmi. Klaim itu menggoda karena rapi dan dramatis. **Biarkan tercabut.** Yang boleh dipertahankan hanya klaim mekanistik (+16,69 pp dari skema split, model & fitur identik) dan klaim per-paper dengan kutipan langsung.

### F10. Angka 95,59% adalah **run beruntung**, dan ia beredar di banyak dokumen Anda

`HASIL_EKSPERIMEN.md` sudah menandainya dengan benar (rerata sesungguhnya 91,94% ± 3,50). Tapi 95,59% muncul di beberapa tempat tanpa peringatan yang menyertainya.
→ **Aksi:** sisir seluruh dokumen; setiap kemunculan 95,59% harus membawa rerata multi-seed di kalimat yang sama.

---

## Lampiran: urutan kerja yang saya rekomendasikan

**P0 — wajib, sebelum apa pun ditulis (perkiraan 1–2 hari)**

1. HuBERT ×3 seed — kepatuhan judul. Tanpa ini tesis tidak dapat disidangkan sebagaimana berjudul.
2. Naikkan ke ≥5 seed untuk `cnn_asp` dan `wav2vec2` (memperbaiki F1).
3. Bangun ulang ensemble dengan protokol seed yang ditetapkan di muka (memperbaiki F2).
4. Intervensi kausal MP3 pada real test set (menutup A8 dan F4). ~20 menit, figur terbaik dalam tesis.
5. Ukur ulang `lead_sil` pada for-norm/for-original (memperbaiki F3).
6. Hitung ulang AUC probe **per partisi** (memperbaiki F5).
7. Kurva sensitivitas salah-spesifikasi prior (menutup A7).

**P1 — kontribusi inti (perkiraan 3–5 hari)**

8. Dekomposisi Cllr / min-Cllr / calibration loss untuk seluruh run tersimpan. Biaya GPU nol.
9. Sweep SNR −5…30 dB × {augmentasi, tanpa} × 4 arsitektur, dengan asimetri FN/FP dan dekomposisi Cllr per level.
10. Generalisasi lintas-keluarga-noise (latih keluarga A, uji keluarga B).
11. **Kalibrasi induktif** — AS-norm / quantile mapping / ambang terkondisi. Ini kebaruannya. Target: merebut kembali sebagian besar selisih 83,40 → 91,94 tanpa transduktif.
12. Perbaiki rancangan `wavval`: **bagi** 652 fake WAV-native antara training dan validasi alih-alih memindahkan semuanya. Ini kemungkinan besar kemenangan termurah yang tersisa — memulihkan akurasi induktif sambil mempertahankan std ±1,41.

**P2 — pelengkap (perkiraan 2–3 hari)**

13. Ekstrak for-rerec, evaluasi mismatch dengan confound diungkap eksplisit.
14. Unfreeze encoder SSL (menutup A6).
15. Dengarkan seluruh 26 berkas salah dari ensemble — 52 detik audio, menghasilkan taksonomi error yang tidak bisa dibeli dengan compute berapa pun.

**Yang saya sarankan JANGAN dikerjakan**, karena riset hulu sudah menunjukkan dampaknya nol pada konteks ini: perbaikan `ComputeDeltas(win_length=400)` (pipeline Anda tidak memakainya), kompensasi encoder delay ffmpeg (Anda memakai FFT-lowpass, bukan pipe ffmpeg), supervisi empat-kelas (dua kelas kosong pada konfigurasi codec terbaik Anda), ablasi SpAArSIST (back-end Anda bukan berbasis graf), dan low-pass 4 kHz sebagai mitigasi (menurunkan akurasi tanpa jaminan generalisasi).