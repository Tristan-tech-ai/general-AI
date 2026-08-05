# Analisis Kritis — Lapisan Kedua

*Dokumen pendamping untuk [ANALISIS_DAN_RENCANA.md](ANALISIS_DAN_RENCANA.md).*

Dokumen pertama menjawab pertanyaan yang diajukan: "bagaimana menaikkan akurasi". Dokumen ini menjawab pertanyaan yang **tidak** diajukan tapi lebih menentukan nasib tesis ini — temuan dari membaca ulang proposal secara adversarial, ditambah beberapa reframing yang menurut saya lebih berharga daripada mengejar 0,5 poin akurasi.

---

## 0. Cara membaca dokumen ini — status epistemik

Setiap klaim di bawah diberi label. Ini penting: sebagian analisis saya berbasis teks proposal (bisa saya buktikan), sebagian berbasis pengetahuan domain (bisa salah), dan sebagian adalah tebakan yang **harus** Anda uji sendiri.

| Label | Arti |
|---|---|
| ✅ **VERIFIED** | Dikutip langsung dari PROPOSAL_TESIS.pdf, dengan nomor halaman. Bisa Anda cek sendiri. |
| 🔬 **TESTABLE** | Hipotesis saya. Saya sertakan cara mengujinya dalam < 1 jam. Jangan percaya sampai diuji. |
| 📚 **DOMAIN** | Pengetahuan umum bidang anti-spoofing. Saya tidak punya akses ke dataset/kode Anda untuk memverifikasinya di sini. Tingkat keyakinan saya dicantumkan. |
| ❓ **UNKNOWN** | Saya tidak tahu, dan tidak akan berpura-pura tahu. Daftar lengkap ada di [§6](#6-apa-yang-saya-tidak-tahu). |

Saya tidak punya akses ke dataset FoR, kode Anda, atau log training Anda. Semua di bawah ini adalah pembacaan dokumen + inferensi. **Yang berlabel 🔬 dan 📚 wajib Anda verifikasi sebelum masuk naskah tesis.**

---

## 1. Ketegangan yang belum ada yang menamainya

✅ **VERIFIED.** Judul tesis: *"**Analisis Performa** Arsitektur..."*. Tujuan penelitian (hal. 6): *"mengevaluasi dan **membandingkan** performansi antara arsitektur..."*. Alasan konfigurasi seragam (hal. 68): *"...perbedaan hasil klasifikasi yang diperoleh lebih mencerminkan pengaruh karakteristik arsitektur masing-masing model daripada perbedaan pengaturan pelatihan."*

**Masalahnya:** permintaan "naikkan akurasi semaksimal mungkin" secara langsung bertabrakan dengan tujuan penelitian yang tertulis.

- Kalau Anda mengoptimasi tiap model habis-habisan (LR berbeda, arsitektur head berbeda, augmentasi berbeda), Anda **tidak lagi membandingkan arsitektur** — Anda membandingkan seberapa keras Anda mengoptimasi masing-masing. Tujuan tesis gugur.
- Kalau Anda mempertahankan konfigurasi seragam demi keadilan, Anda **mengunci tiga dari empat model pada konfigurasi yang merusaknya** (LR 1e-3). Akurasi gugur.

Ini bukan masalah teknis, ini masalah desain penelitian, dan **proposal tidak menyadarinya.**

**Jalan keluar yang saya rekomendasikan — "anggaran tuning seragam":**

> Tiap arsitektur mendapat **jumlah percobaan hyperparameter yang identik** (misal 8 run), pada **ruang pencarian yang identik secara relatif** (LR ∈ [base/10, base×10] di mana base ditentukan oleh konvensi tiap kelas model), dengan **data, split, augmentasi, dan protokol evaluasi yang identik**. Yang dibandingkan adalah performa terbaik masing-masing dalam anggaran yang sama.

Ini adil, standar di literatur benchmarking, menjawab keberatan penguji, **dan** membebaskan Anda menaikkan akurasi. Tulis paragraf ini menggantikan paragraf hal. 68. Ini satu-satunya cara memenuhi dua tujuan sekaligus.

**Konsekuensi tambahan yang harus Anda terima:** setelah perubahan ini, tesis tidak lagi bisa mengklaim "arsitektur X terbaik" secara absolut — hanya "terbaik dalam anggaran tuning 8 run pada FoR-2sec". Itu klaim yang lebih kecil tapi **benar**, dan penguji yang kompeten akan lebih menghargainya.

---

## 2. Enam temuan yang tidak ada di daftar permintaan

### F1 — Bab metodologi mendeskripsikan *pre-training*, bukan penelitian Anda

✅ **VERIFIED — ini temuan paling serius di dokumen ini.**

Hal. 59 (§4.4) menyatakan niat yang benar:
> *"...digunakan sebagai model dasar untuk melakukan **transfer learning** pada tugas klasifikasi audio asli dan audio deepfake."*

Tapi hal. 65 (§4.3.3 HuBERT) mendeskripsikan sesuatu yang sama sekali berbeda:
> *"...keluaran dari Transformer digunakan untuk menghasilkan Prediction terhadap Accoustic Unit pada segmen audio yang di-mask. Prediction ini kemudian dibandingkan dengan Pseudo-Label yang diperoleh dari Acoustic Unit Discovery System. **Selisih antara hasil Prediction dan Pseudo-Label digunakan sebagai sinyal pembelajaran untuk memperbarui parameter model selama proses pelatihan.**"*

Itu adalah deskripsi **pre-training HuBERT oleh Meta AI pada 60.000 jam LibriLight**, bukan deskripsi penelitian Anda. Dalam penelitian Anda, Acoustic Unit Discovery System tidak dijalankan, pseudo-label tidak dibuat, dan masked-unit loss tidak dihitung. Anda memuat checkpoint yang sudah jadi dan melatih classification head dengan cross-entropy pada label real/fake.

Hal yang sama terjadi pada Wav2Vec2 (hal. 60–61): masking laten, quantization, dan pemilihan target dijelaskan sebagai bagian dari alur pemodelan penelitian.

**Mengapa ini penting — tiga alasan berurutan keparahan:**

1. **Penguji akan menanyakannya.** "Di mana Acoustic Unit Discovery System dalam kode Anda?" adalah pertanyaan yang tidak bisa dijawab. Ini pertanyaan yang wajar untuk penguji yang paham HuBERT.
2. **Bab hasil tidak akan nyambung dengan bab metodologi.** Metodologi menjanjikan alur A, hasil melaporkan alur B.
3. 🔬 **Ada kemungkinan nyata ini bukan sekadar kesalahan tulis, tapi cerminan kesalahan implementasi** — misalnya menggunakan `HubertModel` mentah tanpa classification head yang benar, atau salah memahami apa yang dilatih. **Cek sekarang:** print `model.__class__.__name__` dan `sum(p.numel() for p in model.parameters() if p.requires_grad)`. Untuk fine-tuning yang benar Anda harus melihat `HubertForSequenceClassification` (atau head kustom Anda) dan jumlah parameter trainable yang masuk akal.

**Perbaikan naskah:** pisahkan tegas menjadi dua subbab per model — *"(a) Pra-pelatihan (dilakukan oleh pengembang model, tidak diulang dalam penelitian ini)"* dan *"(b) Fine-tuning (dilakukan dalam penelitian ini)"*. Subbab (b) harus menyebut: layer mana yang dibekukan, arsitektur head, loss function, dan jumlah parameter yang dilatih. Saat ini informasi itu tidak ada sama sekali untuk keempat model.

---

### F2 — AST kemungkinan besar sedang dijalankan dengan 80% input berupa padding

🔬 **TESTABLE — kandidat bug terbesar kedua setelah learning rate.**

✅ Fakta dari proposal: audio berdurasi **2 detik** (hal. 55), AST menerima log-Mel spectrogram yang dibagi menjadi patch (hal. 62–63), AST memakai **12 layer Transformer** dan positional embedding (hal. 63).

📚 Fakta domain (keyakinan tinggi): checkpoint AST standar (`MIT/ast-finetuned-audioset-10-10-0.4593`) dilatih pada AudioSet dengan `max_length = 1024` frame ≈ **10,24 detik**, 128 mel bin, patch 16×16 dengan stride 10 → **1212 patch**, dan positional embedding berukuran 1212 yang ikut dilatih.

**Konsekuensi aritmetiknya:** audio 2 detik menghasilkan ~200 frame. Jika `max_length` dibiarkan default 1024, maka **~80% masukan adalah padding nol**, dan positional embedding untuk posisi 200–1212 (yang membawa informasi terlatih) dipetakan ke wilayah kosong. Model menghabiskan sebagian besar kapasitas atensinya pada keheningan buatan.

**Cek 5 menit:**
```python
inputs = feature_extractor(audio_2s, sampling_rate=16000, return_tensors="pt")
print(inputs["input_values"].shape)      # ingin (1, 200, 128) — BUKAN (1, 1024, 128)
print(model.config.max_length)           # ingin 200 — BUKAN 1024
print(model.audio_spectrogram_transformer.embeddings.position_embeddings.shape)
print(feature_extractor.mean, feature_extractor.std)   # ingin ≈ -4.2677393, 4.5689974
```

**Perbaikan:**
```python
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification
cfg = ASTConfig.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593",
                                max_length=200, num_labels=2)
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593", config=cfg,
    ignore_mismatched_sizes=True)      # positional embedding akan di-interpolasi
fe = ASTFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593", max_length=200)
```
⚠️ `ignore_mismatched_sizes=True` akan **menginisialisasi ulang** positional embedding secara acak, bukan menginterpolasinya. Untuk hasil terbaik, interpolasi manual bobot posisi lama (1212 → 228 patch) dengan `F.interpolate` — ini menyelamatkan informasi posisi terlatih dan biasanya bernilai beberapa poin akurasi.

**Isu terpisah yang sama pentingnya — normalisasi bertabrakan.** ✅ Proposal hal. 56–57 menerapkan **peak amplitude normalization** pada waveform. 📚 Namun `ASTFeatureExtractor` melakukan normalisasinya sendiri dengan mean/std AudioSet (−4,27 / 4,57) pada log-Mel, dan `Wav2Vec2FeatureExtractor`/`HubertFeatureExtractor` untuk checkpoint `-large-ll60k` mensyaratkan `do_normalize=True` (zero-mean unit-variance pada waveform). Peak normalization di hulu **mengubah statistik yang diasumsikan feature extractor**. Efeknya bervariasi dari kecil sampai merusak, dan mudah luput karena tidak ada error yang muncul.

**Cek:** `print(feature_extractor.do_normalize)` untuk tiap model, dan pastikan Anda tidak menormalisasi dua kali dengan skema yang berbeda.

**Estimasi dampak F2 jika terbukti:** 📚 keyakinan sedang — **+2 sampai +6 poin untuk AST**, dan menjelaskan mengapa AST mungkin terlihat lebih lemah dari yang seharusnya. Jika AST Anda saat ini underperform dibanding tiga model lain, ini tersangka utamanya, bukan arsitekturnya.

---

### F3 — Berapa file yang sebenarnya salah? Dengarkan semuanya.

Ini nasihat paling murah dan paling terabaikan dalam dokumen ini.

Aritmetika anggaran error, dengan `n` = ukuran test set:

| n test | Error @ 98% | Error @ 99% | Error @ 99,5% |
|---|---|---|---|
| 1.000 | 20 | 10 | 5 |
| 2.000 | 40 | 20 | 10 |
| 4.400 | 88 | 44 | 22 |
| 10.000 | 200 | 100 | 50 |

❓ Saya tidak tahu ukuran test set Anda — verifikasi sendiri. Tapi pada skala mana pun di tabel ini, **jumlah error Anda muat dalam satu sesi mendengarkan.** 88 file × 2 detik = 3 menit audio. Dengan jeda dan pencatatan: sekitar 45 menit kerja.

**Lakukan ini sebelum eksperimen apa pun berikutnya:**

```python
import pandas as pd, numpy as np
probs = model.predict_proba(X_test)[:, 1]
err = pd.DataFrame({"file": test_files, "y": y_test,
                    "p": probs, "conf": np.abs(probs - 0.5)})
err = err[(err.p > 0.5) != (err.y == 1)].sort_values("conf", ascending=False)
err.to_csv("kesalahan_terurut.csv", index=False)   # paling atas = salah & paling yakin
```

Dengarkan 20 teratas (yang salah dengan keyakinan tertinggi — ini yang paling informatif), dan kategorikan. Hipotesis kategori yang mungkin Anda temukan:

| Kategori | Artinya | Tindakan |
|---|---|---|
| Audio hampir seluruhnya hening / potongan buruk | Masalah **data**, bukan model | Perbaiki VAD/trimming — bisa langsung memangkas separuh error |
| **Label salah** | Masalah **anotasi** | Perbaiki. Batas atas akurasi Anda = 100% − noise label |
| Real yang terdengar robotik / TTS yang sangat natural | Masalah **model** yang sesungguhnya | Baru di sini arsitektur/augmentasi relevan |
| Terkonsentrasi pada 1–2 pembicara | **Kebocoran/bias pembicara** | Konfirmasi temuan T0-1 |
| Terkonsentrasi pada 1 sistem TTS | **Generalisasi antar-sintesizer** | Sangat menarik — ini bahan subbab tersendiri |

**Mengapa ini penting:** jika 40 dari 88 error Anda ternyata file rusak atau salah label, maka **batas atas akurasi yang mungkin dicapai adalah ~99,1%**, dan seluruh Tier 2–3 di dokumen pertama akan berebut sisa 48 file. Anda perlu tahu itu **sebelum** menghabiskan enam minggu. Tidak ada teknik di dokumen mana pun yang bisa memperbaiki label yang salah.

📚 Keyakinan tinggi: pada hampir semua proyek yang mencapai 98%, sebagian besar error tersisa adalah masalah data, bukan masalah model. Ini alasan utama mengapa mengejar 99,5% lewat hyperparameter search biasanya sia-sia.

---

### F4 — Perbandingan empat model mungkin tidak dapat difalsifikasi pada ukuran sampel Anda

Ini ground truth statistik murni — bisa dihitung, tidak perlu dipercaya.

Interval kepercayaan 95% untuk akurasi 98% (normal approximation, `1,96 × √(p(1−p)/n)`):

| n test | ± (poin persentase) | CI 95% pada 98,0% |
|---|---|---|
| 1.000 | ±0,87 | [97,1 – 98,9] |
| 2.000 | ±0,61 | [97,4 – 98,6] |
| 4.400 | ±0,41 | [97,6 – 98,4] |
| 10.000 | ±0,27 | [97,7 – 98,3] |

**Implikasinya keras:** jika Wav2Vec2 mendapat 98,3% dan AST 97,9% pada n = 4.400, **interval kepercayaan keduanya bertumpang tindih hampir sepenuhnya.** Kalimat "Wav2Vec2 mengungguli AST" tidak didukung data. Padahal itu persis jenis kalimat yang akan menjadi kesimpulan tesis.

**Tiga cara keluar, urut dari terbaik:**

1. **Gunakan uji berpasangan, bukan CI independen.** Keempat model dievaluasi pada **file yang sama persis**, jadi datanya berpasangan. **Uji McNemar** jauh lebih sensitif daripada membandingkan dua CI, karena hanya melihat sampel yang tidak disepakati (*discordant pairs*).
   ```python
   from statsmodels.stats.contingency_tables import mcnemar
   # b = benar model A & salah model B ; c = salah A & benar B
   tabel = [[a, b], [c, d]]
   print(mcnemar(tabel, exact=True))    # exact=True bila b+c < 25
   ```
   Laporkan p-value untuk **keenam pasangan** model, dengan koreksi Holm–Bonferroni untuk perbandingan ganda.

2. **Pindah ke metrik dengan resolusi lebih tinggi.** Akurasi membuang informasi (membinerkan skor kontinu). **EER dan AUC memakai seluruh distribusi skor** dan jauh lebih stabil di rezim akurasi tinggi. Ini alasan teknis — bukan sekadar konvensi — mengapa bidang anti-spoofing memakai EER.

3. **Perbesar sinyal, bukan sampel.** Perbedaan antar arsitektur yang tidak terlihat pada audio bersih akan **melebar dramatis** pada SNR 0–5 dB. Di sana selisihnya bisa 10–20 poin, jauh di atas derau statistik. **Ini argumen terkuat untuk memindahkan pusat gravitasi tesis ke evaluasi ber-noise** — bukan karena lebih menarik, tapi karena di situlah perbandingan Anda **secara statistik dapat difalsifikasi.**

Poin 3 layak masuk ke naskah tesis sebagai justifikasi metodologis. Itu argumen yang kuat dan jarang ditulis mahasiswa S2.

---

### F5 — Kriteria ">94%" membandingkan varian dataset yang berbeda

✅ **VERIFIED.** Hal. 71: *"Apabila akurasi yang diukur tidak melebihi capaian penelitian sebelumnya sebesar 94%, maka harus dilakukan penyesuaian lebih lanjut."*

Angka 94% itu merujuk ke ref [19] MFAAN (94,47%) dan ref [20] CNN-LSTM (94,7%) — ✅ keduanya disebut proposal menggunakan "dataset Fake-or-Real (FoR)" (hal. 9–10, 13).

**Masalahnya:** ✅ proposal sendiri mencatat FoR punya **empat varian** — `for-original`, `for-norm`, `for-2sec`, `for-rerec` (hal. 4). Anda memakai `for-2sec`. ❓ Proposal **tidak pernah menyebutkan varian mana** yang dipakai ref [19] dan [20].

📚 Keyakinan sedang: varian ini berbeda tingkat kesulitannya secara substansial. `for-2sec` (potongan pendek, ternormalisasi, seimbang) umumnya **lebih mudah** daripada `for-rerec` (rekam ulang lewat pengeras suara — jauh lebih sulit), dan berbeda karakteristik dari `for-original`.

**Konsekuensi:** angka 98% Anda pada `for-2sec` dibandingkan dengan 94,7% mereka pada varian tak diketahui adalah perbandingan yang tidak valid — dan ini adalah **kriteria kelulusan yang ditulis sendiri oleh proposal**. Kalau ternyata ref [20] memakai `for-rerec`, maka 98% Anda tidak "mengalahkan" apa pun.

**Tindakan (wajib, ~1 jam):**
1. Buka paper ref [19] dan [20], cari varian dan protokol split yang mereka pakai. Catat.
2. Jika berbeda dari Anda: **jangan hapus tabel perbandingannya** — tambahkan kolom "Varian FoR" dan "Protokol split", lalu beri catatan kaki bahwa perbandingan bersifat indikatif, bukan head-to-head.
3. Jika sama: bagus, perbandingan valid, dan sebutkan itu eksplisit sebagai kekuatan.
4. Cara terkuat: **latih ulang CNN-LSTM Anda pada varian yang mereka pakai** sebagai jembatan pembanding. Anda sudah punya kode CNN-LSTM; biayanya rendah, dan hasilnya menjadi satu-satunya angka head-to-head yang sah dalam tesis.

Tabel perbandingan state of the art adalah bagian yang paling sering diserang penguji. Kolom "Varian" dan "Protokol" mengubahnya dari titik lemah menjadi titik kuat.

---

### F6 — Kriteria evaluasi bertentangan dengan dirinya sendiri

✅ **VERIFIED.** Tiga pernyataan dalam empat halaman berurutan:

| Halaman | Pernyataan | Standar tersirat |
|---|---|---|
| 70 | *"Nilai di atas 50% dalam metrik evaluasi seperti presisi, perolehan, dan skor F1 dianggap **baik** karena menunjukkan bahwa model berperforma lebih baik daripada tebakan acak."* | **50%** |
| 71 | *"Apabila akurasi yang diukur tidak melebihi capaian penelitian sebelumnya sebesar 94%, maka harus dilakukan penyesuaian lebih lanjut."* | **94%** |
| 72 | *"...dengan harapan mencapai akurasi **lebih dari 97%**."* | **97%** |

Tiga standar berbeda untuk hal yang sama. Yang pertama secara khusus bermasalah: pada tugas biner seimbang dengan state of the art 94%, menyebut 50% sebagai "baik" adalah pernyataan yang akan langsung disorot penguji. Kalimat itu tampaknya diadopsi dari template metodologi tugas klasifikasi umum dan tidak disesuaikan dengan konteks.

**Perbaikan:** hapus kalimat 50%. Ganti seluruh blok dengan tabel kriteria bertingkat per kondisi uji ([§7 dokumen pertama](ANALISIS_DAN_RENCANA.md#7-kriteria-sukses--target-metrik)). Kriteria tunggal tidak masuk akal untuk penelitian yang menguji beberapa kondisi noise — akurasi 90% pada SNR 0 dB jauh lebih mengesankan daripada 98% pada audio bersih, dan kriteria tunggal tidak bisa menangkap itu.

---

## 3. Reframing: tiga kontribusi yang lebih berharga daripada +1,5 poin akurasi

Ini bagian yang benar-benar "di luar prompt". Kalau saya menulis tesis ini, di sinilah saya menaruh energi.

### K1 — Kurva keruntuhan informasi: pada SNR berapa deteksi deepfake menjadi mustahil?

📚 **Argumen fisik (keyakinan tinggi).** Deteksi deepfake bekerja dengan menemukan **artefak vocoder**: distorsi fase, ketidakkonsistenan harmonik, energi pita tinggi yang tidak wajar, dan pola periodisitas buatan. 📚 Banyak dari artefak ini berada di frekuensi tinggi dan beramplitudo rendah.

Noise aditif memiliki sifat yang tepat merusak hal itu: pada SNR rendah, **noise mengubur artefak beramplitudo rendah sebelum mengubur konten wicara beramplitudo tinggi.** Artinya informasi diskriminatif hilang **lebih cepat** daripada inteligibilitas.

**Konsekuensi yang bisa diuji:** harus ada **titik keruntuhan** — SNR di mana akurasi jatuh dari tinggi ke mendekati kebetulan dalam rentang sempit, bukan menurun landai. Dan titik itu **berbeda untuk tiap arsitektur**, tergantung pita frekuensi mana yang diandalkan masing-masing.

**Ini adalah pertanyaan penelitian yang jauh lebih tajam daripada "model mana yang paling akurat":**

> *Pada tingkat SNR berapa masing-masing arsitektur kehilangan kemampuan diskriminatifnya, dan apakah titik keruntuhan itu ditentukan oleh arsitektur atau oleh pita frekuensi artefak yang diandalkannya?*

**Cara mengeksekusi (sudah 90% tercakup rencana E4, tinggal ditambah dua hal):**
1. Evaluasi pada grid SNR rapat: **−5, 0, 5, 10, 15, 20, 25, 30, ∞ dB** (bukan hanya 4 titik).
2. Plot akurasi/EER vs SNR untuk keempat model dalam satu grafik. Cari titik infleksi.
3. Tambahan yang membuatnya jadi analisis, bukan sekadar tabel: **ulangi dengan noise yang dibatasi pita** — noise hanya di 0–2 kHz, hanya 2–4 kHz, hanya 4–8 kHz, pada SNR tetap. Model yang runtuh saat noise pita tinggi diberikan adalah model yang mengandalkan artefak pita tinggi. **Ini mengukur secara langsung *apa* yang dipakai tiap model** — bukan sekadar seberapa baik performanya.

**Nilai:** ini mengubah tesis dari "benchmark empat model" (kontribusi rendah, mudah diserang sebagai tidak baru) menjadi **karakterisasi batas informasi deteksi deepfake pada audio bergangguan** (kontribusi konseptual, sulit diserang, dan langsung menjawab rumusan masalah di hal. 6). Biaya tambahan dibanding rencana E4: sekitar dua hari.

### K2 — Deteksi deepfake bukan klasifikasi biner; ini masalah open-set

📚 **Keyakinan tinggi, didukung data di proposal sendiri.**

Kelas "real" tertutup dan stabil: suara manusia hari ini sama dengan suara manusia tahun lalu. Kelas "fake" **terbuka dan bergerak** — sistem TTS baru muncul terus-menerus, dan masing-masing meninggalkan artefak berbeda.

✅ Bukti kuat untuk ini ada di proposal Anda sendiri: ref [14] melaporkan EER In-the-Wild **31,14%** untuk model yang berkinerja sangat baik (EER 4,06%) pada ASVspoof 2019. ✅ Ref [22] melaporkan Wav2Vec2 hanya **65,28%** pada BanglaFake. ✅ Ref [21] melaporkan HuBERT EER **20,45%** pada ASVspoof 5. Pola yang sama berulang tiga kali di tabel Anda: **performa runtuh saat sistem sintesis berubah.**

📚 Konteks penting: FoR dirilis sekitar 2019 dan dibangun dari sistem TTS era itu. Model deteksi yang dilatih di FoR belajar mengenali artefak TTS 2018–2019 — bukan artefak vocoder neural modern (HiFi-GAN, VITS, dan seterusnya). Akurasi 98% pada FoR **tidak memberi informasi apa pun** tentang kemampuan mendeteksi deepfake yang dibuat hari ini.

**Reframing arsitektural yang mengikuti — dan ini ide yang tidak ada di daftar permintaan:**

Alih-alih melatih classifier biner, latih **model satu-kelas** yang memodelkan seperti apa suara *asli*, dan perlakukan semua yang menyimpang sebagai palsu:

```python
class OCSoftmax(nn.Module):
    """One-Class Softmax (Zhang et al.) — baseline kuat & sederhana untuk
       anti-spoofing open-set. Real ditarik ke satu pusat; fake didorong keluar,
       TANPA mengasumsikan fake punya struktur tunggal."""
    def __init__(self, dim, m_real=0.9, m_fake=0.2, alpha=20.0):
        super().__init__()
        self.center = nn.Parameter(torch.randn(1, dim)); nn.init.kaiming_uniform_(self.center)
        self.m_real, self.m_fake, self.alpha = m_real, m_fake, alpha
    def forward(self, x, labels):                    # labels: 1 = real, 0 = fake
        w = F.normalize(self.center, dim=1)
        x = F.normalize(x, dim=1)
        cos = x @ w.T                                 # (B, 1)
        m = torch.where(labels.bool(), self.m_real, self.m_fake).unsqueeze(1)
        scores = torch.where(labels.bool().unsqueeze(1), m - cos, cos - m)
        return F.softplus(self.alpha * scores).mean(), cos.squeeze(1)
```

📚 Keyakinan sedang-tinggi: OC-Softmax secara konsisten mengungguli cross-entropy biner pada **evaluasi lintas dataset** di literatur anti-spoofing, sambil setara pada data in-domain.

**Mengapa ini kontribusi yang kuat untuk tesis Anda:** Anda bisa melaporkan **matriks 4 arsitektur × 2 fungsi objektif** (CE vs OC-Softmax), diuji in-domain dan cross-dataset. Temuan yang saya perkirakan: **CE menang tipis di FoR, OC-Softmax menang telak di cross-dataset.** Itu temuan yang bersih, dapat dipublikasikan, dan menambah kedalaman signifikan dengan biaya rendah (satu loss function, tanpa ubah arsitektur). Biaya: ~2 hari untuk keempat model.

### K3 — Peta atensi: apakah keempat model melihat hal yang sama?

🔬 **TESTABLE, dan hasilnya mengubah keputusan Anda berikutnya.**

Ini bukan sekadar "explainability untuk mempercantik tesis" — ini **diagnostik prediktif** yang memberi tahu Anda, sebelum menghabiskan waktu, apakah ensembling (T2-5) akan berhasil.

**Logikanya:** ensemble hanya membantu bila error antar model **tidak berkorelasi**. Jika keempat model belajar mengeksploitasi artefak yang sama, mereka akan salah pada file yang sama, dan ensembling hanya akan memberi +0,1 poin — tiga hari kerja terbuang.

**Cara mengukur — tiga tingkat, dari termurah:**

1. **Korelasi error (30 menit, tanpa training).** Anda sudah punya prediksi keempat model. Hitung:
   ```python
   import itertools, numpy as np
   E = {m: (pred[m] != y_test).astype(int) for m in models}      # vektor error biner
   for a, b in itertools.combinations(models, 2):
       phi = np.corrcoef(E[a], E[b])[0, 1]                        # koefisien phi
       jac = (E[a] & E[b]).sum() / max((E[a] | E[b]).sum(), 1)    # Jaccard error
       print(f"{a:10s} vs {b:10s}  phi={phi:.3f}  jaccard={jac:.3f}")
   ```
   **Aturan keputusan:** φ rata-rata < 0,5 → ensembling akan sangat efektif, kerjakan. φ > 0,75 → keempat model melihat hal yang sama; ensembling hampir sia-sia, alihkan waktu ke K1 atau K2. Ini keputusan seharga tiga hari kerja, didapat dalam 30 menit.

2. **Saliency spektral (setengah hari).** Untuk AST dan CNN-LSTM, hitung gradien output terhadap input log-Mel, rata-ratakan di seluruh test set, dan plot sebagai heatmap 128 mel × waktu. Pertanyaan: apakah energi saliency terkonsentrasi di pita tinggi (mel bin 90–128, ≈ 6–8 kHz)? 📚 Prediksi saya: ya, dan itu menjelaskan sensitivitas terhadap noise dan kompresi sekaligus.

3. **Layer probing (satu hari).** Untuk Wav2Vec2 dan HuBERT: bekukan encoder, latih *linear probe* terpisah di atas **setiap** layer (12 dan 24 probe). Plot akurasi vs indeks layer. 📚 Prediksi saya (keyakinan sedang): puncaknya di **layer tengah**, bukan layer terakhir — layer akhir SSL cenderung fonetik/semantik, sedangkan artefak deepfake bersifat akustik tingkat rendah. Jika terbukti, itu **membenarkan langsung** rekomendasi layer-weighting (T2-3) dengan bukti dari data Anda sendiri, bukan sekadar mengutip ref [18].

**Nilai gabungan:** tiga hasil ini mengisi bab pembahasan dengan analisis mekanistik, bukan sekadar tabel angka. Perbedaan antara tesis yang menjawab "berapa" dan tesis yang menjawab "mengapa".

---

## 4. Pre-registrasi: prediksi saya sebelum Anda menjalankan eksperimen

Menulis prediksi **sebelum** eksperimen, lalu melaporkan mana yang benar dan mana yang meleset, adalah praktik ilmiah yang kuat dan sangat jarang dilakukan di tesis S2. Penguji akan memperhatikannya. Silakan pakai tabel ini apa adanya, atau ganti dengan prediksi Anda sendiri — yang penting ditulis sebelum hasil keluar.

| # | Prediksi | Alasan | Keyakinan |
|---|---|---|---|
| P1 | Pada FoR-2sec bersih, keempat model berada dalam rentang **2 poin** satu sama lain, dan sebagian besar pasangan **tidak berbeda signifikan** (McNemar p > 0,05) | Tugasnya sudah hampir jenuh; ada langit-langit dataset, bukan langit-langit arsitektur | Tinggi |
| P2 | **AST akan paling tahan terhadap noise lingkungan yang belum pernah dilihat**, meski mungkin kalah pada audio bersih | 📚 AST dipra-latih pada **AudioSet** — audio YouTube dunia nyata yang secara inheren berisik dan beragam. Wav2Vec2 dan HuBERT dipra-latih pada **LibriLight/Librispeech** — audiobook yang hampir seluruhnya bersih. Distribusi pra-pelatihan menentukan ketahanan domain | **Sedang-tinggi — ini prediksi paling kontraintuitif dan paling berharga di tabel ini** |
| P3 | **CNN-LSTM akan mengejutkan pada noise berat** relatif terhadap posisinya pada audio bersih | Dilatih dari nol pada distribusi teraugmentasi Anda; tidak punya *domain gap* pra-pelatihan yang harus diatasi | Sedang |
| P4 | HuBERT Large **tidak akan mengungguli** Wav2Vec2 Base secara sepadan dengan 3× ukurannya | 2 detik audio ≈ 99 frame laten. Model 317 juta parameter untuk 99 frame adalah over-parameterisasi ekstrem; keunggulan kapasitas tidak bisa terpakai | Sedang-tinggi |
| P5 | Ensembling memberi **< +1 poin** pada audio bersih tetapi **> +3 poin** pada noise berat | Pada audio bersih semua model benar pada file yang sama (error berkorelasi); pada noise mereka gagal secara berbeda | Sedang |
| P6 | Cross-dataset ke In-the-Wild akan menghasilkan **EER > 25%** untuk keempat model | ✅ Ref [14] di proposal Anda melaporkan 31,14% baseline; ✅ ref [22] melaporkan 65,28% akurasi lintas bahasa. Polanya konsisten | Tinggi |
| P7 | Setelah audit T0, **> 30% error yang tersisa adalah masalah data**, bukan model | 📚 Pola yang hampir universal pada sistem yang mencapai 98% | Sedang |
| P8 | Memperbaiki learning rate (T1-1) memberi peningkatan lebih besar daripada gabungan seluruh Tier 2 | Rezim LR yang salah adalah kegagalan orde pertama; sisanya orde kedua | Sedang-tinggi |

**P2 layak diperhatikan khusus.** Jika benar, ini temuan yang bagus dan dapat dipublikasikan: *"ketahanan terhadap noise pada deteksi deepfake lebih ditentukan oleh distribusi data pra-pelatihan daripada oleh arsitektur."* Itu pernyataan yang lebih kuat dan lebih berguna daripada "model X 98,3%, model Y 97,9%" — dan tesis Anda, dengan empat model dari empat rezim pra-pelatihan berbeda, **kebetulan berada dalam posisi sempurna untuk mengujinya.** Itu keunggulan desain yang belum dimanfaatkan proposal.

Jika P2 salah, itu juga hasil yang layak dilaporkan. Prediksi yang meleset dan dilaporkan jujur menunjukkan kematangan; prediksi yang tidak pernah ditulis tidak menunjukkan apa pun.

---

## 5. Lima tes ground-truth yang bisa dijalankan hari ini

Semua di bawah ini murah, falsifiable, dan menghasilkan angka konkret. **Jalankan sebelum eksperimen berikutnya.** Total ~3 jam. Nilai informasinya jauh melebihi tiga hari hyperparameter tuning.

### Tes 1 — Uji "digital silence" (5 menit)
📚 TTS sering menghasilkan keheningan digital sempurna (sampel bernilai persis 0,0), sedangkan rekaman mikrofon selalu punya *noise floor*. Jika ini membedakan kedua kelas, model Anda mungkin sedang menghitung nol, bukan mendeteksi deepfake.
```python
import soundfile as sf, numpy as np
def zero_frac(p):
    x, _ = sf.read(p); return float((np.abs(x) < 1e-6).mean())
# Bandingkan distribusinya per kelas; hitung AUC dari fitur TUNGGAL ini
from sklearn.metrics import roc_auc_score
print("AUC hanya dari fraksi sampel nol:", roc_auc_score(labels, [zero_frac(p) for p in files]))
```
**Interpretasi:** AUC > 0,70 → shortcut serius. AUC > 0,90 → sebagian besar "98%" Anda kemungkinan berasal dari sini.

### Tes 2 — Uji 5 fitur trivial (15 menit)
```python
feats = np.array([[dur_nonsilence(p), rms(p), peak(p), zero_frac(p), dc_offset(p)]
                  for p in files])
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
print("Akurasi 5 fitur trivial:", cross_val_score(LogisticRegression(max_iter=1000),
                                                  feats, labels, cv=5).mean())
```
**Interpretasi:** > 0,75 → dataset punya bias struktural yang harus dinetralkan sebelum angka apa pun bermakna.

### Tes 3 — Uji label acak (30 menit)
Acak label training, latih penuh, evaluasi pada test set berlabel benar. Akurasi **harus** ≈ 50%. Jika > 60%, ada kebocoran struktural yang sangat serius (biasanya berarti test set tercemar train set).

### Tes 4 — Uji band-limit (20 menit)
Low-pass semua audio pada 4 kHz, latih ulang, evaluasi.
**Interpretasi:** akurasi jatuh dari 98% ke ~70% → model bergantung sepenuhnya pada artefak pita tinggi. 📚 Itu **sah secara ilmiah** tapi rapuh — model semacam ini akan gagal total pada audio telepon (band 300–3400 Hz) dan pada MP3 bitrate rendah. Ini fakta yang sangat layak dilaporkan dalam pembahasan, dan menjelaskan hasil noise Anda sebelum Anda menjalankannya.

### Tes 5 — Uji pembicara tumpang tindih (30 menit)
```python
# Ekstrak speaker embedding dengan model pra-latih
from speechbrain.inference import EncoderClassifier
enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")
# Untuk tiap file test, cari kemiripan kosinus maksimum terhadap seluruh file train
# Plot histogram. Puncak di > 0,9 = pembicara yang sama muncul di kedua split.
```
**Interpretasi:** jika > 20% file test punya kecocokan pembicara di train dengan cos-sim > 0,9, maka split Anda tidak speaker-disjoint dan sebagian akurasi berasal dari hafalan identitas pembicara.

**Buat satu tabel ringkas hasil kelima tes ini dan masukkan ke tesis sebagai subbab "Validasi Integritas Dataset".** Hampir tidak ada tesis S2 yang punya bagian seperti ini. Bagian ini sendirian bisa mengubah persepsi penguji terhadap kualitas keseluruhan pekerjaan — dan lebih penting lagi, melindungi Anda dari pertanyaan yang tidak bisa dijawab di ruang sidang.

---

## 6. Apa yang saya TIDAK tahu

Ground truth berarti juga jujur tentang batas pengetahuan. ❓ **Berikut hal-hal yang tidak bisa saya verifikasi, dan yang analisis saya di atas terpaksa berasumsi:**

| Yang tidak saya ketahui | Mengapa penting | Cara Anda mencari tahu |
|---|---|---|
| Dari mana angka **98%** berasal — model mana, kondisi apa, split apa | Menentukan apakah seluruh dokumen ini menyasar target yang benar | Cek log/notebook Anda |
| **Ukuran aktual** train/val/test Anda | Menentukan lebar interval kepercayaan (§F4) dan jumlah error yang harus didengarkan (§F3) | `len()` pada tiap split |
| Apakah split Anda **speaker-disjoint** | Penentu tunggal terbesar validitas angka 98% | Tes 5 |
| **Kode Anda yang sebenarnya** — checkpoint HF apa, feature extractor apa, head apa, layer mana yang dibekukan | Semua klaim bug (F1, F2) bersifat inferensi dari teks proposal, bukan dari kode | Baca notebook Anda sendiri dengan checklist F2 |
| Varian FoR dan protokol yang dipakai **ref [19] dan [20]** | Menentukan apakah kriteria kelulusan ">94%" (hal. 71) valid | Buka kedua paper |
| **Level SNR** yang dipakai untuk test set ber-noise | ✅ Proposal menyebut 15–30 dB hanya untuk augmentasi *training* (hal. 57); untuk *testing* hanya disebut "penambahan noise" tanpa angka (hal. 55) | Tetapkan sekarang, pakai grid di K1 |
| Apakah augmentasi diterapkan **on-the-fly** atau di-precompute sekali | On-the-fly memberi variasi jauh lebih besar dan biasanya bernilai beberapa poin | Cek dataloader Anda |
| Apakah `for-rerec` bisa diakses | ✅ Proposal menyebutnya (hal. 4) tapi tidak memakainya (hal. 6) — padahal itu persis skenario "rekam ulang" yang dijadikan motivasi di latar belakang (hal. 1) | Cek folder dataset Anda |
| Kuota GPU Colab yang tersisa | Menentukan berapa banyak dari roadmap yang realistis | — |

📚 Semua estimasi dampak numerik di kedua dokumen (**"+0,5 sampai +2 poin"** dan sejenisnya) adalah **rentang perkiraan berbasis pola umum literatur**, bukan hasil pengukuran pada sistem Anda. Perlakukan sebagai urutan prioritas, **bukan sebagai janji**. Satu-satunya cara mengetahui angka sebenarnya adalah ablation pada data Anda sendiri — itulah sebabnya E5 (ablation) ada di roadmap.

---

## 7. Kalau saya hanya punya waktu satu minggu

Urutan yang saya jalankan, dengan alasannya:

| Hari | Aksi | Alasan |
|---|---|---|
| 1 pagi | Tes 1–5 (§5) | Menentukan apakah 98% itu nyata. Segalanya bergantung pada ini. |
| 1 siang | Audit kode terhadap checklist F2 (AST `max_length`, normalisasi ganda, `do_normalize`) | Bug paling mungkin kedua; ditemukan dalam hitungan menit |
| 1 sore | Ekspor 88 error teratas, dengarkan, kategorikan (§F3) | Memberi tahu apakah sisa pekerjaan adalah masalah model atau masalah data |
| 2 | Perbaiki LR per model (T1-1) + AMP + early stopping on EER | Perbaikan berdampak terbesar dengan usaha terkecil |
| 3 | Jalankan ulang keempat model. Hitung korelasi error (K3 tingkat 1) | Menentukan apakah ensembling layak — keputusan seharga 3 hari |
| 4 | EER/DET/McNemar/threshold tuning (T1-3, T1-4) + uji signifikansi 6 pasangan | Membuat perbandingan Anda dapat difalsifikasi |
| 5 | Bangun protokol noise unseen (T0-3) + jalankan grid SNR −5…30 dB (K1) | Hasil inti tesis; di sinilah perbedaan antar model akhirnya terlihat |
| 6 | Attentive pooling + layer weighting (T2-3); layer probing (K3 tingkat 3) | Peningkatan terbaik per jam kerja, plus bukti untuk pembahasan |
| 7 | Score fusion (T2-5a) **bila** korelasi error hari 3 mendukung. Tulis semuanya. | Angka terbaik, hanya jika data hari 3 membenarkannya |

Perhatikan yang **tidak** ada di daftar: hyperparameter search, cross-validation, dataset tambahan, model baru. Semuanya bernilai — tapi tak satu pun sebanding dengan mengetahui bahwa 98% Anda nyata, bahwa AST tidak sedang berjalan dengan 80% padding, dan bahwa perbandingan empat model Anda dapat dibuktikan secara statistik.

---

## Satu paragraf penutup

Proposal ini solid secara struktur: rumusan masalah jelas, pemilihan empat arsitektur beralasan kuat dan mewakili empat paradigma yang benar-benar berbeda, dan tinjauan pustakanya relevan. Kelemahannya terkonsentrasi di satu tempat — **BAB IV terlalu tipis tepat di bagian yang menjadi klaim utama tesis.** Rumusan masalah (hal. 6) menempatkan noise sebagai pusat penelitian, tetapi metodologi mengalokasikan satu paragraf untuknya (hal. 55) tanpa menyebut satu pun level SNR pengujian, tanpa memisahkan noise *seen* dari *unseen*, dan tanpa protokol degradasi. Sementara itu, `for-rerec` — varian yang secara harfiah dirancang untuk skenario rekam-ulang yang dijadikan motivasi di halaman pertama — tersedia dan tidak dipakai. Memperbaiki ketimpangan itu bernilai jauh lebih besar bagi tesis ini daripada setiap poin akurasi yang dibahas di dokumen pertama.
