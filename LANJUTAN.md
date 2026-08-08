# Titik Lanjut, Status & Cara Melanjutkan Tanpa Mengulang

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


*Terakhir diperbarui: 4 Agustus 2026. Baca dokumen ini lebih dulu saat melanjutkan.*

---

## 0. STATUS TERKINI (5 Agustus 2026)

✅ **Workflow riset SELESAI PENUH**, 183 agent, 0 error, 961 temuan, 13,8 juta token subagent.
Hasil sintesis sudah diekstrak ke berkas (tidak perlu menjalankan ulang):

| Berkas | Isi |
|---|---|
| `RISET_PETA_NOVELTY.md` | 3 usulan novelty + arsitektur final + tabel prioritas + batas atas realistis |
| `RISET_RENCANA_EKSPERIMEN.md` | Rencana E0–E9 dengan anggaran GPU terukur, dataset, repo, jebakan |
| `RISET_CELAH_DAN_KRITIK.md` | 7 celah ilmiah terbuka + 11 serangan penguji (A1–A11) + cara bertahan |
| `RISET_VERIFIKASI.md` | 1 bertahan / 25 direfutasi dari verifikasi adversarial |
| `TEMUAN_RISET.md` | 1.599 temuan mentah terpanen dari journal |
| `VERIFIKASI_RUJUKAN.md` | Protokol split ref [13]/[19]/[20], butir #7 selesai |

⚠️ **Catatan tingkat kelolosan:** hanya **1 dari 26** temuan lolos verifikasi adversarial
3-lensa. Artinya kekuatan tesis ini berasal dari **data yang sudah diukur sendiri**,
bukan dari literatur. Perlakukan `TEMUAN_RISET.md` sebagai sumber ide, bukan sumber klaim.

### Lubang yang ditemukan sintesis dan sedang ditutup
- ⏳ **HuBERT belum pernah dijalankan** padahal ada di judul tesis, sedang berjalan (3 seed)
- ⚠️ **A1 (serangan penguji):** augmentasi codec dirancang *setelah* mengaudit test set →
 bentuk *test-set peeking*. Perlu dijawab eksplisit di naskah, lihat `RISET_CELAH_DAN_KRITIK.md` §3

---

## 1. Melanjutkan workflow riset (SUDAH SELESAI, arsip)

Workflow riset multi-agent dihentikan di tengah jalan. **Seluruh hasil agent yang
sudah selesai tersimpan dan akan dipakai ulang dari cache**, tidak ada yang
diulang.

| | |
|---|---|
| Run ID | `wf_b19d7541-a60` |
| Agent selesai | **112** dari ~150 |
| Fase tercapai | Sweep (34 dimensi) ✅ · DeepDive ✅ · Refute (sebagian) ⏸ · Synthesize ⏸ |
| Script | `C:\Users\Tristan\.claude\projects\C--Users-Tristan-Downloads\2eb65358-beca-4011-b948-1c0d9186e535\workflows\scripts\add-novelty-hunt-wf_b19d7541-a60.js` |
| Transkrip | `...\subagents\workflows\wf_b19d7541-a60\` (172 MB, journal.jsonl 214 entri) |

**Perintah untuk melanjutkan** (minta Claude menjalankan tool `Workflow`):

```
Workflow({
 scriptPath: "C:\\Users\\Tristan\\.claude\\projects\\C--Users-Tristan-Downloads\\2eb65358-beca-4011-b948-1c0d9186e535\\workflows\\scripts\\add-novelty-hunt-wf_b19d7541-a60.js",
 resumeFromRunId: "wf_b19d7541-a60"
})
```

⚠️ Resume hanya berlaku **dalam sesi yang sama**. Bila sesi sudah berganti, hasil
112 agent tetap ada di `journal.jsonl` dan dapat dibaca manual, tetapi eksekusi
harus dimulai ulang. Kalau ingin isinya tidak hilang, ekstrak dulu:
`Read journal.jsonl` lalu simpan ringkasannya ke berkas.

---

## 2. Yang SUDAH selesai, jangan diulang

### Data (sudah lokal, jangan unduh lagi)

| Berkas | Ukuran | Status |
|---|---|---|
| `for-2sec.tar.gz` | 1.000 MB | ✅ terunduh & terekstrak ke `data/for-2seconds/` |
| `for-rerec.tar.gz` | 1.558 MB | ✅ terunduh, **belum diekstrak** |
| `data/for-2seconds/` | 17.870 wav | ✅ siap pakai |
| `manifest.csv` | 17.870 baris | ✅ dibangun |

> Link Google Drive asli **sudah kena kuota** ("have had many accesses"). Sumber
> resmi yang berfungsi: `https://bil.eecs.yorku.ca/share/for-2sec.tar.gz`
> (juga tersedia `for-norm` 5.945 MB dan `for-original` 7.926 MB bila diperlukan).

### Environment (terverifikasi berjalan)

```
Python 3.14.3 · PyTorch 2.11.0+cu128 · transformers 5.14.1
RTX 5060 Ti 16 GB (Blackwell sm_120) · bf16 matmul OK · 14,8 GiB VRAM bebas
Ryzen 5 7500F 6C/12T · 32 GB RAM · 458 GB disk bebas
```
Jalankan `py check_env.py` untuk memverifikasi ulang kapan saja.
Catatan: `librosa`/`numba` **tidak** dipakai (belum stabil di Python 3.14), semua
audio lewat `soundfile` + `torchaudio` + numpy.

### Audit dataset (selesai, hasil deterministik, tidak perlu diulang)

Skrip: `audit.py`, `probe_codec.py`, `probe_shift.py`, `probe_split.py`
Laporan: `audit_report.md`, `probe_*_report.md`, `TEMUAN_GROUND_TRUTH.md`

Temuan inti yang sudah terkunci:
- Split resmi = 13.956 / 2.826 / **1.088** (bukan 60/20/20 seperti proposal hal. 55)
- 0 duplikat, format seragam 100% (16 kHz mono 2,000 s)
- **90,7% fake di training berasal MP3; 0% fake di testing** → pintasan codec
- Energi >6 kHz: real training 5,8× lebih tinggi dari real testing → pergeseran domain
- Split acak vs resmi (RandomForest fitur trivial): 95,91% vs 79,23% = **+16,69 pp**

### Eksperimen deep learning (16 run tersimpan di `runs/`)

Angka final yang jujur, `cnn_asp`, split resmi, augmentasi codec, ambang
prior-matched, batch 32, 10 epoch, 3 seed:

| Strategi validasi | rerata | std |
|---|---|---|
| Validasi resmi | **91,94%** | ±3,50 |
| Validasi resmi + augmentasi | **92,28%** | ±2,96 |
| Validasi cocok-domain (`wavval`) | **83,40%** | ±1,41 |

Run tunggal lain (n=1, **jangan dijadikan peringkat**):
Wav2Vec2 90,99% · AST 83,46% · CNN-BiLSTM 80,88% · split acak 99,75–99,94%

Lihat `HASIL_EKSPERIMEN.md` untuk tabel lengkap dan koreksi.

### Kode (selesai & teruji)

```
forlib/data.py manifest, 4 skema split, augmentasi (codec/noise/reverb/gain)
forlib/models.py Wav2Vec2/HuBERT/WavLM/AST/CNN-LSTM + 2 varian ablation
forlib/metrics.py EER, DET, McNemar, Holm-Bonferroni, ECE, temperature scaling,
 prior-matched threshold
train.py loop pelatihan lengkap
recompute.py hitung ulang semua run dari skor tersimpan (cepat)
show.py tabel ringkas semua run
smoke_test.py uji semua arsitektur dapat dibangun & backward
```

Bug proposal yang sudah diperbaiki di kode: LR per model (bukan seragam 1e-3),
AST `max_length=200` + interpolasi positional embedding (terverifikasi
`pos_emb=(1,230,768)`), encoder SSL beku, normalisasi loudness (bukan peak),
attentive stats pooling, layer weighting, ambang dari validasi, EER sebagai
metrik seleksi, AMP bf16, early stopping.

---

## 3. Yang BELUM selesai, daftar kerja berikutnya

Urut prioritas.

| # | Tugas | Perintah / catatan |
|---|---|---|
| ~~1~~ | ~~Multi-seed 4 model~~ | ✅ **SELESAI**, 12 run, hasil di `PERBANDINGAN.md` |
| ~~2~~ | ~~Uji McNemar + Holm-Bonferroni~~ | ✅ **SELESAI**, `py compare.py official codec` |
| ~~9~~ | ~~Ensemble + korelasi error~~ | ✅ **SELESAI**, ensemble 4 arsitektur = **97,61%**, φ error 0,06–0,22 |
| 3 | **Unfreezing encoder SSL**, belum diuji sama sekali, bisa mengubah peringkat | tambahkan `--unfreeze` |
| 4 | Perbaiki rancangan `wavval`, bagi 652 fake WAV, jangan pindahkan semua | edit `forlib/data.py` skema `wavval` |
| 5 | Ekstrak & evaluasi `for-rerec` (skenario rekam-ulang) | `for-rerec.tar.gz` sudah lokal |
| 6 | Evaluasi noise unseen bertingkat SNR −5…30 dB | augmentasi sudah ada, tinggal skrip evaluasi per-kondisi |
| 7 | **Verifikasi protokol split ref [13]/[19]/[20]** ⚠️ wajib sebelum klaim soal literatur | baca 3 paper, catat varian FoR + skema split |
| 8 | Uji pengacakan fase (menguji klaim fase di ARSITEKTUR.md §2.3) | belum diimplementasikan |
| 9 | Ensemble 4 model + korelasi error | setelah #1 selesai |
| 10 | Analisis error: dengarkan berkas yang salah (~80 berkas) | `runs/*/test_scores.npy` + `manifest.csv` |

---

## 4. Dokumen yang sudah ditulis

| Berkas | Isi | Status |
|---|---|---|
| `TEMUAN_GROUND_TRUTH.md` | Audit empiris dataset, **paling penting** | ✅ final |
| `HASIL_EKSPERIMEN.md` | Hasil eksperimen + koreksi multi-seed | ✅ diperbarui |
| `ARSITEKTUR.md` | Analisis arsitektur dari prinsip pertama | ⚠️ 2 klaim terbantah data (dicatat di §8 TEMUAN) |
| `ANALISIS_DAN_RENCANA.md` | Rencana 23 item + roadmap | ⚠️ asumsi A1 salah (dikoreksi di TEMUAN §0) |
| `ANALISIS_KRITIS.md` | Audit validitas + reframing | ⚠️ sebagian kekhawatiran terbantah (duplikat: nol) |
| `LANJUTAN.md` | Dokumen ini | ✅ |

---

## 5. Kalimat kontribusi tesis (draf, sudah didukung data)

> Dengan arsitektur, data, dan hyperparameter yang identik, protokol pembagian
> data acak menghasilkan akurasi 99,9% sementara partisi resmi menghasilkan
> ~50% pada dataset Fake-or-Real. Selisih tersebut berasal dari korelasi semu
> antara provenance codec dan label kelas: 90,7% sampel deepfake pada data latih
> berasal dari berkas MP3 sedangkan 0% sampel deepfake pada data uji berasal dari
> MP3. Augmentasi codec yang diterapkan seragam pada kedua kelas menetralkan
> korelasi tersebut dan menurunkan EER dari 28,1% menjadi 8,1%, sehingga akurasi
> pada protokol resmi mencapai 92,3% ± 3,0 atas tiga inisialisasi.

⚠️ Klaim tentang penelitian sebelumnya **belum boleh** dituliskan sampai butir #7
di atas selesai.
