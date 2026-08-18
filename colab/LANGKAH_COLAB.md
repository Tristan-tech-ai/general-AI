# Menjalankan Repositori Ini di Google Colab

Ringkasan langkah. Rinciannya ada di dalam notebook, dan tiap sel di sana sudah
memuat penjelasan mengapa ia perlu dijalankan.

## Yang sudah disiapkan

| Berkas | Isi |
|---|---|
| `colab/Colab_Deteksi_Deepfake.ipynb` | notebook, 39 sel, siap diunggah ke Colab |
| `colab/colab_patch.py` | penyesuai presisi GPU untuk `train.py`, dapat dibatalkan |
| `colab/golden_head.json` | sidik jari sha256 tiap berkas markdown pada commit `ef6a2b0` |
| `cek_dataset.py` | memverifikasi dataset di mesin mana pun identik dengan yang dipakai di sini |
| `dataset_acuan.json` | sidik jari acuan dataset, ikut ke dalam bundle |
| `colab_bundle.zip` | seluruh kode, skor 159 run, `manifest.csv`, dan acuan dataset, sekitar 9 MB, ada di akar folder proyek |

## Langkah

Repositori <https://github.com/Tristan-tech-ai/general-AI> sudah publik, jadi
tidak perlu token dan tidak perlu mengunggah apa pun.

1. Buka <https://colab.research.google.com>, pilih **File → Upload notebook**,
   lalu ambil `colab/Colab_Deteksi_Deepfake.ipynb`.
2. Untuk Jalur B, atur **Runtime → Change runtime type → T4 GPU**. Jalur A
   tidak memerlukannya.
3. Jalankan bagian 0 sampai 3 secara berurutan. Pada bagian 1 jalankan
   **Cara 1** saja, yaitu klon dari GitHub. **Lewati sel Cara 2 dan Cara 3**,
   keduanya cuma cadangan.
4. Setelah itu pilih jalurnya:
   - **Jalur A** (bagian 4 sampai 6), sekitar dua menit, tanpa GPU dan tanpa
     dataset.
   - **Jalur B** (bagian 7 sampai 11), perlu GPU dan mengunduh dataset 1 GB.
5. Simpan hasilnya lewat bagian 12 sebelum sesi berakhir.

Nomor sel untuk Jalur A: **2, 4, 10, 12, 14, 16, 19, 20**. Sel 6 dan 8
dilewati.

Diuji dari klon anonim: 29 dari 29 skrip berhasil dalam 61 detik,
perbandingan sha256 memberi 40 identik dan 7 berubah, dan `NASKAH.pdf`
terbangun utuh 19 halaman.

### Kalau kode di GitHub berubah

Di sesi Colab berikutnya, klon barunya otomatis membawa versi terbaru. Kalau
runtime lama masih hidup dan ingin memperbarui tanpa mengulang dari awal,
jalankan `!git pull` dari dalam folder proyek.

## Mengapa Jalur A tidak memerlukan dataset

Skor uji seluruh 159 run sudah ikut ter-commit sebagai `runs/*/test_scores.npy`,
totalnya hanya 17 MB. Bobot modelnya yang berukuran 120 GB memang tidak ikut,
tetapi seluruh tabel, pengujian statistik, gambar, dan kedua PDF dihitung dari
skor itu. Yang bisa dijalankan di Colab tanpa dataset: 29 skrip pelaporan,
termasuk `naskah.py` yang membangun `NASKAH.pdf` utuh.

## Empat skrip yang sengaja tidak ikut

| Skrip | Sebab |
|---|---|
| `extract_findings.py` | membaca `journal.jsonl` dari sesi workflow lama yang tidak ada di Colab. Di mesin lokal skrip inilah yang menimpa `TEMUAN_RISET.md` sehingga banner penarikan klaim hilang dan 3.609 tanda pisah em masuk kembali |
| `hubert_summary.py` | `glob("runs/hubert*")` ikut menyeret run split acak (3.574 berkas uji) ke dalam ensemble partisi resmi (1.088 berkas) |
| `verify_sota_collapse.py` | memerlukan checkpoint Nes2Net-X di `ckpt/`, tidak ikut repositori |
| `audit.py`, `probe_*.py`, `eval_*.py`, `hard_samples.py`, `analyze_errors.py` | membaca berkas audio, hanya bisa di Jalur B |

## Tiga hal yang berbeda di Colab dan sudah ditangani notebook

**Presisi.** `train.py` memaksa `bfloat16` di setiap GPU. Itu benar pada
RTX 5060 Ti berarsitektur Blackwell, tetapi Tesla T4 tidak memiliki bfloat16 di
perangkat kerasnya dan PyTorch akan meng-emulasinya tanpa satu pun peringatan.
`colab/colab_patch.py` membaca compute capability kartu lalu memilih `float16`
beserta GradScaler bila perlu. Pada kartu yang memang mendukung bfloat16,
perhitungannya tetap persis sama.

**Manifest.** `manifest.csv` di dalam bundle dibuat di Windows dan memuat
pemisah jalur backslash, yang di Linux terbaca sebagai satu nama berkas
panjang. Untuk Jalur A hal itu tidak masalah karena `naskah.py` dan
`gambar_paper.py` hanya memakai kolom split, label, dan is_mp3. Untuk Jalur B
notebook membangunnya ulang, dan itu wajib.

**Jumlah worker.** Bawaan `train.py` adalah enam worker, sedangkan Colab
tingkat gratis hanya punya dua core. Notebook selalu memakai `--workers 2`.

## Menjaga dataset tetap sama di device lain

Dataset tidak ikut bundle karena ukurannya 1 GB, jadi tiap device mengambilnya
sendiri. Dua pengaman membuat "sama" menjadi hal yang dibuktikan, bukan
diharapkan.

| Pengaman | Nilai acuan | Waktu periksa |
|---|---|---|
| sha256 arsip `for-2sec.tar.gz` | `acd09881757832b5e9435d93d05e88ed366b41139ba63d47bbfa61344d706a16` | beberapa detik |
| ukuran arsip | 1.048.591.372 byte | seketika |
| sidik pohon hasil ekstrak | `baf65aaf01f07540145f77d735e0686611f75c1d008f46ad60cff61a15a27a38` | 14 detik untuk 17.870 berkas |

Sidik pohon adalah sha256 atas daftar terurut berisi jalur relatif dan sha256
tiap berkas wav. Jalurnya dinormalkan ke bentuk POSIX, sehingga nilainya sama
di Windows maupun di Linux. Sudah diuji: mengekstrak ulang arsip ke direktori
lain menghasilkan sidik yang identik.

**Device pertama, sekali saja.** Jalankan bagian 7 notebook. Arsip diunduh dari
York University, sha256-nya diperiksa, lalu disalin ke
`Drive/MyDrive/dataset-for/for-2sec.tar.gz`. Lima sampai sepuluh menit.

**Device kedua dan seterusnya.** Jalankan bagian 7 yang sama persis, tanpa
mengubah apa pun. Sel menemukan arsip di Drive, menyalinnya ke runtime, dan
melewati unduhan. Satu sampai dua menit. Syaratnya hanya masuk Colab dengan
akun Google yang sama.

Yang membuat ini terjamin: device kedua memakai berkas yang sama persis, bukan
unduhan baru, dan sha256-nya tetap diperiksa ulang sebelum diekstrak. Kalau
Drive merusak berkasnya, sel berhenti alih-alih diam-diam melanjutkan.

Tanpa Drive alurnya tetap aman, hanya lebih lambat: arsip diunduh ulang lalu
dibandingkan dengan `dataset_acuan.json`.

Memeriksa kapan saja, di mesin mana pun:

```bash
py cek_dataset.py                 # arsip dan pohon
py cek_dataset.py --hanya-arsip   # hanya arsip, beberapa detik
py cek_dataset.py --tulis         # tulis acuan baru, hanya bila dataset diganti
```

## Aturan yang tidak boleh dilanggar

Seluruh pelatihan di Colab menulis ke **`runs_colab/`**, bukan `runs/`. Jangan
pernah menggabungkan keduanya. Presisi dan perangkat kerasnya berbeda, sehingga
menggabungkannya akan melaporkan ragam antar perangkat sebagai ragam antar
inisialisasi acak. Itu persis kelas kekeliruan yang tujuh kali terjadi dalam
penelitian ini, dan `cek_konfigurasi.py` tidak dapat menangkapnya karena ia
hanya memeriksa jumlah epoch dan ukuran batch.

## Membangun ulang bundle

Bundle dibuat dari commit terakhir, bukan dari working tree, supaya Colab
menjadi ruang bersih. Untuk membuatnya ulang setelah ada commit baru:

```bash
py colab/buat_bundle.py
```

## Yang akan terlihat di bagian 5

Notebook membandingkan hasil bangunan ulang dengan sidik jari commit
`ef6a2b0`. Hasil yang diharapkan **40 identik, 7 berubah**. Ketujuh berkas itu
sudah tidak sinkron di dalam repositori sebelum Colab menyentuhnya: skor
delapan seed HuBERT sudah masuk ke `runs/`, tetapi sebagian dokumen hasil masih
memuat angka dari tiga seed. Colab hanya menampakkannya.

Perhatikan juga bahwa `HASIL_NOVELTY_PROBE.md` hasil bangunan ulang kehilangan
banner "KLAIM INI SUDAH DITARIK" dan kembali memuat koefisien korelasi yang
dihitung tanpa merata-ratakan seed per konfigurasi. Angka yang benar ada di
`HASIL_UJI_KORELASI.md`, yaitu r = -0,048 dengan p = 0,895 atas sepuluh
konfigurasi.
