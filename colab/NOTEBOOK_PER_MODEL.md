# Dua Belas Notebook per Model

Dokumen ini menjelaskan rangkaian notebook Google Colab yang menyertai
penelitian deteksi deepfake audio pada repositori ini. Rangkaian ini dibuat
untuk **menjelaskan prosesnya**, bukan untuk mengejar angka hasil secepat
mungkin. Untuk keperluan yang kedua tersedia berkas terpisah,
`Colab_JalurB_Otomatis.ipynb`, yang menjalankan seluruh tahap dalam satu sel.

Empat arsitektur yang dibahas adalah keempat model yang disebut pada judul
penelitian: **Wav2Vec2, AST, HuBERT, dan CNN-LSTM**. Tiap model punya tiga
notebook, sehingga seluruhnya dua belas.

---

## Daftar berkas

| Model | Proses pelatihan | AdamW lawan NAdam | Dropout tetap lawan adaptif |
|---|---|---|---|
| Wav2Vec2 | [Proses_Wav2Vec2](https://colab.research.google.com/github/Tristan-tech-ai/general-AI/blob/main/colab/Proses_Wav2Vec2.ipynb) | [Optimizer_Wav2Vec2](https://colab.research.google.com/github/Tristan-tech-ai/general-AI/blob/main/colab/Optimizer_Wav2Vec2.ipynb) | [Dropout_Wav2Vec2](https://colab.research.google.com/github/Tristan-tech-ai/general-AI/blob/main/colab/Dropout_Wav2Vec2.ipynb) |
| AST | [Proses_AST](https://colab.research.google.com/github/Tristan-tech-ai/general-AI/blob/main/colab/Proses_AST.ipynb) | [Optimizer_AST](https://colab.research.google.com/github/Tristan-tech-ai/general-AI/blob/main/colab/Optimizer_AST.ipynb) | [Dropout_AST](https://colab.research.google.com/github/Tristan-tech-ai/general-AI/blob/main/colab/Dropout_AST.ipynb) |
| HuBERT | [Proses_HuBERT](https://colab.research.google.com/github/Tristan-tech-ai/general-AI/blob/main/colab/Proses_HuBERT.ipynb) | [Optimizer_HuBERT](https://colab.research.google.com/github/Tristan-tech-ai/general-AI/blob/main/colab/Optimizer_HuBERT.ipynb) | [Dropout_HuBERT](https://colab.research.google.com/github/Tristan-tech-ai/general-AI/blob/main/colab/Dropout_HuBERT.ipynb) |
| CNN-LSTM | [Proses_CNN_LSTM](https://colab.research.google.com/github/Tristan-tech-ai/general-AI/blob/main/colab/Proses_CNN_LSTM.ipynb) | [Optimizer_CNN_LSTM](https://colab.research.google.com/github/Tristan-tech-ai/general-AI/blob/main/colab/Optimizer_CNN_LSTM.ipynb) | [Dropout_CNN_LSTM](https://colab.research.google.com/github/Tristan-tech-ai/general-AI/blob/main/colab/Dropout_CNN_LSTM.ipynb) |

Seluruh tautan membuka Colab langsung dari repositori. Tidak ada berkas yang
perlu diunduh atau diunggah lebih dahulu.

---

## Sebelum menjalankan

**Kartu grafis.** Menu `Runtime` lalu `Change runtime type`, pilih **T4 GPU**,
lalu `Save`. Tanpa itu langkah pertama akan berhenti dan menjelaskan sebabnya.

**Dataset.** Dataset Fake-or-Real potongan dua detik berukuran sekitar 1,1 GB
setelah diekstrak, diambil dari arsip yang jauh lebih besar. Notebook mencari
sumbernya di Google Drive menurut urutan berikut, dan berhenti pada yang
pertama ditemukan:

1. cache ringkas hasil sesi sebelumnya, `dataset-for/for-2seconds.tar`
2. arsip penelitian yang sudah ada, `PenelitianAudioDeepfake/Dataset/FoR.zip`
3. arsip resmi, `dataset-for/for-2sec.tar.gz`
4. bila tidak satu pun ada, diunduh dari York University

Pada sesi pertama, biarkan `BUAT_CACHE_DATASET` bernilai `True`. Sesi
berikutnya akan memakai cache itu dan siap dalam sekitar satu menit, alih alih
mengulang ekstraksi.

**Satu sesi Colab, satu ekstraksi.** Menjalankan notebook kedua berarti sesi
baru. Cache di atas ada justru untuk itu.

---

## Isi tiap jenis notebook

### Proses, delapan langkah

Menjelaskan pelatihan satu model dari awal sampai akhir, satu sel per langkah.

| Langkah | Isi |
|---|---|
| 1 | Memeriksa runtime, mengambil kode, memasang dependensi |
| 2 | Mengambil dan **memverifikasi** dataset |
| 3 | Memutar dan menggambar satu suara asli dan satu suara palsu |
| 4 | **Apa yang sebenarnya masuk ke model ini** |
| 5 | Bentuk modelnya dan berapa parameter yang benar benar dilatih |
| 6 | Bagaimana data sengaja dirusak, beserta alasannya |
| 7 | Melatih |
| 8 | Membaca hasil, kalibrasi, dan kesalahannya |

Langkah 4 **berbeda isinya untuk tiap model**, karena di sinilah letak
perbedaan paling mendasar di antara keempatnya:

* **Wav2Vec2 dan HuBERT** menerima gelombang suara apa adanya, 32.000 angka,
  tanpa ada yang dibuang. Sel ini mencetak bentuk keluaran seluruh lapisan
  encoder, 13 lapisan untuk Wav2Vec2 dan 25 untuk HuBERT, lalu menggambar isi
  lapisan pertama dan terakhir berdampingan.
* **AST dan CNN-LSTM** menerima gambar suara, sehingga fase sinyal terbuang dan
  frekuensi tinggi dipadatkan. Sel ini **menghitung sendiri** besar pemadatan
  itu dari pengaturan yang benar benar dipakai model tersebut, dan menghasilkan
  angka sekitar 12 kali lebih kasar di frekuensi tinggi dibanding di frekuensi
  rendah. Angka itu hasil hitungan, bukan kutipan.

Langkah 8 memutar tiga berkas yang salah ditebak dengan keyakinan tertinggi,
lengkap dengan nama berkas dan skornya.

### Optimizer, AdamW lawan NAdam

Melatih model yang sama **dua kali**. Data, split, augmentasi, jumlah putaran,
batch, dan inisialisasi acak seluruhnya identik. Satu satunya yang berbeda
adalah optimizer-nya.

Perlu dicatat bahwa pembandingnya **bukan Adam polos, melainkan AdamW**, yaitu
Adam dengan peluruhan bobot terpisah, karena itulah yang dipakai pada seluruh
hasil penelitian ini sejak awal. Agar perbandingannya bersih, NAdam juga
dijalankan dengan peluruhan bobot terpisah, sehingga yang berbeda benar benar
hanya suku momentum Nesterov-nya, bukan dua hal sekaligus.

Langkah 3 memperagakan perbedaan mekanis keduanya pada permukaan dua dimensi
berbentuk lembah sempit, dengan titik awal, laju, dan jumlah langkah yang sama.
Peragaan itu menunjukkan **apa** yang berbeda, dan tidak dimaksudkan sebagai
bukti mana yang lebih baik untuk tugas ini.

### Dropout, tetap lawan adaptif

Melatih model yang sama dua kali dengan cara pembandingan yang sama.

Sisi pembanding memakai dropout biasa dengan laju tetap 0,2, yaitu angka yang
ditetapkan manusia dan tidak berubah selama pelatihan. Sisi yang diuji memakai
**Concrete Dropout** (Gal, Hron, dan Kendall, NeurIPS 2017), yang menjadikan
laju itu parameter yang ikut dilatih.

Dua hal yang membuat metode ini bekerja, dan keduanya diperagakan pada langkah 3:

1. Menjatuhkan unit adalah keputusan biner, dan keputusan biner tidak dapat
   diturunkan sehingga gradien tidak mengalir ke lajunya. Undian Bernoulli
   karena itu diganti relaksasi kontinu yang dapat diturunkan.
2. Tanpa penahan, gradien akan selalu menekan laju ke nol, sebab model yang
   tidak pernah menjatuhkan apa pun selalu mencatat loss latih lebih kecil.
   Penahannya adalah suku entropi yang menarik laju kembali ke arah 0,5.

Laju akhirnya adalah titik seimbang antara kedua tarikan itu, dan titik itu
ditentukan data. Notebook melaporkan laju yang akhirnya dipilih model beserta
arahnya terhadap tebakan awal 0,2.

---

## Perkiraan waktu

Angka di bawah untuk sepuluh putaran pada kartu T4 gratis.

| Model | Notebook Proses | Notebook perbandingan |
|---|---|---|
| CNN-LSTM | hitungan menit | belasan menit |
| Wav2Vec2 | belasan menit | sekitar setengah jam |
| AST | belasan menit | sekitar setengah jam |
| HuBERT | sekitar satu jam | sekitar dua jam |

Notebook perbandingan melatih dua kali, sehingga waktunya dua kali lipat.
Untuk memperlihatkan jalannya proses saja, `EPOCHS` dapat diturunkan menjadi 2
atau 3. Pada notebook perbandingan, setelan itu diisi sekali di langkah 4 dan
diambil kembali oleh langkah 5, sehingga tidak mungkin tanpa sengaja
membandingkan sepuluh putaran melawan tiga putaran.

---

## Batas yang perlu diketahui sebelum membaca angkanya

Ini bagian terpenting dari dokumen ini.

**Satu inisialisasi acak tidak cukup untuk menyimpulkan apa pun.** Menjalankan
model yang sama persis dengan hanya mengganti inisialisasi acaknya sudah
menghasilkan hasil yang berbeda. Pada penelitian ini, HuBERT dijalankan delapan
kali dengan pengaturan identik dan menghasilkan akurasi antara **93,93 dan
99,45 persen**, yaitu rentang 5,52 poin dari satu model yang sama.

Ragam antar inisialisasi yang terukur pada konfigurasi pembanding:

| Model | Akurasi rerata | Simpangan antar inisialisasi |
|---|---|---|
| Wav2Vec2 | 90,75 % | ±0,51 |
| AST | 86,43 % | ±2,94 |
| HuBERT | 97,29 % | ±2,16 |
| CNN-LSTM | 83,52 % | ±2,28 |

Kedua notebook perbandingan menutup dengan membandingkan selisih yang terukur
terhadap angka simpangan di atas, lalu menyatakan secara eksplisit apakah
selisihnya sudah melampaui ragam atau belum. Bila belum, notebook mengatakannya
terus terang alih alih membiarkan angka itu dibaca sebagai kesimpulan.

**Cara menutup perbandingan dengan benar:** jalankan tiap sisi dengan
`SEED = 42`, lalu `1337`, lalu `2024`; ambil rerata beserta simpangannya; dan
sebut berbeda hanya bila selisih rerata melampaui simpangan itu.

**Hasil Colab tidak boleh digabung dengan hasil lokal.** Notebook menulis ke
`runs_colab/`, sengaja terpisah dari `runs/`. Kartu grafis dan format bilangan
keduanya berbeda, sehingga selisih yang berasal dari perangkat keras akan
terbaca seolah olah selisih akibat inisialisasi acak.

---

## Verifikasi dataset

Dua lapis pemeriksaan berjalan otomatis sebelum pelatihan dimulai:

1. **Sidik jari arsip.** Kode sha256 arsip dibandingkan dengan yang tercatat
   pada `dataset_acuan.json`.
2. **Sidik jari pohon berkas.** Tiap berkas wav diperiksa satu per satu lalu
   diringkas menjadi satu kode. Pemeriksaan ini tidak bergantung pada nama
   folder maupun asal arsipnya, karena yang dibandingkan isi berkasnya.

Bila salah satu tidak cocok, notebook berhenti dan menolak melanjutkan. Itu
disengaja: angka yang dihasilkan dari dataset yang berbeda tidak sebanding
dengan angka yang dilaporkan pada repositori ini.

---

## Untuk pengembang

Kedua belas notebook **dibangkitkan**, tidak ditulis satu per satu. Sebagian
besar selnya identik, sehingga menulisnya dua belas kali berarti satu perbaikan
harus disalin ke dua belas tempat.

```
py colab/buat_notebook_proses.py
```

Berkas pendukung:

| Berkas | Isi |
|---|---|
| `colab/buat_notebook_proses.py` | generator kedua belas notebook |
| `colab/siapkan.py` | penyiapan bersama: runtime, kode, dataset, verifikasi |
| `colab/colab_patch.py` | penyesuai presisi GPU untuk `train.py`, dapat dibatalkan |

Dua saklar baru pada `train.py` yang mendukung notebook perbandingan:

```
--optimizer {adamw,nadam}     bawaan adamw, sesuai seluruh hasil terdahulu
--dropout   {tetap,adaptif}   bawaan tetap, laju 0,2
```

Keduanya ikut ke dalam nama folder keluaran sebagai penanda `NAD` dan `DA`.
Tanpa penanda itu, dua run dengan pengaturan berbeda akan memakai nama folder
yang sama dan yang belakangan menimpa yang duluan tanpa peringatan.
