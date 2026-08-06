# Koreksi: Replikasi Proposal Tidak Pernah Benar-Benar Melatih Encoder

Status: **run ulang sedang diantrekan.** Dokumen ini ditulis lebih dahulu agar
angka yang sudah terlanjur ada di repositori tidak dibaca sebagai angka final.

## Apa yang keliru

Proposal menetapkan satu learning rate 0,001 untuk seluruh model. Untuk model
pra-latih, hal itu berarti encoder ikut dilatih. Bendera `--uniform-lr` dibuat
untuk meniru perilaku tersebut, dan log pelatihan mencetak "encoder DILATIH".

Kenyataannya encoder tidak pernah menerima satu pun gradien.

Ketiga kelas berbasis self-supervised membungkus forward encoder seperti ini:

```python
ctx = torch.no_grad() if self.frozen else torch.enable_grad()
with ctx:
    out = self.encoder(x, output_hidden_states=True)
```

Bendera `--uniform-lr` menyetel `requires_grad = True` pada seluruh parameter,
tetapi tidak pernah mengubah atribut `self.frozen`. Selama atribut itu masih
`True`, forward encoder tetap berjalan di dalam `torch.no_grad()`, sehingga
tidak ada graf komputasi yang tersimpan dan tidak ada gradien yang mengalir ke
belakang. Optimizer memang menerima grup parameter encoder, namun `grad`-nya
selalu `None` sehingga AdamW melewatinya tanpa suara.

## Bagaimana ini ketahuan

Tangga ablasi menjalankan dua konfigurasi yang seharusnya sangat berbeda:

| Langkah | Konfigurasi | Akurasi @0,5 | Akurasi @prior | EER | AUC |
|---|---|---|---|---|---|
| L2 | LR seragam 0,001, encoder "dilatih" | 51,29 | 75,37 | 24,36 | 0,8361 |
| L3 | LR per model, encoder dibekukan | 51,29 | 75,37 | 24,36 | 0,8361 |

Angkanya sama sampai empat desimal. Pemeriksaan lanjutan menunjukkan kedua
berkas skor identik secara bitwise, yaitu `np.array_equal` bernilai benar dengan
selisih maksimum 0,0. Dua konfigurasi yang berbeda tidak mungkin menghasilkan
keluaran yang identik bitwise kecuali keduanya sebenarnya konfigurasi yang sama.

## Pembuktian perbaikan

Uji langsung pada Wav2Vec2, menghitung berapa parameter encoder yang menerima
gradien setelah satu kali backward:

| Keadaan | Parameter encoder bergradien |
|---|---|
| `frozen=True`, perilaku lama | 0 dari 211 |
| `requires_grad=True` saja, yaitu bug-nya | 0 dari 211 |
| ditambah `frozen=False`, yaitu perbaikannya | 210 dari 211 |

Perbaikannya ada di `train.py`, yaitu menyetel `model.frozen = False` ketika
`--uniform-lr` diberikan.

## Apa yang terpengaruh dan apa yang tidak

Terpengaruh, yaitu seluruh run yang tag-nya memuat `ULR`:

* baris "Proposal apa adanya" pada matriks 2x2, untuk AST dan WavLM
* kolom "proposal" pada dekomposisi ambang dan pelatihan
* tabel replikasi proposal pada split acak, yaitu angka 99,80 sampai 99,94 persen
* langkah L2 pada tangga ablasi

Tidak terpengaruh, karena pada konfigurasi ini encoder memang sengaja dibekukan
dan `frozen=True` adalah perilaku yang benar:

* seluruh konfigurasi "diperbaiki", yaitu run dengan tag `full`
* CNN-LSTM dan CNN+ASP, yang tidak memiliki atribut `frozen`
* audit kebocoran codec MP3 pada dataset
* efek protokol pembagian data, yang diukur dengan CNN+ASP
* dekomposisi kegagalan noise menjadi kalibrasi dan daya pisah
* keruntuhan model anti-spoofing publik di luar domain
* seluruh eksperimen band-gain dan RawBoost
* pengujian lintas generasi text-to-speech

Dengan kata lain, temuan pokok penelitian ini tidak bergantung pada arm yang
keliru. Yang perlu diukur ulang adalah pembanding baseline-nya.

## Perkiraan arah perubahan, dicatat sebelum run ulang dijalankan

Melatih encoder transformer pra-latih pada learning rate 0,001 selama 20 epoch
tanpa early stopping adalah laju yang sangat tinggi untuk fine-tuning. Dugaan
yang dicatat di muka adalah bahwa hasilnya akan lebih buruk daripada versi
berencoder beku pada partisi resmi, karena representasi pra-latih kemungkinan
besar rusak. Bila dugaan itu benar, selisih antara proposal dan versi rekayasa
akan melebar, bukan menyempit.

## Hasil sebenarnya: dugaan itu keliru, dan arahnya berkebalikan

AST pada partisi resmi, seed 42, batch 32, seluruh setelan lain sama:

| | Akurasi @0,5 | Akurasi @prior | EER | AUC | Epoch terbaik |
|---|---|---|---|---|---|
| Sebelum perbaikan, encoder tidak pernah dilatih | 51,56 | 83,18 | 16,82 | 0,9099 | 3 |
| Sesudah perbaikan, encoder benar-benar dilatih | **73,16** | **92,56** | **7,44** | **0,9780** | 16 |

Fine-tuning pada laju 0,001 tidak merusak representasi pra-latih. Equal error
rate justru turun lebih dari setengah dan AUC naik menjadi 0,9780. Epoch terbaik
bergeser dari 3 menjadi 16, yang berarti model memang terus belajar sepanjang
pelatihan dan bukan langsung overfit seperti yang diduga.

## Akibatnya bagi kesimpulan penelitian

Konfigurasi rekayasa pada partisi resmi mencapai 89,15 persen. Konfigurasi
proposal yang diimplementasikan dengan benar mencapai 92,56 persen. Dengan kata
lain **konfigurasi rekayasa tertinggal 3,41 poin persentase** dari baseline yang
seharusnya dikalahkannya. Pada ambang 0,5 selisihnya lebih lebar lagi, yaitu
65,72 lawan 73,16.

Angka "nilai rekayasa +37,59 poin persentase" yang sempat dilaporkan bukan nilai
rekayasa. Angka itu artefak dari baseline yang lumpuh karena bug.

Sebab teknisnya dapat ditunjuk dengan jelas. Paket rekayasa membekukan encoder.
Keputusan itu ternyata membuang lebih banyak daripada yang dibeli oleh early
stopping, augmentasi penuh, agregasi berbobot antar lapisan, dan normalisasi
loudness digabung sekalian.

Langkah lanjutan yang jelas adalah menggabungkan sisi terbaik keduanya, yaitu
encoder yang ikut dilatih seperti pada proposal ditambah early stopping,
augmentasi penuh, dan agregasi berbobot antar lapisan. Kombinasi itu belum
pernah dijalankan sama sekali dan dijadwalkan pada `run_unfreeze.ps1`.

## Catatan tentang cara temuan ini muncul

Bug ini tidak ditemukan lewat pemeriksaan kode, melainkan lewat tangga ablasi
yang kebetulan menempatkan dua konfigurasi berbeda secara berdampingan. Tanpa
ablasi itu, angka +37,59 poin persentase akan masuk ke naskah akhir sebagai
temuan utama, dan kesimpulan penelitian akan terbalik dari yang sebenarnya.

Ini juga alasan mengapa dugaan ditulis lebih dahulu. Bila dugaan tidak dicatat
di muka, sangat mudah menyusun penjelasan setelah melihat hasil dan mengaku
sudah menduganya sejak awal.
