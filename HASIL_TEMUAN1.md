# Pemecahan Temuan Pembuka

Temuan pertama menyatakan bahwa protokol pembagian data menentukan hasil, dengan bukti selisih hampir lima puluh poin persentase pada arsitektur, data, dan hyperparameter yang disebut identik. Tabel berikut memisahkan selisih itu menjadi sebab-sebabnya.

Pernyataan bahwa hyperparameternya identik ternyata tidak benar. Run yang menghasilkan angka pada split acak dijalankan selama enam epoch, sedangkan run yang menghasilkan angka pada partisi resmi dijalankan selama **satu** epoch. Keduanya berasal dari tahap paling awal penelitian, ketika nama direktori belum memuat penanda batch dan epoch, sehingga perbedaan itu tidak terlihat dari nama berkasnya dan tidak pernah diperiksa. Perbandingan aslinya karena itu bukan perbandingan terkontrol sama sekali. Baris ketiga dan keempat menjalankan keduanya pada konfigurasi yang seragam.

| Konfigurasi dan split | n | Akurasi @0,5 | Akurasi @prior | AUC |
|---|---|---|---|---|
| run asli, 6 epoch batch 64, split acak | 1 | 99.94 | 99.75 | 1.0000 |
| run asli, 1 epoch batch 64, partisi resmi | 1 | 50.00 | 71.88 | 0.7946 |
| 10 epoch batch 32, split acak | 3 | 99.93 (0.07) | 99.48 (0.47) | 1.0000 |
| 10 epoch batch 32, partisi resmi | 3 | 50.03 (0.05) | 92.56 (3.72) | 0.9756 |

## Tiga sebab yang terpisah

| Sebab | Besaran | Dapat diperbaiki tanpa mengubah protokol |
|---|---|---|
| Ambang keputusan tidak lagi cocok | 42.52 poin | ya, cukup dengan menyesuaikan ambang |
| Model kurang terlatih pada run asli | 20.68 poin | ya, cukup dengan menambah epoch |
| Protokol pembagian data itu sendiri | 6.92 poin | tidak |

Selisih yang dilaporkan semula 49.94 poin persentase. Dari jumlah itu, hanya 6.92 poin merupakan sifat protokolnya, yaitu bagian yang tetap ada setelah model dilatih penuh dan ambangnya disesuaikan.

Selisih protokol tersebut diuji dengan uji t Welch pada ambang prior-matched dan menghasilkan nilai p sebesar 0.0822, yang berarti belum terbukti berbeda.

## Bacaan

Efek protokol pembagian data tetap ada dan terbaca pada penurunan area under curve dari 1.0000 menjadi 0.9756, yang tidak dapat diperbaiki oleh pengaturan ambang. Besarannya jauh lebih kecil daripada yang semula dilaporkan.

Sebaliknya, temuan mengenai kegagalan kalibrasi menjadi jauh lebih kuat. Pada partisi resmi, model yang terlatih penuh memisahkan kedua kelas dengan area under curve 0.9756, namun pada ambang tetap 0,5 hanya mencapai 50.03 persen. Selisih itu sepenuhnya merupakan kegagalan kalibrasi, dan besarnya 42.52 poin persentase.

Dengan kata lain, angka yang semula dipakai untuk menunjukkan bahwa protokol pembagian data menentukan hasil sebenarnya lebih banyak menunjukkan bahwa ambang keputusan menentukan hasil. Kedua pernyataan sama-sama merupakan peringatan terhadap pelaporan akurasi tunggal, namun keduanya menunjuk sebab yang berbeda dan menuntut perbaikan yang berbeda pula.
