# Audit Provenance Codec pada Fake-or-Real

Perhitungan ini membaca nama berkas dan label pada manifest secara langsung. Tidak ada model, pelatihan, maupun keacakan yang terlibat, sehingga hasilnya tidak memiliki ragam antar inisialisasi dan tidak memerlukan pengujian statistik. Inilah sebabnya temuan ini bertahan tanpa syarat sementara sebagian besar temuan lain dalam penelitian ini tidak.

| Partisi resmi | Kelas | Total berkas | Berasal MP3 | Persen |
|---|---|---|---|---|
| testing | asli | 544 | 0 | **0.0** |
| testing | palsu | 544 | 0 | **0.0** |
| training | asli | 6978 | 0 | **0.0** |
| training | palsu | 6978 | 6326 | **90.7** |
| validation | asli | 1413 | 0 | **0.0** |
| validation | palsu | 1413 | 1266 | **89.6** |

Pada data latih, 6326 dari 6978 sampel palsu berasal dari berkas MP3, yaitu 90.7 persen. Pada data uji, 0 dari 544 sampel palsu berasal dari MP3, yaitu 0.0 persen. Tidak ada satu pun sampel asli yang berasal dari MP3 pada partisi mana pun.

Akibatnya, sebuah model yang belajar mengenali jejak kompresi akan mencapai akurasi tinggi pada data latih dan validasi, lalu kehilangan seluruh isyarat itu pada data uji. Isyarat yang dipelajari bukan jejak sintesis melainkan riwayat berkas, dan riwayat itu berkorelasi dengan label hanya pada sebagian partisi.
