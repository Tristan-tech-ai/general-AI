# Apakah Selisihnya Lebih Besar daripada Ragam Antar Inisialisasi?

Tiap baris membandingkan dua konfigurasi pada partisi resmi dengan ambang prior-matched, memakai uji t Welch yang tidak mengandaikan ragam kedua kelompok sama. Ukuran sampel kecil, yaitu paling banyak tiga inisialisasi per sel, sehingga uji ini berdaya rendah. Nilai p yang besar berarti belum terbukti berbeda, dan bukan terbukti sama.

Enam perbandingan diuji sekaligus, sehingga nilai p mentah tidak dapat dibaca apa adanya. Menguji enam hipotesis pada ambang 0,05 memberi peluang sekitar 26 persen untuk mendapatkan setidaknya satu hasil yang tampak bermakna semata karena kebetulan. Karena itu koreksi Holm-Bonferroni diterapkan, dan keputusan diambil dari nilai p terkoreksi.

| Perbandingan | n | Rerata A | Rerata B | Selisih | p mentah | p Holm | Bacaan |
|---|---|---|---|---|---|---|---|
| AST: encoder dilatih vs dibekukan | 3/3 | 93.38 | 89.15 | +4.23 | 0.1597 | 0.4792 | **belum terbukti berbeda** |
| AST: encoder dilatih vs proposal | 3/3 | 93.38 | 93.57 | -0.18 | 0.9064 | 0.9064 | **belum terbukti berbeda** |
| WavLM: encoder dibekukan vs dilatih | 3/3 | 98.62 | 97.30 | +1.32 | 0.3135 | 0.6269 | **belum terbukti berbeda** |
| HuBERT: encoder dilatih vs dibekukan | 3/3 | 97.49 | 94.67 | +2.82 | 0.0500 | 0.1998 | **belum terbukti berbeda** |
| WavLM: rekayasa dibekukan vs proposal | 3/3 | 98.62 | 63.73 | +34.90 | 0.0087 | 0.0520 | **di garis batas, belum meyakinkan** |
| HuBERT: rekayasa dilatih vs proposal | 3/3 | 97.49 | 52.14 | +45.34 | 0.0090 | 0.0520 | **di garis batas, belum meyakinkan** |

## Bacaan

Tabel ini memuat satu situasi yang mudah disalahbaca ke dua arah sekaligus, dan karena itu perlu dijelaskan dengan hati-hati.

Dua perbandingan dengan selisih terbesar, yaitu 34,90 dan 45,34 poin persentase, berhenti pada nilai p terkoreksi 0,0520. Angka itu tepat di atas ambang 0,05 yang lazim dipakai. Menyimpulkan dari situ bahwa selisih 45 poin persentase tidak nyata jelas keliru. Penyebab nilai p tersebut bukan efek yang kecil melainkan derajat bebas yang sangat sedikit. Dengan tiga inisialisasi per sel, uji t Welch hanya memiliki sekitar dua derajat bebas, sementara sel konfigurasi proposal juga memiliki simpangan baku yang besar. Nilai p mentahnya 0,0087 dan 0,0090, dan koreksi Holm atas enam perbandingan menaikkannya menjadi tepat di atas ambang.

Kesalahan ke arah sebaliknya juga perlu dihindari. Nilai p yang lolos ambang tidak akan membuat selisih itu lebih nyata daripada sekarang, dan besaran efek sudah lebih dari tujuh kali simpangan baku gabungannya. Yang sebenarnya dibutuhkan bukan penafsiran yang lebih longgar melainkan inisialisasi tambahan, dan itu dijadwalkan pada run_seeds4.ps1.

Sebaliknya, perbandingan antara membekukan dan melatih encoder menghasilkan selisih yang berada pada orde yang sama dengan simpangan bakunya sendiri. Untuk kelompok itu, penelitian ini tidak berhak menyatakan bahwa satu perlakuan lebih baik daripada yang lain, dan menambah inisialisasi belum tentu mengubahnya.

Konsekuensinya bagi keseluruhan penelitian cukup besar dan perlu dinyatakan terus terang. Beberapa kesimpulan yang sempat ditarik lebih awal, ketika tiap sel baru dijalankan sekali, ternyata tidak bertahan setelah ragam antar inisialisasi diukur. Yang tersisa sebagai temuan yang kokoh adalah hal-hal yang selisihnya berpuluh poin, bukan berbilang poin.
