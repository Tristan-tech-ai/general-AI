# Apakah Selisihnya Lebih Besar daripada Ragam Antar Inisialisasi?

Tiap baris membandingkan dua konfigurasi pada partisi resmi dengan ambang prior-matched, memakai uji t Welch yang tidak mengandaikan ragam kedua kelompok sama. Ukuran sampel kecil, yaitu paling banyak tiga inisialisasi per sel, sehingga uji ini berdaya rendah. Nilai p yang besar berarti belum terbukti berbeda, dan bukan terbukti sama.

Enam perbandingan diuji sekaligus, sehingga nilai p mentah tidak dapat dibaca apa adanya. Menguji enam hipotesis pada ambang 0,05 memberi peluang sekitar 26 persen untuk mendapatkan setidaknya satu hasil yang tampak bermakna semata karena kebetulan. Karena itu koreksi Holm-Bonferroni diterapkan, dan keputusan diambil dari nilai p terkoreksi.

| Perbandingan | n | Rerata A | Rerata B | Selisih | p mentah | p Holm | Bacaan |
|---|---|---|---|---|---|---|---|
| AST: encoder dilatih vs dibekukan | 3/3 | 93.38 | 89.15 | +4.23 | 0.1597 | 0.6389 | **belum terbukti berbeda** |
| AST: encoder dilatih vs proposal | 3/3 | 93.38 | 93.57 | -0.18 | 0.9064 | 0.9064 | **belum terbukti berbeda** |
| WavLM: encoder dibekukan vs dilatih | 5/3 | 98.36 | 97.30 | +1.06 | 0.3964 | 0.7978 | **belum terbukti berbeda** |
| HuBERT: encoder dilatih vs dibekukan | 5/3 | 96.19 | 94.67 | +1.53 | 0.2659 | 0.7978 | **belum terbukti berbeda** |
| WavLM: rekayasa dibekukan vs proposal | 5/5 | 98.36 | 63.29 | +35.07 | 0.0000 | 0.0002 | **selisih melampaui ragam** |
| HuBERT: rekayasa dilatih vs proposal | 5/5 | 96.19 | 51.64 | +44.56 | 0.0000 | 0.0002 | **selisih melampaui ragam** |

## Bacaan

Hasilnya terbelah bersih menjadi dua kelompok yang tidak saling berdekatan.

Kelompok pertama adalah perbandingan antara konfigurasi proposal dan konfigurasi rekayasa pada kedua model swa-selia berukuran besar. Selisihnya berpuluh poin persentase dan nilai p terkoreksinya jauh di bawah ambang, sehingga kesimpulannya kokoh. Perlu dicatat bahwa pada tahap sebelumnya, ketika tiap sel baru memiliki tiga inisialisasi, kedua perbandingan ini justru berhenti pada nilai p terkoreksi 0,0520 yaitu tepat di atas ambang. Penyebabnya bukan efek yang kecil melainkan derajat bebas yang sangat sedikit dan simpangan baku yang besar pada sel konfigurasi proposal. Menambah inisialisasi keempat dan kelima menyelesaikannya, dan itu memang tanggapan yang tepat terhadap nilai p yang berhenti di ambang.

Kelompok kedua adalah perbandingan antara membekukan dan melatih encoder. Selisihnya berada pada orde yang sama dengan simpangan bakunya sendiri, dan tidak satu pun terbukti berbeda meskipun sebagian sel sudah memiliki empat atau lima inisialisasi. Untuk kelompok ini, penelitian ini tidak berhak menyatakan bahwa satu perlakuan lebih baik daripada yang lain.

Kesimpulan keseluruhannya sempit tetapi jelas. Yang menentukan hasil pada partisi resmi adalah besaran learning rate relatif terhadap encodernya, bukan keputusan membekukan atau melatih encoder itu sendiri.

Konsekuensinya bagi keseluruhan penelitian cukup besar dan perlu dinyatakan terus terang. Beberapa kesimpulan yang sempat ditarik lebih awal, ketika tiap sel baru dijalankan sekali, ternyata tidak bertahan setelah ragam antar inisialisasi diukur. Yang tersisa sebagai temuan yang kokoh adalah hal-hal yang selisihnya berpuluh poin, bukan berbilang poin.
