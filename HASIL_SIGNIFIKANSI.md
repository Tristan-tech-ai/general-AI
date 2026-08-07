# Apakah Selisihnya Lebih Besar daripada Ragam Antar Inisialisasi?

Tiap baris membandingkan dua konfigurasi pada partisi resmi dengan ambang prior-matched, memakai uji t Welch yang tidak mengandaikan ragam kedua kelompok sama. Ukuran sampel kecil, yaitu paling banyak tiga inisialisasi per sel, sehingga uji ini berdaya rendah. Nilai p yang besar berarti belum terbukti berbeda, dan bukan terbukti sama.

| Perbandingan | n | Rerata A | Rerata B | Selisih | p | Bacaan |
|---|---|---|---|---|---|---|
| AST: encoder dilatih vs dibekukan | 3/1 | 93.38 | 89.15 | +4.23 | tidak dapat diuji | perlu minimal dua inisialisasi di kedua sisi |
| AST: encoder dilatih vs proposal | 3/1 | 93.38 | 92.56 | +0.83 | tidak dapat diuji | perlu minimal dua inisialisasi di kedua sisi |
| WavLM: encoder dibekukan vs dilatih | 3/3 | 98.62 | 97.30 | +1.32 | 0.313 | **belum terbukti berbeda** |
| HuBERT: encoder dilatih vs dibekukan | 1/1 | 98.16 | 93.93 | +4.23 | tidak dapat diuji | perlu minimal dua inisialisasi di kedua sisi |
| WavLM: rekayasa dibekukan vs proposal | 3/1 | 98.62 | 56.99 | +41.64 | tidak dapat diuji | perlu minimal dua inisialisasi di kedua sisi |
| HuBERT: rekayasa dilatih vs proposal | 1/1 | 98.16 | 50.46 | +47.70 | tidak dapat diuji | perlu minimal dua inisialisasi di kedua sisi |

## Bacaan

Perbandingan yang melibatkan konfigurasi proposal pada kedua model swa-selia berukuran besar terpisah sangat jauh, yaitu puluhan poin persentase, sehingga kesimpulannya tidak mungkin dibalik oleh ragam antar inisialisasi. Sebaliknya, perbandingan antara membekukan dan melatih encoder menghasilkan selisih yang berada pada orde yang sama dengan simpangan bakunya sendiri. Untuk kelompok kedua ini, penelitian ini tidak berhak menyatakan bahwa satu perlakuan lebih baik daripada yang lain.

Konsekuensinya bagi keseluruhan penelitian cukup besar dan perlu dinyatakan terus terang. Beberapa kesimpulan yang sempat ditarik lebih awal, ketika tiap sel baru dijalankan sekali, ternyata tidak bertahan setelah ragam antar inisialisasi diukur. Yang tersisa sebagai temuan yang kokoh adalah hal-hal yang selisihnya berpuluh poin, bukan berbilang poin.
