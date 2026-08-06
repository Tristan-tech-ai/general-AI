# Dekomposisi Nilai Rekayasa: Pelatihan atau Ambang?

Matriks 2x2 melaporkan selisih +37,59 poin persentase pada AST dan +43,01 poin pada WavLM. Selisih itu membandingkan konfigurasi proposal pada ambang 0,5 dengan konfigurasi diperbaiki pada ambang prior-matched, sehingga dua variabel berubah bersamaan. Bagian ini memisahkan keduanya dari skor yang sama persis, tanpa melatih ulang.

## Empat sel dari skor yang sama

| Arsitektur | n | A. proposal @ 0,5 | B. proposal @ prior | C. rekayasa @ 0,5 | D. rekayasa @ prior |
|---|---|---|---|---|---|
| ast | 1/1 | 73.16 | **92.56** | **65.72** | **89.15** |
| wavlm | 1/3 | 53.31 | **56.99** | **83.15** | **98.62** |

## Sumbangan tiap sumbu (poin persentase)

| Arsitektur | Ambang saja (B-A) | Pelatihan saja (C-A) | Interaksi | Total (D-A) |
|---|---|---|---|---|
| ast | **+19.39** | **-7.44** | +4.04 | +15.99 |
| wavlm | **+3.68** | **+29.84** | +11.80 | +45.31 |

## Daya pisah, yang sama sekali tidak bergantung pada ambang

| Arsitektur | AUC proposal | AUC rekayasa | EER proposal | EER rekayasa | Penurunan EER relatif |
|---|---|---|---|---|---|
| ast | 0.9780 | **0.9586** | 7.44 | **10.85** | **-45.7 persen** |
| wavlm | 0.6569 | **0.9992** | 43.01 | **1.41** | **+96.7 persen** |

## Bacaan

Sebagian besar selisih yang tampak besar pada matriks 2x2 berasal dari penetapan ambang, bukan dari cara model dilatih. Pada WavLM, konfigurasi proposal yang hanya diganti ambangnya sudah mencapai angka yang praktis sama dengan konfigurasi yang direkayasa penuh. Ini konsisten dengan temuan pada bagian noise, bahwa kegagalan model pada kondisi yang bergeser sebagian besar merupakan kegagalan kalibrasi dan bukan kegagalan pengenalan. Mekanisme yang sama ternyata juga berlaku pada sumbu pergeseran protokol.

Meski begitu, sumbangan pelatihan tidak nol dan tidak boleh diabaikan. Pada ambang 0,5 yang sama persis, konfigurasi yang direkayasa tetap unggul, yang berarti pelatihan memperbaiki kalibrasi skornya sendiri sehingga ambang bawaan menjadi lebih tepat. Daya pisahnya juga naik, dan kenaikan itu terlihat pada AUC dan EER yang sama sekali tidak bergantung pada pilihan ambang.

Kesimpulan yang jujur adalah bahwa rekayasa ini bernilai, namun nilainya sebagian besar terletak pada kalibrasi keputusan dan hanya sebagian kecil pada peningkatan daya pisah. Melaporkan +43 poin sebagai hasil perbaikan arsitektur akan menyesatkan.
