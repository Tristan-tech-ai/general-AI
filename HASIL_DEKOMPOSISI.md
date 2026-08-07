# Dekomposisi Nilai Rekayasa: Pelatihan atau Ambang?

Perbandingan antara konfigurasi proposal dan konfigurasi diperbaiki pada matriks 2x2 mengubah dua variabel sekaligus, yaitu cara model dilatih dan cara ambang keputusan ditetapkan, karena tiap konfigurasi dievaluasi pada ambangnya masing-masing. Bagian ini memisahkan keduanya dari skor yang sama persis, tanpa melatih ulang apa pun.

## Empat sel dari skor yang sama

| Arsitektur | n | A. proposal @ 0,5 | B. proposal @ prior | C. rekayasa @ 0,5 | D. rekayasa @ prior |
|---|---|---|---|---|---|
| ast | 3/3 | 74.85 | **93.57** | **58.82** | **89.15** |
| wavlm | 4/4 | 54.18 | **63.40** | **82.24** | **98.55** |
| hubert | 3/6 | 52.27 | **52.14** | **76.95** | **95.01** |

## Sumbangan tiap sumbu (poin persentase)

| Arsitektur | Ambang saja (B-A) | Pelatihan saja (C-A) | Interaksi | Total (D-A) |
|---|---|---|---|---|
| ast | **+18.72** | **-16.02** | +11.61 | +14.31 |
| wavlm | **+9.21** | **+28.06** | +7.10 | +44.37 |
| hubert | **-0.12** | **+24.68** | +18.18 | +42.74 |

## Daya pisah, yang sama sekali tidak bergantung pada ambang

| Arsitektur | AUC proposal | AUC rekayasa | EER proposal | EER rekayasa | Penurunan EER relatif |
|---|---|---|---|---|---|
| ast | 0.9824 | **0.9540** | 6.50 | **10.81** | **-66.5 persen** |
| wavlm | 0.7171 | **0.9989** | 36.63 | **1.47** | **+96.0 persen** |
| hubert | 0.5605 | **0.9916** | 48.22 | **4.99** | **+89.6 persen** |

## Bacaan

Sebagian besar selisih yang tampak besar pada matriks 2x2 berasal dari penetapan ambang, bukan dari cara model dilatih. Pada WavLM, konfigurasi proposal yang hanya diganti ambangnya sudah mencapai angka yang praktis sama dengan konfigurasi yang direkayasa penuh. Ini konsisten dengan temuan pada bagian noise, bahwa kegagalan model pada kondisi yang bergeser sebagian besar merupakan kegagalan kalibrasi dan bukan kegagalan pengenalan. Mekanisme yang sama ternyata juga berlaku pada sumbu pergeseran protokol.

Meski begitu, sumbangan pelatihan tidak nol dan tidak boleh diabaikan. Pada ambang 0,5 yang sama persis, konfigurasi yang direkayasa tetap unggul, yang berarti pelatihan memperbaiki kalibrasi skornya sendiri sehingga ambang bawaan menjadi lebih tepat. Daya pisahnya juga naik, dan kenaikan itu terlihat pada AUC dan EER yang sama sekali tidak bergantung pada pilihan ambang.

Kesimpulan yang jujur adalah bahwa rekayasa ini bernilai, namun nilainya sebagian besar terletak pada kalibrasi keputusan dan hanya sebagian kecil pada peningkatan daya pisah. Melaporkan +43 poin sebagai hasil perbaikan arsitektur akan menyesatkan.
