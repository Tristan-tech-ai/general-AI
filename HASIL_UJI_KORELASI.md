# Apakah Korelasi yang Dilaporkan Bertahan pada Ukuran Sampelnya?

Dua korelasi dilaporkan dalam penelitian ini tanpa nilai p maupun selang kepercayaan. Pada ukuran sampel kecil, koefisien korelasi memiliki sebaran yang sangat lebar sehingga nilai besar sekalipun dapat muncul dari data yang sebenarnya tidak berhubungan. Nilai p di bawah dihitung dengan uji permutasi lengkap, yang eksak untuk ukuran sampel sekecil ini dan tidak mengandaikan kenormalan. Selang kepercayaan dihitung dengan bootstrap persentil.

## Akurasi Fake-or-Real terhadap recall TTS generasi terbaru

Titik data: 10 konfigurasi, yaitu pasangan arsitektur dan strategi augmentasi, masing-masing dirata-ratakan atas inisialisasi acak yang tersedia.

| Konfigurasi | Akurasi FoR | Recall TTS 2025-2026 |
|---|---|---|
| wavlm + fullbgrb | 98.90 | 92.69 |
| wavlm + fullbg | 98.65 | 92.56 |
| wavlm + full | 98.62 | 88.39 |
| nes2net + fullrb | 98.50 | 87.64 |
| nes2net + fullbgrb | 97.46 | 93.67 |
| nes2net + fullbg | 97.12 | 93.50 |
| nes2net + soft | 96.75 | 80.94 |
| hubert + fullbg | 95.71 | 91.33 |
| hubert + full | 95.34 | 85.67 |
| nes2net + full | 93.75 | 94.97 |

Koefisien korelasi Pearson r = -0.048 dengan n = 10. Nilai p dua sisi dari uji permutasi acak adalah 0.8951.
Selang kepercayaan bootstrap 95 persen membentang dari -0.664 sampai 0.629.

Selang kepercayaannya memuat nol. Arah hubungan karena itu belum dapat ditetapkan dari data ini saja, sekalipun koefisien titiknya bertanda negatif.

Perlu ditegaskan bahwa yang menopang temuan mengenai hubungan terbalik ini bukan koefisien korelasinya, melainkan mekanismenya yang terdokumentasi secara terpisah, yaitu audit kebocoran codec pada dataset, eksperimen augmentasi terkontrol, dan pola kebutaan model terhadap sistem yang tidak dikompresi. Koefisien korelasi pada ukuran sampel sekecil ini sebaiknya dibaca sebagai ringkasan deskriptif, bukan sebagai bukti.
