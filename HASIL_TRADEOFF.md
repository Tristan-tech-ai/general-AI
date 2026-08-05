# Apakah Band-Gain Memecah Trade-off FoR vs TTS Modern?

Arsitektur identik (Nes2Net-X), data identik, 3 seed per strategi. Yang berbeda hanya augmentasi.

| strategi augmentasi | n | akurasi FoR | recall TTS 2025-26 | recall TTS-2019 non-MP3 |
|---|---|---|---|---|
| aug penuh (basis) | 3 | 93.75% +/-5.15 | **94.97%** +/-5.34 | 82.1% +/-22.95 |
| aug penuh + RawBoost | 3 | 98.50% +/-0.41 | **87.64%** +/-6.41 | 60.7% +/-28.44 |
| **aug penuh + band-gain (usulan)** | 3 | 97.12% +/-2.77 | **93.50%** +/-4.77 | 92.1% +/-4.19 |
| preset 'soft' (cacat: 4 variabel berubah) | 3 | 96.75% +/-0.77 | **80.94%** +/-15.78 | 66.1% +/-42.57 |

## Ablasi variabel-tunggal terhadap basis `aug penuh`

| perbandingan | d-akurasi FoR | d-TTS modern | d-TTS-2019 non-MP3 |
|---|---|---|---|
| + RawBoost | **+4.75 pp** | **-7.33 pp** | **-21.4 pp** |
| + band-gain | **+3.37 pp** | **-1.47 pp** | **+10.0 pp** |

Simpangan baku TTS-2019 non-MP3: basis +/-23.0, RawBoost +/-28.4, **band-gain +/-4.2**

**Band-gain sebagian besar memecah trade-off.** Akurasi FoR naik +3.37 pp sementara deteksi TTS modern hanya bergeser -1.47 pp - di dalam derau antar-seed (+/-5.34). RawBoost, pada kenaikan FoR yang serupa, kehilangan 7.33 pp.

Sinyal terkuatnya ada di TTS-2019 non-MP3, proksi ketergantungan pintasan codec: band-gain +10.0 pp dengan simpangan baku runtuh dari +/-23.0 ke +/-4.2, sedangkan RawBoost -21.4 pp. Ini persis yang diprediksi diagnosis mekanistiknya - menetralkan LEVEL energi HF menghapus pintasan, sementara menghancurkan STRUKTUR HALUS HF (low-pass, RawBoost) ikut menghapus buktinya.

> Preset `soft` sebelumnya tampak gagal karena mengubah empat variabel sekaligus. Hasil di sini berasal dari ablasi variabel-tunggal (`fullbg`) dan menyimpulkan sebaliknya.
