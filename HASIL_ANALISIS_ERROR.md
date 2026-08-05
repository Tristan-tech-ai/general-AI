# Analisis Error Tersisa — Ensemble HuBERT Terbaik

Ensemble 3 run HuBERT Large (augmentasi per-epoch), split resmi.
Akurasi **99.63%**, EER 0.37%, **4 berkas salah dari 1088**.

Pertanyaan: apakah sisa error ini *bisa* diperbaiki, atau plafon dataset?

| # | berkas | label benar | prediksi | skor | durasi bicara | RMS | energi>6kHz | zero-frac |
|---|---|---|---|---|---|---|---|---|
| 1 | `file1097.wav_16k.wav_norm.wav_mono.wav_silen` | **real** | fake | 0.0726 | 2.000s | 0.1574 | 0.00281 | 0.004 |
| 2 | `file1409.wav_16k.wav_norm.wav_mono.wav_silen` | **real** | fake | 0.0655 | 2.000s | 0.1601 | 0.01867 | 0.006 |
| 3 | `file1369.wav_16k.wav_norm.wav_mono.wav_silen` | **fake** | real | 0.0502 | 2.000s | 0.1744 | 0.00025 | 0.002 |
| 4 | `file932.wav_16k.wav_norm.wav_mono.wav_silenc` | **fake** | real | 0.0305 | 2.000s | 0.1927 | 0.00053 | 0.002 |

## Pembanding: berkas yang diklasifikasi BENAR (sampel n=217)

| properti | error (rerata) | benar (rerata) | benar (p5–p95) | anomali? |
|---|---|---|---|---|
| durasi bicara (s) | 2.0000 | 1.9975 | 1.9818 – 2.0000 | tidak |
| RMS | 0.1712 | 0.1926 | 0.1205 – 0.2611 | tidak |
| energi >6 kHz | 0.0056 | 0.0049 | 0.0001 – 0.0160 | tidak |
| fraksi hening | 0.0031 | 0.0059 | 0.0007 – 0.0201 | tidak |

## Keyakinan model pada error

- Ambang keputusan: 0.0550
- Skor error: 0.0726, 0.0655, 0.0502, 0.0305
- Error yang **dekat ambang** (|skor−ambang| < 0,15): **4/4**

Seluruh error berada dekat ambang keputusan. Artinya model **ragu**, bukan
**yakin-salah**. Ini pola khas sampel yang secara intrinsik ambigu, bukan
pola kegagalan sistematis yang dapat diperbaiki arsitektur.
