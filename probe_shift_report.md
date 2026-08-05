# Pergeseran Distribusi Train → Test pada FoR-2sec

## 1. Energi pita tinggi (>6 kHz) per split × kelas × provenance

| split | kelas | provenance | n | energi >6kHz (mean) |
|---|---|---|---|---|
| training | real | dari WAV | 6978 | 0.03355 |
| training | fake | dari WAV | 652 | 0.02872 |
| training | fake | dari MP3 | 6326 | 0.00802 |
| validation | real | dari WAV | 1413 | 0.03097 |
| validation | fake | dari WAV | 147 | 0.03813 |
| validation | fake | dari MP3 | 1266 | 0.00832 |
| testing | real | dari WAV | 544 | 0.00575 |
| testing | fake | dari WAV | 544 | 0.00431 |

## 2. Inti masalah

- Training: real = **0.03355**, fake(MP3) = **0.00802** → rasio **4.18×**
- Training: real = 0.03355, fake(WAV) = **0.02872** → rasio **1.17×**
- Testing : real = **0.00575**, fake(WAV) = **0.00431** → rasio **1.33×**

Selama pelatihan, 90,7% sampel fake adalah turunan MP3 dengan energi pita
tinggi yang tertekan. Model belajar aturan **"HF rendah ⇒ fake"**.
Pada pengujian, TIDAK ADA fake yang berasal dari MP3 — aturan itu menjadi
tidak berlaku, bahkan menyesatkan.

## 3. Uji intervensi: apakah normalisasi pita menyembuhkan?

| representasi | akurasi train | akurasi test | celah generalisasi |
|---|---|---|---|
| Profil spektral penuh 0–8 kHz | 100.00% | **69.39%** | 30.61 pp |
| Hanya 0–4 kHz (pintasan HF dibuang) | 100.00% | **75.37%** | 24.63 pp |
| Ternormalisasi per-berkas (tilt dibuang) | 100.00% | **64.15%** | 35.85 pp |
| 0–4 kHz + ternormalisasi | 100.00% | **71.42%** | 28.58 pp |

Celah generalisasi yang besar = model menghafal pintasan yang tidak ada di test set.
