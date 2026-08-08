# Probe: Apakah FoR-2sec Memisahkan Codec, Bukan Deepfake?

## P1, Provenance nama berkas

| kelas | total | mengandung `.mp3` | persen |
|---|---|---|---|
| real | 8935 | 0 | **0.00%** |
| fake | 8935 | 7592 | **84.97%** |

AUC dari fitur tunggal `'.mp3' ada di nama berkas`: **0.9248**

## P2/P3, Cutoff spektral (batas atas pita frekuensi)

| kelas | cutoff −60 dB (Hz) | cutoff −40 dB (Hz) |
|---|---|---|
| real | 7555 ± 676 | 6682 ± 1633 |
| fake | 7107 ± 853 | 6274 ± 1510 |

AUC `cutoff60` sebagai fitur tunggal: **0.2845**

AUC `cutoff40` sebagai fitur tunggal: **0.3564**

## P4, Klasifikasi HANYA dari profil 32-bin energi spektral

Tidak ada informasi fonetik, tidak ada fase, tidak ada temporal, hanya *bentuk* spektrum rata-rata.

| model | akurasi test |
|---|---|
| LogReg (32 bin spektral) | **55.51%** |
| RandomForest (32 bin spektral) | **69.39%** |

## P5, Setelah pintasan dinetralkan (buang seluruh pita > 4 kHz)

| model | akurasi test (hanya 0–4 kHz) | Δ vs pita penuh |
|---|---|---|
| LogReg | **65.62%** | +10.11 pp |
| RandomForest | **75.37%** | +5.97 pp |

## Profil energi rata-rata per pita (fraksi energi total)

| pita (Hz) | real | fake | rasio real/fake |
|---|---|---|---|
| 0–250 | 2.629e-01 | 3.373e-01 | 0.78× |
| 500–750 | 2.121e-01 | 1.523e-01 | 1.39× |
| 1000–1250 | 2.458e-02 | 2.878e-02 | 0.85× |
| 1500–1750 | 1.346e-02 | 1.152e-02 | 1.17× |
| 2000–2250 | 5.290e-03 | 4.370e-03 | 1.21× |
| 2500–2750 | 5.783e-03 | 3.553e-03 | 1.63× |
| 3000–3250 | 3.663e-03 | 2.371e-03 | 1.55× |
| 3500–3750 | 3.772e-03 | 1.653e-03 | 2.28× |
| 4000–4250 | 2.811e-03 | 1.256e-03 | 2.24× |
| 4500–4750 | 2.649e-03 | 1.093e-03 | 2.42× |
| 5000–5250 | 2.707e-03 | 9.846e-04 | 2.75× |
| 5500–5750 | 2.876e-03 | 1.318e-03 | 2.18× |
| 6000–6250 | 2.891e-03 | 1.125e-03 | 2.57× |
| 6250–6500 | 3.269e-03 | 1.342e-03 | 2.44× |
| 6500–6750 | 4.166e-03 | 1.317e-03 | 3.16× |
| 6750–7000 | 5.027e-03 | 1.645e-03 | 3.06× |
| 7000–7250 | 6.180e-03 | 1.633e-03 | 3.79× |
| 7250–7500 | 6.725e-03 | 1.764e-03 | 3.81× |
| 7500–7750 | 3.166e-03 | 1.011e-03 | 3.13× |
| 7750–8000 | 2.602e-05 | 8.547e-06 | 3.04× |
