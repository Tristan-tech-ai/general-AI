# Audit Integritas Dataset FoR-2sec

Total berkas dianalisis: **17870** (gagal dibaca: 0)

## 1. Komposisi split

| split | real | fake | total |
|---|---|---|---|
| training | 6978 | 6978 | 13956 |
| validation | 1413 | 1413 | 2826 |
| testing | 544 | 544 | 1088 |

## 2. Keseragaman format

- Sampling rate: `{16000: 17870}`
- Kanal: `{1: 17870}`
- Durasi: min=2.000s  median=2.000s  maks=2.000s  std=0.0000s
  - real: mean=2.0000s std=0.0000s
  - fake: mean=2.0000s std=0.0000s

## 3. T3 — Duplikat eksak & kebocoran lintas split

- Grup duplikat byte-identik: **0** (total 0 berkas)
- Duplikat yang MELINTASI split (kebocoran train/test): **0**
- Duplikat dengan label berbeda (konflik label): **0**

  ✅ Tidak ada duplikat byte-identik yang melintasi split.

## 4. T1/T2/T4 — Daya diskriminatif fitur non-semantik

AUC 0,5 = tidak informatif; AUC → 1,0 atau → 0,0 = sangat memisahkan kelas.
**AUC tinggi pada fitur trivial = pintasan (shortcut) dataset.**

| fitur | AUC | mean(real) | mean(fake) | tafsir |
|---|---|---|---|---|
| `exact_zero` | 0.5882 | 0.0021744 | 0.014594 | ✅ netral |
| `zero_frac` | 0.5882 | 0.0021744 | 0.014594 | ✅ netral |
| `lead_sil` | 0.5001 | 2.0985e-07 | 8.1142e-07 | ✅ netral |
| `trail_sil` | 0.4823 | 0.0093576 | 0.0083013 | ✅ netral |
| `speech_dur` | 0.5176 | 1.9906 | 1.9917 | ✅ netral |
| `peak` | 0.6024 | 0.94891 | 0.98894 | 🟡 lemah |
| `rms` | 0.6980 | 0.1538 | 0.18173 | 🟡 lemah |
| `dc` | 0.6354 | -0.0019903 | 0.006546 | 🟡 lemah |
| `clip_frac` | 0.6169 | 0.00091736 | 4.9014e-05 | 🟡 lemah |
| `zcr` | 0.4590 | 0.14781 | 0.13954 | ✅ netral |
| `e_0_1k` | 0.6053 | 0.84833 | 0.89592 | 🟡 lemah |
| `e_1_4k` | 0.4640 | 0.098358 | 0.085251 | ✅ netral |
| `e_4_6k` | 0.3586 | 0.021867 | 0.0089853 | 🟡 lemah |
| `e_6_8k` | 0.3613 | 0.03145 | 0.0098441 | 🟡 lemah |
| `centroid` | 0.3419 | 853.82 | 584.24 | 🟡 lemah |
| `rolloff95` | 0.3701 | 2906.6 | 1869.1 | 🟡 lemah |
| `hf_slope` | 0.3235 | -0.0047516 | -0.010472 | 🟡 lemah |
| `dur` | 0.5000 | 2 | 2 | ✅ netral |

### Klasifikasi HANYA dari fitur trivial (tanpa melihat isi wicara)

Dilatih pada split `training` resmi, diuji pada split `testing` resmi.

| model | akurasi test | tafsir |
|---|---|---|
| LogReg (18 fitur trivial) | **53.12%** | ✅ wajar |
| RandomForest (18 fitur trivial) | **77.48%** | 🟠 pintasan signifikan |

Fitur trivial tunggal paling diskriminatif: **`rms`** (AUC 0.6980)

## 5. T5 — Pola nama berkas (jejak sumber / pembicara)

**real** — 8935 berkas. Contoh: `file1000.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav`, `file10003.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav`, `file10006.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav`
  prefiks unik: 8800; 8 terbanyak: [('file1039.wav', 2), ('file1075.wav', 2), ('file1097.wav', 2), ('file117.wav', 2), ('file1177.wav', 2), ('file1180.wav', 2), ('file1196.wav', 2), ('file121.wav', 2)]
**fake** — 8935 berkas. Contoh: `file10005.mp3.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav`, `file10007.mp3.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav`, `file10009.mp3.wav_16k.wav_norm.wav_mono.wav_silence.wav_2sec.wav`
  prefiks unik: 8921; 8 terbanyak: [('file1010.wav', 2), ('file1097.wav', 2), ('file1339.wav', 2), ('file1354.wav', 2), ('file1408.wav', 2), ('file155.wav', 2), ('file1703.wav', 2), ('file2162.wav', 2)]

## 6. Resolusi statistik test set

Ukuran test set resmi: **1088** berkas.

| akurasi | jumlah error | CI 95% (±) |
|---|---|---|
| 95.0% | 54 berkas | ±1.30 pp |
| 97.0% | 33 berkas | ±1.01 pp |
| 98.0% | 22 berkas | ±0.83 pp |
| 99.0% | 11 berkas | ±0.59 pp |
| 99.5% | 5 berkas | ±0.42 pp |
| 100.0% | 0 berkas | ±0.00 pp |

**Implikasi:** pada 1088 berkas uji, selisih 1 berkas = 0.092 pp. Dua model yang berbeda < 1.7 pp tidak dapat dibedakan secara statistik tanpa uji berpasangan (McNemar).
