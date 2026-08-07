# Tabel Ringkas Hasil (angka terukur, bukan perkiraan)

Seluruh angka berasal dari berkas hasil di repositori dan dapat direproduksi. Format: rerata (simpangan baku) dalam persen, n = jumlah inisialisasi acak.

## 1. Efek protokol pembagian data

Model, data, dan hyperparameter identik. Hanya cara pembagian data berbeda.

| Konfigurasi | Split | n | Akurasi | EER |
|---|---|---|---|---|
| CNN+ASP tanpa augmentasi | acak 60/20/20 | 1 | **99.75** | 0.03 |
| CNN+ASP tanpa augmentasi | resmi FoR | 1 | **71.88** | 28.12 |
| Replikasi proposal: ast | resmi FoR | 3 | **74.85** (1.47) | 6.50 |
| Replikasi proposal: wavlm | resmi FoR | 5 | **53.40** (3.94) | 36.73 |
| Replikasi proposal: hubert | resmi FoR | 5 | **51.47** (1.89) | 48.60 |
| Replikasi proposal: cnnlstm_proposal | acak 60/20/20 | 1 | **99.83** | 0.03 |
| Replikasi proposal: wav2vec2 | acak 60/20/20 | 1 | **99.78** | 0.08 |
| Replikasi proposal: ast | acak 60/20/20 | 1 | **99.69** | 0.14 |
| Replikasi proposal: hubert | acak 60/20/20 | 1 | **99.69** | 0.39 |
| Replikasi proposal: wavlm | acak 60/20/20 | 1 | **99.52** | 0.31 |

Metrik replikasi proposal dilaporkan pada ambang 0,5 seperti yang tersirat di proposal. Angka dalam kurung adalah simpangan baku antar inisialisasi acak.

## 2. Kinerja pada partisi resmi FoR

| Arsitektur | Augmentasi | n | Akurasi | EER |
|---|---|---|---|---|
| wavlm | fullbgrb | 3 | **98.90** (0.18) | 1.10 |
| wavlm | fullbg | 3 | **98.65** (0.37) | 1.29 |
| nes2net | fullrb | 3 | **98.50** (0.41) | 1.44 |
| wavlm | full | 5 | **98.36** (0.63) | 1.65 |
| nes2net | fullbgrb | 3 | **97.46** (2.10) | 2.54 |
| wavlm | fullUF | 3 | **97.30** (1.71) | 2.70 |
| hubert | codec | 8 | **97.29** (2.16) | 2.71 |
| nes2net | fullbg | 3 | **97.12** (2.77) | 2.88 |
| nes2net | soft | 3 | **96.75** (0.77) | 3.25 |
| wavlm | codec | 3 | **96.51** (2.41) | 3.55 |

## 3. Deteksi TTS komersial 2025-2026

Diukur pada spesifisitas 95 persen yang disamakan untuk semua model, memakai 1.500 berkas asli In-the-Wild sebagai acuan ambang.

| Arsitektur | Augmentasi | n | Recall TTS 2025-2026 |
|---|---|---|---|
| nes2net | full | 3 | **94.97** (5.34) |
| nes2net | fullbgrb | 3 | **93.67** (3.47) |
| nes2net | fullbg | 3 | **93.50** (4.77) |
| wavlm | fullbgrb | 3 | **92.69** (4.68) |
| wavlm | fullbg | 3 | **92.56** (2.15) |
| hubert | fullbg | 3 | **91.33** (4.04) |
| wavlm | full | 3 | **88.39** (7.29) |
| nes2net | fullrb | 3 | **87.64** (6.41) |
| hubert | full | 3 | **85.67** (6.11) |
| nes2net | soft | 3 | **80.94** (15.78) |

## 4. Ketahanan terhadap noise

Noise DEMAND, korpus yang tidak dipakai saat pelatihan.

| Arsitektur dan augmentasi | bersih | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| wavlm[full] | 98.6 | 98.0 | 98.0 | 97.5 | 95.5 |
| hubert[full] | 95.4 | 93.1 | 91.2 | 89.5 | 84.1 |
| cnn_asp[full] | 95.1 | 88.2 | 86.8 | 84.6 | 82.7 |
| wavlm[codec] | 96.4 | 92.6 | 89.4 | 86.6 | 76.2 |
| cnn_asp[codec] | 91.9 | 74.7 | 72.2 | 69.0 | 65.3 |
| ast[codec] | 86.4 | 76.5 | 68.2 | 65.7 | 61.9 |
| hubert[codec] | 97.1 | 92.6 | 87.0 | 75.2 | 61.0 |
| wav2vec2[codec] | 90.7 | 84.7 | 75.3 | 67.2 | 60.4 |
| cnnlstm[codec] | 83.5 | 60.6 | 58.2 | 56.2 | 54.1 |

## 5. Angka kunci untuk dikutip

| Klaim | Angka | Sumber |
|---|---|---|
| Sampel palsu di data latih FoR yang berasal MP3 | 90,7 persen | audit_report.md |
| Sampel palsu di data uji FoR yang berasal MP3 | 0 persen | probe_codec_report.md |
| Selisih akurasi akibat protokol split saja | sekitar 50 poin | probe_split_report.md |
| Korelasi akurasi FoR dengan recall TTS modern | r = -0,542 | HASIL_NOVELTY_PROBE.md |
| Spesifisitas model SOTA publik di luar domain | 0,00 persen | HASIL_SOTA_COLLAPSE.md |
| Korelasi ceiling band-gain (3 arsitektur) | r = -0,980 | HASIL_LINTAS_ARSITEKTUR.md |
