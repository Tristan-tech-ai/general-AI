# Matriks 2x2: Apakah Rekayasa Melewati Baseline?

Setiap matriks memakai satu arsitektur, batch dan seed yang sama. Hanya konfigurasi pelatihan dan skema pembagian data yang berubah, sehingga tiap perbandingan bersifat satu variabel.

Konfigurasi proposal: learning rate 0,001 seragam dengan encoder ikut dilatih, 20 epoch tanpa early stopping, normalisasi peak, augmentasi noise SNR 15 sampai 30 dB, ambang keputusan 0,5.

Konfigurasi diperbaiki: learning rate per model dengan encoder dibekukan dan agregasi berbobot antar lapisan, 10 epoch dengan early stopping pada EER, normalisasi loudness, augmentasi penuh, ambang prior-matched.

## ast

| Konfigurasi | Split acak 60/20/20 | Partisi resmi FoR | Selisih antar split |
|---|---|---|---|
| Proposal apa adanya | 99.69% | **74.85% (1.47)** | +24.85 pp |
| Diperbaiki (rekayasa) | 99.75% | **89.15% (3.49)** | +10.59 pp |

**Nilai rekayasa pada protokol resmi: +14.31 poin persentase** (74.85% menjadi 89.15%).
Pada split acak selisihnya hanya +0.06 poin (99.69% menjadi 99.75%), yang menunjukkan bahwa protokol longgar tidak dapat membedakan kedua konfigurasi.

## wavlm

| Konfigurasi | Split acak 60/20/20 | Partisi resmi FoR | Selisih antar split |
|---|---|---|---|
| Proposal apa adanya | 99.52% | **53.40% (3.94)** | +46.12 pp |
| Diperbaiki (rekayasa) | 99.75% | **98.36% (0.63)** | +1.38 pp |

**Nilai rekayasa pada protokol resmi: +44.96 poin persentase** (53.40% menjadi 98.36%).
Pada split acak selisihnya hanya +0.22 poin (99.52% menjadi 99.75%), yang menunjukkan bahwa protokol longgar tidak dapat membedakan kedua konfigurasi.

## hubert

| Konfigurasi | Split acak 60/20/20 | Partisi resmi FoR | Selisih antar split |
|---|---|---|---|
| Proposal apa adanya | 99.69% | **51.47% (1.89)** | +48.22 pp |
| Diperbaiki (rekayasa) | belum ada | **95.01% (0.97)** | n/a |

**Nilai rekayasa pada protokol resmi: +43.54 poin persentase** (51.47% menjadi 95.01%).

## nes2net

| Konfigurasi | Split acak 60/20/20 | Partisi resmi FoR | Selisih antar split |
|---|---|---|---|
| Proposal apa adanya | belum ada | **belum ada** | n/a |
| Diperbaiki (rekayasa) | belum ada | **93.75% (5.15)** | n/a |

## Kesimpulan

| Arsitektur | Proposal pada partisi resmi | Diperbaiki pada partisi resmi | Selisih |
|---|---|---|---|
| ast | 74.85% | **89.15%** | **+14.31 pp** |
| wavlm | 53.40% | **98.36%** | **+44.96 pp** |
| hubert | 51.47% | **95.01%** | **+43.54 pp** |

Rerata selisih +34.27 poin persentase pada protokol yang sama persis. Rekayasa memberi perbaikan nyata, dan perbaikan itu tidak terlihat sama sekali bila hanya melihat kolom split acak.
