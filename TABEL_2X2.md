# Matriks 2x2: Apakah Rekayasa Melewati Baseline?

Setiap matriks memakai satu arsitektur, batch dan seed yang sama. Hanya konfigurasi pelatihan dan skema pembagian data yang berubah, sehingga tiap perbandingan bersifat satu variabel.

Konfigurasi proposal: learning rate 0,001 seragam dengan encoder ikut dilatih, 20 epoch tanpa early stopping, normalisasi peak, augmentasi noise SNR 15 sampai 30 dB, ambang keputusan 0,5.

Konfigurasi diperbaiki: learning rate per model dengan encoder dibekukan dan agregasi berbobot antar lapisan, 10 epoch dengan early stopping pada EER, normalisasi loudness, augmentasi penuh, ambang prior-matched.

## ast

| Konfigurasi | Split acak 60/20/20 | Partisi resmi FoR | Selisih antar split |
|---|---|---|---|
| Proposal apa adanya | 99.69% | **73.16%** | +26.53 pp |
| Diperbaiki (rekayasa) | 99.75% | **89.15%** | +10.59 pp |

**Nilai rekayasa pada protokol resmi: +15.99 poin persentase** (73.16% menjadi 89.15%).
Pada split acak selisihnya hanya +0.06 poin (99.69% menjadi 99.75%), yang menunjukkan bahwa protokol longgar tidak dapat membedakan kedua konfigurasi.

## wavlm

| Konfigurasi | Split acak 60/20/20 | Partisi resmi FoR | Selisih antar split |
|---|---|---|---|
| Proposal apa adanya | 99.52% | **53.31%** | +46.22 pp |
| Diperbaiki (rekayasa) | 99.75% | **98.62% (0.64)** | +1.13 pp |

**Nilai rekayasa pada protokol resmi: +45.31 poin persentase** (53.31% menjadi 98.62%).
Pada split acak selisihnya hanya +0.22 poin (99.52% menjadi 99.75%), yang menunjukkan bahwa protokol longgar tidak dapat membedakan kedua konfigurasi.

## hubert

| Konfigurasi | Split acak 60/20/20 | Partisi resmi FoR | Selisih antar split |
|---|---|---|---|
| Proposal apa adanya | 99.69% | **belum ada** | n/a |
| Diperbaiki (rekayasa) | belum ada | **95.34% (0.88)** | n/a |

## nes2net

| Konfigurasi | Split acak 60/20/20 | Partisi resmi FoR | Selisih antar split |
|---|---|---|---|
| Proposal apa adanya | belum ada | **belum ada** | n/a |
| Diperbaiki (rekayasa) | belum ada | **93.75% (5.15)** | n/a |

## Kesimpulan

| Arsitektur | Proposal pada partisi resmi | Diperbaiki pada partisi resmi | Selisih |
|---|---|---|---|
| ast | 73.16% | **89.15%** | **+15.99 pp** |
| wavlm | 53.31% | **98.62%** | **+45.31 pp** |

Rerata selisih +30.65 poin persentase pada protokol yang sama persis. Rekayasa memberi perbaikan nyata, dan perbaikan itu tidak terlihat sama sekali bila hanya melihat kolom split acak.
