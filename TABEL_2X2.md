# Matriks 2x2: Apakah Rekayasa Melewati Baseline?

Setiap matriks memakai satu arsitektur, batch dan seed yang sama. Hanya konfigurasi pelatihan dan skema pembagian data yang berubah, sehingga tiap perbandingan bersifat satu variabel.

Konfigurasi proposal: learning rate 0,001 seragam dengan encoder ikut dilatih, 20 epoch tanpa early stopping, normalisasi peak, augmentasi noise SNR 15 sampai 30 dB, ambang keputusan 0,5.

Konfigurasi diperbaiki: learning rate per model dengan encoder dibekukan dan agregasi berbobot antar lapisan, 10 epoch dengan early stopping pada EER, normalisasi loudness, augmentasi penuh, ambang prior-matched.

## ast

| Konfigurasi | Split acak 60/20/20 | Partisi resmi FoR | Selisih antar split |
|---|---|---|---|
| Proposal apa adanya | 99.86% | **51.56%** | +48.30 pp |
| Diperbaiki (rekayasa) | 99.75% | **89.15%** | +10.59 pp |

**Nilai rekayasa pada protokol resmi: +37.59 poin persentase** (51.56% menjadi 89.15%).
Pada split acak selisihnya hanya -0.11 poin (99.86% menjadi 99.75%), yang menunjukkan bahwa protokol longgar tidak dapat membedakan kedua konfigurasi.

## wavlm

| Konfigurasi | Split acak 60/20/20 | Partisi resmi FoR | Selisih antar split |
|---|---|---|---|
| Proposal apa adanya | 100.00% | **55.61%** | +44.39 pp |
| Diperbaiki (rekayasa) | belum ada | **98.62% (0.64)** | n/a |

**Nilai rekayasa pada protokol resmi: +43.01 poin persentase** (55.61% menjadi 98.62%).

## hubert

| Konfigurasi | Split acak 60/20/20 | Partisi resmi FoR | Selisih antar split |
|---|---|---|---|
| Proposal apa adanya | 99.94% | **belum ada** | n/a |
| Diperbaiki (rekayasa) | belum ada | **95.34% (0.88)** | n/a |

## nes2net

| Konfigurasi | Split acak 60/20/20 | Partisi resmi FoR | Selisih antar split |
|---|---|---|---|
| Proposal apa adanya | belum ada | **belum ada** | n/a |
| Diperbaiki (rekayasa) | belum ada | **93.75% (5.15)** | n/a |

## Kesimpulan

| Arsitektur | Proposal pada partisi resmi | Diperbaiki pada partisi resmi | Selisih |
|---|---|---|---|
| ast | 51.56% | **89.15%** | **+37.59 pp** |
| wavlm | 55.61% | **98.62%** | **+43.01 pp** |

Rerata selisih +40.30 poin persentase pada protokol yang sama persis. Rekayasa memberi perbaikan nyata, dan perbaikan itu tidak terlihat sama sekali bila hanya melihat kolom split acak.
