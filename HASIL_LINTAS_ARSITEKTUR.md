# Uji Lintas-Arsitektur: Apakah Band-Gain Bekerja di Luar Nes2Net?

Band-gain dirancang dari diagnosis tentang SINYAL (level vs struktur pita tinggi), bukan tentang arsitektur. Bila diagnosisnya benar, efeknya harus muncul juga di arsitektur lain.

| arsitektur | augmentasi | n | akurasi FoR | TTS 2025-26 | TTS-2019 non-MP3 |
|---|---|---|---|---|---|
| `hubert` | full | 1 | 94.39%  | **84.58%**  | 2.3%  |
| `nes2net` | full | 3 | 93.75% +/-5.15 | **94.97%** +/-5.34 | 82.1% +/-22.95 |
| `nes2net` | fullbg | 3 | 97.12% +/-2.77 | **93.50%** +/-4.77 | 92.1% +/-4.19 |
| `nes2net` | fullbgrb | 3 | 97.46% +/-2.10 | **93.67%** +/-3.47 | 85.0% +/-11.54 |
| `nes2net` | fullrb | 3 | 98.50% +/-0.41 | **87.64%** +/-6.41 | 60.7% +/-28.44 |
| `nes2net` | soft | 3 | 96.75% +/-0.77 | **80.94%** +/-15.78 | 66.1% +/-42.57 |
| `wavlm` | full | 3 | 98.62% +/-0.64 | **88.39%** +/-7.29 | 97.5% +/-3.11 |
| `wavlm` | fullbg | 3 | 98.65% +/-0.37 | **92.56%** +/-2.15 | 94.3% +/-9.37 |

## Efek band-gain per arsitektur (vs basis `full`)

| arsitektur | d-akurasi FoR | d-TTS modern | d-TTS-2019 non-MP3 |
|---|---|---|---|
| `nes2net` | **+3.37 pp** | **-1.47 pp** | **+10.0 pp** |
| `wavlm` | **+0.03 pp** | **+4.17 pp** | **-3.2 pp** |

## Kesimpulan

Arah efek band-gain **tidak konsisten** lintas arsitektur (nes2net +10.0 pp, wavlm -3.2 pp). Klaim mekanistik level-vs-struktur belum dapat digeneralisasi; hasil pada Nes2Net mungkin spesifik arsitektur.

## Kombinasi band-gain + RawBoost (Nes2Net-X)

| varian | akurasi FoR | TTS modern | TTS-2019 non-MP3 |
|---|---|---|---|
| band-gain saja | 97.12% | **93.50%** | 92.1% |
| RawBoost saja | 98.50% | **87.64%** | 60.7% |
| keduanya | 97.46% | **93.67%** | 85.0% |

Kombinasi mempertahankan keunggulan generalisasi band-gain: TTS modern 93.67% vs 93.50% (band-gain saja) dan 87.64% (RawBoost saja).
