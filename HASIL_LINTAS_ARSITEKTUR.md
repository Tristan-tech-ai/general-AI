# Uji Lintas-Arsitektur: Apakah Band-Gain Bekerja di Luar Nes2Net?

Band-gain dirancang dari diagnosis tentang SINYAL (level vs struktur pita tinggi), bukan tentang arsitektur. Bila diagnosisnya benar, efeknya harus muncul juga di arsitektur lain.

| arsitektur | augmentasi | n | akurasi FoR | TTS 2025-26 | TTS-2019 non-MP3 |
|---|---|---|---|---|---|
| `hubert` | full@b16e10 | 3 | 95.34% +/-0.88 | **85.67%** +/-6.11 | 29.2% +/-28.96 |
| `hubert` | fullbg@b16e10 | 3 | 95.71% +/-1.29 | **91.33%** +/-4.04 | 58.4% +/-5.23 |
| `nes2net` | full@b16e10 | 3 | 93.75% +/-5.15 | **94.97%** +/-5.34 | 82.1% +/-22.95 |
| `nes2net` | fullbg@b16e10 | 3 | 97.12% +/-2.77 | **93.50%** +/-4.77 | 92.1% +/-4.19 |
| `nes2net` | fullbgrb@b16e10 | 3 | 97.46% +/-2.10 | **93.67%** +/-3.47 | 85.0% +/-11.54 |
| `nes2net` | fullrb@b16e10 | 3 | 98.50% +/-0.41 | **87.64%** +/-6.41 | 60.7% +/-28.44 |
| `nes2net` | soft@b16e10 | 3 | 96.75% +/-0.77 | **80.94%** +/-15.78 | 66.1% +/-42.57 |
| `wavlm` | full@b16e10 | 3 | 98.62% +/-0.64 | **88.39%** +/-7.29 | 97.5% +/-3.11 |
| `wavlm` | fullbg@b16e10 | 3 | 98.65% +/-0.37 | **92.56%** +/-2.15 | 94.3% +/-9.37 |
| `wavlm` | fullbgrb@b16e10 | 3 | 98.90% +/-0.18 | **92.69%** +/-4.68 | 99.3% +/-0.80 |

## Efek band-gain per arsitektur (vs basis `full`)

| arsitektur | d-akurasi FoR | d-TTS modern | d-TTS-2019 non-MP3 |
|---|---|---|---|

## Cakupan

Sebanyak 53 run pada partisi resmi tidak masuk analisis ini karena nama tag-nya di luar pola yang ditangani, yaitu konfigurasi yang ditambahkan setelah skrip ini ditulis seperti encoder yang dilatih dan replikasi proposal. Analisis ini memang menyangkut augmentasi band-gain dan tidak memerlukannya, namun jumlahnya dicatat di sini supaya pelewatan tersebut tidak berlangsung tanpa diketahui.

Tag yang dilewati: `ast_official_fullUFENC0.001_b32e10_s42`, `ast_official_fullUF_b32e10_s1337`, `ast_official_fullUF_b32e10_s2024`, `ast_official_fullUF_b32e10_s42`, `ast_official_proposalULRPK_b32e20_s1337`, `ast_official_proposalULRPK_b32e20_s2024`, `ast_official_proposalULRPK_b32e20_s42`, `ast_official_proposalULR_b32e20_s1337`, dan 45 lainnya.
