# Uji Lintas-Arsitektur: Apakah Band-Gain Bekerja di Luar Nes2Net?

Band-gain dirancang dari diagnosis tentang SINYAL (level vs struktur pita tinggi), bukan tentang arsitektur. Bila diagnosisnya benar, efeknya harus muncul juga di arsitektur lain.

| arsitektur | augmentasi | n | akurasi FoR | TTS 2025-26 | TTS-2019 non-MP3 |
|---|---|---|---|---|---|
| `hubert` | full | 3 | 94.67% +/-1.12 | **85.67%** +/-6.11 | 29.2% +/-28.96 |
| `hubert` | fullbg | 3 | 95.71% +/-1.29 | **91.33%** +/-4.04 | 58.4% +/-5.23 |
| `nes2net` | full | 3 | 93.75% +/-5.15 | **94.97%** +/-5.34 | 82.1% +/-22.95 |
| `nes2net` | fullbg | 3 | 97.12% +/-2.77 | **93.50%** +/-4.77 | 92.1% +/-4.19 |
| `nes2net` | fullbgrb | 3 | 97.46% +/-2.10 | **93.67%** +/-3.47 | 85.0% +/-11.54 |
| `nes2net` | fullrb | 3 | 98.50% +/-0.41 | **87.64%** +/-6.41 | 60.7% +/-28.44 |
| `nes2net` | soft | 3 | 96.75% +/-0.77 | **80.94%** +/-15.78 | 66.1% +/-42.57 |
| `wavlm` | full | 3 | 98.62% +/-0.64 | **88.39%** +/-7.29 | 97.5% +/-3.11 |
| `wavlm` | fullbg | 3 | 98.65% +/-0.37 | **92.56%** +/-2.15 | 94.3% +/-9.37 |
| `wavlm` | fullbgrb | 3 | 98.90% +/-0.18 | **92.69%** +/-4.68 | 99.3% +/-0.80 |

## Efek band-gain per arsitektur (vs basis `full`)

| arsitektur | d-akurasi FoR | d-TTS modern | d-TTS-2019 non-MP3 |
|---|---|---|---|
| `hubert` | **+1.04 pp** | **+5.67 pp** | **+29.2 pp** |
| `nes2net` | **+3.37 pp** | **-1.47 pp** | **+10.0 pp** |
| `wavlm` | **+0.03 pp** | **+4.17 pp** | **-3.2 pp** |

## Uji hipotesis ceiling

Hipotesis yang diajukan sebelum menjalankan HuBERT: band-gain memperbaiki sumbu yang masih punya ruang, dan tidak menolong bila sumbu itu sudah mendekati batas atas. Karena band-gain bekerja dengan menghapus ketergantungan pintasan codec, besarnya perbaikan seharusnya berbanding terbalik dengan titik awal pada proksi pintasan.

| arsitektur | recall awal TTS-2019 non-MP3 | perubahan | std sebelum | std sesudah |
|---|---|---|---|---|
| `hubert` | 29.2% | **+29.2 pp** | +/-29.0 | **+/-5.2** |
| `nes2net` | 82.1% | **+10.0 pp** | +/-23.0 | **+/-4.2** |
| `wavlm` | 97.5% | **-3.2 pp** | +/-3.1 | **+/-9.4** |

Korelasi antara titik awal dan besar perbaikan: **r = -0.980**

Polanya monoton dan sangat kuat: makin rendah titik awal, makin besar perbaikannya. Ini konsisten dengan hipotesis ceiling dan mendukung bahwa band-gain memang bekerja lewat penghapusan ketergantungan pintasan, bukan lewat efek yang khas satu arsitektur.

> Catatan kehati-hatian: hipotesis ini disusun setelah melihat hasil Nes2Net dan WavLM, lalu diuji pada HuBERT dengan prediksi yang dituliskan lebih dulu. Konfirmasinya karena itu bermakna, tetapi hanya berbasis tiga arsitektur.

## Kombinasi band-gain + RawBoost (Nes2Net-X)

| varian | akurasi FoR | TTS modern | TTS-2019 non-MP3 |
|---|---|---|---|
| band-gain saja | 97.12% | **93.50%** | 92.1% |
| RawBoost saja | 98.50% | **87.64%** | 60.7% |
| keduanya | 97.46% | **93.67%** | 85.0% |

Kombinasi mempertahankan keunggulan generalisasi band-gain: TTS modern 93.67% vs 93.50% (band-gain saja) dan 87.64% (RawBoost saja).

## Cakupan

Sebanyak 30 run pada partisi resmi tidak masuk analisis ini karena nama tag-nya di luar pola yang ditangani, yaitu konfigurasi yang ditambahkan setelah skrip ini ditulis seperti encoder yang dilatih dan replikasi proposal. Analisis ini memang menyangkut augmentasi band-gain dan tidak memerlukannya, namun jumlahnya dicatat di sini supaya pelewatan tersebut tidak berlangsung tanpa diketahui.

Tag yang dilewati: `ast_official_fullUFENC0.001_b32e10_s42`, `ast_official_fullUF_b32e10_s1337`, `ast_official_fullUF_b32e10_s2024`, `ast_official_fullUF_b32e10_s42`, `ast_official_proposalULRPK_b32e20_s1337`, `ast_official_proposalULRPK_b32e20_s2024`, `ast_official_proposalULRPK_b32e20_s42`, `ast_official_proposalULR_b32e20_s1337`, dan 22 lainnya.
