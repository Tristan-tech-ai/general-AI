# Grid SNR, Ketahanan terhadap Noise Lingkungan yang Belum Pernah Dilihat

Model dilatih dengan augmentasi **colored noise sintetis**; diuji dengan **DEMAND** (6 lingkungan nyata: dapur, taman, kantor, kafetaria, lalu lintas, bus). Kedua korpus sepenuhnya terpisah.

Test set 1.088 berkas, 9 arsitektur × 3 seed. Nilai = rerata ± simpangan baku atas seed.

## 1. Akurasi vs SNR (ambang prior-matched)

| arsitektur | bersih | 30 dB | 25 dB | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB | -5 dB |
|---|---|---|---|---|---|---|---|---|---|
| `ast[codec]` | 86.4 | 83.2 | 80.2 | 76.5 | 72.4 | 68.2 | 65.7 | 61.9 | 57.1 |
| `cnn_asp[codec]` | 91.9 | 78.5 | 76.1 | 74.7 | 73.9 | 72.2 | 69.0 | 65.3 | 59.8 |
| `cnn_asp[full]` | 95.1 | 91.5 | 90.2 | 88.2 | 87.9 | 86.8 | 84.6 | 82.7 | 78.5 |
| `cnnlstm[codec]` | 83.5 | 66.5 | 62.9 | 60.6 | 59.7 | 58.2 | 56.2 | 54.1 | 52.0 |
| `hubert[codec]` | 97.1 | 93.5 | 92.7 | 92.6 | 91.6 | 87.0 | 75.2 | 61.0 | 54.0 |
| `hubert[full]` | 95.4 | 94.5 | 94.0 | 93.1 | 92.2 | 91.2 | 89.5 | 84.1 | 77.5 |
| `wav2vec2[codec]` | 90.7 | 87.9 | 86.3 | 84.7 | 82.5 | 75.3 | 67.2 | 60.4 | 57.8 |
| `wavlm[codec]` | 96.4 | 94.9 | 93.7 | 92.6 | 90.5 | 89.4 | 86.6 | 76.2 | 66.2 |
| `wavlm[full]` | 98.6 | 98.5 | 98.1 | 98.0 | 98.0 | 98.0 | 97.5 | 95.5 | 89.2 |

## 2. EER vs SNR

| arsitektur | bersih | 30 dB | 25 dB | 20 dB | 15 dB | 10 dB | 5 dB | 0 dB | -5 dB |
|---|---|---|---|---|---|---|---|---|---|
| `ast[codec]` | 13.4 | 16.9 | 19.7 | 23.3 | 27.6 | 31.8 | 34.2 | 38.2 | 42.6 |
| `cnn_asp[codec]` | 8.1 | 21.5 | 23.9 | 25.2 | 26.1 | 27.8 | 30.9 | 34.9 | 40.1 |
| `cnn_asp[full]` | 4.9 | 8.5 | 10.0 | 11.8 | 12.1 | 13.1 | 15.4 | 17.4 | 21.5 |
| `cnnlstm[codec]` | 16.5 | 33.5 | 37.3 | 39.2 | 40.4 | 41.7 | 43.8 | 45.9 | 48.0 |
| `hubert[codec]` | 2.9 | 6.5 | 7.3 | 7.7 | 8.4 | 12.9 | 24.8 | 39.0 | 46.0 |
| `hubert[full]` | 4.6 | 5.5 | 6.0 | 6.8 | 7.7 | 8.8 | 10.5 | 15.9 | 22.6 |
| `wav2vec2[codec]` | 9.3 | 12.1 | 13.7 | 15.4 | 17.4 | 24.7 | 32.8 | 39.6 | 42.1 |
| `wavlm[codec]` | 3.6 | 5.1 | 6.3 | 7.4 | 9.3 | 10.4 | 13.4 | 23.3 | 33.7 |
| `wavlm[full]` | 1.3 | 1.6 | 1.8 | 2.0 | 2.0 | 2.0 | 2.5 | 4.4 | 10.7 |

## 3. Degradasi dari kondisi bersih (poin persentase)

| arsitektur | bersih | @10 dB | @0 dB | @−5 dB | **turun bersih→0 dB** |
|---|---|---|---|---|---|
| `ast[codec]` | 86.4% | 68.2% | 61.9% | 57.1% | **−24.6 pp** |
| `cnn_asp[codec]` | 91.9% | 72.2% | 65.3% | 59.8% | **−26.7 pp** |
| `cnn_asp[full]` | 95.1% | 86.8% | 82.7% | 78.5% | **−12.4 pp** |
| `cnnlstm[codec]` | 83.5% | 58.2% | 54.1% | 52.0% | **−29.4 pp** |
| `hubert[codec]` | 97.1% | 87.0% | 61.0% | 54.0% | **−36.1 pp** |
| `hubert[full]` | 95.4% | 91.2% | 84.1% | 77.5% | **−11.4 pp** |
| `wav2vec2[codec]` | 90.7% | 75.3% | 60.4% | 57.8% | **−30.4 pp** |
| `wavlm[codec]` | 96.4% | 89.4% | 76.2% | 66.2% | **−20.2 pp** |
| `wavlm[full]` | 98.6% | 98.0% | 95.5% | 89.2% | **−3.1 pp** |

**Paling tahan (degradasi terkecil bersih→0 dB):** `wavlm[full]` (−3.1 pp) · **paling rapuh:** `hubert[codec]` (−36.1 pp)

## 4. Dekomposisi: diskriminasi vs kalibrasi

`acc_fix` memakai ambang yang dibekukan dari kondisi bersih; `acc_pm` memakai ambang prior-matched per kondisi. Selisihnya adalah akurasi yang hilang **semata karena ambang meleset**, bukan karena model kehilangan daya pisah.

| arsitektur | SNR | acc (ambang beku) | acc (prior-matched) | **hilang krn ambang** | AUC |
|---|---|---|---|---|---|
| `ast[codec]` | 20 dB | 70.5% | 76.5% | **+6.0 pp** | 0.845 |
| `ast[codec]` | 10 dB | 62.3% | 68.2% | **+5.8 pp** | 0.749 |
| `ast[codec]` | 0 dB | 55.2% | 61.9% | **+6.7 pp** | 0.673 |
| `cnn_asp[codec]` | 20 dB | 70.4% | 74.7% | **+4.3 pp** | 0.824 |
| `cnn_asp[codec]` | 10 dB | 69.5% | 72.2% | **+2.6 pp** | 0.796 |
| `cnn_asp[codec]` | 0 dB | 57.1% | 65.3% | **+8.1 pp** | 0.703 |
| `cnn_asp[full]` | 20 dB | 88.2% | 88.2% | **+0.0 pp** | 0.952 |
| `cnn_asp[full]` | 10 dB | 81.6% | 86.8% | **+5.1 pp** | 0.932 |
| `cnn_asp[full]` | 0 dB | 75.6% | 82.7% | **+7.2 pp** | 0.897 |
| `cnnlstm[codec]` | 20 dB | 60.7% | 60.6% | **-0.1 pp** | 0.646 |
| `cnnlstm[codec]` | 10 dB | 58.7% | 58.2% | **-0.5 pp** | 0.605 |
| `cnnlstm[codec]` | 0 dB | 54.0% | 54.1% | **+0.2 pp** | 0.545 |
| `hubert[codec]` | 20 dB | 88.6% | 92.6% | **+4.0 pp** | 0.977 |
| `hubert[codec]` | 10 dB | 83.8% | 87.0% | **+3.2 pp** | 0.947 |
| `hubert[codec]` | 0 dB | 66.7% | 61.0% | **-5.7 pp** | 0.648 |
| `hubert[full]` | 20 dB | 92.5% | 93.1% | **+0.7 pp** | 0.983 |
| `hubert[full]` | 10 dB | 89.6% | 91.2% | **+1.7 pp** | 0.975 |
| `hubert[full]` | 0 dB | 78.4% | 84.1% | **+5.7 pp** | 0.922 |
| `wav2vec2[codec]` | 20 dB | 78.2% | 84.7% | **+6.4 pp** | 0.929 |
| `wav2vec2[codec]` | 10 dB | 63.7% | 75.3% | **+11.7 pp** | 0.843 |
| `wav2vec2[codec]` | 0 dB | 55.9% | 60.4% | **+4.5 pp** | 0.648 |
| `wavlm[codec]` | 20 dB | 75.1% | 92.6% | **+17.6 pp** | 0.981 |
| `wavlm[codec]` | 10 dB | 59.8% | 89.4% | **+29.6 pp** | 0.962 |
| `wavlm[codec]` | 0 dB | 50.0% | 76.2% | **+26.2 pp** | 0.840 |
| `wavlm[full]` | 20 dB | 98.0% | 98.0% | **-0.0 pp** | 0.998 |
| `wavlm[full]` | 10 dB | 97.8% | 98.0% | **+0.2 pp** | 0.997 |
| `wavlm[full]` | 0 dB | 95.4% | 95.5% | **+0.1 pp** | 0.988 |

Rerata akurasi yang dapat dipulihkan hanya dengan mengoreksi ambang: **+5.6 pp** (maks +29.6 pp).

## 5. Titik runtuh (SNR saat akurasi turun di bawah 80%)

| arsitektur | SNR runtuh |
|---|---|
| `ast[codec]` | 20 dB |
| `cnn_asp[codec]` | 30 dB |
| `cnn_asp[full]` | -5 dB |
| `cnnlstm[codec]` | 30 dB |
| `hubert[codec]` | 5 dB |
| `hubert[full]` | -5 dB |
| `wav2vec2[codec]` | 10 dB |
| `wavlm[codec]` | 0 dB |
| `wavlm[full]` | tidak runtuh |
