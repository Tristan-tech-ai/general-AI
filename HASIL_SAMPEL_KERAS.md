# Uji Penentu: Apakah Sisa Error Bersifat Struktural?

Sampel uji: 1088. Ensemble HuBERT salah pada **4** berkas.

Pertanyaan: adakah arsitektur lain yang benar pada berkas-berkas itu?

## Prediksi tiap arsitektur pada sampel keras

| idx | berkas | label | `ast` | `cnn_asp` | `cnnlstm` | `hubert` | `wav2vec2` | `wavlm` | ada yg benar? |
|---|---|---|---|---|---|---|---|---|---|
| 19 | `file1097.wav_16k.wav_norm.` | **real** | ❌0.170 | ❌0.388 | ❌0.523 | ❌0.073 | ❌0.050 | ✅0.008 | ✅ ADA |
| 117 | `file1409.wav_16k.wav_norm.` | **real** | ❌0.142 | ✅0.062 | ✅0.042 | ❌0.065 | ✅0.025 | ✅0.014 | ✅ ADA |
| 644 | `file1369.wav_16k.wav_norm.` | **fake** | ✅0.272 | ✅0.292 | ✅0.137 | ❌0.050 | ✅0.072 | ❌0.013 | ✅ ADA |
| 1073 | `file932.wav_16k.wav_norm.w` | **fake** | ✅0.295 | ✅0.459 | ✅0.180 | ❌0.031 | ✅0.031 | ✅0.026 | ✅ ADA |

**Sampel yang salah di SELURUH 6 arsitektur: 0/4**

## Cek lebih ketat: seluruh 21 run individual

| idx | label | run yang BENAR | dari total |
|---|---|---|---|
| 19 | real | hubert/s1337, wavlm/s2024, wavlm/s42 | 3/21 |
| 117 | real | cnn_asp/s1337, cnn_asp/s2024, cnn_asp/s42, cnn_asp/s1337, cnn_asp/s2024, cnn_asp/s42… | 12/21 |
| 644 | fake | ast/s1337, ast/s2024, ast/s42, cnn_asp/s1337, cnn_asp/s42, cnn_asp/s1337… | 13/21 |
| 1073 | fake | ast/s1337, ast/s2024, ast/s42, cnn_asp/s1337, cnn_asp/s2024, cnn_asp/s42… | 17/21 |

**Sampel yang salah di SELURUH 21 run: 0**

## Kesimpulan

**4 dari 4** sampel BISA diklasifikasi benar oleh setidaknya satu run. Artinya masih ada ruang: fusi/gating yang lebih cerdas berpotensi menaikkan hasil hingga **100.00%**.
