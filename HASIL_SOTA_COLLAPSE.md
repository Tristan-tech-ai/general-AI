# Verifikasi: Runtuhnya Kalibrasi Model SOTA di Luar Domain

Model: Nes2Net-X + XLS-R, checkpoint resmi, EER **1,49%** pada ASVspoof 2021 DF.
Diuji ZERO-SHOT pada FoR-2sec (1.088 berkas seimbang), tanpa adaptasi apa pun.

## Confusion matrix pada ambang 0,5

| | prediksi: asli | prediksi: palsu |
|---|---|---|
| **asli** (n=544) | 0 | **544** |
| **palsu** (n=544) | 122 | 422 |

- Akurasi: 38.79%
- Recall (spoof): 77.57%
- Spesifisitas (asli dikenali benar): **0.00%**
- Proporsi SELURUH berkas ditandai palsu: **88.79%**

## Distribusi skor

| kelas | n | min | p25 | median | p75 | maks |
|---|---|---|---|---|---|---|
| asli | 544 | 0.9493 | 0.9961 | 0.9989 | 0.9996 | 1.0000 |
| palsu | 544 | 0.0112 | 0.5648 | 0.8434 | 0.9574 | 0.9996 |

AUC = **0.0233**, EER = **91.08%**

AUC di bawah 0,5 berarti pengurutan skornya **TERBALIK**: model memberi skor 'palsu' LEBIH TINGGI kepada audio ASLI daripada audio palsu. Bila polaritas dibalik, AUC menjadi **0.9767** — jadi model tetap sangat diskriminatif, hanya arahnya terbalik pada korpus ini.

## Tafsir

Implikasi untuk klaim kebaruan: literatur anti-spoofing melaporkan EER *dalam domain*, yang tidak menangkap keruntuhan kalibrasi lintas korpus seperti ini. Mengukurnya eksplisit — recall DAN spesifisitas pada korpus asing, bukan hanya EER in-domain — adalah celah evaluasi yang nyata.