# Matriks Arsitektur terhadap Perlakuan Encoder

Seluruh angka diukur pada partisi resmi Fake-or-Real dengan ambang prior-matched. Tiga baris pertama memakai paket rekayasa yang sama persis, yaitu 10 epoch dengan early stopping pada equal error rate, augmentasi penuh, normalisasi loudness, dan agregasi berbobot antar lapisan. Hanya perlakuan encoder yang berbeda di antara ketiganya, sehingga perbandingannya bersifat satu variabel. Baris keempat disertakan sebagai acuan, yaitu konfigurasi proposal apa adanya.

## AST, 86 juta parameter, pra-latih terselia

| Perlakuan encoder | n | Akurasi | AUC | EER |
|---|---|---|---|---|
| Encoder dibekukan | 3 | 89.15 (3.49) | 0.9540 | 10.81 |
| Encoder dilatih, laju wajar per model | 3 | 93.38 (1.85) | 0.9811 | 6.68 |
| Encoder dilatih, laju 0,001 | 1 | 88.51 | 0.9513 | 11.49 |
| Proposal apa adanya, laju 0,001 seragam | 3 | **93.57 (1.75)** | 0.9824 | 6.50 |

## WavLM Large, 300 juta parameter, swa-selia

| Perlakuan encoder | n | Akurasi | AUC | EER |
|---|---|---|---|---|
| Encoder dibekukan | 5 | **98.36 (0.63)** | 0.9985 | 1.65 |
| Encoder dilatih, laju wajar per model | 3 | 97.30 (1.71) | 0.9959 | 2.70 |
| Encoder dilatih, laju 0,001 | 1 | 80.06 | 0.8963 | 19.94 |
| Proposal apa adanya, laju 0,001 seragam | 5 | 63.29 (4.19) | 0.7096 | 36.73 |

## HuBERT Large, 300 juta parameter, swa-selia

| Perlakuan encoder | n | Akurasi | AUC | EER |
|---|---|---|---|---|
| Encoder dibekukan | 3 | 94.67 (1.12) | 0.9907 | 5.33 |
| Encoder dilatih, laju wajar per model | 4 | **97.13 (1.30)** | 0.9961 | 2.92 |
| Encoder dilatih, laju 0,001 | 1 | 43.66 | 0.4837 | 55.97 |
| Proposal apa adanya, laju 0,001 seragam | 4 | 52.87 (6.76) | 0.5726 | 47.43 |

## Bacaan

Pada AST, melatih encoder pada laju wajar lebih baik daripada membekukannya, dengan selisih +4.23 poin persentase.

Pada WavLM Large, melatih encoder pada laju wajar lebih buruk daripada membekukannya, dengan selisih -1.06 poin persentase.

Pada HuBERT Large, melatih encoder pada laju wajar lebih baik daripada membekukannya, dengan selisih +2.46 poin persentase.

Arah pengaruhnya tidak sama antar arsitektur. Tidak ada satu perlakuan encoder yang benar untuk semuanya. Inilah sebabnya baik penyeragaman learning rate pada proposal maupun penyeragaman pembekuan encoder pada konfigurasi rekayasa sama-sama menghasilkan kerugian pada arsitektur yang tidak cocok dengan pilihan tersebut. Keputusan ini seharusnya ditetapkan per arsitektur dan dipilih menggunakan data validasi, bukan diseragamkan di muka.
