# Matriks Arsitektur terhadap Perlakuan Encoder

Seluruh angka diukur pada partisi resmi Fake-or-Real dengan ambang prior-matched. Tiga baris pertama memakai paket rekayasa yang sama persis, yaitu 10 epoch dengan early stopping pada equal error rate, augmentasi penuh, normalisasi loudness, dan agregasi berbobot antar lapisan. Hanya perlakuan encoder yang berbeda di antara ketiganya, sehingga perbandingannya bersifat satu variabel. Baris keempat disertakan sebagai acuan, yaitu konfigurasi proposal apa adanya.

## AST, 86 juta parameter, pra-latih terselia

| Perlakuan encoder | n | Akurasi | AUC | EER |
|---|---|---|---|---|
| Encoder dibekukan | 1 | 89.15 | 0.9586 | 10.85 |
| Encoder dilatih, laju wajar per model | 1 | **93.20** | 0.9817 | 6.80 |
| Encoder dilatih, laju 0,001 | 1 | 88.51 | 0.9513 | 11.49 |
| Proposal apa adanya, laju 0,001 seragam | 1 | 92.56 | 0.9780 | 7.44 |

## WavLM Large, 300 juta parameter, swa-selia

| Perlakuan encoder | n | Akurasi | AUC | EER |
|---|---|---|---|---|
| Encoder dibekukan | 3 | **98.62 (0.64)** | 0.9992 | 1.41 |
| Encoder dilatih, laju wajar per model | 1 | 96.14 | 0.9929 | 3.86 |
| Encoder dilatih, laju 0,001 | 1 | 80.06 | 0.8963 | 19.94 |
| Proposal apa adanya, laju 0,001 seragam | 1 | 56.99 | 0.6569 | 43.01 |

## HuBERT Large, 300 juta parameter, swa-selia

| Perlakuan encoder | n | Akurasi | AUC | EER |
|---|---|---|---|---|
| Encoder dibekukan | 1 | 93.93 | 0.9903 | 6.07 |
| Encoder dilatih, laju wajar per model | 1 | **98.16** | 0.9978 | 1.84 |
| Encoder dilatih, laju 0,001 | | belum ada | | |
| Proposal apa adanya, laju 0,001 seragam | 1 | 50.46 | 0.5325 | 50.37 |

## Bacaan

Pada AST, melatih encoder pada laju wajar lebih baik daripada membekukannya, dengan selisih +4.04 poin persentase.

Pada WavLM Large, melatih encoder pada laju wajar lebih buruk daripada membekukannya, dengan selisih -2.48 poin persentase.

Pada HuBERT Large, melatih encoder pada laju wajar lebih baik daripada membekukannya, dengan selisih +4.23 poin persentase.

Arah pengaruhnya tidak sama antar arsitektur. Tidak ada satu perlakuan encoder yang benar untuk semuanya. Inilah sebabnya baik penyeragaman learning rate pada proposal maupun penyeragaman pembekuan encoder pada konfigurasi rekayasa sama-sama menghasilkan kerugian pada arsitektur yang tidak cocok dengan pilihan tersebut. Keputusan ini seharusnya ditetapkan per arsitektur dan dipilih menggunakan data validasi, bukan diseragamkan di muka.
