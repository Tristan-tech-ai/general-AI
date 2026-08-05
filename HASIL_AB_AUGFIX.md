# A/B: Perbaikan Bug Augmentasi Statis (P0-1) pada HuBERT Large

Bug: RNG augmentasi di-seed hanya dari (nama berkas, seed run) sehingga tiap
berkas menerima satu varian augmentasi yang sama di seluruh epoch.
Perbaikan: epoch ikut masuk ke seed + `persistent_workers=False` pada loader latih.

Semua run: HuBERT Large, split resmi, augmentasi codec, batch 16, 10 epoch,
ambang prior-matched, seed {42, 1337, 2024}.

| kondisi | n | akurasi | std | EER | AUC | terbaik | salah (terbaik) |
|---|---|---|---|---|---|---|---|
| Augmentasi **beku** (bug) | 3 | **97.37%** | ±0.65 | 2.63% | 0.9971 | **97.98%** | 22/1088 |
| Augmentasi **per-epoch** (fix) | 3 | **96.97%** | ±2.80 | 3.03% | 0.9955 | **99.45%** | 6/1088 |

## Selisih

- Akurasi rerata: **-0.40 pp**
- Akurasi terbaik: **+1.47 pp**
- EER rerata: **+0.40 pp**
- Simpangan baku: 0.65 → 2.80 pp

- Welch t-test rerata: t=-0.240, **p=0.831** → **tidak signifikan** pada n=3

**Pembacaan jujur:** perbaikan ini **tidak** menaikkan rerata secara
signifikan pada n=3, tetapi menaikkan **plafon** dan **variansi** sekaligus.
Itu konsisten dengan mekanismenya: keragaman augmentasi yang lebih besar
memperluas ruang eksplorasi, sehingga run terbaik menjadi lebih baik dan
run terburuk menjadi lebih buruk. Untuk pelaporan tesis, konsekuensinya
adalah **butuh lebih banyak seed**, bukan lebih sedikit.

## Ensemble antar-seed

| kondisi | akurasi | EER | AUC | salah |
|---|---|---|---|---|
| Augmentasi beku | **98.35%** | 1.65% | 0.9992 | 18/1088 |
| Augmentasi per-epoch | **99.63%** | 0.37% | 0.9998 | 4/1088 |
| **Gabungan seluruh 6 run HuBERT** | **99.45%** | 0.46% | 0.9999 | 6/1088 |
