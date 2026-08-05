# Eksperimen Penentu: Split Resmi vs Split Acak

Classifier identik (RandomForest 400 pohon), fitur identik (38 fitur spektral+statistik global).

**Tidak ada informasi fonetik, tidak ada fase, tidak ada model deep learning.**

Satu-satunya yang berubah: cara data dibagi.

| skema split | n latih | n uji | akurasi uji |
|---|---|---|---|
| **Resmi FoR** (training+val → testing) | 16782 | 1088 | **79.23%** |
| **Acak 60/20/20** (rencana proposal) | 10722 | 3574 | **95.91%** |
| Acak, hanya dalam training resmi (kontrol) | 11164 | 2792 | **96.92%** |

## Kesimpulan

Selisih split acak vs split resmi: **+16.69 poin persentase**
pada model yang sama persis dan fitur yang sama persis.

🔴 **HIPOTESIS TERKONFIRMASI.** Split acak menaikkan akurasi secara dramatis
tanpa peningkatan kemampuan model sedikit pun. Kenaikan itu sepenuhnya berasal
dari kebocoran domain: partisi resmi FoR sengaja menempatkan sumber rekaman
yang berbeda di test set, dan split acak menghancurkan pemisahan tersebut.

Artinya: angka 95.9% dari fitur spektral sepele saja sudah
mendekati angka yang dilaporkan penelitian deep learning sebelumnya pada FoR
(93,50% / 94,47% / 94,7%). Ini indikasi kuat bahwa sebagian besar hasil
terpublikasi pada FoR memakai split acak dan mengukur kebocoran domain,
bukan kemampuan deteksi deepfake.
