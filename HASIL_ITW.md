# Generalisasi Lintas-Korpus: FoR-2sec → In-the-Wild

Model dilatih **hanya** pada FoR-2sec, diuji tanpa adaptasi apa pun pada In-the-Wild (31779 berkas, 11816 spoof = 37.2%, 54 pembicara).

Skor per berkas = rata-rata probabilitas atas hingga 5 jendela 2 detik. Ambang memakai prior asli (0.372), bukan 0,5.

| arsitektur | augmentasi | n seed | akurasi | EER | AUC |
|---|---|---|---|---|---|
| `wavlm` | full | 3 | **81.11%** ±2.02 | 20.55% | 0.8710 |
| `ast` | codec | 3 | **83.16%** ±2.47 | 19.49% | 0.8684 |

Konteks literatur: ref [14] dalam proposal melaporkan EER In-the-Wild **31,14%** untuk sistem baseline yang kuat di domainnya sendiri (EER 4,06% pada ASVspoof 2019).