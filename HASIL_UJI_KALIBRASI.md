# Apakah Kegagalan di Bawah Noise Merupakan Kegagalan Kalibrasi?

Selisih diukur antara akurasi pada ambang yang dibekukan dari kondisi bersih dan akurasi pada ambang prior-matched, pada model dan berkas yang sama persis. Karena kedua nilai berasal dari inisialisasi acak yang sama, pengujiannya memakai uji t berpasangan. Selisih yang besar berarti daya pisah model masih ada dan yang bergeser hanyalah letak ambangnya.

| Arsitektur | SNR | n | AUC | Pemulihan ambang | p mentah | p Holm |
|---|---|---|---|---|---|---|
| wavlm[codec] | 10 dB | 3 | 0.9615 | +29.6 (7.9) | 0.0232 | 0.3250 |
| wavlm[codec] | 0 dB | 3 | 0.8400 | +26.2 (4.2) | 0.0084 | 0.1342 |
| wav2vec2[codec] | 10 dB | 3 | 0.8434 | +11.7 (2.3) | 0.0130 | 0.1954 |
| cnn_asp[codec] | 0 dB | 3 | 0.7030 | +8.1 (9.0) | 0.2562 | 1.0000 |
| ast[codec] | 0 dB | 3 | 0.6730 | +6.7 (4.4) | 0.1213 | 1.0000 |
| ast[codec] | 10 dB | 3 | 0.7492 | +5.8 (7.5) | 0.3097 | 1.0000 |
| hubert[full] | 0 dB | 3 | 0.9219 | +5.7 (7.3) | 0.3124 | 1.0000 |
| wav2vec2[codec] | 0 dB | 3 | 0.6480 | +4.5 (2.3) | 0.0797 | 1.0000 |
| hubert[codec] | 10 dB | 3 | 0.9467 | +3.2 (5.5) | 0.4172 | 1.0000 |
| cnn_asp[codec] | 10 dB | 3 | 0.7963 | +2.6 (2.4) | 0.1981 | 1.0000 |
| hubert[full] | 10 dB | 3 | 0.9748 | +1.7 (1.8) | 0.2381 | 1.0000 |
| cnnlstm[codec] | 0 dB | 3 | 0.5451 | +0.2 (4.1) | 0.9455 | 1.0000 |
| wavlm[full] | 10 dB | 3 | 0.9974 | +0.2 (0.4) | 0.6035 | 1.0000 |
| wavlm[full] | 0 dB | 3 | 0.9883 | +0.1 (0.3) | 0.6220 | 1.0000 |
| cnnlstm[codec] | 10 dB | 3 | 0.6051 | -0.5 (1.9) | 0.7003 | 1.0000 |
| hubert[codec] | 0 dB | 3 | 0.6479 | -5.7 (4.0) | 0.1337 | 1.0000 |

## Bacaan

Pada nilai p mentah, 3 dari 16 perbandingan melampaui ambang lima persen, yaitu wavlm[codec] pada 0 dB, wav2vec2[codec] pada 10 dB, wavlm[codec] pada 10 dB. Setelah koreksi Holm-Bonferroni atas sepuluh perbandingan, nilai p terkecil menjadi 0.134, yang berada tepat di atas ambang.

Perbandingan ini berbeda dari klaim lain dalam penelitian yang gagal bertahan, dan perbedaannya perlu dinyatakan agar tidak terbaca sebagai pembelaan. Pertama, arah efeknya konsisten pada dua tingkat noise yang terpisah untuk arsitektur yang sama. Kedua, besarannya berpuluh poin persentase, bukan berbilang poin. Ketiga, mekanismenya dapat diperiksa secara langsung lewat area under curve, yang tidak bergantung pada ambang sama sekali. WavLM mempertahankan area under curve sekitar 0,96 pada 10 dB, sehingga daya pisahnya memang masih ada dan pernyataan bahwa yang bergeser adalah ambangnya dapat diverifikasi tanpa uji statistik.

Meskipun demikian, nilai p terkoreksi yang berada tepat di atas ambang berarti temuan ini belum dapat dinyatakan mapan pada tingkat kekakuan yang sama dengan temuan mengenai learning rate. Statusnya berada di antara keduanya, dan dilaporkan demikian.
