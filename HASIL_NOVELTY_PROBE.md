# Dua Probe Novelty

## Probe A, Inversi polaritas lintas korpus

AUC < 0,5 berarti model memberi skor 'palsu' lebih tinggi kepada audio ASLI. Ini bukan sekadar performa buruk; ini pembalikan arah keputusan.

Total pengukuran AUC yang diperiksa: **80**
Yang terbalik (AUC < 0,5): **3**

| model | dataset | AUC | AUC bila dibalik |
|---|---|---|---|
| Nes2Net-X SOTA (ASVspoof) | FoR-2sec | **0.0233** | 0.9767 |
| ?[none] | for-rerec | **0.3137** | 0.6863 |
| ?[codec] | for-rerec | **0.4997** | 0.5003 |

Inversi terjadi HANYA pada model yang dilatih di korpus lain lalu diuji lintas korpus. Model yang dilatih pada FoR tidak menunjukkannya di FoR, jadi ini spesifik pergeseran domain, bukan cacat arsitektur.

## Probe B, Ketergantungan pintasan sebagai prediktor generalisasi

Proksi ketergantungan pintasan: recall pada TTS **2019 non-MP3** (Tacotron2, SpeedySpeech, VITS). Model yang belajar jejak MP3 dari FoR akan buta terhadap TTS lama yang TIDAK dikompresi MP3.

| model | recall TTS-2019 non-MP3 | recall TTS 2025-26 | selisih | akurasi FoR |
|---|---|---|---|---|
| wavlm[full] | 99.2% | 81.2% | +18.1 pp | 98.6% |
| hubert[full] | 2.3% | 84.6% | -82.2 pp | 95.3% |
| nes2net[full] | 93.8% | 98.5% | -4.7 pp | 93.8% |
| nes2net[fullrb] | 66.4% | 91.0% | -24.6 pp | 98.5% |
| nes2net[fullrb] | 29.8% | 80.3% | -50.5 pp | 98.5% |
| nes2net_lastlayer[full] | 73.9% | 92.1% | -18.2 pp | 92.0% |
| wavlm[codec] | 41.0% | 73.4% | -32.4 pp | 96.5% |

- Korelasi **akurasi FoR** vs recall TTS modern: **r = -0.542**
- Korelasi **recall TTS-2019 non-MP3** vs recall TTS modern: **r = +0.467**

> **KLAIM INI SUDAH DITARIK.** Angka r = -0,542 dihitung atas tujuh run tanpa merata-ratakan seed per konfigurasi. Dihitung ulang atas sepuluh konfigurasi dengan seed dirata-ratakan, hasilnya **r = -0,048**, uji permutasi **p = 0,895**, selang bootstrap **[-0,664, +0,629]** yang memuat nol. Lihat [HASIL_UJI_KORELASI.md](HASIL_UJI_KORELASI.md).
>
> Yang tetap berdiri adalah pernyataan yang lebih lemah, yaitu bahwa akurasi FoR **tidak memprediksi** kemampuan mendeteksi TTS modern. Pernyataan itu ditopang audit kebocoran codec yang dihitung langsung dari berkas, bukan oleh koefisien korelasi. Pernyataan bahwa keduanya berkorelasi negatif secara sistematis tidak didukung data.

Mekanismenya sudah terdokumentasi di proyek ini: 90,7% sampel palsu di data latih FoR berasal dari MP3. Akurasi FoR yang tinggi sebagian diperoleh dengan mengeksploitasi jejak itu, dan jejak itu tidak ada pada TTS modern.

Bukti paling telanjang ada pada `hubert[full]`: akurasi FoR 95,3% namun recall hanya **2,3%** pada TTS 2019 yang tidak dikompresi MP3. Model itu praktis mendeteksi MP3, bukan sintesis.

> Catatan: hanya 7 model. Korelasi pada n sekecil ini bersifat indikatif, belum kesimpulan.
