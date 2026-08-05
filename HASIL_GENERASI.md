# Deteksi Lintas-Generasi TTS pada Titik Operasi Tersamakan

Setiap model dikalibrasi agar **spesifisitas = 95%** pada 1.500 berkas asli In-the-Wild, baru kemudian recall diukur pada tiap sistem TTS (MLAAD, 300 berkas per sistem).

Kalibrasi ini penting: tanpanya, detektor yang selalu menjawab 'palsu' akan mencatat recall 100% — persis yang terjadi pada checkpoint SOTA (spesifisitas 0%). Di sini seluruh model dibandingkan pada FPR 5% yang sama.

| sistem TTS | generasi | nes2net_official_fullbg_b16e10_s42 | nes2net_official_fullbg_b16e10_s1337 | nes2net_official_fullbg_b16e10_s2024 | nes2net_official_soft_b16e10_s42 | nes2net_official_soft_b16e10_s1337 | nes2net_official_soft_b16e10_s2024 | Nes2Net-X + aug penuh | nes2net_official_full_b16e10_s1337 | nes2net_official_full_b16e10_s2024 | nes2net_official_fullrb_b16e10_s42 | nes2net_official_fullrb_b16e10_s1337 | nes2net_official_fullrb_b16e10_s2024 | WavLM + aug penuh | HuBERT + aug penuh |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Griffin-Lim | 2019 (era FoR) | 99.0% | 92.0% | 100.0% | 90.7% | 87.7% | 98.3% | 99.7% | 98.3% | 96.7% | 99.0% | 82.0% | 93.0% | 97.0% | 94.0% |
| Tacotron2-DDC | 2019 (era FoR) | 80.3% | 92.3% | 92.3% | 15.7% | 89.3% | 87.0% | 90.3% | 94.7% | 53.7% | 49.0% | 28.7% | 81.7% | 98.0% | 4.0% |
| SpeedySpeech | 2019 (era FoR) | 95.3% | 97.7% | 93.3% | 16.3% | 95.7% | 94.0% | 97.3% | 99.3% | 45.7% | 89.3% | 25.7% | 92.3% | 99.7% | 1.3% |
| VITS | 2021-2022 | 86.3% | 95.0% | 96.7% | 19.0% | 90.0% | 88.3% | 93.7% | 96.7% | 67.7% | 61.0% | 35.0% | 83.3% | 100.0% | 1.7% |
| XTTS-v2 | 2021-2022 | 91.7% | 81.3% | 97.0% | 58.7% | 85.0% | 92.7% | 97.3% | 98.3% | 79.7% | 91.7% | 67.3% | 95.0% | 87.0% | 76.0% |
| Bark | 2021-2022 | 81.0% | 69.7% | 79.7% | 37.3% | 71.7% | 83.0% | 90.0% | 82.3% | 70.0% | 74.7% | 59.7% | 88.3% | 67.0% | 75.7% |
| F5-TTS | 2024-2025 open | 97.0% | 96.3% | 100.0% | 80.7% | 90.7% | 99.0% | 100.0% | 99.7% | 95.7% | 99.0% | 94.0% | 97.0% | 92.3% | 90.7% |
| Kokoro | 2024-2025 open | 96.7% | 96.3% | 99.0% | 70.3% | 97.3% | 99.7% | 99.7% | 97.3% | 98.0% | 99.7% | 95.7% | 97.0% | 95.7% | 95.3% |
| Sesame-CSM | 2024-2025 open | 88.7% | 67.0% | 77.0% | 41.0% | 75.3% | 86.0% | 92.3% | 89.7% | 67.0% | 79.3% | 48.0% | 87.0% | 71.7% | 86.7% |
| Orpheus | 2024-2025 open | 98.3% | 97.3% | 97.0% | 53.3% | 95.0% | 96.3% | 99.7% | 98.3% | 90.7% | 97.0% | 89.0% | 94.0% | 83.7% | 83.3% |
| ElevenLabs-v3 | 2025-2026 komersial | 96.7% | 91.3% | 97.0% | 72.3% | 91.7% | 99.7% | 97.7% | 96.0% | 91.7% | 95.3% | 90.7% | 96.0% | 93.3% | 87.0% |
| Chatterbox | 2025-2026 komersial | 89.0% | 90.3% | 100.0% | 62.7% | 77.0% | 96.0% | 99.3% | 98.3% | 88.0% | 86.0% | 81.0% | 93.3% | 85.7% | 76.0% |
| OpenAI TTS-1 HD | 2025-2026 komersial | 97.7% | 97.7% | 100.0% | 76.7% | 98.0% | 99.3% | 99.7% | 99.7% | 96.3% | 99.3% | 93.0% | 95.7% | 95.0% | 98.7% |
| Higgs-Audio-V2 | 2025-2026 komersial | 87.0% | 77.7% | 97.7% | 45.0% | 66.0% | 87.0% | 97.3% | 96.3% | 79.3% | 83.3% | 56.3% | 81.7% | 50.7% | 76.7% |

## Rerata per generasi

| model | 2019 (era FoR) | 2021-2022 | 2024-2025 open | 2025-2026 komersial | **turun 2019→2026** |
|---|---|---|---|---|---|
| nes2net_official_fullbg_b16e10_s42 | 91.6% | 86.3% | 95.2% | 92.6% | **-1.0 pp** |
| nes2net_official_fullbg_b16e10_s1337 | 94.0% | 82.0% | 89.2% | 89.2% | **+4.8 pp** |
| nes2net_official_fullbg_b16e10_s2024 | 95.2% | 91.1% | 93.2% | 98.7% | **-3.4 pp** |
| nes2net_official_soft_b16e10_s42 | 40.9% | 38.3% | 61.3% | 64.2% | **-23.3 pp** |
| nes2net_official_soft_b16e10_s1337 | 90.9% | 82.2% | 89.6% | 83.2% | **+7.7 pp** |
| nes2net_official_soft_b16e10_s2024 | 93.1% | 88.0% | 95.2% | 95.5% | **-2.4 pp** |
| Nes2Net-X + aug penuh | 95.8% | 93.7% | 97.9% | 98.5% | **-2.7 pp** |
| nes2net_official_full_b16e10_s1337 | 97.4% | 92.4% | 96.2% | 97.6% | **-0.1 pp** |
| nes2net_official_full_b16e10_s2024 | 65.3% | 72.4% | 87.8% | 88.8% | **-23.5 pp** |
| nes2net_official_fullrb_b16e10_s42 | 79.1% | 75.8% | 93.8% | 91.0% | **-11.9 pp** |
| nes2net_official_fullrb_b16e10_s1337 | 45.4% | 54.0% | 81.7% | 80.3% | **-34.8 pp** |
| nes2net_official_fullrb_b16e10_s2024 | 89.0% | 88.9% | 93.8% | 91.7% | **-2.7 pp** |
| WavLM + aug penuh | 98.2% | 84.7% | 85.8% | 81.2% | **+17.1 pp** |
| HuBERT + aug penuh | 33.1% | 51.1% | 89.0% | 84.6% | **-51.5 pp** |
