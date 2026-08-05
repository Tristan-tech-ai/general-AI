# Deteksi Lintas-Generasi TTS pada Titik Operasi Tersamakan

Setiap model dikalibrasi agar **spesifisitas = 95%** pada 1.500 berkas asli In-the-Wild, baru kemudian recall diukur pada tiap sistem TTS (MLAAD, 300 berkas per sistem).

Kalibrasi ini penting: tanpanya, detektor yang selalu menjawab 'palsu' akan mencatat recall 100% — persis yang terjadi pada checkpoint SOTA (spesifisitas 0%). Di sini seluruh model dibandingkan pada FPR 5% yang sama.

| sistem TTS | generasi | WavLM + aug penuh | HuBERT + aug penuh | Nes2Net-X + aug penuh | nes2net_official_fullrb_b16e10_s42 | nes2net_official_fullrb_b16e10_s1337 | nes2net_lastlayer_official_full_b16e10_s1337 | WavLM + codec saja |
|---|---|---|---|---|---|---|---|---|
| Griffin-Lim | 2019 (era FoR) | 97.0% | 94.0% | 99.7% | 99.0% | 82.0% | 89.0% | 99.0% |
| Tacotron2-DDC | 2019 (era FoR) | 98.0% | 4.0% | 90.3% | 49.0% | 28.7% | 66.3% | 28.7% |
| SpeedySpeech | 2019 (era FoR) | 99.7% | 1.3% | 97.3% | 89.3% | 25.7% | 88.3% | 78.7% |
| VITS | 2021-2022 | 100.0% | 1.7% | 93.7% | 61.0% | 35.0% | 67.0% | 15.7% |
| XTTS-v2 | 2021-2022 | 87.0% | 76.0% | 97.3% | 91.7% | 67.3% | 97.3% | 66.3% |
| Bark | 2021-2022 | 67.0% | 75.7% | 90.0% | 74.7% | 59.7% | 87.7% | 80.3% |
| F5-TTS | 2024-2025 open | 92.3% | 90.7% | 100.0% | 99.0% | 94.0% | 98.0% | 91.0% |
| Kokoro | 2024-2025 open | 95.7% | 95.3% | 99.7% | 99.7% | 95.7% | 92.7% | 90.7% |
| Sesame-CSM | 2024-2025 open | 71.7% | 86.7% | 92.3% | 79.3% | 48.0% | 92.0% | 71.7% |
| Orpheus | 2024-2025 open | 83.7% | 83.3% | 99.7% | 97.0% | 89.0% | 94.0% | 72.0% |
| ElevenLabs-v3 | 2025-2026 komersial | 93.3% | 87.0% | 97.7% | 95.3% | 90.7% | 98.0% | 84.0% |
| Chatterbox | 2025-2026 komersial | 85.7% | 76.0% | 99.3% | 86.0% | 81.0% | 95.0% | 52.7% |
| OpenAI TTS-1 HD | 2025-2026 komersial | 95.0% | 98.7% | 99.7% | 99.3% | 93.0% | 97.7% | 98.0% |
| Higgs-Audio-V2 | 2025-2026 komersial | 50.7% | 76.7% | 97.3% | 83.3% | 56.3% | 77.7% | 59.0% |

## Rerata per generasi

| model | 2019 (era FoR) | 2021-2022 | 2024-2025 open | 2025-2026 komersial | **turun 2019→2026** |
|---|---|---|---|---|---|
| WavLM + aug penuh | 98.2% | 84.7% | 85.8% | 81.2% | **+17.1 pp** |
| HuBERT + aug penuh | 33.1% | 51.1% | 89.0% | 84.6% | **-51.5 pp** |
| Nes2Net-X + aug penuh | 95.8% | 93.7% | 97.9% | 98.5% | **-2.7 pp** |
| nes2net_official_fullrb_b16e10_s42 | 79.1% | 75.8% | 93.8% | 91.0% | **-11.9 pp** |
| nes2net_official_fullrb_b16e10_s1337 | 45.4% | 54.0% | 81.7% | 80.3% | **-34.8 pp** |
| nes2net_lastlayer_official_full_b16e10_s1337 | 81.2% | 84.0% | 94.2% | 92.1% | **-10.9 pp** |
| WavLM + codec saja | 68.8% | 54.1% | 81.3% | 73.4% | **-4.6 pp** |
