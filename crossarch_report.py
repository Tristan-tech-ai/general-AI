"""Uji lintas-arsitektur: apakah band-gain bekerja di WavLM juga, bukan hanya Nes2Net?"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.metrics import full_metrics, prior_matched_threshold

HERE = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(os.path.join(HERE, "generations_results.json"), encoding="utf-8"))
OLD_NONMP3 = ["tts_models_en_ljspeech_tacotron2-DDC",
              "tts_models_en_ljspeech_speedy-speech",
              "tts_models_en_ljspeech_vits"]


def parse(tag):
    m = re.match(r"^(.+?)_official_([a-z]+?)(?:_b\d+e\d+)?_s(\d+)$", tag)
    return None if not m else (m.group(1), m.group(2), m.group(3))


foracc = {}
for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
    f = os.path.join(d, "test_scores.npy")
    p = parse(os.path.basename(d))
    if not (os.path.exists(f) and p):
        continue
    y, s, _ = np.load(f)
    if len(y) != 1088:
        continue
    foracc[p] = full_metrics(y.astype(int), s,
                             prior_matched_threshold(s, 0.5))["accuracy"]

rows = defaultdict(list)
for tag, r in G.items():
    p = parse(tag)
    if not p:
        continue
    modern = [v["recall"] for v in r["tts"].values()
              if v["era"] == "2025-2026 komersial"]
    old = [r["tts"][k]["recall"] for k in OLD_NONMP3 if k in r["tts"]]
    fa = foracc.get(p)
    if modern and fa is not None:
        rows[(p[0], p[1])].append({"for": fa, "modern": np.mean(modern),
                                   "old": np.mean(old) if old else np.nan})

L = []
def out(s=""):
    print(s); L.append(s)

out("# Uji Lintas-Arsitektur: Apakah Band-Gain Bekerja di Luar Nes2Net?\n")
out("Band-gain dirancang dari diagnosis tentang SINYAL (level vs struktur pita "
    "tinggi), bukan tentang arsitektur. Bila diagnosisnya benar, efeknya harus "
    "muncul juga di arsitektur lain.\n")

out("| arsitektur | augmentasi | n | akurasi FoR | TTS 2025-26 | TTS-2019 non-MP3 |")
out("|---|---|---|---|---|---|")
S = {}
for k in sorted(rows, key=lambda t: (t[0], t[1])):
    v = rows[k]
    fa = np.array([x["for"] for x in v]) * 100
    md = np.array([x["modern"] for x in v]) * 100
    ol = np.array([x["old"] for x in v]) * 100
    S[k] = (fa, md, ol)
    sd = lambda a: ("+/-%.2f" % a.std(ddof=1)) if len(a) > 1 else ""
    out("| `%s` | %s | %d | %.2f%% %s | **%.2f%%** %s | %.1f%% %s |"
        % (k[0], k[1], len(v), fa.mean(), sd(fa), md.mean(), sd(md),
           ol.mean(), sd(ol)))
out("")

out("## Efek band-gain per arsitektur (vs basis `full`)\n")
out("| arsitektur | d-akurasi FoR | d-TTS modern | d-TTS-2019 non-MP3 |")
out("|---|---|---|---|")
ok = []
for arch in ["hubert", "nes2net", "wavlm"]:
    b = S.get((arch, "full"))
    g_ = S.get((arch, "fullbg"))
    if not (b and g_):
        continue
    d_for = g_[0].mean() - b[0].mean()
    d_md = g_[1].mean() - b[1].mean()
    d_ol = g_[2].mean() - b[2].mean()
    ok.append((arch, d_for, d_md, d_ol, b[2].mean(), b[2].std(ddof=1),
               g_[2].std(ddof=1)))
    out("| `%s` | **%+.2f pp** | **%+.2f pp** | **%+.1f pp** |"
        % (arch, d_for, d_md, d_ol))
out("")

if len(ok) >= 3:
    out("## Uji hipotesis ceiling\n")
    out("Hipotesis yang diajukan sebelum menjalankan HuBERT: band-gain "
        "memperbaiki sumbu yang masih punya ruang, dan tidak menolong bila sumbu "
        "itu sudah mendekati batas atas. Karena band-gain bekerja dengan "
        "menghapus ketergantungan pintasan codec, besarnya perbaikan seharusnya "
        "berbanding terbalik dengan titik awal pada proksi pintasan.\n")
    out("| arsitektur | recall awal TTS-2019 non-MP3 | perubahan | std sebelum | std sesudah |")
    out("|---|---|---|---|---|")
    for a, _, _, d_ol, base, sd0, sd1 in sorted(ok, key=lambda x: x[4]):
        out("| `%s` | %.1f%% | **%+.1f pp** | +/-%.1f | **+/-%.1f** |"
            % (a, base, d_ol, sd0, sd1))
    out("")
    base = np.array([x[4] for x in ok])
    dlt = np.array([x[3] for x in ok])
    r = float(np.corrcoef(base, dlt)[0, 1])
    out("Korelasi antara titik awal dan besar perbaikan: **r = %+.3f**\n" % r)
    if r < -0.9:
        out("Polanya monoton dan sangat kuat: makin rendah titik awal, makin "
            "besar perbaikannya. Ini konsisten dengan hipotesis ceiling dan "
            "mendukung bahwa band-gain memang bekerja lewat penghapusan "
            "ketergantungan pintasan, bukan lewat efek yang khas satu arsitektur.\n")
        out("> Catatan kehati-hatian: hipotesis ini disusun setelah melihat hasil "
            "Nes2Net dan WavLM, lalu diuji pada HuBERT dengan prediksi yang "
            "dituliskan lebih dulu. Konfirmasinya karena itu bermakna, tetapi "
            "hanya berbasis tiga arsitektur.\n")
    else:
        out("Pola ceiling belum tegas pada tiga arsitektur ini (r = %+.3f).\n" % r)

# kombinasi
kb = S.get(("nes2net", "fullbgrb"))
if kb:
    bb = S.get(("nes2net", "fullbg"))
    rb = S.get(("nes2net", "fullrb"))
    out("## Kombinasi band-gain + RawBoost (Nes2Net-X)\n")
    out("| varian | akurasi FoR | TTS modern | TTS-2019 non-MP3 |")
    out("|---|---|---|---|")
    for nm, v in [("band-gain saja", bb), ("RawBoost saja", rb),
                  ("keduanya", kb)]:
        if v:
            out("| %s | %.2f%% | **%.2f%%** | %.1f%% |"
                % (nm, v[0].mean(), v[1].mean(), v[2].mean()))
    out("")
    if bb and rb:
        out("Kombinasi %s keunggulan generalisasi band-gain: TTS modern %.2f%% "
            "vs %.2f%% (band-gain saja) dan %.2f%% (RawBoost saja).\n"
            % ("mempertahankan" if kb[1].mean() >= bb[1].mean() - 2 else "kehilangan",
               kb[1].mean(), bb[1].mean(), rb[1].mean()))

open(os.path.join(HERE, "HASIL_LINTAS_ARSITEKTUR.md"), "w",
     encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_LINTAS_ARSITEKTUR.md")
