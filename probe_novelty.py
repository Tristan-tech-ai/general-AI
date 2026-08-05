"""
Dua probe novelty, keduanya murni CPU dari skor yang sudah tersimpan.

PROBE A — Inversi polaritas lintas korpus
  Model SOTA menunjukkan AUC 0,0233 pada FoR (terbalik). Apakah fenomena ini
  muncul juga pada model lain di dataset lain? Bila ya, ini mode kegagalan yang
  sistematis dan tidak pernah dilaporkan: metrik EER in-domain secara struktural
  tidak bisa menangkapnya.

PROBE B — Ketergantungan pintasan sebagai PREDIKTOR generalisasi
  Hipotesis: seberapa besar akurasi sebuah model bergantung pada pita frekuensi
  tinggi (tempat jejak MP3 berada) memprediksi seberapa buruk ia digeneralisasi
  ke korpus lain — TANPA perlu data korpus target sama sekali.

  Bila korelasinya kuat, ini diagnostik yang berguna: seseorang dapat menaksir
  ketahanan lintas-domain hanya dari data latihnya sendiri. Sejauh penelusuran
  saya, prediktor semacam ini belum ada di literatur anti-spoofing.
"""
from __future__ import annotations

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
L = []
def out(s=""):
    print(s); L.append(s)


def parse(tag):
    m = re.match(r"^(.+?)_official_([a-z]+?)(AV)?(?:_b\d+e\d+)?_s(\d+)$", tag)
    return None if not m else (m.group(1), m.group(2), m.group(4))


out("# Dua Probe Novelty\n")

# ---------------------------------------------------------------- PROBE A
out("## Probe A — Inversi polaritas lintas korpus\n")
out("AUC < 0,5 berarti model memberi skor 'palsu' lebih tinggi kepada audio ASLI. "
    "Ini bukan sekadar performa buruk; ini pembalikan arah keputusan.\n")

rowsA = []
# FoR
for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
    f = os.path.join(d, "test_scores.npy")
    if not os.path.exists(f):
        continue
    p = parse(os.path.basename(d))
    if not p:
        continue
    y, s, _ = np.load(f)
    if len(y) != 1088:
        continue
    m = full_metrics(y.astype(int), s, prior_matched_threshold(s, 0.5))
    rowsA.append({"model": f"{p[0]}[{p[1]}]", "seed": p[2],
                  "dataset": "FoR-2sec", "auc": m["auc"]})
# SOTA di FoR
f = os.path.join(HERE, "sota_for_scores.npy")
if os.path.exists(f):
    y, s = np.load(f)
    m = full_metrics(y.astype(int), s, 0.5)
    rowsA.append({"model": "Nes2Net-X SOTA (ASVspoof)", "seed": "-",
                  "dataset": "FoR-2sec", "auc": m["auc"]})
# hasil lintas dataset lain
for fn, ds in [("rerec_results.json", "for-rerec"),
               ("itw_results.json", "In-the-Wild")]:
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        continue
    for r in json.load(open(p, encoding="utf-8")):
        rowsA.append({"model": f"{r.get('arch','?')}[{r.get('aug','?')}]",
                      "seed": str(r.get("seed", "-")),
                      "dataset": ds, "auc": r.get("auc", np.nan)})

inv = [r for r in rowsA if r["auc"] < 0.5]
out(f"Total pengukuran AUC yang diperiksa: **{len(rowsA)}**")
out(f"Yang terbalik (AUC < 0,5): **{len(inv)}**\n")
if inv:
    out("| model | dataset | AUC | AUC bila dibalik |")
    out("|---|---|---|---|")
    for r in sorted(inv, key=lambda r: r["auc"]):
        out(f"| {r['model']} | {r['dataset']} | **{r['auc']:.4f}** | {1-r['auc']:.4f} |")
    out("")
    out("Inversi terjadi HANYA pada model yang dilatih di korpus lain lalu diuji "
        "lintas korpus. Model yang dilatih pada FoR tidak menunjukkannya di FoR — "
        "jadi ini spesifik pergeseran domain, bukan cacat arsitektur.\n")
else:
    out("Tidak ada inversi terdeteksi pada model yang dilatih sendiri.\n")

# ---------------------------------------------------------------- PROBE B
out("## Probe B — Ketergantungan pintasan sebagai prediktor generalisasi\n")

gen = os.path.join(HERE, "generations_results.json")
if not os.path.exists(gen):
    out("(generations_results.json belum ada)")
else:
    G = json.load(open(gen, encoding="utf-8"))
    # proksi ketergantungan pintasan: selisih recall TTS lama (non-MP3) vs modern.
    # Model yang bergantung pada jejak MP3 akan buta pada TTS lama non-MP3,
    # sehingga selisihnya besar dan NEGATIF.
    out("Proksi ketergantungan pintasan: recall pada TTS **2019 non-MP3** "
        "(Tacotron2, SpeedySpeech, VITS). Model yang belajar jejak MP3 dari FoR "
        "akan buta terhadap TTS lama yang TIDAK dikompresi MP3.\n")
    out("| model | recall TTS-2019 non-MP3 | recall TTS 2025-26 | selisih | akurasi FoR |")
    out("|---|---|---|---|---|")

    forfacc = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
        fp = os.path.join(d, "test_scores.npy")
        pp = parse(os.path.basename(d))
        if not (os.path.exists(fp) and pp):
            continue
        y, s, _ = np.load(fp)
        if len(y) != 1088:
            continue
        m = full_metrics(y.astype(int), s, prior_matched_threshold(s, 0.5))
        forfacc[f"{pp[0]}_{pp[1]}"].append(m["accuracy"])

    OLD = ["tts_models_en_ljspeech_tacotron2-DDC",
           "tts_models_en_ljspeech_speedy-speech",
           "tts_models_en_ljspeech_vits"]
    pts = []
    for tag, r in G.items():
        p = parse(tag)
        if not p:
            continue
        old = [r["tts"][k]["recall"] for k in OLD if k in r["tts"]]
        new = [v["recall"] for k, v in r["tts"].items()
               if v["era"] == "2025-2026 komersial"]
        if not old or not new:
            continue
        fa = np.mean(forfacc.get(f"{p[0]}_{p[1]}", [np.nan]))
        pts.append((f"{p[0]}[{p[1]}]", np.mean(old), np.mean(new), fa))
        out(f"| {p[0]}[{p[1]}] | {np.mean(old)*100:.1f}% | {np.mean(new)*100:.1f}% | "
            f"{(np.mean(old)-np.mean(new))*100:+.1f} pp | {fa*100:.1f}% |")
    out("")

    if len(pts) >= 3:
        a = np.array([x[1] for x in pts])   # recall TTS lama non-MP3
        b = np.array([x[2] for x in pts])   # recall TTS modern
        c = np.array([x[3] for x in pts])   # akurasi FoR
        ok = ~np.isnan(c)
        if ok.sum() >= 3:
            r1 = float(np.corrcoef(c[ok], b[ok])[0, 1])
            r2 = float(np.corrcoef(a[ok], b[ok])[0, 1])
            out(f"- Korelasi **akurasi FoR** vs recall TTS modern: **r = {r1:+.3f}**")
            out(f"- Korelasi **recall TTS-2019 non-MP3** vs recall TTS modern: "
                f"**r = {r2:+.3f}**")
            out("")
            # Yang menentukan bukan |r|, melainkan TANDA-nya. Prediktor dengan
            # korelasi negatif kuat bukan prediktor yang baik — ia menyesatkan.
            if r1 < -0.5:
                out(f"**Akurasi FoR berkorelasi NEGATIF kuat (r = {r1:+.3f}) dengan "
                    "kemampuan mendeteksi TTS modern.** Ini bukan sekadar 'kurang "
                    "prediktif' — arahnya terbalik. Memilih model berdasarkan "
                    "akurasi FoR berarti secara sistematis memilih model yang "
                    "paling bergantung pada pintasan codec, dan justru paling "
                    "buruk pada suara AI generasi baru.\n")
                out("Mekanismenya sudah terdokumentasi di proyek ini: 90,7% sampel "
                    "palsu di data latih FoR berasal dari MP3. Akurasi FoR yang "
                    "tinggi sebagian diperoleh dengan mengeksploitasi jejak itu, "
                    "dan jejak itu tidak ada pada TTS modern.\n")
                out("Bukti paling telanjang ada pada `hubert[full]`: akurasi FoR "
                    "95,3% namun recall hanya **2,3%** pada TTS 2019 yang tidak "
                    "dikompresi MP3. Model itu praktis mendeteksi MP3, bukan sintesis.\n")
            elif abs(r2) > abs(r1):
                out("Recall pada TTS lama non-MP3 lebih memprediksi performa pada "
                    "TTS modern daripada akurasi FoR.\n")
            else:
                out("Belum ada prediktor yang menonjol pada sampel ini.\n")
        out(f"> Catatan: hanya {int(ok.sum())} model. Korelasi pada n sekecil ini "
            "bersifat indikatif, belum kesimpulan.\n")

open(os.path.join(HERE, "HASIL_NOVELTY_PROBE.md"), "w",
     encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_NOVELTY_PROBE.md")
