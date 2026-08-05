"""
UJI PENENTU untuk pertanyaan "sudah maksimal secara struktural?"

Ambil sampel yang salah pada ensemble HuBERT terbaik, lalu periksa apakah ADA
arsitektur mana pun (dari 21 run tersimpan: hubert, wavlm, wav2vec2, ast,
cnn_asp, cnnlstm) yang mengklasifikasikannya dengan benar.

  * Bila TIDAK ADA satu pun model yang benar  -> plafon dataset/label.
    Tidak ada arsitektur yang akan memperbaikinya; berhenti mengejar akurasi.
  * Bila ADA yang benar -> masih ada ruang; fusi/gating yang lebih baik bisa menutupnya.
"""
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.data import load_manifest, make_splits
from forlib.metrics import prior_matched_threshold

rows = load_manifest("manifest.csv")
_, _, te = make_splits(rows, "official")

runs = defaultdict(list)
for d in sorted(glob.glob("runs/*")):
    f = os.path.join(d, "test_scores.npy")
    if not os.path.exists(f):
        continue
    m = re.match(r"^(.+?)_official_codec(?:AV)?(?:_b\d+e\d+)?_s(\d+)$",
                 os.path.basename(d))
    if not m:
        continue
    y, p, _ = np.load(f)
    if len(y) != 1088:
        continue
    runs[m.group(1)].append({"seed": int(m.group(2)), "p": p})
    Y = y.astype(int)

per = {k: np.mean([r["p"] for r in rs], axis=0) for k, rs in runs.items()}
P_h = per["hubert"]
thr_h = prior_matched_threshold(P_h, 0.5)
wrong = np.flatnonzero((P_h >= thr_h).astype(int) != Y)

L = []
def out(s=""):
    print(s); L.append(s)

out("# Uji Penentu: Apakah Sisa Error Bersifat Struktural?\n")
out(f"Sampel uji: {len(Y)}. Ensemble HuBERT salah pada **{len(wrong)}** berkas.\n")
out("Pertanyaan: adakah arsitektur lain yang benar pada berkas-berkas itu?\n")

archs = sorted(per)
out("## Prediksi tiap arsitektur pada sampel keras\n")
hdr = "| idx | berkas | label | " + " | ".join(f"`{a}`" for a in archs) + " | ada yg benar? |"
out(hdr)
out("|" + "---|" * (len(archs) + 4))

n_none = 0
hard = []
for i in wrong:
    cells, any_ok = [], False
    for a in archs:
        t = prior_matched_threshold(per[a], 0.5)
        ok = int(per[a][i] >= t) == Y[i]
        any_ok |= ok
        cells.append(("✅" if ok else "❌") + f"{per[a][i]:.3f}")
    if not any_ok:
        n_none += 1
        hard.append(i)
    out(f"| {i} | `{te[i]['fname'][:26]}` | **{'fake' if Y[i] else 'real'}** | "
        + " | ".join(cells) + f" | {'✅ ADA' if any_ok else '❌ TIDAK ADA'} |")
out("")

out(f"**Sampel yang salah di SELURUH {len(archs)} arsitektur: {n_none}/{len(wrong)}**\n")

# juga cek: apakah ada run INDIVIDUAL (bukan ensemble arsitektur) yang benar
out("## Cek lebih ketat: seluruh 21 run individual\n")
allruns = [(k, r) for k, rs in runs.items() for r in rs]
out("| idx | label | run yang BENAR | dari total |")
out("|---|---|---|---|")
truly_hard = []
for i in wrong:
    good = []
    for k, r in allruns:
        t = prior_matched_threshold(r["p"], 0.5)
        if int(r["p"][i] >= t) == Y[i]:
            good.append(f"{k}/s{r['seed']}")
    if not good:
        truly_hard.append(i)
    out(f"| {i} | {'fake' if Y[i] else 'real'} | "
        f"{', '.join(good[:6]) + ('…' if len(good) > 6 else '') if good else '**TIDAK ADA**'} | "
        f"{len(good)}/{len(allruns)} |")
out("")
out(f"**Sampel yang salah di SELURUH {len(allruns)} run: {len(truly_hard)}**\n")

# properti akustik sampel yang benar-benar keras
if truly_hard:
    out("## Properti akustik sampel yang tak pernah benar\n")
    out("| idx | label | berkas | RMS | energi>6kHz | rolloff95 | zero-frac |")
    out("|---|---|---|---|---|---|---|")
    for i in truly_hard:
        r = te[i]
        x, sr = sf.read(r["path"], dtype="float64")
        if x.ndim > 1:
            x = x.mean(axis=1)
        S = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), 1.0 / sr)
        tot = S.sum() + 1e-20
        cs = np.cumsum(S) / tot
        ro = float(f[min(int(np.searchsorted(cs, 0.95)), len(f) - 1)])
        out(f"| {i} | {'fake' if Y[i] else 'real'} | `{r['fname'][:30]}` | "
            f"{float(np.sqrt((x**2).mean())):.4f} | {float(S[f>=6000].sum()/tot):.5f} | "
            f"{ro:.0f} Hz | {float((np.abs(x)<1e-6).mean()):.4f} |")
    out("")

out("## Kesimpulan\n")
if len(truly_hard) == len(wrong):
    out(f"Seluruh {len(wrong)} sampel yang tersisa **salah pada setiap satu dari "
        f"{len(allruns)} run** yang mencakup 6 arsitektur berbeda (SSL waveform, "
        f"Transformer spektrogram, CNN, CNN-LSTM) dan 4 rezim pra-pelatihan.")
    out("")
    out("**Ini penghalang struktural.** Menambah arsitektur, seed, atau kapasitas "
        "tidak akan memperbaikinya. Kemungkinan penyebab: label keliru, atau sampel "
        "yang secara akustik memang tidak dapat dipisahkan pada segmen 2 detik.")
    out("")
    out(f"Plafon empiris pada protokol ini: **{(len(Y)-len(truly_hard))/len(Y)*100:.2f}%** "
        f"({len(truly_hard)} error tak terhindarkan dari {len(Y)}).")
else:
    n_fix = len(wrong) - len(truly_hard)
    out(f"**{n_fix} dari {len(wrong)}** sampel BISA diklasifikasi benar oleh setidaknya "
        f"satu run. Artinya masih ada ruang: fusi/gating yang lebih cerdas berpotensi "
        f"menaikkan hasil hingga "
        f"**{(len(Y)-len(truly_hard))/len(Y)*100:.2f}%**.")
out("")

open("HASIL_SAMPEL_KERAS.md", "w", encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_SAMPEL_KERAS.md")
