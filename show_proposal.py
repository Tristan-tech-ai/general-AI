"""Ringkas hasil replikasi proposal apa adanya, dan bandingkan dengan versi diperbaiki."""
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

out("# Replikasi Metodologi Proposal Apa Adanya\n")
out("Seluruh angka di halaman ini berasal dari run setelah perbaikan bug "
    "encoder beku yang diuraikan di KOREKSI_REPLIKASI_PROPOSAL.md. Sebelum "
    "perbaikan itu, encoder tidak pernah menerima gradien meskipun proposal "
    "menetapkan satu learning rate untuk seluruh model.\n")
out("Konfigurasi persis seperti tertulis di proposal: split acak 60/20/20, "
    "learning rate 0,001 seragam untuk semua model, 20 epoch tanpa early "
    "stopping, normalisasi peak amplitudo, augmentasi noise SNR 15 sampai 30 dB, "
    "batch 32, ambang keputusan 0,5.\n")

# Hanya split acak. Sejak arm proposal juga dijalankan pada partisi resmi,
# pola glob lama ikut menarik run partisi resmi ke dalam tabel ini dan membuat
# dua protokol yang berbeda tampak berada dalam satu kolom.
rows = []
for d in sorted(glob.glob(os.path.join(HERE, "runs", "*_random_*ULRPK*"))):
    fj = os.path.join(d, "results.json")
    fs = os.path.join(d, "test_scores.npy")
    if not (os.path.exists(fj) and os.path.exists(fs)):
        continue
    r = json.load(open(fj, encoding="utf-8"))
    y, p, _ = np.load(fs)
    y = y.astype(int)
    m05 = full_metrics(y, p, 0.5)
    rows.append({"model": r["args"]["model"], "n": m05["n"],
                 "acc": m05["accuracy"], "eer": m05["eer"], "auc": m05["auc"],
                 "f1": m05["f1"], "prec": m05["precision"],
                 "rec": m05["recall"], "tp": m05["tp"], "tn": m05["tn"],
                 "fp": m05["fp"], "fn": m05["fn"],
                 "epoch": r.get("best_epoch")})

if not rows:
    out("Belum ada hasil.")
else:
    out("| model | akurasi | presisi | recall | F1 | AUC | EER |")
    out("|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: -x["acc"]):
        out("| %s | **%.2f%%** | %.2f%% | %.2f%% | %.2f%% | %.4f | %.2f%% |"
            % (r["model"], r["acc"] * 100, r["prec"] * 100, r["rec"] * 100,
               r["f1"] * 100, r["auc"], r["eer"] * 100))
    out("")
    out("Metrik dihitung pada ambang 0,5 seperti yang tersirat di proposal. "
        "Jumlah berkas uji %d (split acak 20 persen).\n" % rows[0]["n"])

    out("## Confusion matrix\n")
    out("| model | TP | TN | FP | FN |")
    out("|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: -x["acc"]):
        out("| %s | %d | %d | %d | %d |"
            % (r["model"], r["tp"], r["tn"], r["fp"], r["fn"]))
    out("")

# Konfigurasi proposal yang sama, tetapi diuji pada partisi resmi.
# Digabungkan per arsitektur, bukan per inisialisasi. Menampilkan tiap
# inisialisasi sebagai baris terpisah membuat sebaran tampak seperti perbedaan
# antar model, padahal seluruh baris berasal dari konfigurasi yang sama.
ofs = defaultdict(lambda: {"a05": [], "apm": [], "auc": [], "eer": []})
for d in sorted(glob.glob(os.path.join(HERE, "runs", "*_official_*ULRPK*"))):
    fs = os.path.join(d, "test_scores.npy")
    fj = os.path.join(d, "results.json")
    if not (os.path.exists(fs) and os.path.exists(fj)):
        continue
    r = json.load(open(fj, encoding="utf-8"))
    y, p, _ = np.load(fs)
    y = y.astype(int)
    m05 = full_metrics(y, p, 0.5)
    g = ofs[r["args"]["model"]]
    g["a05"].append(m05["accuracy"] * 100)
    g["apm"].append(full_metrics(y, p, prior_matched_threshold(p, 0.5))["accuracy"] * 100)
    g["auc"].append(m05["auc"])
    g["eer"].append(m05["eer"] * 100)

if ofs:
    out("## Konfigurasi proposal yang sama, diuji pada partisi resmi\n")
    out("Perbedaan dengan tabel di atas hanya pada cara data dibagi. Seluruh "
        "setelan pelatihan identik. Angka dalam kurung adalah simpangan baku "
        "antar inisialisasi acak.\n")
    out("| model | n | akurasi @0,5 | akurasi @prior | AUC | EER |")
    out("|---|---|---|---|---|---|")
    for nm, g in sorted(ofs.items(), key=lambda t: -np.mean(t[1]["apm"])):
        n = len(g["apm"])
        sd = np.std(g["apm"], ddof=1) if n > 1 else 0.0
        pm = ("%.2f%% (%.2f)" % (np.mean(g["apm"]), sd) if n > 1
              else "%.2f%%" % np.mean(g["apm"]))
        out("| %s | %d | %.2f%% | **%s** | %.4f | %.2f%% |"
            % (nm, n, np.mean(g["a05"]), pm, np.mean(g["auc"]),
               np.mean(g["eer"])))
    out("")
    out("Selisih antara kedua tabel jauh lebih besar daripada selisih antar "
        "arsitektur di dalam masing-masing tabel. Pada split acak seluruh model "
        "berkerumun dalam rentang setengah poin persentase, sedangkan pada "
        "partisi resmi rentangnya puluhan poin. Pemeringkatan yang dihasilkan "
        "kedua protokol juga tidak sama, sehingga memilih model berdasarkan "
        "split acak dapat menghasilkan pilihan yang keliru.\n")

# pembanding: versi diperbaiki pada protokol resmi
out("## Pembanding: metodologi yang diperbaiki\n")
g = defaultdict(list)
for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
    f = os.path.join(d, "test_scores.npy")
    m = re.match(r"^(.+?)_official_([a-z]+?)(?:_b\d+e\d+)?_s(\d+)$",
                 os.path.basename(d))
    if not (os.path.exists(f) and m):
        continue
    y, p, _ = np.load(f)
    if len(y) != 1088:
        continue
    met = full_metrics(y.astype(int), p, prior_matched_threshold(p, 0.5))
    g[(m.group(1), m.group(2))].append(met["accuracy"])

best = sorted(((k, np.mean(v), len(v)) for k, v in g.items() if len(v) >= 3),
              key=lambda t: -t[1])[:5]
if best:
    out("| model dan augmentasi | n seed | akurasi pada partisi resmi |")
    out("|---|---|---|")
    for (a, au), acc, n in best:
        out("| %s + %s | %d | **%.2f%%** |" % (a, au, n, acc * 100))
    out("")

out("Perlu dicatat bahwa kedua kolom tidak setara. Replikasi proposal diuji "
    "pada split acak, sedangkan versi diperbaiki diuji pada partisi resmi yang "
    "memisahkan domain rekaman. Angka yang lebih tinggi pada split acak tidak "
    "berarti model lebih baik, melainkan bahwa tugasnya lebih mudah. Inilah inti "
    "temuan penelitian ini.\n")

open(os.path.join(HERE, "HASIL_REPLIKASI_PROPOSAL.md"), "w",
     encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_REPLIKASI_PROPOSAL.md")
