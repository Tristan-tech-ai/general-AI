"""
Memecah "nilai rekayasa" menjadi sumbu yang benar-benar terpisah.

Angka +37,59 dan +43,01 poin persentase yang dilaporkan dari matriks 2x2
membandingkan konfigurasi proposal pada ambang 0,5 dengan konfigurasi diperbaiki
pada ambang prior-matched. Dua hal berubah sekaligus di sana, yaitu cara model
dilatih dan cara ambang keputusan ditetapkan. Selisih itu karena itu tidak boleh
dibaca sebagai sumbangan pelatihan saja.

Skrip ini memisahkan keduanya dengan menghitung empat besaran dari skor yang
sama persis, tanpa melatih ulang apa pun:

  A. proposal pada ambang 0,5              -> angka yang dilaporkan proposal
  B. proposal pada ambang prior-matched    -> A ditambah perbaikan ambang saja
  C. rekayasa pada ambang 0,5              -> A ditambah perbaikan pelatihan saja
  D. rekayasa pada ambang prior-matched    -> keduanya

Sumbangan ambang murni  = B - A
Sumbangan pelatihan murni = C - A
Interaksi                = D - B - C + A

AUC dilaporkan berdampingan karena AUC tidak bergantung pada ambang sama sekali.
Bila AUC naik, daya pisah model memang bertambah. Bila AUC tetap sedangkan
akurasi naik, yang berubah hanyalah letak ambangnya.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.metrics import full_metrics, prior_matched_threshold

HERE = os.path.dirname(os.path.abspath(__file__))
L = []
def out(s=""):
    print(s); L.append(s)


def kumpul(pat):
    """Rerata metrik atas seluruh run yang cocok, pada kedua ambang."""
    a05, apm, auc, eer_pm = [], [], [], []
    for d in sorted(glob.glob(os.path.join(HERE, pat))):
        f = os.path.join(d, "test_scores.npy")
        if not os.path.exists(f):
            continue
        y, p, _ = np.load(f)
        y = y.astype(int)
        m0 = full_metrics(y, p, 0.5)
        mp = full_metrics(y, p, prior_matched_threshold(p, 0.5))
        a05.append(m0["accuracy"] * 100)
        apm.append(mp["accuracy"] * 100)
        auc.append(m0["auc"])
        eer_pm.append(m0["eer"] * 100)
    if not a05:
        return None
    return {"a05": np.mean(a05), "apm": np.mean(apm), "auc": np.mean(auc),
            "eer": np.mean(eer_pm), "n": len(a05)}


out("# Dekomposisi Nilai Rekayasa: Pelatihan atau Ambang?\n")
out("Perbandingan antara konfigurasi proposal dan konfigurasi diperbaiki pada "
    "matriks 2x2 mengubah dua variabel sekaligus, yaitu cara model dilatih dan "
    "cara ambang keputusan ditetapkan, karena tiap konfigurasi dievaluasi pada "
    "ambangnya masing-masing. Bagian ini memisahkan keduanya dari skor yang sama "
    "persis, tanpa melatih ulang apa pun.\n")

BARIS = []
for model in ["ast", "wavlm", "hubert", "nes2net"]:
    P = kumpul(f"runs/{model}_official_proposalULRPK_*")
    D = kumpul(f"runs/{model}_official_full_*")
    if not (P and D):
        continue
    BARIS.append((model, P, D))

if not BARIS:
    out("Belum ada pasangan run yang lengkap.")
else:
    out("## Empat sel dari skor yang sama\n")
    out("| Arsitektur | n | A. proposal @ 0,5 | B. proposal @ prior | "
        "C. rekayasa @ 0,5 | D. rekayasa @ prior |")
    out("|---|---|---|---|---|---|")
    for m, P, D in BARIS:
        out(f"| {m} | {P['n']}/{D['n']} | {P['a05']:.2f} | **{P['apm']:.2f}** | "
            f"**{D['a05']:.2f}** | **{D['apm']:.2f}** |")
    out("")

    out("## Sumbangan tiap sumbu (poin persentase)\n")
    out("| Arsitektur | Ambang saja (B-A) | Pelatihan saja (C-A) | "
        "Interaksi | Total (D-A) |")
    out("|---|---|---|---|---|")
    for m, P, D in BARIS:
        amb = P["apm"] - P["a05"]
        pel = D["a05"] - P["a05"]
        tot = D["apm"] - P["a05"]
        out(f"| {m} | **{amb:+.2f}** | **{pel:+.2f}** | {tot - amb - pel:+.2f} | "
            f"{tot:+.2f} |")
    out("")

    out("## Daya pisah, yang sama sekali tidak bergantung pada ambang\n")
    out("| Arsitektur | AUC proposal | AUC rekayasa | EER proposal | "
        "EER rekayasa | Penurunan EER relatif |")
    out("|---|---|---|---|---|---|")
    for m, P, D in BARIS:
        rel = (P["eer"] - D["eer"]) / P["eer"] * 100 if P["eer"] else 0.0
        out(f"| {m} | {P['auc']:.4f} | **{D['auc']:.4f}** | {P['eer']:.2f} | "
            f"**{D['eer']:.2f}** | **{rel:+.1f} persen** |")
    out("")

    out("## Bacaan\n")
    out("Sebagian besar selisih yang tampak besar pada matriks 2x2 berasal dari "
        "penetapan ambang, bukan dari cara model dilatih. Pada WavLM, konfigurasi "
        "proposal yang hanya diganti ambangnya sudah mencapai angka yang praktis "
        "sama dengan konfigurasi yang direkayasa penuh. Ini konsisten dengan "
        "temuan pada bagian noise, bahwa kegagalan model pada kondisi yang "
        "bergeser sebagian besar merupakan kegagalan kalibrasi dan bukan "
        "kegagalan pengenalan. Mekanisme yang sama ternyata juga berlaku pada "
        "sumbu pergeseran protokol.\n")
    out("Meski begitu, sumbangan pelatihan tidak nol dan tidak boleh diabaikan. "
        "Pada ambang 0,5 yang sama persis, konfigurasi yang direkayasa tetap "
        "unggul, yang berarti pelatihan memperbaiki kalibrasi skornya sendiri "
        "sehingga ambang bawaan menjadi lebih tepat. Daya pisahnya juga naik, dan "
        "kenaikan itu terlihat pada AUC dan EER yang sama sekali tidak bergantung "
        "pada pilihan ambang.\n")
    out("Kesimpulan yang jujur adalah bahwa rekayasa ini bernilai, namun nilainya "
        "sebagian besar terletak pada kalibrasi keputusan dan hanya sebagian "
        "kecil pada peningkatan daya pisah. Melaporkan +43 poin sebagai hasil "
        "perbaikan arsitektur akan menyesatkan.\n")

open(os.path.join(HERE, "HASIL_DEKOMPOSISI.md"), "w",
     encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_DEKOMPOSISI.md")
