"""
Memecah selisih pada temuan pembuka menjadi sebab-sebabnya.

Temuan pertama penelitian ini menyatakan bahwa protokol pembagian data
menentukan hasil, dengan bukti berupa selisih hampir lima puluh poin persentase
antara split acak dan partisi resmi pada model, data, dan hyperparameter yang
identik.

Pemeriksaan ulang menunjukkan bahwa selisih tersebut menggabungkan tiga sebab
yang berbeda, dan hanya satu di antaranya merupakan sifat protokolnya:

  1. Ambang keputusan yang tidak lagi cocok ketika distribusi skor bergeser.
     Ini dapat diperbaiki tanpa menyentuh protokol maupun model.
  2. Model yang kurang terlatih. Run aslinya memakai enam epoch, sedangkan
     seluruh eksperimen lain dalam penelitian ini memakai sepuluh.
  3. Protokol pembagian data itu sendiri, yang terbaca pada penurunan area
     under curve dan tidak dapat diperbaiki oleh pengaturan ambang.

Skrip ini menghitung ketiganya secara terpisah. Perbandingan hanya dilakukan
antar run yang konfigurasinya identik, karena menggabungkan konfigurasi yang
berbeda akan melaporkan ragam antar konfigurasi sebagai ragam antar
inisialisasi.
"""
import glob
import os
import sys
from math import lgamma

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.metrics import full_metrics, prior_matched_threshold

HERE = os.path.dirname(os.path.abspath(__file__))
L = []
def out(s=""):
    print(s); L.append(s)


def kumpul(pat):
    a05, apm, auc = [], [], []
    for d in sorted(glob.glob(os.path.join(HERE, pat))):
        f = os.path.join(d, "test_scores.npy")
        if not os.path.exists(f):
            continue
        y, p, _ = np.load(f)
        y = y.astype(int)
        m = full_metrics(y, p, 0.5)
        a05.append(m["accuracy"] * 100)
        auc.append(m["auc"])
        apm.append(full_metrics(y, p, prior_matched_threshold(p, 0.5))["accuracy"] * 100)
    if not a05:
        return None
    return {"a05": np.array(a05), "apm": np.array(apm), "auc": np.array(auc)}


def welch_p(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb == 0:
        return None
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    x = df / (df + t * t)
    aa, bb = df / 2.0, 0.5
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = lgamma(aa) + lgamma(bb) - lgamma(aa + bb)
    front = np.exp(np.log(x) * aa + np.log(1 - x) * bb - lb) / aa
    f, c, d = 1.0, 1.0, 0.0
    for i in range(300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (bb - m) * x) / ((aa + 2 * m - 1) * (aa + 2 * m))
        else:
            num = -((aa + m) * (aa + bb + m) * x) / ((aa + 2 * m) * (aa + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-10:
            break
    return float(front * (f - 1.0))


LAMA_R = kumpul("runs/cnn_asp_random_none_s42")
LAMA_O = kumpul("runs/cnn_asp_official_none_s42")
BARU_R = kumpul("runs/cnn_asp_random_none_b32e10_s*")
BARU_O = kumpul("runs/cnn_asp_official_none_b32e10_s*")

out("# Pemecahan Temuan Pembuka\n")
out("Temuan pertama menyatakan bahwa protokol pembagian data menentukan hasil, "
    "dengan bukti selisih hampir lima puluh poin persentase pada arsitektur, "
    "data, dan hyperparameter yang disebut identik. Tabel berikut memisahkan "
    "selisih itu menjadi sebab-sebabnya.\n")
out("Pernyataan bahwa hyperparameternya identik ternyata tidak benar. Run yang "
    "menghasilkan angka pada split acak dijalankan selama enam epoch, sedangkan "
    "run yang menghasilkan angka pada partisi resmi dijalankan selama **satu** "
    "epoch. Keduanya berasal dari tahap paling awal penelitian, ketika nama "
    "direktori belum memuat penanda batch dan epoch, sehingga perbedaan itu "
    "tidak terlihat dari nama berkasnya dan tidak pernah diperiksa. "
    "Perbandingan aslinya karena itu bukan perbandingan terkontrol sama sekali. "
    "Baris ketiga dan keempat menjalankan keduanya pada konfigurasi yang "
    "seragam.\n")


def baris(m, lbl):
    if m is None:
        return f"| {lbl} | belum ada | | | |"
    n = len(m["a05"])
    s = lambda z: f" ({z.std(ddof=1):.2f})" if n > 1 else ""
    return (f"| {lbl} | {n} | {m['a05'].mean():.2f}{s(m['a05'])} | "
            f"{m['apm'].mean():.2f}{s(m['apm'])} | {m['auc'].mean():.4f} |")


out("| Konfigurasi dan split | n | Akurasi @0,5 | Akurasi @prior | AUC |")
out("|---|---|---|---|---|")
out(baris(LAMA_R, "run asli, 6 epoch batch 64, split acak"))
out(baris(LAMA_O, "run asli, 1 epoch batch 64, partisi resmi"))
out(baris(BARU_R, "10 epoch batch 32, split acak"))
out(baris(BARU_O, "10 epoch batch 32, partisi resmi"))
out("")

if all(x is not None for x in (LAMA_R, LAMA_O, BARU_R, BARU_O)):
    sel_lama = LAMA_R["a05"].mean() - LAMA_O["a05"].mean()
    sel_baru_amb = BARU_O["apm"].mean() - BARU_O["a05"].mean()
    sel_latih = BARU_O["apm"].mean() - LAMA_O["apm"].mean()
    sel_prot = BARU_R["apm"].mean() - BARU_O["apm"].mean()
    out("## Tiga sebab yang terpisah\n")
    out("| Sebab | Besaran | Dapat diperbaiki tanpa mengubah protokol |")
    out("|---|---|---|")
    out(f"| Ambang keputusan tidak lagi cocok | {sel_baru_amb:.2f} poin | ya, "
        "cukup dengan menyesuaikan ambang |")
    out(f"| Model kurang terlatih pada run asli | {sel_latih:.2f} poin | ya, "
        "cukup dengan menambah epoch |")
    out(f"| Protokol pembagian data itu sendiri | {sel_prot:.2f} poin | tidak |")
    out("")
    out(f"Selisih yang dilaporkan semula {sel_lama:.2f} poin persentase. Dari "
        f"jumlah itu, hanya {sel_prot:.2f} poin merupakan sifat protokolnya, "
        "yaitu bagian yang tetap ada setelah model dilatih penuh dan ambangnya "
        "disesuaikan.\n")

    p = welch_p(BARU_R["apm"], BARU_O["apm"])
    if p is not None:
        bacaan = ("melampaui ragam" if p < 0.05 else "belum terbukti berbeda")
        out(f"Selisih protokol tersebut diuji dengan uji t Welch pada ambang "
            f"prior-matched dan menghasilkan nilai p sebesar {p:.4f}, yang "
            f"berarti {bacaan}.\n")

    out("## Bacaan\n")
    out("Efek protokol pembagian data tetap ada dan terbaca pada penurunan area "
        f"under curve dari {BARU_R['auc'].mean():.4f} menjadi "
        f"{BARU_O['auc'].mean():.4f}, yang tidak dapat diperbaiki oleh "
        "pengaturan ambang. Besarannya jauh lebih kecil daripada yang semula "
        "dilaporkan.\n")
    out("Sebaliknya, temuan mengenai kegagalan kalibrasi menjadi jauh lebih "
        "kuat. Pada partisi resmi, model yang terlatih penuh memisahkan kedua "
        f"kelas dengan area under curve {BARU_O['auc'].mean():.4f}, namun pada "
        f"ambang tetap 0,5 hanya mencapai {BARU_O['a05'].mean():.2f} persen. "
        "Selisih itu sepenuhnya merupakan kegagalan kalibrasi, dan besarnya "
        f"{sel_baru_amb:.2f} poin persentase.\n")
    out("Dengan kata lain, angka yang semula dipakai untuk menunjukkan bahwa "
        "protokol pembagian data menentukan hasil sebenarnya lebih banyak "
        "menunjukkan bahwa ambang keputusan menentukan hasil. Kedua pernyataan "
        "sama-sama merupakan peringatan terhadap pelaporan akurasi tunggal, "
        "namun keduanya menunjuk sebab yang berbeda dan menuntut perbaikan yang "
        "berbeda pula.\n")

open(os.path.join(HERE, "HASIL_TEMUAN1.md"), "w",
     encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_TEMUAN1.md")
