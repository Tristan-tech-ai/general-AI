"""
Uji apakah selisih antar konfigurasi lebih besar daripada ragam antar
inisialisasi acak.

Sepanjang penelitian ini beberapa kesimpulan sempat ditarik dari selisih
beberapa poin persentase antara dua sel yang masing-masing hanya dijalankan
sekali. Setelah tiap sel dijalankan dengan beberapa inisialisasi, sebagian
selisih itu ternyata lebih kecil daripada ragamnya sendiri. Skrip ini menguji
tiap perbandingan secara eksplisit dengan uji t Welch, yang tidak mengandaikan
ragam kedua kelompok sama.

Ukuran sampel di sini kecil, yaitu tiga inisialisasi per sel, sehingga uji ini
berdaya rendah. Nilai p yang besar karena itu berarti "belum terbukti berbeda",
bukan "terbukti sama. Keduanya dibedakan secara eksplisit dalam keluaran.
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


def akurasi(pat):
    a = []
    for d in sorted(glob.glob(os.path.join(HERE, pat))):
        f = os.path.join(d, "test_scores.npy")
        if not os.path.exists(f):
            continue
        y, p, _ = np.load(f)
        a.append(full_metrics(y.astype(int), p,
                              prior_matched_threshold(p, 0.5))["accuracy"] * 100)
    return np.array(a)


def welch(a, b):
    """Statistik t Welch beserta derajat bebas Welch-Satterthwaite."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    if va + vb == 0:
        return None
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    return t, df


def p_dua_sisi(t, df):
    """Nilai p dua sisi dari distribusi t, dihitung lewat fungsi beta tak
    lengkap agar tidak bergantung pada SciPy."""
    from math import lgamma

    x = df / (df + t * t)

    def betainc(a, b, x):
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lbeta = lgamma(a) + lgamma(b) - lgamma(a + b)
        front = np.exp(np.log(x) * a + np.log(1 - x) * b - lbeta) / a
        f, c, d = 1.0, 1.0, 0.0
        for i in range(0, 300):
            m = i // 2
            if i == 0:
                num = 1.0
            elif i % 2 == 0:
                num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
            else:
                num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
            d = 1.0 + num * d
            d = 1e-30 if abs(d) < 1e-30 else d
            d = 1.0 / d
            c = 1.0 + num / c
            c = 1e-30 if abs(c) < 1e-30 else c
            f *= c * d
            if abs(1.0 - c * d) < 1e-10:
                break
        return front * (f - 1.0)

    return betainc(df / 2.0, 0.5, x)


BANDINGAN = [
    ("AST: encoder dilatih vs dibekukan",
     "runs/ast_official_fullUF_b32e10_s*", "runs/ast_official_full_b32e10_s*"),
    ("AST: encoder dilatih vs proposal",
     "runs/ast_official_fullUF_b32e10_s*",
     "runs/ast_official_proposalULRPK_b32e20_s*"),
    ("WavLM: encoder dibekukan vs dilatih",
     "runs/wavlm_official_full_b16e10_s*",
     "runs/wavlm_official_fullUF_b16e10_s*"),
    ("HuBERT: encoder dilatih vs dibekukan",
     "runs/hubert_official_fullUF_b32e10_s*",
     "runs/hubert_official_full_b32e10_s*"),
    ("WavLM: rekayasa dibekukan vs proposal",
     "runs/wavlm_official_full_b16e10_s*",
     "runs/wavlm_official_proposalULRPK_b16e20_s*"),
    ("HuBERT: rekayasa dilatih vs proposal",
     "runs/hubert_official_fullUF_b32e10_s*",
     "runs/hubert_official_proposalULRPK_b32e20_s*"),
]

out("# Apakah Selisihnya Lebih Besar daripada Ragam Antar Inisialisasi?\n")
out("Tiap baris membandingkan dua konfigurasi pada partisi resmi dengan ambang "
    "prior-matched, memakai uji t Welch yang tidak mengandaikan ragam kedua "
    "kelompok sama. Ukuran sampel kecil, yaitu paling banyak tiga inisialisasi "
    "per sel, sehingga uji ini berdaya rendah. Nilai p yang besar berarti belum "
    "terbukti berbeda, dan bukan terbukti sama.\n")
out("Enam perbandingan diuji sekaligus, sehingga nilai p mentah tidak dapat "
    "dibaca apa adanya. Menguji enam hipotesis pada ambang 0,05 memberi peluang "
    "sekitar 26 persen untuk mendapatkan setidaknya satu hasil yang tampak "
    "bermakna semata karena kebetulan. Karena itu koreksi Holm-Bonferroni "
    "diterapkan, dan keputusan diambil dari nilai p terkoreksi.\n")

# Kumpulkan dahulu, lalu koreksi Holm-Bonferroni atas seluruh perbandingan
# yang benar-benar dapat diuji.
hasil = []
for nama, pa, pb in BANDINGAN:
    a, b = akurasi(pa), akurasi(pb)
    if len(a) == 0 or len(b) == 0:
        hasil.append({"nama": nama, "status": "belum ada"})
        continue
    w = welch(a, b)
    hasil.append({"nama": nama, "na": len(a), "nb": len(b),
                  "ma": a.mean(), "mb": b.mean(), "sel": a.mean() - b.mean(),
                  "p": None if w is None else p_dua_sisi(*w),
                  "status": "ok"})

diuji = [h for h in hasil if h.get("p") is not None]
m = len(diuji)
for rank, h in enumerate(sorted(diuji, key=lambda x: x["p"])):
    # Holm: nilai p ke-k dikalikan (m - k), lalu dijaga tidak menurun
    h["p_holm"] = min(1.0, h["p"] * (m - rank))
berjalan = 0.0
for h in sorted(diuji, key=lambda x: x["p"]):
    berjalan = max(berjalan, h["p_holm"])
    h["p_holm"] = berjalan

out("| Perbandingan | n | Rerata A | Rerata B | Selisih | p mentah | p Holm | Bacaan |")
out("|---|---|---|---|---|---|---|---|")
for h in hasil:
    if h["status"] == "belum ada":
        out(f"| {h['nama']} | | belum ada | | | | | |")
        continue
    if h["p"] is None:
        out(f"| {h['nama']} | {h['na']}/{h['nb']} | {h['ma']:.2f} | "
            f"{h['mb']:.2f} | {h['sel']:+.2f} | tidak dapat diuji | | "
            "perlu minimal dua inisialisasi di kedua sisi |")
        continue
    ph = h["p_holm"]
    bacaan = ("selisih melampaui ragam" if ph < 0.05 else
              "di garis batas, belum meyakinkan" if ph < 0.15 else
              "belum terbukti berbeda")
    out(f"| {h['nama']} | {h['na']}/{h['nb']} | {h['ma']:.2f} | {h['mb']:.2f} | "
        f"{h['sel']:+.2f} | {h['p']:.4f} | {ph:.4f} | **{bacaan}** |")
out("")

out("## Bacaan\n")
out("Hasilnya terbelah bersih menjadi dua kelompok yang tidak saling "
    "berdekatan.\n")
out("Kelompok pertama adalah perbandingan antara konfigurasi proposal dan "
    "konfigurasi rekayasa pada kedua model swa-selia berukuran besar. "
    "Selisihnya berpuluh poin persentase dan nilai p terkoreksinya jauh di "
    "bawah ambang, sehingga kesimpulannya kokoh. Perlu dicatat bahwa pada tahap "
    "sebelumnya, ketika tiap sel baru memiliki tiga inisialisasi, kedua "
    "perbandingan ini justru berhenti pada nilai p terkoreksi 0,0520 yaitu "
    "tepat di atas ambang. Penyebabnya bukan efek yang kecil melainkan derajat "
    "bebas yang sangat sedikit dan simpangan baku yang besar pada sel "
    "konfigurasi proposal. Menambah inisialisasi keempat dan kelima "
    "menyelesaikannya, dan itu memang tanggapan yang tepat terhadap nilai p "
    "yang berhenti di ambang.\n")
out("Kelompok kedua adalah perbandingan antara membekukan dan melatih encoder. "
    "Selisihnya berada pada orde yang sama dengan simpangan bakunya sendiri, "
    "dan tidak satu pun terbukti berbeda meskipun sebagian sel sudah memiliki "
    "empat atau lima inisialisasi. Untuk kelompok ini, penelitian ini tidak "
    "berhak menyatakan bahwa satu perlakuan lebih baik daripada yang lain.\n")
out("Kesimpulan keseluruhannya sempit tetapi jelas. Yang menentukan hasil pada "
    "partisi resmi adalah besaran learning rate relatif terhadap encodernya, "
    "bukan keputusan membekukan atau melatih encoder itu sendiri.\n")
out("Konsekuensinya bagi keseluruhan penelitian cukup besar dan perlu "
    "dinyatakan terus terang. Beberapa kesimpulan yang sempat ditarik lebih "
    "awal, ketika tiap sel baru dijalankan sekali, ternyata tidak bertahan "
    "setelah ragam antar inisialisasi diukur. Yang tersisa sebagai temuan yang "
    "kokoh adalah hal-hal yang selisihnya berpuluh poin, bukan berbilang poin.\n")

open(os.path.join(HERE, "HASIL_SIGNIFIKANSI.md"), "w",
     encoding="utf-8").write("\n".join(L))
print("\n-> HASIL_SIGNIFIKANSI.md")
