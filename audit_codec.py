"""
Menghitung kebocoran provenance codec langsung dari manifest.

Angka 90,7 persen dan 0 persen yang dikutip sebagai temuan kedua penelitian ini
sebelumnya hanya ada di dalam teks, tanpa berkas yang menghasilkannya. Skrip ini
menutup celah itu, sehingga angka tersebut memiliki sumber yang dapat dijalankan
ulang seperti angka lainnya.

Perhitungan ini tidak melibatkan model, pelatihan, maupun keacakan apa pun. Ia
hanya membaca nama berkas dan label pada manifest, sehingga hasilnya tidak
memiliki ragam dan tidak memerlukan pengujian statistik.
"""
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
L = []
def out(s=""):
    print(s); L.append(s)

rows = list(csv.DictReader(open(os.path.join(HERE, "manifest.csv"),
                                encoding="utf-8")))
c = defaultdict(lambda: [0, 0])
for r in rows:
    kelas = "asli" if r["label"] == "0" else "palsu"
    c[(r["split_official"], kelas)][0] += 1
    if r["is_mp3"] in ("1", "True", "true"):
        c[(r["split_official"], kelas)][1] += 1

out("# Audit Provenance Codec pada Fake-or-Real\n")
out("Perhitungan ini membaca nama berkas dan label pada manifest secara "
    "langsung. Tidak ada model, pelatihan, maupun keacakan yang terlibat, "
    "sehingga hasilnya tidak memiliki ragam antar inisialisasi dan tidak "
    "memerlukan pengujian statistik. Inilah sebabnya temuan ini bertahan tanpa "
    "syarat sementara sebagian besar temuan lain dalam penelitian ini tidak.\n")
out("| Partisi resmi | Kelas | Total berkas | Berasal MP3 | Persen |")
out("|---|---|---|---|---|")
for k in sorted(c):
    tot, n = c[k]
    out(f"| {k[0]} | {k[1]} | {tot} | {n} | **{100 * n / tot:.1f}** |")
out("")

lt, ln = c[("training", "palsu")]
ut, un = c[("testing", "palsu")]
out(f"Pada data latih, {ln} dari {lt} sampel palsu berasal dari berkas MP3, "
    f"yaitu {100 * ln / lt:.1f} persen. Pada data uji, {un} dari {ut} sampel "
    f"palsu berasal dari MP3, yaitu {100 * un / ut:.1f} persen. Tidak ada satu "
    "pun sampel asli yang berasal dari MP3 pada partisi mana pun.\n")
out("Akibatnya, sebuah model yang belajar mengenali jejak kompresi akan "
    "mencapai akurasi tinggi pada data latih dan validasi, lalu kehilangan "
    "seluruh isyarat itu pada data uji. Isyarat yang dipelajari bukan jejak "
    "sintesis melainkan riwayat berkas, dan riwayat itu berkorelasi dengan "
    "label hanya pada sebagian partisi.\n")

open(os.path.join(HERE, "audit_codec_report.md"), "w",
     encoding="utf-8").write("\n".join(L))
print("\n-> audit_codec_report.md")
