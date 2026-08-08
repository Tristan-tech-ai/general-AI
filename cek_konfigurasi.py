"""
Pengaman: memeriksa apakah ada kelompok run yang memuat lebih dari satu
konfigurasi pelatihan.

Tujuh kekeliruan dalam penelitian ini berasal dari pola yang sama, yaitu asumsi
yang benar ketika ditulis lalu menjadi salah karena data baru masuk, dan
seluruhnya luput karena tidak ada yang memberi tahu bahwa asumsinya berubah.
Yang paling merusak adalah dua run yang dibandingkan sebagai pasangan terkontrol
padahal jumlah epoch-nya berbeda enam kali lipat.

Skrip ini menutup celah itu secara struktural, bukan dengan kewaspadaan. Ia
mengelompokkan seluruh run menurut arsitektur, skema pembagian data, dan
augmentasi, lalu memeriksa apakah satu kelompok memuat lebih dari satu pasangan
jumlah epoch dan ukuran batch. Bila ada, skrip keluar dengan status gagal
sehingga rangkaian pembuatan laporan berhenti dan kelompok tersebut harus
ditangani lebih dahulu.

Kelompok yang memang sengaja memuat beberapa konfigurasi dapat didaftarkan pada
DIKECUALIKAN beserta alasannya, sehingga pengecualian itu tercatat dan tidak
berlangsung diam-diam.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# Kelompok yang boleh memuat lebih dari satu konfigurasi, beserta alasannya.
# Setiap penambahan di sini harus disertai alasan yang dapat diperiksa.
DIKECUALIKAN = {
    ("cnn_asp", "random", "none"):
        "run enam epoch dari tahap awal sengaja disimpan sebagai catatan "
        "sejarah di samping run sepuluh epoch, lihat HASIL_TEMUAN1.md",
    ("cnn_asp", "official", "none"):
        "run satu epoch dari tahap awal sengaja disimpan sebagai catatan "
        "sejarah di samping run sepuluh epoch, lihat HASIL_TEMUAN1.md",
    ("ast", "official", "proposal"):
        "dua langkah tangga ablasi yang memang berbeda jumlah epoch, yaitu L3 "
        "dengan dua puluh epoch dan L4 dengan sepuluh epoch",
    ("cnn_asp", "wavval", "codec"):
        "run empat belas epoch dari tahap awal, tidak dipakai dalam naskah",
    ("hubert", "official", "full"):
        "dijalankan pada batch enam belas untuk pengujian lintas generasi dan "
        "batch tiga puluh dua untuk matriks perlakuan encoder",
}

POLA = re.compile(r"^(.+?)_(official|random|clean_val|wavval)_"
                  r"([a-zA-Z0-9.]+?)(?:_b\d+e\d+)?_s(\d+)$")


def main():
    grup = defaultdict(set)
    for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
        if not os.path.exists(os.path.join(d, "test_scores.npy")):
            continue
        m = POLA.match(os.path.basename(d))
        if not m:
            continue
        fj = os.path.join(d, "results.json")
        if not os.path.exists(fj):
            continue
        a = json.load(open(fj, encoding="utf-8"))["args"]
        grup[(m.group(1), m.group(2), m.group(3))].add(
            (a.get("epochs"), a.get("batch")))

    tercampur = {k: v for k, v in grup.items() if len(v) > 1}
    baru = {k: v for k, v in tercampur.items() if k not in DIKECUALIKAN}

    print(f"kelompok run diperiksa : {len(grup)}")
    print(f"memuat lebih dari satu konfigurasi : {len(tercampur)}")
    print(f"sudah didaftarkan sebagai pengecualian : "
          f"{len(tercampur) - len(baru)}")

    if tercampur:
        print("\nkelompok yang tercampur:")
        for k, v in sorted(tercampur.items()):
            tanda = "terdaftar" if k in DIKECUALIKAN else "BARU"
            print(f"  [{tanda}] {'_'.join(k)}: {sorted(v)}")
            if k in DIKECUALIKAN:
                print(f"           alasan: {DIKECUALIKAN[k]}")

    if baru:
        print("\nGAGAL. Kelompok di atas yang bertanda BARU memuat lebih dari "
              "satu konfigurasi pelatihan tanpa alasan yang tercatat.")
        print("Menggabungkannya akan melaporkan ragam antar konfigurasi sebagai "
              "ragam antar inisialisasi acak.")
        print("Perbaiki dengan menyamakan konfigurasinya, atau daftarkan pada "
              "DIKECUALIKAN di cek_konfigurasi.py beserta alasannya.")
        sys.exit(1)

    print("\nOK. Tidak ada kelompok yang tercampur tanpa alasan yang tercatat.")


if __name__ == "__main__":
    main()
