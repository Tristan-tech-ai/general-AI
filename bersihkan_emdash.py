"""
Mengganti tanda pisah em pada seluruh dokumen Markdown.

Penggantian buta menjadi koma menghasilkan kalimat yang janggal, karena tanda
pisah em dipakai untuk beberapa keperluan yang berbeda. Skrip ini membedakan
keperluan tersebut menurut konteksnya:

  angka em angka        rentang, diganti menjadi "sampai"
  awal baris            penanda daftar, diganti menjadi tanda hubung
  spasi em spasi        sisipan penjelas, diganti menjadi koma
  tanpa spasi           kata majemuk, diganti menjadi tanda hubung

Setelah penggantian, koma ganda dan spasi ganda yang mungkin muncul dirapikan.
Jumlah penggantian per berkas dilaporkan supaya perubahannya dapat ditelusuri.
"""
import glob
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EM = "—"


def bersihkan(t):
    n_awal = t.count(EM)
    # 1. rentang angka, misalnya "2019 em 2020"
    t = re.sub(r"(\d)\s*" + EM + r"\s*(\d)", r"\1 sampai \2", t)
    # 2. penanda daftar di awal baris
    t = re.sub(r"(?m)^(\s*)" + EM + r"\s+", r"\1- ", t)
    # 3. sisipan penjelas dengan spasi di kedua sisi
    t = re.sub(r"\s+" + EM + r"\s+", ", ", t)
    # 4. sisa: kata majemuk atau tempelan, jadi tanda hubung
    t = t.replace(EM, "-")
    # 5. rapikan akibat penggantian
    t = re.sub(r",\s*,+", ",", t)
    t = re.sub(r",(\s*[.;:!?])", r"\1", t)
    t = re.sub(r"\(\s*,\s*", "(", t)
    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t, n_awal


def main():
    total, berubah = 0, []
    for f in sorted(glob.glob(os.path.join(HERE, "*.md"))):
        t = io.open(f, encoding="utf-8").read()
        if EM not in t:
            continue
        baru, n = bersihkan(t)
        io.open(f, "w", encoding="utf-8").write(baru)
        sisa = baru.count(EM)
        berubah.append((os.path.basename(f), n, sisa))
        total += n

    if not berubah:
        print("tidak ada tanda pisah em yang perlu diganti")
        return
    print("| berkas | diganti | sisa |")
    print("|---|---|---|")
    for nm, n, sisa in berubah:
        print(f"| {nm} | {n} | {sisa} |")
    print(f"\ntotal diganti: {total}")
    if any(s for _, _, s in berubah):
        print("PERINGATAN: masih ada sisa, periksa manual")
        sys.exit(1)


if __name__ == "__main__":
    main()
