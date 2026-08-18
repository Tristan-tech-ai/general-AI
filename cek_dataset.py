"""
Memastikan dataset yang dipakai di mesin lain benar-benar dataset yang sama.

Seluruh angka dalam penelitian ini dihitung dari 17.870 berkas audio yang
diunduh dari York University. Ketika pekerjaan berpindah ke Google Colab, atau
ke komputer lain, berkas itu diunduh ulang. Tidak ada yang menjamin unduhan
kedua identik dengan yang pertama: unduhan dapat terpotong lalu dilanjutkan
dengan cara yang salah, cermin penyimpanan dapat berbeda, dan penerbitnya dapat
memperbarui arsipnya tanpa mengubah namanya.

Skrip ini menutup celah itu dengan dua sidik jari yang saling melengkapi.

  ARSIP  sha256 atas berkas for-2sec.tar.gz apa adanya. Pemeriksaan ini
         memakan beberapa detik dan sudah cukup untuk hampir semua keperluan.

  POHON  sha256 atas daftar terurut berisi jalur relatif dan sha256 tiap berkas
         wav hasil ekstrak. Jalurnya dinormalkan ke bentuk POSIX dan relatif
         terhadap akar dataset, sehingga sidik jarinya sama di Windows maupun
         di Linux. Pemeriksaan ini memakan satu sampai dua menit dan menangkap
         hal yang tidak tertangkap sidik arsip, misalnya berkas yang rusak
         setelah ekstraksi atau folder yang tanpa sengaja tercampur.

Acuannya disimpan di dataset_acuan.json dan ikut ter-commit, sehingga mesin
mana pun dapat membuktikan bahwa datanya sama tanpa perlu mengunduh ulang.

Pemakaian:
    py cek_dataset.py                 verifikasi terhadap acuan
    py cek_dataset.py --hanya-arsip   hanya sidik arsip, beberapa detik
    py cek_dataset.py --tulis         tulis acuan baru dari mesin ini
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ACUAN = os.path.join(HERE, "dataset_acuan.json")

# Arsip dicari di beberapa tempat karena letaknya berbeda antara mesin lokal
# dan Colab.
CALON_ARSIP = ["for-2sec.tar.gz", "data/for-2sec.tar.gz"]
AKAR_POHON = os.path.join("data", "for-2seconds")
PARTISI = [(s, c) for s in ("training", "validation", "testing")
           for c in ("real", "fake")]


def sha256_berkas(path: str, blok: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(blok)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def cari_arsip() -> str | None:
    for rel in CALON_ARSIP:
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            return p
    return None


def sidik_arsip(path: str) -> dict:
    besar = os.path.getsize(path)
    print(f"  membaca {os.path.basename(path)} ({besar / 1e9:.2f} GB) ...")
    return {"nama": os.path.basename(path), "bytes": besar,
            "sha256": sha256_berkas(path)}


def sidik_pohon(akar: str) -> dict:
    """Sidik jari yang tidak bergantung pada sistem berkas maupun urutan baca."""
    baris, total = [], 0
    berkas = []
    for dirpath, _, names in os.walk(akar):
        for n in names:
            if n.lower().endswith(".wav"):
                berkas.append(os.path.join(dirpath, n))
    berkas.sort()
    for i, p in enumerate(berkas, 1):
        rel = os.path.relpath(p, akar).replace(os.sep, "/")
        total += os.path.getsize(p)
        baris.append(f"{rel} {sha256_berkas(p)}")
        if i % 2500 == 0:
            print(f"    {i}/{len(berkas)} berkas ...")
    baris.sort()
    gabung = "\n".join(baris).encode()
    return {"akar": AKAR_POHON.replace(os.sep, "/"), "berkas": len(berkas),
            "bytes": total, "sidik": hashlib.sha256(gabung).hexdigest()}


def hitung_partisi(akar: str) -> dict:
    out = {}
    for s, c in PARTISI:
        d = os.path.join(akar, s, c)
        out[f"{s}/{c}"] = len(os.listdir(d)) if os.path.isdir(d) else 0
    return out


def bandingkan(nama: str, acuan, sekarang) -> bool:
    cocok = acuan == sekarang
    tanda = "COCOK  " if cocok else "BEDA   "
    print(f"  {tanda} {nama}")
    if not cocok:
        print(f"           acuan   : {acuan}")
        print(f"           sekarang: {sekarang}")
    return cocok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tulis", action="store_true",
                    help="tulis acuan baru dari keadaan mesin ini")
    ap.add_argument("--hanya-arsip", action="store_true",
                    help="lewati sidik pohon yang memakan waktu")
    a = ap.parse_args()

    arsip = cari_arsip()
    akar = os.path.join(HERE, AKAR_POHON)
    ada_pohon = os.path.isdir(akar)

    if not arsip and not ada_pohon:
        print("Tidak ada arsip maupun folder dataset. Tidak ada yang diperiksa.")
        print("Dicari di:", ", ".join(CALON_ARSIP + [AKAR_POHON.replace(os.sep, "/")]))
        return 1

    hasil: dict = {}
    if arsip:
        print("sidik arsip:")
        hasil["arsip"] = sidik_arsip(arsip)
        print(f"  sha256 {hasil['arsip']['sha256']}")
    else:
        print("arsip tidak ada di mesin ini, dilewati")

    if ada_pohon and not a.hanya_arsip:
        print("sidik pohon:")
        hasil["pohon"] = sidik_pohon(akar)
        hasil["partisi"] = hitung_partisi(akar)
        print(f"  {hasil['pohon']['berkas']} berkas, "
              f"{hasil['pohon']['bytes'] / 1e9:.2f} GB")
        print(f"  sidik  {hasil['pohon']['sidik']}")
    elif ada_pohon:
        hasil["partisi"] = hitung_partisi(akar)
        print("sidik pohon dilewati atas permintaan")
    else:
        print("folder dataset belum ada, sidik pohon dilewati")

    if a.tulis:
        lama = {}
        if os.path.exists(ACUAN):
            lama = json.load(open(ACUAN, encoding="utf-8"))
        lama.update(hasil)
        lama["sumber"] = ("https://bil.eecs.yorku.ca/share/for-2sec.tar.gz")
        with open(ACUAN, "w", encoding="utf-8", newline="\n") as f:
            json.dump(lama, f, indent=1, ensure_ascii=False)
            f.write("\n")
        print(f"\nacuan ditulis ke {os.path.basename(ACUAN)}")
        return 0

    if not os.path.exists(ACUAN):
        print(f"\n{os.path.basename(ACUAN)} belum ada. "
              f"Buat sekali di mesin yang datasetnya dipercaya:")
        print("    py cek_dataset.py --tulis")
        return 1

    ref = json.load(open(ACUAN, encoding="utf-8"))
    print("\nverifikasi terhadap acuan:")
    ok = True
    if "arsip" in hasil and "arsip" in ref:
        ok &= bandingkan("sha256 arsip", ref["arsip"]["sha256"],
                         hasil["arsip"]["sha256"])
        ok &= bandingkan("ukuran arsip", ref["arsip"]["bytes"],
                         hasil["arsip"]["bytes"])
    if "pohon" in hasil and "pohon" in ref:
        ok &= bandingkan("sidik pohon", ref["pohon"]["sidik"],
                         hasil["pohon"]["sidik"])
        ok &= bandingkan("jumlah berkas", ref["pohon"]["berkas"],
                         hasil["pohon"]["berkas"])
    if "partisi" in hasil and "partisi" in ref:
        ok &= bandingkan("jumlah per partisi", ref["partisi"], hasil["partisi"])

    print()
    if ok:
        print("DATASET SAMA. Angka yang dihasilkan di sini sebanding dengan "
              "angka di repositori.")
        return 0
    print("DATASET BERBEDA. Jangan bandingkan angkanya dengan hasil di "
          "repositori sebelum sebabnya diketahui.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
