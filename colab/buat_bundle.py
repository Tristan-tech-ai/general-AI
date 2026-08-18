"""
Membangun colab_bundle.zip, berkas yang diunggah ke Google Colab.

Isinya diambil dari **commit terakhir**, bukan dari working tree. Alasannya
disengaja: working tree dapat memuat berkas hasil yang sedang tidak sinkron,
sedangkan Colab lebih berguna bila menjadi ruang bersih yang dapat dibandingkan
dengan keadaan yang sudah tercatat di riwayat.

Dua berkas ditambahkan di luar git karena keduanya ada di .gitignore tetapi
dibutuhkan:

  manifest.csv   dibaca naskah.py dan gambar_paper.py, sekitar 3 MB
  colab/         notebook, patch, dan sidik jari, belum tentu sudah di-commit

Isi zip diletakkan di akar arsip tanpa folder pembungkus, sehingga sel unggah
di notebook cukup memanggil extractall("/content/general-ai").

Pemakaian:
    py colab/buat_bundle.py
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
AKAR = os.path.dirname(HERE)
KELUAR = os.path.join(AKAR, "colab_bundle.zip")

# Berkas yang perlu ikut walaupun belum ada di commit terakhir, entah karena
# di-gitignore atau karena baru dibuat dan belum di-commit. Bila kelak sudah
# ikut ter-commit, baris di sini tidak menimbulkan duplikasi karena berkas yang
# sudah ada di arsip git dilewati.
TAMBAHAN = ["manifest.csv", "cek_dataset.py", "dataset_acuan.json"]
FOLDER_TAMBAHAN = ["colab"]
# Jangan pernah ikut: berkas besar dan berkas yang tidak dibutuhkan Colab.
LEWATI_AKHIRAN = (".zip", ".pt", ".pth", ".tar.gz")


def git(*args: str) -> bytes:
    return subprocess.run(["git", "-C", AKAR, *args],
                          capture_output=True, check=True).stdout


def main() -> int:
    komit = git("rev-parse", "--short", "HEAD").decode().strip()
    kotor = git("status", "--porcelain").decode().strip()

    print(f"commit sumber : {komit}")
    if kotor:
        n = len(kotor.splitlines())
        print(f"catatan       : working tree punya {n} berkas berubah, dan "
              f"perubahan itu TIDAK ikut ke dalam bundle")

    # Isi commit diambil lewat git archive supaya persis sama dengan HEAD.
    arsip = git("archive", "--format=tar", "HEAD")

    n_git = n_extra = 0
    with zipfile.ZipFile(KELUAR, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        with tarfile.open(fileobj=io.BytesIO(arsip)) as t:
            for anggota in t.getmembers():
                if not anggota.isfile():
                    continue
                if anggota.name.endswith(LEWATI_AKHIRAN):
                    continue
                data = t.extractfile(anggota)
                if data is None:
                    continue
                z.writestr(anggota.name, data.read())
                n_git += 1

        sudah = set(z.namelist())
        for rel in TAMBAHAN:
            p = os.path.join(AKAR, rel)
            if os.path.exists(p) and rel not in sudah:
                z.write(p, rel)
                n_extra += 1
            elif not os.path.exists(p):
                print(f"PERINGATAN    : {rel} tidak ada, bundle tetap dibuat "
                      f"tetapi naskah.py akan gagal di Colab")

        for folder in FOLDER_TAMBAHAN:
            for akar, _, berkas in os.walk(os.path.join(AKAR, folder)):
                for b in berkas:
                    p = os.path.join(akar, b)
                    rel = os.path.relpath(p, AKAR).replace(os.sep, "/")
                    if rel.endswith(LEWATI_AKHIRAN) or rel in sudah:
                        continue
                    if "__pycache__" in rel:
                        continue
                    z.write(p, rel)
                    n_extra += 1

    besar = os.path.getsize(KELUAR)
    print(f"dari git      : {n_git} berkas")
    print(f"ditambahkan   : {n_extra} berkas (manifest.csv dan colab/)")
    print(f"hasil         : {os.path.relpath(KELUAR, AKAR)}  "
          f"{besar / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
