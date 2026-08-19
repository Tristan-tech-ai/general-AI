"""
Penyiapan bersama untuk keempat notebook proses di Colab.

Notebook proses dibaca orang, bukan hanya dijalankan. Kalau seluruh penyiapan
dataset ditulis ulang di dalam tiap notebook, pembacanya harus melewati seratus
lima puluh baris urusan unduh dan ekstrak sebelum sampai ke bagian yang benar
benar dijelaskan. Karena itu bagian itu dipindahkan ke sini, dan notebook hanya
memanggilnya satu baris.

Isinya sama persis dengan fase 0 sampai 6 pada Colab_JalurB_Otomatis.ipynb,
termasuk urutan prioritas sumber arsip dan pemeriksaan sidik jari, supaya
angka yang keluar dari notebook proses sebanding dengan angka jalur otomatis.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

AKAR = "/content/general-ai"
REPO = "https://github.com/Tristan-tech-ai/general-AI.git"
TUJUAN = "data/for-2seconds"


# --------------------------------------------------------------------------
# Utilitas
# --------------------------------------------------------------------------
def judul(teks):
    print(f"\n{'=' * 68}\n  {teks}\n{'=' * 68}")


def berhenti(pesan):
    print("\n" + "!" * 68)
    print("  BERHENTI")
    print("!" * 68)
    print(pesan)
    raise SystemExit(1)


def jalankan(perintah, cwd=None, tampilkan=True):
    """Alirkan keluaran baris demi baris supaya kemajuan terlihat saat berjalan."""
    p = subprocess.Popen(perintah, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", bufsize=1)
    akhir = ""
    for baris in p.stdout:
        if tampilkan:
            print(baris, end="")
        if baris.strip():
            akhir = baris.strip()
    p.wait()
    return p.returncode, akhir


def _sha256(path, blok=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(blok), b""):
            h.update(b)
    return h.hexdigest()


def _cari_akar(mulai):
    for dp, _, _ in os.walk(mulai):
        if (os.path.isdir(os.path.join(dp, "training", "real"))
                and os.path.isdir(os.path.join(dp, "testing", "fake"))):
            return dp
    return None


# --------------------------------------------------------------------------
# Langkah 1: runtime, kode, dependensi
# --------------------------------------------------------------------------
def periksa_runtime():
    judul("Memeriksa runtime")
    try:
        import torch
    except ImportError:
        berhenti("PyTorch tidak ada. Runtime Colab tampaknya tidak standar.")

    if not torch.cuda.is_available():
        berhenti(
            "GPU tidak aktif, sedangkan pelatihan memerlukannya.\n\n"
            "Perbaikannya:\n"
            "  1. Menu Runtime -> Change runtime type\n"
            "  2. Pilih T4 GPU, lalu Save\n"
            "  3. Runtime akan restart dan seluruh isinya terhapus\n"
            "  4. Jalankan sel ini lagi dari awal")

    cc = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"kemampuan: compute capability {cc[0]}.{cc[1]}, {vram:.1f} GB")
    print(f"presisi  : {'bfloat16 asli' if cc[0] >= 8 else 'float16, bf16 hanya emulasi di kartu ini'}")
    print(f"CPU      : {os.cpu_count()} core")
    _, _, bebas = shutil.disk_usage("/content")
    print(f"disk     : {bebas / 1e9:.0f} GB bebas")
    if bebas < 8e9:
        berhenti("Ruang disk kurang dari 8 GB. Dataset perlu sekitar 3 GB.")


def ambil_kode():
    judul("Mengambil kode penelitian")
    if os.path.exists(os.path.join(AKAR, ".git")):
        print("sudah ada, memperbarui ...")
        jalankan(["git", "pull", "--quiet"], cwd=AKAR)
    else:
        shutil.rmtree(AKAR, ignore_errors=True)
        kode, _ = jalankan(["git", "clone", "--quiet", REPO, AKAR])
        if kode != 0:
            berhenti("Klon gagal. Periksa koneksi internet Colab.")
    os.chdir(AKAR)
    if AKAR not in sys.path:
        sys.path.insert(0, AKAR)
    jalankan(["git", "log", "--oneline", "-1"])


def pasang_dependensi(model):
    judul("Memasang dependensi")
    perlu = []
    for modul, paket in [("soundfile", "soundfile")]:
        try:
            __import__(modul)
        except ImportError:
            perlu.append(paket)
    if model not in ("cnn_asp", "cnnlstm"):
        try:
            import transformers
            print(f"transformers {transformers.__version__} sudah ada")
        except ImportError:
            perlu.append("transformers")
    if perlu:
        print("memasang:", ", ".join(perlu))
        jalankan([sys.executable, "-m", "pip", "install", "--quiet", *perlu],
                 tampilkan=False)
    print("dependensi siap")

    kode, _ = jalankan([sys.executable, "cek_konfigurasi.py"])
    if kode != 0:
        berhenti("Pengaman konfigurasi menolak melanjutkan.")


def siapkan(model):
    """Gabungan langkah 1: runtime, kode, dependensi, pengaman konfigurasi."""
    periksa_runtime()
    ambil_kode()
    pasang_dependensi(model)


# --------------------------------------------------------------------------
# Langkah 2: dataset
# --------------------------------------------------------------------------
def siapkan_dataset(buat_cache=True):
    judul("Menyiapkan dataset")

    acuan = json.load(open("dataset_acuan.json", encoding="utf-8"))
    sha_harap = acuan["arsip"]["sha256"]
    bytes_harap = acuan["arsip"]["bytes"]

    if os.path.isdir(os.path.join(TUJUAN, "training", "real")):
        print("dataset sudah terpasang di runtime ini, penyiapan dilewati")
    else:
        print("Meminta izin akses Google Drive.")
        print("Bila muncul jendela permintaan izin, klik Connect to Google Drive.\n")
        from google.colab import drive
        drive.mount("/content/drive")
        DRIVE = "/content/drive/MyDrive"

        cache = f"{DRIVE}/dataset-for/for-2seconds.tar"
        kandidat = [
            ("cache ringkas hasil sesi sebelumnya", cache, "tar"),
            ("arsip penelitian yang sudah ada di Drive ini",
             f"{DRIVE}/PenelitianAudioDeepfake/Dataset/FoR.zip", "zip"),
            ("simpanan arsip resmi", f"{DRIVE}/dataset-for/for-2sec.tar.gz", "targz"),
        ]
        arsip = jenis = asal = None
        for nama, path, j in kandidat:
            if os.path.exists(path):
                arsip, jenis, asal = path, j, nama
                print(f"memakai {nama}")
                print(f"  {path}")
                print(f"  {os.path.getsize(path) / 1e9:.2f} GB, tidak ada yang diunduh")
                break
        if arsip is None:
            print("tidak ada arsip di Drive, mengunduh dari York University ...")
            os.makedirs(f"{DRIVE}/dataset-for", exist_ok=True)
            arsip = f"{DRIVE}/dataset-for/for-2sec.tar.gz"
            jenis, asal = "targz", "unduhan baru"
            kode, _ = jalankan(["wget", "-c", "--no-verbose", "--show-progress",
                                "-O", arsip, acuan["sumber"]])
            if kode != 0:
                berhenti("Unduhan gagal. Jalankan sel ini lagi, wget melanjutkan "
                         "dari bagian yang sudah terunduh.")

        if jenis == "targz":
            print("\nmemeriksa sha256 arsip ...")
            besar, sha = os.path.getsize(arsip), _sha256(arsip)
            print(f"  ukuran : {besar}  (acuan {bytes_harap})")
            print(f"  sha256 : {sha}")
            if besar != bytes_harap or sha != sha_harap:
                berhenti("Arsip berbeda dari acuan. Hapus berkas itu lalu ulangi.")
            print("  arsip sama dengan yang dipakai di mesin lokal")
        else:
            print("\nsha256 arsip dilewati, wadahnya bukan tar.gz resmi. Kesamaan")
            print("isinya diperiksa lewat sidik pohon di langkah berikutnya, yang")
            print("justru lebih dalam karena membandingkan tiap berkas wav.")

        print("\nmengekstrak ...")
        tmp = "data/_arsip"
        os.makedirs("data", exist_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)

        if jenis in ("targz", "tar"):
            bendera = "-xzf" if jenis == "targz" else "-xf"
            jalankan(["tar", bendera, arsip, "-C", "data"], tampilkan=False)
            sumber = _cari_akar("data")
        else:
            # Arsip penelitian memuat seluruh rilis FoR. Mengekstrak semuanya
            # berarti menulis belasan gigabyte untuk mengambil satu koma satu.
            with zipfile.ZipFile(arsip) as z:
                anggota = z.namelist()
                prefiks = None
                for n in anggota:
                    p = n.replace("\\", "/")
                    i = p.find("for-2seconds/")
                    if i >= 0 and "/training/real/" in p[i:]:
                        prefiks = p[: i + len("for-2seconds/")]
                        break
                if prefiks is None:
                    berhenti("Folder for-2seconds tidak ditemukan di dalam zip. "
                             f"Contoh isi: {anggota[:5]}")
                pilih = [n for n in anggota
                         if n.replace("\\", "/").startswith(prefiks)]
                byte_pilih = sum(z.getinfo(n).file_size for n in pilih)
                print(f"  prefiks   : {prefiks}")
                print(f"  diambil   : {len(pilih)} dari {len(anggota)} entri")
                print(f"  ukuran    : {byte_pilih / 1e9:.2f} GB dari "
                      f"{os.path.getsize(arsip) / 1e9:.2f} GB total")
                print("  sisanya varian FoR lain yang tidak dipakai penelitian ini\n")
                for i, n in enumerate(pilih, 1):
                    z.extract(n, tmp)
                    if i % 2500 == 0:
                        print(f"    {i}/{len(pilih)} berkas ...")
            sumber = _cari_akar(tmp)

        if not sumber:
            berhenti("Struktur training/validation/testing tidak ditemukan di "
                     f"dalam arsip. Isi data/: {os.listdir('data')}")
        if os.path.abspath(sumber) != os.path.abspath(TUJUAN):
            shutil.rmtree(TUJUAN, ignore_errors=True)
            shutil.move(sumber, TUJUAN)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"dataset siap, sumbernya {asal}")

        if buat_cache and jenis != "tar" and not os.path.exists(cache):
            print("\nmembuat cache ringkas untuk sesi berikutnya ...")
            print("Tanpa pemampatan, karena berkas wav hampir tidak mengecil bila")
            print("dimampatkan sedangkan waktunya bertambah banyak.")
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            kode, _ = jalankan(["tar", "-cf", cache, "-C", "data", "for-2seconds"],
                               tampilkan=False)
            if kode == 0:
                print(f"  tersimpan: {cache}  ({os.path.getsize(cache) / 1e9:.2f} GB)")
                print("  sesi berikutnya siap dalam sekitar satu menit")
            else:
                print("  gagal membuat cache, tidak apa apa, ini hanya mempercepat")

    total = 0
    print()
    for s in ["training", "validation", "testing"]:
        for c in ["real", "fake"]:
            n = len(os.listdir(f"{TUJUAN}/{s}/{c}"))
            total += n
            print(f"  {s:<11} {c:<5} {n:>6} berkas")
    print(f"  {'TOTAL':<17} {total:>6} berkas   (seharusnya 17.870)")
    if total != 17870:
        berhenti(f"Jumlah berkas {total}, seharusnya 17.870. Datasetnya tidak utuh.")


def verifikasi_dan_manifest():
    judul("Memverifikasi dataset dan membangun manifest")
    kode, _ = jalankan([sys.executable, "cek_dataset.py"])
    if kode != 0:
        berhenti("Dataset BERBEDA dari acuan.\n\n"
                 "Jangan melanjutkan. Angka yang dihasilkan dari dataset yang "
                 "berbeda tidak boleh dibandingkan dengan angka di repositori.")

    if os.path.exists("manifest.csv"):
        os.remove("manifest.csv")
    from forlib.data import build_manifest
    baris = build_manifest(TUJUAN, "manifest.csv")
    print(f"\nmanifest dibangun ulang: {len(baris)} baris")
    print(f"contoh path: {baris[0]['path']}")

    kode, _ = jalankan([sys.executable, "colab/colab_patch.py", "--apply"])
    if kode != 0:
        berhenti("Patch presisi gagal dipasang. Baca keluarannya di atas.")
    return baris


def simpan_ke_drive(pola=("runs_colab/*/results.json", "runs_colab/*/test_scores.npy")):
    import glob
    judul("Menyimpan hasil ke Drive")
    from google.colab import drive
    drive.mount("/content/drive")
    tujuan = "/content/drive/MyDrive/general-ai-hasil-colab"
    n = 0
    for p in pola:
        for src in glob.glob(p):
            dst = os.path.join(tujuan, src)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
    print(f"{n} berkas tersimpan di {tujuan}")
    print("Bobot model tidak ikut disalin karena ukurannya bisa lebih dari satu")
    print("gigabyte, sedangkan seluruh analisis hanya memerlukan skornya.")
