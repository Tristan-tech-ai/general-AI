"""
Menyesuaikan train.py agar aman dijalankan di GPU Google Colab.

Ada satu hal di train.py yang benar di mesin lokal tetapi salah di Colab.
Barisnya berbunyi:

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

RTX 5060 Ti berarsitektur Blackwell dan mendukung bfloat16 secara asli. GPU
yang paling sering diberikan Colab pada tingkat gratis adalah Tesla T4, yang
berarsitektur Turing dan TIDAK memiliki jalur bfloat16 di perangkat kerasnya.
PyTorch tetap menerima permintaan itu dan menjalankannya lewat emulasi,
sehingga pelatihan menjadi jauh lebih lambat tanpa satu pun pesan kesalahan.
Ini persis jenis kekeliruan yang gagal secara diam-diam, sama seperti bug
encoder beku yang diuraikan di KOREKSI_REPLIKASI_PROPOSAL.md.

Skrip ini menggantinya dengan pemilihan yang membaca kemampuan kartu:

    compute capability >= 8.0  ->  bfloat16   (Ampere, Ada, Hopper, Blackwell)
    compute capability <  8.0  ->  float16    (Turing, Volta, Pascal)
    tanpa CUDA                 ->  float32

Karena float16 memiliki rentang eksponen yang jauh lebih sempit daripada
bfloat16, gradiennya dapat meluruh menjadi nol. Karena itu GradScaler ikut
dipasang. Pada bfloat16 dan float32 scaler tersebut dinonaktifkan sehingga
tidak mengubah perhitungan sama sekali, dan urutan operasinya tetap sama.

Pemakaian:
    python colab/colab_patch.py --apply     pasang
    python colab/colab_patch.py --revert    kembalikan dari cadangan
    python colab/colab_patch.py --status    lihat keadaan sekarang

Salinan asli disimpan sebagai train.py.sebelum_colab. Berkas itu tidak
dilacak git, jadi `git checkout train.py` juga mengembalikan keadaan semula.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AKAR = os.path.dirname(HERE)
TARGET = os.path.join(AKAR, "train.py")
CADANGAN = TARGET + ".sebelum_colab"

PENANDA = "# --- colab_patch: presisi menurut kemampuan GPU ---"

LAMA_AMP = (
    '    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32\n'
)

BARU_AMP = (
    "    " + PENANDA + "\n"
    "    if device.type != \"cuda\":\n"
    "        amp_dtype = torch.float32\n"
    "    elif torch.cuda.get_device_capability(0)[0] >= 8:\n"
    "        amp_dtype = torch.bfloat16      # Ampere ke atas: bf16 ada di perangkat keras\n"
    "    else:\n"
    "        amp_dtype = torch.float16       # T4 dan sejenisnya: bf16 hanya emulasi\n"
    "    try:\n"
    "        _skala = torch.amp.GradScaler(\"cuda\",\n"
    "                                      enabled=(amp_dtype == torch.float16))\n"
    "    except (AttributeError, TypeError):                 # PyTorch lama\n"
    "        _skala = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))\n"
    "    print(f\"[colab] presisi {amp_dtype}, GradScaler aktif: {_skala.is_enabled()}\")\n"
)

LAMA_BACKWARD = (
    "            opt.zero_grad(set_to_none=True)\n"
    "            loss.backward()\n"
    "            torch.nn.utils.clip_grad_norm_(\n"
    "                [p for p in model.parameters() if p.requires_grad], 1.0)\n"
    "            opt.step()\n"
)

BARU_BACKWARD = (
    "            opt.zero_grad(set_to_none=True)\n"
    "            # colab_patch: scaler nonaktif pada bf16/fp32 sehingga identik\n"
    "            # dengan perilaku semula\n"
    "            _skala.scale(loss).backward()\n"
    "            _skala.unscale_(opt)\n"
    "            torch.nn.utils.clip_grad_norm_(\n"
    "                [p for p in model.parameters() if p.requires_grad], 1.0)\n"
    "            _skala.step(opt)\n"
    "            _skala.update()\n"
)

GANTI = [("pemilihan presisi", LAMA_AMP, BARU_AMP),
         ("langkah backward", LAMA_BACKWARD, BARU_BACKWARD)]


def _baca() -> str:
    with open(TARGET, encoding="utf-8") as f:
        return f.read()


def _tulis(isi: str) -> None:
    with open(TARGET, "w", encoding="utf-8", newline="\n") as f:
        f.write(isi)


def status() -> bool:
    """True bila train.py sudah dipatch."""
    ada = PENANDA in _baca()
    print("train.py :", "SUDAH dipatch" if ada else "masih asli")
    print("cadangan :", CADANGAN if os.path.exists(CADANGAN) else "belum ada")
    return ada


def apply() -> int:
    isi = _baca()
    if PENANDA in isi:
        print("Sudah dipatch, tidak ada yang diubah.")
        return 0

    # Menolak lebih dulu bila salah satu potongan tidak ditemukan, supaya
    # train.py tidak pernah tertinggal dalam keadaan setengah berubah.
    for nama, lama, _ in GANTI:
        n = isi.count(lama)
        if n != 1:
            print(f"GAGAL. Potongan '{nama}' ditemukan {n} kali, seharusnya tepat "
                  f"satu kali.\ntrain.py tampaknya sudah berubah sejak skrip ini "
                  f"ditulis. Tidak ada yang diubah.")
            return 1

    if not os.path.exists(CADANGAN):
        shutil.copy2(TARGET, CADANGAN)
        print("cadangan dibuat:", os.path.basename(CADANGAN))

    for nama, lama, baru in GANTI:
        isi = isi.replace(lama, baru)
        print("  diganti:", nama)
    _tulis(isi)
    print("Selesai. Kembalikan dengan: python colab/colab_patch.py --revert")
    return 0


def revert() -> int:
    if not os.path.exists(CADANGAN):
        print("Tidak ada cadangan. Pakai: git checkout train.py")
        return 1
    shutil.copy2(CADANGAN, TARGET)
    os.remove(CADANGAN)
    print("train.py dikembalikan dari cadangan.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    g.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(TARGET):
        print("train.py tidak ditemukan di", AKAR)
        return 1
    if a.status:
        status()
        return 0
    return apply() if a.apply else revert()


if __name__ == "__main__":
    sys.exit(main())
