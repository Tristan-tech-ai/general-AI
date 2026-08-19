"""
Membangun dua belas notebook Colab: empat model di judul proposal, dikali tiga
jenis notebook.

  Proses_<Model>.ipynb      menjelaskan proses pelatihan langkah demi langkah
  Optimizer_<Model>.ipynb   membandingkan AdamW lawan NAdam pada model itu
  Dropout_<Model>.ipynb     membandingkan dropout tetap lawan dropout adaptif

Kenapa dibangkitkan, bukan ditulis tangan dua belas kali. Sebagian besar sel
pada dua belas notebook itu identik: penyiapan runtime, penyiapan dataset,
pemanggilan train.py, dan pembacaan hasil. Kalau ditulis tangan, satu perbaikan
harus disalin ke dua belas tempat dan cepat atau lambat ada yang tertinggal.
Di sini bagian bersama ditulis sekali, bagian yang memang berbeda ditulis
terpisah, lalu keduanya digabung.

Jalankan:  py colab/buat_notebook_proses.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_URL = "https://github.com/Tristan-tech-ai/general-AI.git"
GH = "Tristan-tech-ai/general-AI"

PEMBUKA = f"""> **Tentang berkas ini.** Notebook ini bagian dari repositori penelitian
> deteksi deepfake audio di <https://github.com/{GH}>. Seluruh kode yang
> dijalankan di sini diambil langsung dari repositori itu, tidak ada yang
> ditempel ke dalam notebook, sehingga hasilnya sebanding dengan hasil yang
> dilaporkan di sana. Laporan lengkapnya ada pada `NASKAH.pdf`."""


# ==========================================================================
# Spesifikasi per model
# ==========================================================================
SPEK = {
    "wav2vec2": dict(
        nama="Wav2Vec2",
        ckpt="facebook/wav2vec2-base",
        batch=32,
        satu_kalimat=(
            "Model swa-selia yang membaca gelombang suara apa adanya, tanpa "
            "diubah jadi gambar lebih dahulu. Yang dipakai di sini versi Base, "
            "sekitar 95 juta parameter."),
        catatan_penting=(
            "Ini satu satunya model swa-selia berukuran kecil di penelitian ini. "
            "HuBERT yang dibandingkan dengannya berukuran tiga kali lipat, "
            "sehingga selisih hasil keduanya tidak boleh langsung disebut "
            "selisih arsitektur."),
        hasil_lokal="90,75 persen (simpangan 0,51 atas tiga inisialisasi)",
        berat="sedang",
    ),
    "ast": dict(
        nama="AST (Audio Spectrogram Transformer)",
        ckpt="MIT/ast-finetuned-audioset-10-10-0.4593",
        batch=32,
        satu_kalimat=(
            "Transformer yang bekerja pada gambar suara, dipra-latih untuk "
            "mengenali suara sehari hari pada AudioSet: gonggongan, pintu, "
            "musik, mesin."),
        catatan_penting=(
            "Pra-pelatihannya mengajarkan suara itu apa, bukan suara itu asli "
            "atau buatan. Dua pertanyaan yang berbeda, dan itu terlihat pada "
            "hasilnya."),
        hasil_lokal="86,43 persen (simpangan 2,94 atas tiga inisialisasi)",
        berat="sedang",
    ),
    "hubert": dict(
        nama="HuBERT",
        ckpt="facebook/hubert-large-ll60k",
        batch=16,
        satu_kalimat=(
            "Model swa-selia berukuran besar, sekitar 317 juta parameter, "
            "dilatih memprediksi unit tersembunyi pada rekaman suara bersih."),
        catatan_penting=(
            "Tertinggi di antara empat model proposal pada kondisi bersih. "
            "Tetapi begitu audionya diberi derau, model ini justru yang paling "
            "cepat jatuh dari seluruh konfigurasi yang diuji."),
        hasil_lokal="97,29 persen (simpangan 2,16 atas delapan inisialisasi)",
        berat="berat",
    ),
    "cnnlstm": dict(
        nama="CNN-LSTM",
        ckpt="tidak ada, dilatih dari nol",
        batch=32,
        satu_kalimat=(
            "Rancangan konvensional: konvolusi membaca gambar suara, lalu LSTM "
            "membacanya berurutan dari kiri ke kanan sambil mengingat."),
        catatan_penting=(
            "Satu satunya model di sini yang tidak membawa pengetahuan apa pun "
            "dari pra-pelatihan. Seluruh 6,8 juta parameternya dilatih dari nol "
            "hanya dengan 13.956 klip."),
        hasil_lokal="83,52 persen (simpangan 2,28 atas tiga inisialisasi)",
        berat="ringan",
    ),
}

SSL = ("wav2vec2", "hubert")

WAKTU = {
    "ringan": ("Ini yang paling ringan dari empat model proposal, karena tidak "
               "ada encoder besar yang harus dilewati. Sepuluh putaran biasanya "
               "selesai dalam hitungan menit."),
    "sedang": ("Pada kartu T4 gratis, sepuluh putaran biasanya belasan sampai "
               "dua puluhan menit."),
    "berat": ("Model ini yang paling berat dari empat model proposal. Pada kartu "
              "T4 gratis, sepuluh putaran bisa memakan waktu sekitar satu jam."),
}


def berkas(m, jenis):
    awalan = {"proses": "Proses", "optimizer": "Optimizer", "dropout": "Dropout"}[jenis]
    nama = {"wav2vec2": "Wav2Vec2", "ast": "AST",
            "hubert": "HuBERT", "cnnlstm": "CNN_LSTM"}[m]
    return f"{awalan}_{nama}.ipynb"


def tautan(m, jenis):
    return (f"https://colab.research.google.com/github/{GH}/blob/main/colab/"
            f"{berkas(m, jenis)}")


# ==========================================================================
# Pembungkus sel
# ==========================================================================
def sel(tipe, teks):
    baris = teks.split("\n")
    src = [b + "\n" for b in baris[:-1]] + [baris[-1]]
    if tipe == "markdown":
        return {"cell_type": "markdown", "metadata": {}, "source": src}
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


def notebook(cells, nama):
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True, "name": nama},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


# ==========================================================================
# Sel bersama: penyiapan dan dataset
# ==========================================================================
MD_SETUP = """## Langkah 1 · Menyiapkan runtime, kode, dan dependensi

Yang terjadi di sel ini, berurutan:

1. **Memeriksa kartu grafisnya.** Bukan sekadar ada atau tidak, tetapi jenisnya,
   karena kartu lama dan kartu baru memakai format bilangan yang berbeda saat
   melatih. Ini berpengaruh pada hasil, jadi dicatat sejak awal.
2. **Mengambil kode penelitian** dari GitHub. Tidak ada kode yang ditempel di
   dalam notebook ini, semuanya berasal dari repositori yang sama dengan yang
   dipakai di komputer lokal. Itu syarat supaya angkanya bisa dibandingkan.
3. **Memasang pustaka** yang belum dibawa Colab.
4. **Menjalankan pengaman konfigurasi**, yaitu skrip yang menolak melanjutkan
   bila ada pengaturan yang menyimpang dari acuan penelitian."""

MD_DATASET = """## Langkah 2 · Mengambil dan memverifikasi dataset

Dataset yang dipakai adalah **Fake-or-Real, potongan dua detik**: 17.870 berkas
suara, masing masing tepat dua detik, 16.000 sampel per detik, satu kanal.

Yang penting di sini bukan mengunduhnya, tapi **membuktikan datanya sama**.
Kalau dataset di Colab berbeda sedikit saja dari yang dipakai di komputer
lokal, seluruh perbandingan angka menjadi tidak berarti.

Karena itu ada dua lapis pemeriksaan:

1. **Sidik jari arsip.** Kode sha256 dari berkas arsipnya dibandingkan dengan
   yang tercatat di repositori.
2. **Sidik jari pohon berkas.** Ini yang lebih dalam: setiap berkas wav
   diperiksa satu per satu, lalu diringkas menjadi satu kode. Pemeriksaan ini
   tidak peduli berkasnya datang dari mana atau namanya di folder apa. Yang
   dibandingkan isinya.

Kalau salah satu tidak cocok, sel ini **berhenti** dan tidak melanjutkan ke
pelatihan. Itu memang disengaja."""

KODE_DATASET = """from colab.siapkan import siapkan_dataset, verifikasi_dan_manifest

siapkan_dataset(buat_cache=True)
baris = verifikasi_dan_manifest()"""


def kode_setup(m, s):
    return f"""# Notebook ini khusus satu model. Ketiga nilai berikut sengaja ditulis tetap,
# bukan sebagai pilihan, supaya tidak ada yang tergeser tanpa sengaja dan
# angkanya tetap sebanding dengan hasil rujukan di mesin lokal.
MODEL      = "{m}"
BATCH      = {s['batch']}          # batch yang dipakai di mesin lokal untuk model ini
AUGMENTASI = "codec"    # perbaikan kebocoran codec, lihat notebook Proses

import os, subprocess, sys

AKAR = "/content/general-ai"
if not os.path.exists(os.path.join(AKAR, ".git")):
    subprocess.run(["git", "clone", "--quiet", "{REPO_URL}", AKAR], check=True)
os.chdir(AKAR)
sys.path.insert(0, AKAR)

from colab.siapkan import siapkan
siapkan(MODEL)"""


# ==========================================================================
# NOTEBOOK 1: proses
# ==========================================================================
def md_langkah4(m, s):
    if m in SSL:
        return f"""## Langkah 4 · Apa yang sebenarnya dilihat {s['nama']}

Ini bagian yang paling membedakan keempat model, dan paling layak dijelaskan pelan pelan.

{s['nama']} **tidak mengubah suara menjadi gambar**. Ia menerima 32.000 angka
mentah, yaitu tinggi gelombang suara yang diukur 16.000 kali per detik selama
dua detik. Tidak ada satu pun informasi yang dibuang sebelum masuk model.

Ini penting karena sebagian jejak mesin hidup di **fase** sinyal, dan fase
itulah yang hilang begitu suara diubah menjadi gambar spektrum. Model yang
bekerja pada gambar suara secara rancangan buta terhadap bukti itu.

Sel di bawah membangun modelnya, memasukkan satu klip, lalu mencetak bentuk
data di tiap tahap. Yang perlu diperhatikan: encoder mengeluarkan **banyak
lapisan sekaligus**, bukan satu. Model kemudian belajar sendiri lapisan mana
yang paling berguna.

> Sel ini mengunduh bobot `{s['ckpt']}` dari Hugging Face pada
> pemanggilan pertama. Untuk model besar ukurannya lebih dari satu gigabyte,
> jadi wajar bila butuh satu sampai dua menit."""

    if m == "ast":
        return """## Langkah 4 · Apa yang sebenarnya dilihat AST

AST bekerja pada **gambar suara**, bukan gelombangnya. Urutannya begini:

1. Gelombang 32.000 angka diubah menjadi spektrum, yaitu peta energi tiap
   frekuensi di tiap saat.
2. Frekuensinya dipadatkan ke 128 pita mengikuti **skala Mel**, yaitu skala
   yang meniru cara telinga manusia mendengar.
3. Hasilnya dipotong menjadi petak petak kecil, dan tiap petak diperlakukan
   seperti satu kata dalam kalimat oleh Transformer.

Langkah kedua adalah yang harus dijelaskan hati hati. Skala Mel memberi
**banyak** detail pada frekuensi rendah, tempat suara manusia berada, dan
**sedikit** detail pada frekuensi tinggi. Padahal jejak mesin justru banyak
berada di frekuensi tinggi.

Sel di bawah menghitung sendiri seberapa besar pemadatan itu, langsung dari
definisi skala Mel yang dipakai model ini. Angkanya bukan kutipan, tapi hasil
hitungan di depan mata."""

    return """## Langkah 4 · Apa yang sebenarnya dilihat CNN-LSTM

Sama seperti AST, CNN-LSTM bekerja pada **gambar suara**, bukan gelombangnya.
Gelombang 32.000 angka diubah menjadi peta energi 128 pita frekuensi kali
sekitar 200 potongan waktu, lalu peta itu diperlakukan persis seperti foto:
dipindai dengan konvolusi.

Dua hal hilang di langkah ini, dan keduanya perlu disebut:

1. **Fase dibuang.** Yang disimpan hanya seberapa kuat tiap frekuensi, bukan
   bagaimana gelombangnya bergeser. Sebagian jejak mesin hidup di sana.
2. **Frekuensi tinggi dipadatkan.** Skala Mel memberi detail halus pada
   frekuensi rendah dan detail kasar pada frekuensi tinggi.

Sel di bawah menghitung sendiri seberapa besar pemadatan itu, langsung dari
pengaturan yang benar benar dipakai model ini."""


def kode_langkah4(m, s):
    if m in SSL:
        return """import torch
from forlib.models import build_model

# freeze=True persis seperti yang dipakai train.py untuk model swa-selia
model = build_model(MODEL, freeze=True, layer_weighting=True).eval().cuda()

wav = torch.from_numpy(x_asli[:32000].astype("float32"))[None].cuda()
print(f"masukan model : {tuple(wav.shape)}")
print(f"                {wav.shape[1]:,} angka gelombang, tidak ada yang dibuang\\n")

with torch.no_grad():
    keluaran = model.encoder(model._prep(wav), output_hidden_states=True)
hs = keluaran.hidden_states

print(f"encoder mengeluarkan {len(hs)} lapisan sekaligus:")
print(f"  1 lapis embedding + {len(hs) - 1} lapis transformer")
print(f"  bentuk tiap lapis : {tuple(hs[0].shape)}")
print(f"                      = 1 klip, {hs[0].shape[1]} langkah waktu, "
      f"{hs[0].shape[2]} angka per langkah")
print(f"  total angka       : {hs[0].numel() * len(hs):,}")
print()
print("Model belum tahu lapisan mana yang berguna. Ia MEMPELAJARINYA lewat")
print("bobot lapisan, yang akan kita lihat hasilnya di langkah 8.")

fig, ax = plt.subplots(1, 2, figsize=(13, 3.4), layout="constrained")
for a, i, judul_a in [(ax[0], 1, "lapis transformer pertama"),
                      (ax[1], len(hs) - 1, "lapis transformer terakhir")]:
    z = hs[i][0].float().cpu().numpy().T
    a.imshow(z, aspect="auto", origin="lower", cmap="RdBu_r",
             vmin=-3, vmax=3, interpolation="nearest")
    a.set_title(f"{judul_a}  (lapis {i})")
    a.set_xlabel("langkah waktu"); a.set_ylabel("dimensi")
fig.suptitle("Isi encoder untuk klip ASLI tadi. Sumbu tegak bukan frekuensi, "
             "melainkan ciri yang ditemukan sendiri oleh model.")
plt.show()"""

    if m == "ast":
        return """import torch, numpy as np
from forlib.models import build_model

model = build_model("ast", freeze=True, layer_weighting=True).eval().cuda()

wav = torch.from_numpy(x_asli[:32000].astype("float32"))[None].cuda()
with torch.no_grad():
    fb = model._fbank(wav)

print(f"masukan model : {tuple(wav.shape)}   gelombang mentah")
print(f"setelah fbank : {tuple(fb.shape)}")
print(f"                = 1 klip, {fb.shape[1]} bingkai waktu, {fb.shape[2]} pita mel")
print()

cfg = model.encoder.config
f_grid, t_grid = model._grid(cfg)
print(f"petak         : {cfg.patch_size} x {cfg.patch_size}, "
      f"langkah {cfg.frequency_stride} x {cfg.time_stride}")
print(f"jumlah petak  : {f_grid} x {t_grid} = {f_grid * t_grid} petak")
print("tiap petak diperlakukan seperti satu kata oleh Transformer\\n")

# ---- Seberapa besar sebenarnya pemadatan skala Mel? Dihitung, bukan dikutip.
SR_ = 16000
n_mel = cfg.num_mel_bins
mel_maks = 2595.0 * np.log10(1.0 + (SR_ / 2) / 700.0)
tepi_mel = np.linspace(0.0, mel_maks, n_mel + 1)
tepi_hz = 700.0 * (10.0 ** (tepi_mel / 2595.0) - 1.0)
lebar_hz = np.diff(tepi_hz)

print(f"lebar 1 pita mel di frekuensi terendah : {lebar_hz[0]:6.1f} Hz")
print(f"lebar 1 pita mel di frekuensi tertinggi: {lebar_hz[-1]:6.1f} Hz")
print(f"rasio                                  : {lebar_hz[-1] / lebar_hz[0]:6.1f} kali lebih kasar")
print()
print("Artinya detail di frekuensi tinggi dipadatkan belasan kali lebih rapat")
print("daripada di frekuensi rendah. Jejak mesin banyak berada di sana.")

fig, ax = plt.subplots(1, 2, figsize=(13, 3.6), layout="constrained")
ax[0].imshow(fb[0].float().cpu().numpy().T, aspect="auto", origin="lower",
             cmap="magma", interpolation="nearest")
ax[0].set_title(f"Yang masuk ke AST: {fb.shape[1]} bingkai x {fb.shape[2]} pita mel")
ax[0].set_xlabel("bingkai waktu"); ax[0].set_ylabel("pita mel")
for k in range(f_grid + 1):
    ax[0].axhline(k * cfg.frequency_stride, color="w", lw=0.4, alpha=0.5)
for k in range(0, t_grid + 1, 2):
    ax[0].axvline(k * cfg.time_stride, color="w", lw=0.4, alpha=0.5)

ax[1].plot(tepi_hz[:-1], lebar_hz, color="#C2255C", lw=2)
ax[1].set_title("Lebar tiap pita mel, dalam Hz")
ax[1].set_xlabel("frekuensi (Hz)"); ax[1].set_ylabel("lebar 1 pita (Hz)")
ax[1].grid(alpha=0.25)
ax[1].annotate(f"{lebar_hz[-1] / lebar_hz[0]:.1f}x lebih kasar\\ndi frekuensi tinggi",
               xy=(tepi_hz[-2], lebar_hz[-1]), xytext=(1500, lebar_hz[-1] * 0.72),
               arrowprops=dict(arrowstyle="->", color="#495057"), fontsize=9)
plt.show()"""

    return """import torch, numpy as np
from forlib.models import build_model

model = build_model("cnnlstm").eval()

wav = torch.from_numpy(x_asli[:32000].astype("float32"))[None]
with torch.no_grad():
    mel = model.to_db(model.mel(wav))

print(f"masukan model : {tuple(wav.shape)}   gelombang mentah")
print(f"setelah mel   : {tuple(mel.shape)}")
print(f"                = 1 klip, {mel.shape[1]} pita frekuensi, {mel.shape[2]} bingkai waktu")
print()
print("pengaturan yang dipakai:")
print(f"  jendela FFT   : {model.mel.n_fft} sampel")
print(f"  loncatan      : {model.mel.hop_length} sampel  "
      f"({model.mel.hop_length / 16000 * 1000:.0f} ms)")
print(f"  jumlah pita   : {model.mel.n_mels}")
print()

# ---- Seberapa besar sebenarnya pemadatan skala Mel? Dihitung, bukan dikutip.
SR_, n_mel = 16000, model.mel.n_mels
mel_maks = 2595.0 * np.log10(1.0 + (SR_ / 2) / 700.0)
tepi_mel = np.linspace(0.0, mel_maks, n_mel + 1)
tepi_hz = 700.0 * (10.0 ** (tepi_mel / 2595.0) - 1.0)
lebar_hz = np.diff(tepi_hz)
print(f"lebar 1 pita mel di frekuensi terendah : {lebar_hz[0]:6.1f} Hz")
print(f"lebar 1 pita mel di frekuensi tertinggi: {lebar_hz[-1]:6.1f} Hz")
print(f"rasio                                  : {lebar_hz[-1] / lebar_hz[0]:6.1f} kali lebih kasar")

fig, ax = plt.subplots(1, 2, figsize=(13, 3.6), layout="constrained")
ax[0].imshow(mel[0].numpy(), aspect="auto", origin="lower", cmap="magma",
             interpolation="nearest")
ax[0].set_title(f"Yang masuk ke CNN: {mel.shape[1]} pita x {mel.shape[2]} bingkai")
ax[0].set_xlabel("bingkai waktu"); ax[0].set_ylabel("pita mel")

ax[1].plot(tepi_hz[:-1], lebar_hz, color="#C2255C", lw=2)
ax[1].set_title("Lebar tiap pita mel, dalam Hz")
ax[1].set_xlabel("frekuensi (Hz)"); ax[1].set_ylabel("lebar 1 pita (Hz)")
ax[1].grid(alpha=0.25)
ax[1].annotate(f"{lebar_hz[-1] / lebar_hz[0]:.1f}x lebih kasar\\ndi frekuensi tinggi",
               xy=(tepi_hz[-2], lebar_hz[-1]), xytext=(1500, lebar_hz[-1] * 0.72),
               arrowprops=dict(arrowstyle="->", color="#495057"), fontsize=9)
plt.show()"""


def md_langkah5(m, s):
    if m in SSL:
        return f"""## Langkah 5 · Bentuk modelnya dan berapa yang benar benar dilatih

Ini angka yang paling sering mengejutkan orang, dan paling layak disorot.

{s['nama']} membawa ratusan juta parameter, tapi **hampir semuanya dibekukan**.
Yang benar benar dilatih hanya bagian pengambil keputusan di ujungnya.

Alasannya bisa dihitung. Data latihnya 13.956 klip, sekitar 7,8 jam suara.
Melatih ulang ratusan juta parameter dengan data sesedikit itu hampir pasti
menghafal, bukan belajar.

Cara membaca angka itu: **telinga model tidak diajari ulang.** Telinganya
sudah terlatih bertahun tahun sebelum bertemu data ini, dan yang ditambahkan
hanya satu kemampuan baru, yaitu menjawab pertanyaan asli atau palsu."""

    if m == "ast":
        return """## Langkah 5 · Bentuk modelnya dan berapa yang benar benar dilatih

AST membawa sekitar 86 juta parameter dari pra-pelatihan AudioSet, dan
**hampir semuanya dibekukan**. Yang dilatih hanya bagian pengambil keputusan
di ujungnya.

Alasannya sama seperti pada model swa-selia: data latihnya hanya 13.956 klip,
sekitar 7,8 jam. Melatih ulang puluhan juta parameter dengan data sesedikit itu
hampir pasti menghafal, bukan belajar."""

    return """## Langkah 5 · Bentuk modelnya dan berapa yang benar benar dilatih

Di sinilah CNN-LSTM berbeda paling tajam dari tiga model lainnya.

Model ini **tidak membawa apa apa dari pra-pelatihan**. Tidak ada bobot yang
diunduh, tidak ada pengetahuan sebelumnya. Seluruh parameternya dimulai dari
angka acak dan dilatih hanya dengan 13.956 klip yang ada.

Kalau tiga model lain diibaratkan orang yang telinganya sudah terlatih
bertahun tahun lalu diajari satu tugas baru, model ini adalah orang yang baru
belajar mendengar hari ini, sekaligus disuruh mengerjakan tugasnya.

Perhatikan juga jalur bentuk datanya di bawah. Setelah konvolusi, sisa waktunya
tinggal sekitar 25 potongan. LSTM diminta mengingat urutan dari 25 potongan
yang mewakili dua detik suara. Pertanyaan yang wajar diajukan, dan memang
diuji di penelitian ini: apakah ada yang layak diingat di situ?"""


def kode_langkah5(m, s):
    umum = """tot = sum(p.numel() for p in model.parameters())
lat = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"total parameter : {tot:>12,}")
print(f"dilatih         : {lat:>12,}   ({lat / tot * 100:5.2f} persen)")
print(f"dibekukan       : {tot - lat:>12,}   ({(tot - lat) / tot * 100:5.2f} persen)")
print()
print(f"{'bagian':<16}{'parameter':>14}{'dilatih':>10}")
print("-" * 40)
for nama_bagian, modul in model.named_children():
    n = sum(p.numel() for p in modul.parameters())
    d = sum(p.numel() for p in modul.parameters() if p.requires_grad)
    if n:
        print(f"{nama_bagian:<16}{n:>14,}{'ya' if d else 'tidak':>10}")
print()
"""
    if m in SSL:
        alur = """print("Jalur bentuk data dari gelombang sampai jawaban:\\n")
with torch.no_grad():
    h_all = model.encoder(model._prep(wav), output_hidden_states=True).hidden_states
    h = model.lw(h_all)
    print(f"  gelombang masuk        {tuple(wav.shape)}")
    print(f"  keluaran encoder       {len(h_all)} x {tuple(h_all[0].shape)}")
    print(f"  digabung pakai bobot   {tuple(h.shape)}")
    h = model.bottleneck(h.transpose(1, 2))
    print(f"  conv penyempit         {tuple(h.shape)}")
    z = model.pool(h)
    print(f"  attentive pooling      {tuple(z.shape)}   <- waktu diringkas habis")
    o = model.head(z)
    print(f"  kepala keputusan       {tuple(o.shape)}   <- 2 angka: asli, palsu")
    p_ = torch.softmax(o, dim=1)[0]
    print(f"\\n  tebakan model saat ini : asli {p_[0]:.3f} / palsu {p_[1]:.3f}")
    print("  (masih asal asalan, model ini belum dilatih sama sekali)")"""
    elif m == "ast":
        alur = """print("Jalur bentuk data dari gelombang sampai jawaban:\\n")
with torch.no_grad():
    x = model._fbank(wav)
    print(f"  gelombang masuk        {tuple(wav.shape)}")
    print(f"  gambar suara (fbank)   {tuple(x.shape)}")
    h_all = model.encoder(x, output_hidden_states=True).hidden_states
    h = model.lw(h_all)
    print(f"  keluaran encoder       {len(h_all)} x {tuple(h_all[0].shape)}")
    print(f"  digabung pakai bobot   {tuple(h.shape)}")
    h = model.bottleneck(h.transpose(1, 2))
    print(f"  conv penyempit         {tuple(h.shape)}")
    z = model.pool(h)
    print(f"  attentive pooling      {tuple(z.shape)}")
    o = model.head(z)
    print(f"  kepala keputusan       {tuple(o.shape)}   <- 2 angka: asli, palsu")
    p_ = torch.softmax(o, dim=1)[0]
    print(f"\\n  tebakan model saat ini : asli {p_[0]:.3f} / palsu {p_[1]:.3f}")
    print("  (masih asal asalan, model ini belum dilatih sama sekali)")"""
    else:
        alur = """print("Jalur bentuk data dari gelombang sampai jawaban:\\n")
with torch.no_grad():
    m_ = model.to_db(model.mel(wav))
    m_ = (m_ - m_.mean(dim=(1, 2), keepdim=True)) / (m_.std(dim=(1, 2), keepdim=True) + 1e-5)
    print(f"  gelombang masuk        {tuple(wav.shape)}")
    print(f"  gambar suara (mel)     {tuple(m_.shape)}")
    x = model.cnn(m_.unsqueeze(1))
    print(f"  setelah 3 blok CNN     {tuple(x.shape)}   <- (klip, kanal, pita, waktu)")
    B, C, Fq, T = x.shape
    x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fq)
    print(f"  disusun jadi urutan    {tuple(x.shape)}   <- {T} potongan waktu")
    x, _ = model.lstm(x)
    print(f"  setelah BiLSTM         {tuple(x.shape)}")
    z = model.pool(x.transpose(1, 2))
    print(f"  attentive pooling      {tuple(z.shape)}")
    o = model.head(z)
    print(f"  kepala keputusan       {tuple(o.shape)}   <- 2 angka: asli, palsu")
    p_ = torch.softmax(o, dim=1)[0]
    print(f"\\n  tebakan model saat ini : asli {p_[0]:.3f} / palsu {p_[1]:.3f}")
    print("  (masih asal asalan, model ini belum dilatih sama sekali)")"""
    return umum + alur


MD_DENGAR = """## Langkah 3 · Mendengarkan dan melihat datanya

Sebelum membicarakan model, ada baiknya mendengar dulu apa yang harus
dibedakan. Sel ini mengambil satu suara asli dan satu suara palsu dari **data
uji**, memutarnya, lalu menggambarnya dengan dua cara.

Keduanya umumnya terdengar sangat mirip, dan gambarnya pun terlihat mirip.
Itu memang inti persoalannya. Bila perbedaannya kentara bagi telinga manusia,
tugas ini tidak memerlukan model sama sekali.

Dua gambar yang ditampilkan:

- **Gelombang**, yaitu tinggi sinyal terhadap waktu. Ini bentuk aslinya.
- **Spektrum**, yaitu peta energi tiap frekuensi di tiap saat. Warna terang
  berarti energi besar. Bagian atas gambar adalah frekuensi tinggi, dan di
  sanalah jejak mesin biasanya berada."""

KODE_DENGAR = """import numpy as np, soundfile as sf, matplotlib.pyplot as plt
from scipy.signal import stft
from IPython.display import Audio, display
from forlib.data import load_manifest

rows = load_manifest("manifest.csv")
uji = [r for r in rows if r["split_official"] == "testing"]
asli  = [r for r in uji if r["label"] == 0][7]
palsu = [r for r in uji if r["label"] == 1][7]

def baca(r):
    x, sr = sf.read(r["path"], dtype="float32")
    return (x.mean(axis=1) if x.ndim > 1 else x), sr

x_asli, SR = baca(asli)
x_palsu, _ = baca(palsu)

print(f"ASLI  : {asli['fname']}")
display(Audio(x_asli, rate=SR))
print(f"PALSU : {palsu['fname']}")
display(Audio(x_palsu, rate=SR))

def gambar_spek(x):
    f, t, Z = stft(x, fs=SR, nperseg=512, noverlap=384)
    return f, t, 20 * np.log10(np.abs(Z) + 1e-8)

fig, ax = plt.subplots(2, 2, figsize=(13, 5.6), layout="constrained")
for kol, (x, judul_k, warna) in enumerate(
        [(x_asli, "ASLI", "#1864AB"), (x_palsu, "PALSU", "#C2255C")]):
    ax[0, kol].plot(np.arange(len(x)) / SR, x, lw=0.4, color=warna)
    ax[0, kol].set_title(f"{judul_k} - gelombang")
    ax[0, kol].set_xlabel("detik"); ax[0, kol].set_ylim(-1, 1)
    f, t, S = gambar_spek(x)
    ax[1, kol].imshow(S, aspect="auto", origin="lower", cmap="magma",
                      extent=[0, len(x) / SR, 0, SR / 2000],
                      vmin=S.max() - 80, vmax=S.max(), interpolation="nearest")
    ax[1, kol].set_title(f"{judul_k} - spektrum")
    ax[1, kol].set_xlabel("detik"); ax[1, kol].set_ylabel("kHz")
plt.show()

print("Kalau keduanya terdengar dan terlihat mirip, itu memang yang diharapkan.")
print("Perbedaannya ada, tapi terlalu halus untuk telinga dan mata manusia.")"""

MD_AUG = """## Langkah 6 · Bagaimana data sengaja dirusak, dan kenapa

Ini bagian yang paling layak dijelaskan panjang, karena di sinilah letak
temuan yang tidak ada di rencana awal penelitian.

Waktu datanya diperiksa satu per satu, ketahuan ada masalah serius. Di **data
latih**, hampir semua suara palsu berasal dari berkas MP3, sedangkan di **data
uji** tidak ada satu pun. Sel di bawah menghitung angkanya langsung dari
datanya, jadi bukan kutipan.

Kenapa itu masalah. Kompresi MP3 memotong frekuensi tinggi. Kalau hampir semua
suara palsu di data latih terpotong frekuensi tingginya sedangkan suara aslinya
tidak, model tidak perlu belajar mengenali jejak mesin sama sekali. Cukup
belajar satu aturan pendek: **frekuensi tingginya hilang berarti palsu.**

Aturan itu bekerja sempurna di data latih dan **gagal total** di data uji,
karena di sana tidak ada satu pun yang berasal dari MP3.

Perbaikannya sederhana dan bisa dijelaskan dalam satu kalimat: **potong
frekuensi tinggi secara acak pada kedua kelas.** Suara asli juga dipotong,
suara palsu juga dipotong, jumlahnya acak. Setelah itu memotong frekuensi
tinggi tidak lagi menandakan apa apa, dan model terpaksa mencari jejak yang
sesungguhnya."""

KODE_AUG = """from forlib.data import codec_augment

# ---- Berapa besar sebenarnya kebocorannya? Dihitung dari datanya sendiri.
tr_palsu = [r for r in rows if r["split_official"] == "training" and r["label"] == 1]
te_palsu = [r for r in rows if r["split_official"] == "testing" and r["label"] == 1]
p_tr = sum(r["is_mp3"] for r in tr_palsu) / len(tr_palsu) * 100
p_te = sum(r["is_mp3"] for r in te_palsu) / len(te_palsu) * 100
print(f"suara PALSU di data latih yang berasal dari MP3 : {p_tr:5.1f} persen  "
      f"({sum(r['is_mp3'] for r in tr_palsu)} dari {len(tr_palsu)})")
print(f"suara PALSU di data uji   yang berasal dari MP3 : {p_te:5.1f} persen  "
      f"({sum(r['is_mp3'] for r in te_palsu)} dari {len(te_palsu)})")
print()
print("Selisih sebesar itu adalah jalan pintas yang menganga. Model yang")
print("memakainya akan terlihat sangat pintar saat latihan lalu jatuh saat diuji.")
print()

rng = np.random.default_rng(0)
x_rusak = codec_augment(x_asli.astype(np.float64), rng).astype(np.float32)

print("Suara ASLI sebelum dirusak:")
display(Audio(x_asli, rate=SR))
print("Suara ASLI yang sama, setelah frekuensi tingginya dipotong acak:")
display(Audio(x_rusak, rate=SR))

def rerata_spektrum(x):
    f, _, Z = stft(x, fs=SR, nperseg=512, noverlap=384)
    return f, 20 * np.log10(np.abs(Z).mean(axis=1) + 1e-8)

fig, ax = plt.subplots(1, 2, figsize=(13, 3.6), layout="constrained")
f, s0 = rerata_spektrum(x_asli)
_, s1 = rerata_spektrum(x_rusak)
ax[0].plot(f / 1000, s0, color="#1864AB", lw=1.4, label="sebelum")
ax[0].plot(f / 1000, s1, color="#E8590C", lw=1.4, label="setelah dipotong")
ax[0].set_title("Rerata energi tiap frekuensi")
ax[0].set_xlabel("kHz"); ax[0].set_ylabel("dB"); ax[0].legend(); ax[0].grid(alpha=0.25)

ax[1].bar(["latih", "uji"], [p_tr, p_te], color=["#C2255C", "#2F9E44"], width=0.5)
ax[1].set_title("Persen suara PALSU yang berasal dari MP3")
ax[1].set_ylabel("persen"); ax[1].set_ylim(0, 100)
for i, v in enumerate([p_tr, p_te]):
    ax[1].text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=11)
plt.show()

print("Titik potongnya diacak antara 3.000 dan 7.800 Hz, berbeda tiap berkas")
print("dan tiap putaran latihan, dan dikenakan pada kedua kelas tanpa pandang bulu.")"""


def bangun_proses(m):
    s = SPEK[m]
    c = [sel("markdown", f"""# Proses pelatihan {s['nama']}

**{s['satu_kalimat']}**

{PEMBUKA}

Notebook ini sengaja dipecah menjadi delapan langkah terpisah, satu sel per
langkah, supaya prosesnya bisa dijelaskan sambil berjalan. Notebook ini bukan
untuk mengejar hasil secepat mungkin.

| langkah | isi | perkiraan waktu |
|---|---|---|
| 1 | Menyiapkan runtime, kode, dan dependensi | 1 sampai 2 menit |
| 2 | Mengambil dan memverifikasi dataset | 1 menit dari cache, 10 menit dari arsip |
| 3 | Mendengarkan dan melihat datanya | seketika |
| 4 | Apa yang sebenarnya dilihat model ini | 1 sampai 2 menit |
| 5 | Bentuk modelnya dan berapa yang dilatih | seketika |
| 6 | Bagaimana data sengaja dirusak, dan kenapa | seketika |
| 7 | Melatih | lihat catatan di langkah 7 |
| 8 | Membaca hasil dan kesalahannya | seketika |

**Sebelum mulai:** menu `Runtime` lalu `Change runtime type`, pilih **T4 GPU**,
lalu `Save`. Tanpa itu langkah 1 akan berhenti dan memberi tahu.

> **Catatan tentang model ini.** {s['catatan_penting']}

Dua notebook pendamping membandingkan pilihan pelatihan pada model yang sama:
`{berkas(m, 'optimizer')}` dan `{berkas(m, 'dropout')}`.""")]

    c += [sel("markdown", MD_SETUP), sel("code", kode_setup(m, s))]
    c += [sel("markdown", MD_DATASET), sel("code", KODE_DATASET)]
    c += [sel("markdown", MD_DENGAR), sel("code", KODE_DENGAR)]
    c += [sel("markdown", md_langkah4(m, s)), sel("code", kode_langkah4(m, s))]
    c += [sel("markdown", md_langkah5(m, s)), sel("code", kode_langkah5(m, s))]
    c += [sel("markdown", MD_AUG), sel("code", KODE_AUG)]

    c.append(sel("markdown", f"""## Langkah 7 · Melatih

Sekarang barulah pelatihannya berjalan. Yang terjadi di tiap putaran:

1. Data latih diacak urutannya, lalu dibagi menjadi rombongan berisi
   {s['batch']} klip.
2. Tiap klip dibaca dari berkas, dirusak secara acak seperti di langkah 6, lalu
   disamakan kerasnya.
3. Model menebak, tebakannya dibandingkan dengan jawaban benar, lalu bagian
   yang dilatih digeser sedikit ke arah yang lebih benar.
4. Sesudah satu putaran penuh, model diuji pada data validasi. Kalau hasilnya
   lebih baik dari putaran sebelumnya, bobotnya disimpan.

Yang disimpan pada akhirnya adalah **putaran terbaik menurut data validasi**,
bukan putaran terakhir. Data uji tidak pernah dilihat selama pelatihan, satu
kali pun.

> **Waktu.** {WAKTU[s['berat']]} Untuk memperlihatkan prosesnya saja, turunkan
> `EPOCHS` menjadi 2 atau 3.

Hasilnya ditulis ke `runs_colab/`, sengaja terpisah dari `runs/` yang berisi
hasil komputer lokal. Keduanya tidak boleh dicampur, karena kartu grafis dan
format bilangannya berbeda, dan selisih akibat perangkat keras akan terbaca
seolah olah selisih akibat inisialisasi acak."""))

    c.append(sel("code", """#@title Setelan pelatihan { display-mode: "form" }
EPOCHS = 10 #@param {type:"slider", min:1, max:20, step:1}
SEED = 42 #@param {type:"integer"}
SIMPAN_HASIL_KE_DRIVE = False #@param {type:"boolean"}

import time
from colab.siapkan import jalankan, berhenti

# Tag dibentuk dengan aturan yang sama seperti train.py, sehingga langkah 8
# tahu persis folder mana yang harus dibaca tanpa menebak lewat pencarian.
TAG = f"{MODEL}_official_{AUGMENTASI}_b{BATCH}e{EPOCHS}_s{SEED}"

print(f"model      : {MODEL}")
print(f"augmentasi : {AUGMENTASI}")
print(f"epoch      : {EPOCHS}   batch: {BATCH}   seed: {SEED}")
print(f"keluaran   : runs_colab/{TAG}\\n")

t0 = time.time()
kode, _ = jalankan([
    sys.executable, "train.py",
    "--model", MODEL, "--split", "official", "--augment", AUGMENTASI,
    "--epochs", str(EPOCHS), "--batch", str(BATCH), "--workers", "2",
    "--seed", str(SEED), "--out", "runs_colab",
])
if kode != 0:
    berhenti("Pelatihan gagal. Kalau pesannya menyebut kehabisan memori GPU, "
             "turunkan BATCH menjadi 8 lalu jalankan sel ini lagi.")
print(f"\\npelatihan selesai dalam {(time.time() - t0) / 60:.1f} menit")

if SIMPAN_HASIL_KE_DRIVE:
    from colab.siapkan import simpan_ke_drive
    simpan_ke_drive()"""))

    lw_md = ("" if m == "cnnlstm" else """

**Bobot lapisan.** Encoder mengeluarkan belasan lapisan sekaligus, dan model
mempelajari sendiri lapisan mana yang berguna. Grafik batangnya ditampilkan di
bawah. Kalau bobotnya menumpuk di lapisan tengah, itu bacaan yang menarik:
lapisan paling akhir dipra-latih untuk tugas lain, sedangkan lapisan tengah
menyimpan lebih banyak ciri akustik mentah.""")

    c.append(sel("markdown", f"""## Langkah 8 · Membaca hasil dan kesalahannya

Tiga hal yang dibaca di sini, dan urutannya penting.

**Ambang keputusan.** Model mengeluarkan angka antara 0 dan 1, bukan jawaban
ya atau tidak. Angka itu harus dipotong di suatu titik. Memotong di 0,5 terasa
wajar tetapi sebenarnya sewenang wenang. Sel ini melaporkan dua duanya: pada
ambang 0,5, dan pada ambang yang disesuaikan dengan perbandingan kelas di data
uji. Selisih keduanya adalah ukuran langsung seberapa meleset kalibrasinya.

**AUC dan EER.** Dua ukuran ini tidak bergantung pada ambang sama sekali. Kalau
akurasinya jelek tetapi AUC-nya tinggi, artinya model sebenarnya bisa
membedakan, hanya salah menaruh garis potongnya. Itu masalah yang jauh lebih
ringan, dan membedakan keduanya adalah salah satu temuan penelitian ini.

**Kesalahan yang paling percaya diri.** Sel ini memutar berkas yang salah
ditebak model dengan keyakinan paling tinggi. Bagian ini memperlihatkan secara
langsung jenis berkas seperti apa yang menipu model, sesuatu yang tidak
terbaca dari angka ringkasan mana pun.{lw_md}"""))

    c.append(sel("code", """import json
import numpy as np
from forlib.metrics import full_metrics, prior_matched_threshold

d = f"runs_colab/{TAG}"
if not os.path.exists(f"{d}/results.json"):
    raise SystemExit(f"Belum ada hasil di {d}. Jalankan langkah 7 lebih dahulu.")

y, p, _ = np.load(f"{d}/test_scores.npy")
y = y.astype(int)
res = json.load(open(f"{d}/results.json"))

m05 = full_metrics(y, p, 0.5)
mpm = full_metrics(y, p, prior_matched_threshold(p, 0.5))

print(f"hasil dari : {d}")
print(f"putaran terbaik menurut validasi : epoch {res['best_epoch']}\\n")
print(f"{'':<26}{'ambang 0,5':>13}{'ambang disesuaikan':>21}")
print("-" * 60)
for nama_m, k in [("akurasi", "accuracy"), ("recall (palsu terdeteksi)", "recall"),
                  ("spesifisitas (asli aman)", "specificity"), ("F1", "f1")]:
    print(f"{nama_m:<26}{m05[k] * 100:>12.2f}%{mpm[k] * 100:>20.2f}%")
print("-" * 60)
print(f"{'AUC':<26}{m05['auc']:>13.4f}   <- tidak bergantung ambang")
print(f"{'EER':<26}{m05['eer'] * 100:>12.2f}%   <- tidak bergantung ambang")
print()
selisih = (mpm["accuracy"] - m05["accuracy"]) * 100
print(f"Akurasi yang hilang semata karena ambangnya meleset: {selisih:+.2f} poin.")
print("Kalau angka ini besar sedangkan AUC tinggi, yang gagal adalah")
print("kalibrasinya, bukan kemampuan modelnya membedakan.")

if "layer_weights" in res:
    w = np.array(res["layer_weights"])
    fig, ax = plt.subplots(figsize=(9, 2.8), layout="constrained")
    ax.bar(range(len(w)), w, color="#1864AB")
    ax.set_title("Bobot yang dipelajari untuk tiap lapisan encoder")
    ax.set_xlabel("lapisan (0 = embedding, terakhir = paling atas)")
    ax.set_ylabel("bobot")
    ax.axhline(1 / len(w), color="#868E96", ls="--", lw=1)
    ax.text(0.2, 1 / len(w) * 1.06, "garis putus putus = bila semua sama rata",
            fontsize=8, color="#495057")
    plt.show()
    print(f"Lapisan paling berguna menurut model: lapis {int(w.argmax())} "
          f"dari {len(w) - 1}.")

te_rows = [r for r in rows if r["split_official"] == "testing"]
amb = prior_matched_threshold(p, 0.5)
tebak = (p >= amb).astype(int)
salah = np.where(tebak != y)[0]
print(f"\\nsalah tebak: {len(salah)} dari {len(y)} berkas "
      f"({len(salah) / len(y) * 100:.2f} persen)")

if len(te_rows) != len(y):
    print(f"\\n(daftar berkas {len(te_rows)} tidak sepadan dengan skor {len(y)}, "
          f"pemutaran contoh kesalahan dilewati)")
elif len(salah):
    for i in salah[np.argsort(-np.abs(p[salah] - amb))][:3]:
        r = te_rows[i]
        print(f"\\n  {r['fname']}")
        print(f"  sebenarnya {'ASLI' if y[i] == 0 else 'PALSU'}, "
              f"ditebak {'PALSU' if tebak[i] == 1 else 'ASLI'}, "
              f"skor {p[i]:.4f} (ambang {amb:.4f})")
        x, sr = sf.read(r["path"], dtype="float32")
        display(Audio(x.mean(axis=1) if x.ndim > 1 else x, rate=sr))"""))

    lain = [SPEK[k]["nama"] for k in SPEK if k != m]
    c.append(sel("markdown", f"""## Penutup

Angka yang keluar dari notebook ini berasal dari **satu inisialisasi acak**.
Itu belum cukup untuk menyimpulkan apa pun.

Di komputer lokal, {s['nama']} pada konfigurasi yang sama menghasilkan
**{s['hasil_lokal']}**. Simpangan itu bukan hiasan. Ia berarti bila notebook
ini dijalankan ulang dengan `SEED` yang berbeda, hasilnya akan bergeser
beberapa poin, padahal tidak ada satu pun hal lain yang berubah.

Karena itu, sebelum membandingkan model ini dengan {', '.join(lain[:-1])}, atau
{lain[-1]}, jalankan langkah 7 sekali lagi dengan `SEED = 1337` lalu sekali
lagi dengan `SEED = 2024`. Baru dari tiga angka itu perbandingannya layak
dibicarakan.

Dua belas notebook di rangkaian ini:

| model | proses | optimizer | dropout |
|---|---|---|---|
{chr(10).join(f"| {SPEK[k]['nama']} | `{berkas(k, 'proses')}` | `{berkas(k, 'optimizer')}` | `{berkas(k, 'dropout')}` |" for k in SPEK)}"""))

    return notebook(c, berkas(m, "proses").replace(".ipynb", ""))


# ==========================================================================
# NOTEBOOK 2 dan 3: perbandingan
# ==========================================================================
BANDING = {
    "optimizer": dict(
        judul="AdamW lawan NAdam",
        pokok="optimizer",
        arm_a=("AdamW", []),
        arm_b=("NAdam", ["--optimizer", "nadam"]),
        akhiran_b="NAD",
    ),
    "dropout": dict(
        judul="Dropout tetap lawan dropout adaptif",
        pokok="dropout",
        arm_a=("dropout tetap 0,2", []),
        arm_b=("dropout adaptif", ["--dropout", "adaptif"]),
        akhiran_b="DA",
    ),
}


MD_KONSEP_OPT = """## Langkah 3 · Apa itu optimizer, dan apa bedanya kedua ini

**Optimizer adalah aturan yang menentukan seberapa jauh dan ke arah mana bobot
model digeser setiap kali ia salah.** Modelnya sendiri tidak berubah. Yang
berubah hanya cara menuruni bukitnya.

Analogi yang biasanya langsung dimengerti: bayangkan menuruni lembah berkabut,
hanya bisa merasakan kemiringan tanah di bawah kaki.

- **Adam** melangkah mengikuti kemiringan, sambil mengingat arah beberapa
  langkah terakhir supaya tidak zigzag. Ingatan arah itu namanya momentum.
- **NAdam** melakukan hal yang sama, tapi menambahkan satu hal: sebelum
  melangkah, ia **melihat dulu ke arah yang sedang dituju momentumnya**, lalu
  mengukur kemiringan **di titik itu**, bukan di tempat ia berdiri sekarang.
  Ini yang disebut momentum Nesterov.

Efeknya: bila lembahnya menikung, NAdam sadar lebih cepat dan tidak terlanjur
melewati tikungannya.

Sel di bawah memperlihatkannya pada permukaan mainan dua dimensi yang bentuknya
menyerupai lembah sempit. Keduanya diberi titik awal, laju, dan jumlah langkah
yang persis sama.

> **Jujur soal batasnya.** Permukaan mainan ini **bukan bukti** bahwa NAdam
> lebih baik untuk tugas kita. Ia hanya memperlihatkan apa yang berbeda secara
> mekanis. Buktinya baru datang dari langkah 4 sampai 6, dan itu pun perlu
> beberapa inisialisasi sebelum layak disebut kesimpulan.

**Satu hal yang perlu diluruskan.** Pembandingnya di sini bukan Adam polos,
melainkan **AdamW**, yaitu Adam dengan peluruhan bobot terpisah. Itu yang
dipakai di seluruh hasil penelitian ini sejak awal. Supaya perbandingannya
adil, NAdam juga dijalankan dengan peluruhan bobot terpisah, sehingga
**satu satunya yang berbeda memang suku Nesterov-nya**, bukan dua hal sekaligus."""

KODE_KONSEP_OPT = """import torch, numpy as np, matplotlib.pyplot as plt

def permukaan(x, y):
    # lembah sempit melengkung: sedikit salah arah langsung terasa
    return (1 - x) ** 2 + 20 * (y - x ** 2) ** 2

def jalur(Opt, **kw):
    p = torch.tensor([-1.5, 2.0], requires_grad=True)
    o = Opt([p], lr=0.05, **kw)
    t = [p.detach().clone().numpy()]
    for _ in range(120):
        o.zero_grad()
        permukaan(p[0], p[1]).backward()
        o.step()
        t.append(p.detach().clone().numpy())
    return np.array(t)

t_adam = jalur(torch.optim.AdamW, weight_decay=0.0)
t_nadam = jalur(torch.optim.NAdam, weight_decay=0.0)

gx, gy = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-0.5, 3, 300))
gz = permukaan(gx, gy)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), layout="constrained")
ax[0].contour(gx, gy, np.log10(gz + 1e-6), levels=25, cmap="Greys", linewidths=0.6)
ax[0].plot(*t_adam.T, color="#1864AB", lw=1.8, label="AdamW")
ax[0].plot(*t_nadam.T, color="#E8590C", lw=1.8, label="NAdam")
ax[0].scatter([-1.5], [2.0], c="k", s=30, zorder=5)
ax[0].scatter([1.0], [1.0], marker="*", c="#2F9E44", s=200, zorder=5,
              label="titik terendah")
ax[0].set_title("Jalur turun, 120 langkah, laju sama")
ax[0].legend(); ax[0].set_xlabel("bobot 1"); ax[0].set_ylabel("bobot 2")

for t, nama_o, warna in [(t_adam, "AdamW", "#1864AB"), (t_nadam, "NAdam", "#E8590C")]:
    ax[1].plot([permukaan(*q) for q in t], color=warna, lw=1.8, label=nama_o)
ax[1].set_yscale("log"); ax[1].set_title("Seberapa salah, tiap langkah")
ax[1].set_xlabel("langkah"); ax[1].set_ylabel("nilai fungsi (skala log)")
ax[1].legend(); ax[1].grid(alpha=0.25)
plt.show()

print(f"setelah 120 langkah  AdamW: {permukaan(*t_adam[-1]):.4f}")
print(f"                     NAdam: {permukaan(*t_nadam[-1]):.4f}")
print()
print("Sekali lagi: ini permukaan mainan. Ia menunjukkan APA yang berbeda,")
print("bukan MANA yang lebih baik untuk deteksi deepfake audio.")"""


MD_KONSEP_DO = """## Langkah 3 · Apa itu dropout, dan apa bedanya kedua ini

**Dropout adalah menjatuhkan sebagian unit secara acak selama pelatihan.**
Di tiap langkah, sebagian unit dipaksa bernilai nol, dipilih acak. Tujuannya
mencegah model bergantung pada segelintir unit saja.

Analogi yang biasanya langsung dimengerti: sebuah tim yang setiap latihan
mengistirahatkan beberapa pemain secara acak. Karena tidak ada yang tahu siapa
yang absen besok, semua orang terpaksa bisa mengerjakan bagian orang lain. Tim
seperti itu tidak runtuh bila satu bintangnya cedera.

Nilai `p = 0.2` berarti dua dari sepuluh unit dijatuhkan tiap langkah. Yang
tersisa dikalikan `1/(1-p)` supaya jumlah totalnya tidak ikut mengecil.

**Masalahnya: dari mana angka 0,2 itu?** Ditebak. Lalu dicoba. Angka itu tidak
pernah berubah selama pelatihan dan tidak pernah tahu apa apa tentang datanya.

**Dropout adaptif menjadikan angka itu sesuatu yang dipelajari**, bukan
ditetapkan. Yang dipakai di sini adalah Concrete Dropout (Gal, Hron, Kendall,
NeurIPS 2017).

Ada satu kendala teknis yang menarik untuk diceritakan. Menjatuhkan unit adalah
keputusan **ya atau tidak**, dan keputusan ya atau tidak tidak punya kemiringan,
sehingga gradien tidak bisa mengalir ke angka `p` itu. Jalan keluarnya:
keputusan biner tadi diganti dengan versi yang **melunak**, yang nilainya bisa
di antara nol dan satu. Versi lunak itu punya kemiringan, sehingga `p` bisa
ikut dilatih seperti bobot biasa. Sel di bawah menggambarkan pelunakan itu.

Satu hal terakhir. Kalau dibiarkan, gradien akan selalu mendorong `p` ke nol,
karena model yang tidak pernah menjatuhkan apa pun selalu terlihat lebih baik
saat latihan. Karena itu ada suku penahan yang menariknya kembali ke arah 0,5.
**Angka akhirnya adalah titik seimbang antara keduanya, dan titik itu ditentukan
data, bukan oleh yang menyetel.**"""

KODE_KONSEP_DO = """import torch, numpy as np, matplotlib.pyplot as plt
from forlib.models import ConcreteDropout

# ---- 1. Dropout biasa: unit mana yang dijatuhkan
torch.manual_seed(0)
x = torch.ones(1, 12)
do = torch.nn.Dropout(0.2).train()
print("satu vektor berisi 12 unit, semuanya bernilai 1,0")
print("dijatuhkan acak 20 persen, tiga langkah berturut turut:\\n")
for langkah in range(3):
    keluar = do(x)[0].numpy()
    tanda = "".join("." if v == 0 else "#" for v in keluar)
    print(f"  langkah {langkah + 1}:  {tanda}   nilai yang bertahan = "
          f"{keluar[keluar > 0][0]:.3f}  (1 / (1 - 0,2) = 1,25)")
print("\\ntitik = dijatuhkan, pagar = bertahan. Berbeda tiap langkah, itu intinya.")

# ---- 2. Bagaimana keputusan biner dilunakkan supaya bisa dilatih
u = np.linspace(1e-4, 1 - 1e-4, 500)
p_ = 0.2
fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), layout="constrained")
for suhu, warna in [(0.5, "#ADB5BD"), (0.1, "#1864AB"), (0.02, "#C2255C")]:
    z = 1 / (1 + np.exp(-(np.log(p_) - np.log(1 - p_)
                          + np.log(u) - np.log(1 - u)) / suhu))
    ax[0].plot(u, z, color=warna, lw=2, label=f"suhu {suhu}")
ax[0].step(u, (u < p_).astype(float), where="post", color="k", ls="--", lw=1.2,
           label="keputusan biner asli")
ax[0].set_title("Keputusan ya atau tidak, dilunakkan")
ax[0].set_xlabel("undian acak"); ax[0].set_ylabel("seberapa dijatuhkan")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)

# ---- 3. Tarik menarik yang menentukan laju akhirnya
pp = np.linspace(0.01, 0.99, 400)
entropi_negatif = pp * np.log(pp) + (1 - pp) * np.log(1 - pp)
ax[1].plot(pp, -entropi_negatif, color="#2F9E44", lw=2,
           label="penahan: menarik ke 0,5")
ax[1].plot(pp, 1.6 * pp, color="#E8590C", lw=2,
           label="loss latih: menarik ke 0 (gambaran)")
ax[1].axvline(0.2, color="#868E96", ls="--", lw=1)
ax[1].text(0.21, 0.05, "titik awal 0,2", fontsize=8, color="#495057")
ax[1].set_title("Dua tarikan yang menentukan laju akhirnya")
ax[1].set_xlabel("laju dropout p"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.25)
plt.show()

# ---- 4. Modul yang benar benar dipakai di pelatihan nanti
cd = ConcreteDropout(p_awal=0.2).train()
_ = cd(torch.ones(4, 256))
print(f"\\nmodul yang dipakai: {type(cd).__name__}")
print(f"laju awalnya      : {float(cd.p):.4f}")
print("Ke mana angka itu bergerak setelah dilatih, itulah yang akan kita lihat")
print("di langkah 6. Model yang menentukannya, bukan kita.")"""


def bangun_banding(m, jenis):
    s, b = SPEK[m], BANDING[jenis]
    nama_a, arg_a = b["arm_a"]
    nama_b, arg_b = b["arm_b"]

    c = [sel("markdown", f"""# {b['judul']} pada {s['nama']}

{PEMBUKA}

Notebook ini menjalankan **model yang sama persis dua kali**. Data sama, split
sama, augmentasi sama, jumlah putaran sama, batch sama, inisialisasi acak sama.
Satu satunya yang berbeda adalah **{b['pokok']}**-nya.

Itu memang syarat perbandingan yang sah. Kalau dua hal berubah sekaligus,
selisih hasilnya tidak bisa dibebankan pada salah satunya.

| langkah | isi |
|---|---|
| 1 | Menyiapkan runtime, kode, dan dependensi |
| 2 | Mengambil dan memverifikasi dataset |
| 3 | Apa itu {b['pokok']}, dan apa bedanya kedua ini |
| 4 | Melatih dengan **{nama_a}** |
| 5 | Melatih dengan **{nama_b}** |
| 6 | Membandingkan keduanya |

**Sebelum mulai:** menu `Runtime` lalu `Change runtime type`, pilih **T4 GPU**,
lalu `Save`.

> **Waktu.** {WAKTU[s['berat']]} Notebook ini melatih **dua kali**, jadi
> kalikan dua. Untuk memperlihatkan prosesnya saja, turunkan `EPOCHS` menjadi
> 3. Yang penting kedua sisi memakai angka yang sama, dan sel di langkah 5
> memaksa itu.

Untuk penjelasan model dan datanya sendiri, lihat `{berkas(m, 'proses')}`.""")]

    c += [sel("markdown", MD_SETUP), sel("code", kode_setup(m, s))]
    c += [sel("markdown", MD_DATASET), sel("code", KODE_DATASET)]

    if jenis == "optimizer":
        c += [sel("markdown", MD_KONSEP_OPT), sel("code", KODE_KONSEP_OPT)]
    else:
        c += [sel("markdown", MD_KONSEP_DO), sel("code", KODE_KONSEP_DO)]

    c.append(sel("markdown", f"""## Langkah 4 · Melatih dengan {nama_a}

Ini sisi pembanding, yaitu pengaturan yang dipakai di seluruh hasil penelitian
ini sejak awal. Angka yang keluar dari sini harus dekat dengan angka lokal
untuk model yang sama, dan bila jauh melenceng itu pertanda ada yang salah
sebelum kita membandingkan apa pun.

Setelan di sel ini berlaku untuk **kedua** sisi. Langkah 5 mengambil angka yang
sama dari sini, sehingga tidak mungkin tanpa sengaja membandingkan sepuluh
putaran melawan tiga putaran."""))

    c.append(sel("code", f"""#@title Setelan pelatihan, berlaku untuk KEDUA sisi {{ display-mode: "form" }}
EPOCHS = 10 #@param {{type:"slider", min:1, max:20, step:1}}
SEED = 42 #@param {{type:"integer"}}
SIMPAN_HASIL_KE_DRIVE = False #@param {{type:"boolean"}}

import time
from colab.siapkan import jalankan, berhenti

# Tag dibentuk dengan aturan yang sama seperti train.py, jadi langkah 6 tahu
# persis dua folder mana yang harus dibandingkan tanpa menebak.
TAG_A = f"{{MODEL}}_official_{{AUGMENTASI}}_b{{BATCH}}e{{EPOCHS}}_s{{SEED}}"
TAG_B = f"{{MODEL}}_official_{{AUGMENTASI}}{b['akhiran_b']}_b{{BATCH}}e{{EPOCHS}}_s{{SEED}}"
NAMA_A, NAMA_B = "{nama_a}", "{nama_b}"

def latih(tambahan, tag, nama_sisi):
    print(f"\\n{{'=' * 68}}\\n  {{nama_sisi}}\\n{{'=' * 68}}")
    print(f"epoch {{EPOCHS}}   batch {{BATCH}}   seed {{SEED}}")
    print(f"keluaran runs_colab/{{tag}}\\n")
    t0 = time.time()
    kode, _ = jalankan([
        sys.executable, "train.py",
        "--model", MODEL, "--split", "official", "--augment", AUGMENTASI,
        "--epochs", str(EPOCHS), "--batch", str(BATCH), "--workers", "2",
        "--seed", str(SEED), "--out", "runs_colab", *tambahan,
    ])
    if kode != 0:
        berhenti("Pelatihan gagal. Kalau pesannya menyebut kehabisan memori "
                 "GPU, turunkan BATCH menjadi 8 lalu jalankan lagi.")
    print(f"\\nselesai dalam {{(time.time() - t0) / 60:.1f}} menit")

latih({arg_a!r}, TAG_A, f"SISI A: {{NAMA_A}}")"""))

    c.append(sel("markdown", f"""## Langkah 5 · Melatih dengan {nama_b}

Sekarang sisi yang diuji. Perintahnya **sama persis** dengan langkah 4 kecuali
satu bendera tambahan: `{' '.join(arg_b)}`.

Seed-nya juga sama, yaitu nilai yang diisi pada langkah 4. Ini penting dan
sering terlewat: bila seed-nya berbeda, selisih yang muncul dapat berasal dari
inisialisasi acaknya, bukan dari {b['pokok']}-nya. Dengan seed yang sama, kedua
model berangkat dari titik awal yang identik."""))

    c.append(sel("code", f"""latih({arg_b!r}, TAG_B, f"SISI B: {{NAMA_B}}")

if SIMPAN_HASIL_KE_DRIVE:
    from colab.siapkan import simpan_ke_drive
    simpan_ke_drive()"""))

    tambahan_do = ("" if jenis != "dropout" else """

**Laju yang dipelajari.** Bagian paling menarik dari perbandingan ini bukan
angka akurasinya, melainkan **berapa laju dropout yang akhirnya dipilih model
itu sendiri**. Kalau ia berhenti jauh di atas 0,2, artinya tebakan manusia tadi
terlalu longgar dan model sebenarnya butuh lebih banyak dijatuhkan. Kalau jauh
di bawah, sebaliknya. Angka itu dicetak di bawah.""")

    c.append(sel("markdown", f"""## Langkah 6 · Membandingkan keduanya

Tiga hal yang dibaca, dan urutannya penting.

**Kurva belajar.** Grafik pertama menunjukkan EER validasi di tiap putaran.
Ini memperlihatkan **bagaimana** keduanya belajar, bukan hanya di mana mereka
berakhir. Dua model bisa berakhir di angka yang sama lewat jalan yang sangat
berbeda, dan yang lebih cepat stabil punya nilainya sendiri.

**Angka akhir pada data uji.** Tabelnya memuat akurasi pada dua ambang, AUC,
dan EER. AUC dan EER tidak bergantung ambang, jadi keduanya pembanding yang
lebih jujur daripada akurasi.

**Ukuran selisihnya.** Ini yang paling penting dan paling sering dilewatkan.
Selisih beberapa poin **belum tentu berarti apa apa**, karena menjalankan model
yang sama dengan inisialisasi acak berbeda pun sudah menghasilkan selisih
sebesar itu. Sel di bawah membandingkan selisih yang terukur dengan ragam antar
inisialisasi yang sudah diukur di penelitian ini, lalu menyatakan terus terang
apakah selisihnya layak disebut nyata atau belum.{tambahan_do}"""))

    c.append(sel("code", f"""import json
import numpy as np
import matplotlib.pyplot as plt
from forlib.metrics import full_metrics, prior_matched_threshold

RAGAM_LOKAL = {{"wav2vec2": 0.51, "ast": 2.94, "hubert": 2.16, "cnnlstm": 2.28}}

def muat(tag):
    d = f"runs_colab/{{tag}}"
    if not os.path.exists(f"{{d}}/results.json"):
        raise SystemExit(f"Belum ada hasil di {{d}}. Jalankan langkah 4 dan 5 dulu.")
    y, p, _ = np.load(f"{{d}}/test_scores.npy")
    return json.load(open(f"{{d}}/results.json")), y.astype(int), p

res_a, y, p_a = muat(TAG_A)
res_b, _, p_b = muat(TAG_B)

# ---- kurva belajar
fig, ax = plt.subplots(1, 2, figsize=(13, 3.6), layout="constrained")
for res, nama_s, warna in [(res_a, NAMA_A, "#1864AB"), (res_b, NAMA_B, "#E8590C")]:
    h = res["history"]
    ax[0].plot([e["epoch"] for e in h], [e["val_eer"] * 100 for e in h],
               marker="o", ms=3.5, color=warna, lw=1.8, label=nama_s)
    ax[1].plot([e["epoch"] for e in h], [e["loss"] for e in h],
               marker="o", ms=3.5, color=warna, lw=1.8, label=nama_s)
ax[0].set_title("EER validasi tiap putaran (makin rendah makin baik)")
ax[0].set_xlabel("putaran"); ax[0].set_ylabel("EER validasi (persen)")
ax[1].set_title("Loss latih tiap putaran")
ax[1].set_xlabel("putaran"); ax[1].set_ylabel("loss")
for a in ax:
    a.legend(); a.grid(alpha=0.25)
plt.show()

# ---- angka akhir
def angka(p):
    return (full_metrics(y, p, 0.5), full_metrics(y, p, prior_matched_threshold(p, 0.5)))

(a05, apm), (b05, bpm) = angka(p_a), angka(p_b)
print(f"{{'ukuran':<28}}{{NAMA_A:>18}}{{NAMA_B:>18}}{{'selisih':>11}}")
print("-" * 76)
baris_tabel = [
    ("akurasi @ ambang 0,5", a05["accuracy"] * 100, b05["accuracy"] * 100, "%"),
    ("akurasi @ disesuaikan", apm["accuracy"] * 100, bpm["accuracy"] * 100, "%"),
    ("AUC", a05["auc"], b05["auc"], ""),
    ("EER (makin kecil baik)", a05["eer"] * 100, b05["eer"] * 100, "%"),
    ("putaran terbaik", res_a["best_epoch"], res_b["best_epoch"], "int"),
]
for nama_u, va, vb, sat in baris_tabel:
    if sat == "%":
        print(f"{{nama_u:<28}}{{va:>17.2f}}%{{vb:>17.2f}}%{{vb - va:>+10.2f}}")
    elif sat == "int":
        print(f"{{nama_u:<28}}{{va:>18d}}{{vb:>18d}}{{vb - va:>+11d}}")
    else:
        print(f"{{nama_u:<28}}{{va:>18.4f}}{{vb:>18.4f}}{{vb - va:>+11.4f}}")
menit_a = sum(e["sec"] for e in res_a["history"]) / 60
menit_b = sum(e["sec"] for e in res_b["history"]) / 60
print(f"{{'waktu latih (menit)':<28}}{{menit_a:>18.1f}}{{menit_b:>18.1f}}{{menit_b - menit_a:>+11.1f}}")
print("-" * 76)

# ---- laju dropout yang dipelajari, kalau ada
if "dropout_dipelajari" in res_b:
    laju = res_b["dropout_dipelajari"]
    print(f"\\nlaju dropout yang DIPELAJARI model: "
          f"{{', '.join(f'{{v:.4f}}' for v in laju)}}")
    print("laju tetap yang dipakai sisi A    : 0,2000")
    arah = "lebih banyak" if laju[0] > 0.2 else "lebih sedikit"
    print(f"Model memilih menjatuhkan {{arah}} unit daripada tebakan manusia.")

# ---- apakah selisihnya layak disebut nyata
selisih = (bpm["accuracy"] - apm["accuracy"]) * 100
ragam = RAGAM_LOKAL[MODEL]
print(f"\\nselisih akurasi yang terukur          : {{selisih:+.2f}} poin")
print(f"ragam antar inisialisasi model ini    : sekitar {{ragam:.2f}} poin")
if abs(selisih) < ragam:
    print("\\nSelisihnya LEBIH KECIL daripada ragam antar inisialisasi.")
    print("Dengan satu inisialisasi, ini BELUM bisa disebut perbedaan.")
    print("Ulangi langkah 4 dan 5 dengan SEED 1337 lalu 2024 sebelum menyimpulkan.")
else:
    print("\\nSelisihnya LEBIH BESAR daripada ragam antar inisialisasi satu model.")
    print("Itu menjanjikan, tetapi satu inisialisasi tetap belum cukup.")
    print("Ulangi dengan SEED 1337 lalu 2024 untuk memastikan.")"""))

    c.append(sel("markdown", f"""## Penutup

Satu inisialisasi tidak cukup untuk menyimpulkan {b['pokok']} mana yang lebih
baik. Itu bukan kehati hatian berlebihan, melainkan temuan yang diukur langsung
di penelitian ini: menjalankan HuBERT delapan kali dengan pengaturan yang persis
sama menghasilkan akurasi antara 93,93 dan 99,45 persen. Jaraknya 5,52 poin,
dari model yang sama.

Untuk {s['nama']}, ragam antar inisialisasi pada konfigurasi ini sekitar
**{s['hasil_lokal'].split('(')[1].rstrip(')')}**.

Cara menutup perbandingan ini dengan benar:

1. Jalankan langkah 4 sampai 6 dengan `SEED = 42`, lalu `1337`, lalu `2024`.
2. Ambil rata rata tiap sisi beserta simpangannya.
3. Baru bandingkan rata ratanya, dan hanya sebut berbeda bila selisihnya
   melampaui simpangan itu.

Sebelum ketiganya selesai, yang didukung data dari notebook ini hanyalah
selisih yang teramati pada satu inisialisasi. Pernyataan bahwa salah satu
{b['pokok']} lebih baik adalah klaim yang berbeda dan belum berdasar. Batas itu
disengaja, karena penelitian ini justru menemukan bahwa sebagian kesimpulan
yang ditarik terlalu dini akhirnya tidak bertahan.

Dua belas notebook di rangkaian ini:

| model | proses | optimizer | dropout |
|---|---|---|---|
{chr(10).join(f"| {SPEK[k]['nama']} | `{berkas(k, 'proses')}` | `{berkas(k, 'optimizer')}` | `{berkas(k, 'dropout')}` |" for k in SPEK)}"""))

    return notebook(c, berkas(m, jenis).replace(".ipynb", ""))


# ==========================================================================
if __name__ == "__main__":
    total = 0
    for m in SPEK:
        for jenis in ("proses", "optimizer", "dropout"):
            nb = bangun_proses(m) if jenis == "proses" else bangun_banding(m, jenis)
            path = os.path.join(HERE, berkas(m, jenis))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(nb, f, ensure_ascii=False, indent=1)
            n_kode = sum(1 for x in nb["cells"] if x["cell_type"] == "code")
            print(f"{berkas(m, jenis):<28} {len(nb['cells']):>2} sel "
                  f"({n_kode} kode)   {os.path.getsize(path) / 1024:>5.0f} KB")
            total += 1
    print(f"\n{total} notebook. Tautan Colab:\n")
    for m in SPEK:
        print(f"  {SPEK[m]['nama']}")
        for jenis in ("proses", "optimizer", "dropout"):
            print(f"    {jenis:<10} {tautan(m, jenis)}")
