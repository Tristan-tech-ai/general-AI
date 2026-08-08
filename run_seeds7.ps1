# Menyamakan konfigurasi temuan pembuka.
#
# KEKELIRUAN YANG DIPERBAIKI. Run asli yang menghasilkan angka 99,94 dan 50,00
# memakai 6 epoch dengan batch 64, karena dijalankan pada tahap awal penelitian
# sebelum penanda batch dan epoch ditambahkan ke nama direktori. Inisialisasi
# tambahan yang dijalankan belakangan memakai 10 epoch dengan batch 32, yaitu
# konfigurasi yang dipakai seluruh eksperimen lain.
#
# Menggabungkan keduanya berarti merata-ratakan dua konfigurasi yang berbeda dan
# menyebutnya ragam antar inisialisasi, padahal sebagiannya ragam antar
# konfigurasi. Kekeliruan itu persis jenis yang dikritik penelitian ini di
# tempat lain, dan terjadi karena skrip sebelumnya disusun tanpa memeriksa
# konfigurasi run aslinya terlebih dahulu.
#
# Perbaikannya adalah menjalankan seed 42 pada konfigurasi yang baru, sehingga
# ketiga inisialisasi berada pada konfigurasi yang sama. Run asli dengan 6 epoch
# tetap disimpan sebagai catatan sejarah dan tidak dihapus.

Set-Location $PSScriptRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"

foreach ($sp in @("random", "official")) {
    Write-Output "##### SEED7 cnn_asp / $sp / seed 42 / konfigurasi baru #####"
    py -u train.py --model cnn_asp --split $sp --augment none `
        --epochs 10 --batch 32 --workers 4 --seed 42 2>&1 |
        Select-Object -Last 6
}
Write-Output "##### SEED7 SELESAI #####"
