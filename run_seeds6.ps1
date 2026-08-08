# Memberi inisialisasi tambahan pada temuan pembuka penelitian.
#
# Temuan pertama, yaitu bahwa protokol pembagian data menentukan hasil, sampai
# saat ini bersandar pada satu inisialisasi acak di kedua sisinya. Temuan itu
# dikutip paling sering di seluruh naskah, namun justru paling sedikit
# dukungannya dalam hal jumlah pengulangan.
#
# Dua inisialisasi tambahan dijalankan untuk masing-masing sisi. Konfigurasinya
# persis seperti yang menghasilkan angka aslinya, yaitu CNN dengan attentive
# statistics pooling tanpa augmentasi sama sekali, sehingga satu-satunya yang
# berbeda antara kedua sisi adalah skema pembagian datanya.

Set-Location $PSScriptRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"

foreach ($s in @(1337, 2024)) {
    foreach ($sp in @("random", "official")) {
        Write-Output "##### SEED6 cnn_asp / $sp / seed $s #####"
        py -u train.py --model cnn_asp --split $sp --augment none `
            --epochs 10 --batch 32 --workers 4 --seed $s 2>&1 |
            Select-Object -Last 6
    }
}
Write-Output "##### SEED6 SELESAI #####"
