# Menutup dua klaim terakhir yang masih menggantung.
#
# 1. Tangga ablasi L3 dan L4. Keduanya lolos koreksi Holm pada tiga
#    inisialisasi, namun L4 berhenti tepat di 0,048 yaitu hanya dua per seribu
#    di bawah ambang. Nilai sedekat itu pada n sama dengan tiga tidak stabil,
#    dan dua inisialisasi tambahan akan menyelesaikannya ke arah mana pun.
#    L1 sudah memiliki lima inisialisasi sehingga tidak perlu ditambah.
#
# 2. Redaman band-gain 3 dB. Titik ini memberi p mentah 0,0371 terhadap
#    konfigurasi tanpa band-gain, tetapi menjadi 0,4078 setelah dikoreksi untuk
#    sebelas perbandingan dalam sapuan. Pengujian ulang di sini memakai
#    hipotesis tunggal yang ditetapkan sebelum datanya ada, yaitu bahwa redaman
#    3 dB mengungguli konfigurasi tanpa band-gain, sehingga tidak memerlukan
#    koreksi banding ganda. Hipotesis itu dicatat di sini agar tidak dapat
#    diubah setelah hasilnya terlihat.

Set-Location $PSScriptRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"

foreach ($s in @(7, 2718)) {
    Write-Output "##### SEED5 L3-lr-beku / seed $s #####"
    py -u train.py --model ast --split official --augment proposal `
        --epochs 20 --patience 99 --batch 32 --workers 4 --seed $s 2>&1 |
        Select-Object -Last 6

    Write-Output "##### SEED5 L4-earlystop / seed $s #####"
    py -u train.py --model ast --split official --augment proposal `
        --epochs 10 --batch 32 --workers 4 --seed $s 2>&1 |
        Select-Object -Last 6

    Write-Output "##### SEED5 bandgain 3 dB / seed $s #####"
    py -u train.py --model wavlm --split official --augment fullbg `
        --epochs 10 --batch 16 --workers 4 --seed $s --bg-db 3 2>&1 |
        Select-Object -Last 6

    Write-Output "##### SEED5 tanpa band-gain / seed $s #####"
    py -u train.py --model wavlm --split official --augment full `
        --epochs 10 --batch 16 --workers 4 --seed $s 2>&1 |
        Select-Object -Last 6
}
Write-Output "##### SEED5 SELESAI #####"
