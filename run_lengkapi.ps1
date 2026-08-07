# Melengkapi dua celah yang tersisa pada matriks arsitektur terhadap perlakuan
# encoder.
#
# 1. HuBERT Large dengan encoder dilatih pada laju 0,001. Sel ini belum pernah
#    dijalankan, sehingga baris HuBERT pada matriks masih berlubang.
# 2. Dua inisialisasi acak tambahan untuk konfigurasi terbaik pada AST, supaya
#    angka 93,20 persen memiliki simpangan baku dan tidak berdiri di atas satu
#    seed saja. Sel WavLM yang sebanding sudah memiliki tiga seed.

Set-Location $PSScriptRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"

Write-Output "##### LENGKAPI hubert / encoder dilatih laju 0,001 #####"
py -u train.py --model hubert --split official --augment full --unfreeze `
    --enc-lr 0.001 --epochs 10 --batch 32 --workers 4 --seed 42 2>&1 |
    Select-Object -Last 6

foreach ($s in @(1337, 2024)) {
    Write-Output "##### LENGKAPI ast / encoder dilatih / seed $s #####"
    py -u train.py --model ast --split official --augment full --unfreeze `
        --epochs 10 --batch 32 --workers 4 --seed $s 2>&1 |
        Select-Object -Last 6
}

Write-Output "##### LENGKAPI SELESAI #####"
