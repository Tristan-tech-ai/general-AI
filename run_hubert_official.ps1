# Menutup celah pada temuan interaksi arsitektur dengan learning rate.
#
# Sejauh ini learning rate seragam 0,001 terbukti menolong AST (86 juta
# parameter, pra-latih terselia) dan merusak WavLM Large (300 juta parameter,
# swa-selia). Dugaan yang muncul adalah bahwa yang menentukan adalah ukuran dan
# jenis pra-pelatihan encoder, bukan identitas modelnya.
#
# HuBERT Large adalah uji yang tepat untuk dugaan itu. Ukurannya sama dengan
# WavLM Large dan sama-sama swa-selia, tetapi korpus dan tujuan pra-pelatihannya
# berbeda. Bila HuBERT juga runtuh, dugaan tersebut menguat. Bila tidak, berarti
# penyebabnya khusus pada WavLM dan harus dicari lebih jauh.
#
# Dugaan dicatat di muka: HuBERT Large akan ikut runtuh pada partisi resmi,
# dengan AUC jatuh jauh di bawah versi berencoder beku.

Set-Location $PSScriptRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"

Write-Output "##### HUBERT proposal / partisi resmi #####"
py -u train.py --model hubert --split official --augment proposal `
    --normalize peak --uniform-lr 0.001 --epochs 20 --patience 99 `
    --batch 32 --workers 4 --seed 42 2>&1 | Select-Object -Last 6

Write-Output "##### HUBERT rekayasa beku / partisi resmi #####"
py -u train.py --model hubert --split official --augment full `
    --epochs 10 --batch 32 --workers 4 --seed 42 2>&1 | Select-Object -Last 6

Write-Output "##### HUBERT encoder dilatih laju wajar / partisi resmi #####"
py -u train.py --model hubert --split official --augment full --unfreeze `
    --epochs 10 --batch 32 --workers 4 --seed 42 2>&1 | Select-Object -Last 6

Write-Output "##### HUBERT SELESAI #####"
