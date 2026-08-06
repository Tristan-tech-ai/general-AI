# Lengkapi matriks 2x2 apple-to-apple untuk dua arsitektur.
#
# Dua konfigurasi yang dibandingkan:
#   PROPOSAL   : LR 0,001 seragam (encoder ikut dilatih), 20 epoch tanpa early
#                stopping, normalisasi peak, augmentasi noise SNR 15-30 dB,
#                ambang keputusan 0,5
#   DIPERBAIKI : LR per model (encoder beku + layer weighting), 10 epoch dengan
#                early stopping pada EER, normalisasi loudness, augmentasi penuh,
#                ambang prior-matched
#
# Keduanya dijalankan pada kedua skema split. Batch dijaga konstan di dalam tiap
# arsitektur supaya setiap perbandingan hanya mengubah satu variabel.

Set-Location $PSScriptRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"

$jobs = @(
    @{m = "ast";   b = 32; split = "official"; cfg = "proposal"},
    @{m = "ast";   b = 32; split = "random";   cfg = "diperbaiki"},
    @{m = "ast";   b = 32; split = "official"; cfg = "diperbaiki"},
    @{m = "wavlm"; b = 16; split = "random";   cfg = "proposal"},
    @{m = "wavlm"; b = 16; split = "official"; cfg = "proposal"},
    @{m = "wavlm"; b = 16; split = "random";   cfg = "diperbaiki"}
)

foreach ($j in $jobs) {
    Write-Output "##### 2x2 $($j.m) / $($j.split) / $($j.cfg) #####"
    if ($j.cfg -eq "proposal") {
        py -u train.py --model $j.m --split $j.split --augment proposal `
            --normalize peak --uniform-lr 0.001 --epochs 20 --patience 99 `
            --batch $j.b --workers 4 --seed 42 2>&1 | Select-Object -Last 3
    } else {
        py -u train.py --model $j.m --split $j.split --augment full `
            --epochs 10 --batch $j.b --workers 4 --seed 42 2>&1 |
            Select-Object -Last 3
    }
}
Write-Output "##### 2x2 SELESAI #####"
