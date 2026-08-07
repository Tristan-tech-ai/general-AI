# Memberi simpangan baku pada sel-sel matriks yang masih bersandar satu seed.
#
# Sel AST yang encodernya dilatih menunjukkan simpangan baku 1,85 dengan rentang
# 3,67 poin persentase antar inisialisasi. Ragam sebesar itu lebih lebar
# daripada selisih antar metodologi pada arsitektur tersebut, sehingga setiap
# sel yang masih n sama dengan satu belum dapat dipakai untuk menyimpulkan apa
# pun.
#
# Prioritasnya adalah sel-sel yang menjadi pembanding langsung bagi klaim yang
# masih hidup, yaitu sel berencoder beku dan sel proposal pada tiap arsitektur.
# Sel dengan laju 0,001 sengaja ditunda karena hasilnya berada di kisaran 43
# sampai 88 persen, yaitu terpisah jauh sehingga ragam antar inisialisasi tidak
# mungkin membalik kesimpulannya.

Set-Location $PSScriptRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"

$jobs = @(
    # pembanding langsung bagi sel AST yang encodernya dilatih
    @{m = "ast";    b = 32; seed = 1337; cfg = "beku"},
    @{m = "ast";    b = 32; seed = 2024; cfg = "beku"},
    # acuan proposal pada tiap arsitektur
    @{m = "ast";    b = 32; seed = 1337; cfg = "proposal"},
    @{m = "ast";    b = 32; seed = 2024; cfg = "proposal"},
    @{m = "wavlm";  b = 16; seed = 1337; cfg = "proposal"},
    @{m = "wavlm";  b = 16; seed = 2024; cfg = "proposal"},
    @{m = "hubert"; b = 32; seed = 1337; cfg = "proposal"},
    @{m = "hubert"; b = 32; seed = 2024; cfg = "proposal"}
)

foreach ($j in $jobs) {
    Write-Output "##### SEED2 $($j.m) / $($j.cfg) / seed $($j.seed) #####"
    if ($j.cfg -eq "proposal") {
        py -u train.py --model $j.m --split official --augment proposal `
            --normalize peak --uniform-lr 0.001 --epochs 20 --patience 99 `
            --batch $j.b --workers 4 --seed $j.seed 2>&1 | Select-Object -Last 6
    } else {
        py -u train.py --model $j.m --split official --augment full `
            --epochs 10 --batch $j.b --workers 4 --seed $j.seed 2>&1 |
            Select-Object -Last 6
    }
}
Write-Output "##### SEED2 SELESAI #####"
