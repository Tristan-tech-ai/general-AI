# Menambah inisialisasi keempat dan kelima pada empat sel yang menopang dua
# perbandingan terpenting.
#
# Dua perbandingan yang selisihnya paling besar, yaitu 34,90 dan 45,34 poin
# persentase, justru berhenti di nilai p terkoreksi 0,0520. Penyebabnya bukan
# efeknya yang kecil melainkan derajat bebas yang sangat sedikit. Dengan tiga
# inisialisasi per sel, uji t Welch hanya memiliki sekitar dua derajat bebas,
# dan koreksi Holm atas enam perbandingan menaikkan nilai p mentah 0,0087 dan
# 0,0090 menjadi tepat di atas ambang.
#
# Menambah dua inisialisasi per sel menaikkan derajat bebas secara berarti dan
# akan menyelesaikan persoalan ini ke arah mana pun datanya mengarah.

Set-Location $PSScriptRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"

$jobs = @(
    @{m = "wavlm";  b = 16; cfg = "proposal"},
    @{m = "wavlm";  b = 16; cfg = "beku"},
    @{m = "hubert"; b = 32; cfg = "proposal"},
    @{m = "hubert"; b = 32; cfg = "dilatih"}
)

foreach ($s in @(7, 2718)) {
    foreach ($j in $jobs) {
        Write-Output "##### SEED4 $($j.m) / $($j.cfg) / seed $s #####"
        if ($j.cfg -eq "proposal") {
            py -u train.py --model $j.m --split official --augment proposal `
                --normalize peak --uniform-lr 0.001 --epochs 20 --patience 99 `
                --batch $j.b --workers 4 --seed $s 2>&1 | Select-Object -Last 6
        } elseif ($j.cfg -eq "dilatih") {
            py -u train.py --model $j.m --split official --augment full --unfreeze `
                --epochs 10 --batch $j.b --workers 4 --seed $s 2>&1 |
                Select-Object -Last 6
        } else {
            py -u train.py --model $j.m --split official --augment full `
                --epochs 10 --batch $j.b --workers 4 --seed $s 2>&1 |
                Select-Object -Last 6
        }
    }
}
Write-Output "##### SEED4 SELESAI #####"
