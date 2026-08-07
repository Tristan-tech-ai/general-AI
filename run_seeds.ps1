# Menambah inisialisasi acak pada sel-sel yang keputusannya masih bergantung
# pada satu seed.
#
# Dua klaim dalam penelitian ini masih berdiri di atas satu inisialisasi:
#   * WavLM Large lebih baik dibekukan daripada dilatih. Sel yang dibekukan
#     memakai tiga seed, sel yang dilatih baru satu.
#   * HuBERT Large lebih baik dilatih daripada dibekukan. Keduanya baru satu.
#
# Tanpa sebaran, selisih beberapa poin persentase tidak dapat dibedakan dari
# ragam antar inisialisasi. Sel AST yang sebanding sudah menunjukkan simpangan
# baku sekitar satu poin, yang cukup besar untuk membalik kesimpulan.

Set-Location $PSScriptRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"

$jobs = @(
    @{m = "wavlm";  b = 16; seed = 1337; uf = $true},
    @{m = "wavlm";  b = 16; seed = 2024; uf = $true},
    @{m = "hubert"; b = 32; seed = 1337; uf = $true},
    @{m = "hubert"; b = 32; seed = 2024; uf = $true},
    # sel HuBERT yang dibekukan juga baru satu seed, padahal ia pembanding
    # langsung bagi kedua sel di atas
    @{m = "hubert"; b = 32; seed = 1337; uf = $false},
    @{m = "hubert"; b = 32; seed = 2024; uf = $false}
)

foreach ($j in $jobs) {
    $mode = if ($j.uf) { "encoder dilatih" } else { "encoder beku" }
    Write-Output "##### SEED $($j.m) / $mode / seed $($j.seed) #####"
    $extra = @()
    if ($j.uf) { $extra = @("--unfreeze") }
    py -u train.py --model $j.m --split official --augment full `
        --epochs 10 --batch $j.b --workers 4 --seed $j.seed @extra 2>&1 |
        Select-Object -Last 6
}
Write-Output "##### SEED SELESAI #####"
