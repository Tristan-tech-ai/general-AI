# Menguji ulang keputusan desain yang ternyata paling merugikan: membekukan encoder.
#
# Replikasi proposal yang sudah dibetulkan menunjukkan bahwa fine-tuning encoder
# jauh lebih kuat daripada membekukannya. Konfigurasi rekayasa kita membekukan
# encoder, dan itu ternyata membuang lebih banyak daripada yang dibeli oleh
# seluruh perbaikan lainnya.
#
# Skrip ini menggabungkan sisi terbaik dari keduanya, yaitu encoder yang ikut
# dilatih seperti pada proposal, ditambah early stopping, augmentasi penuh,
# agregasi berbobot antar lapisan, dan learning rate encoder yang wajar.
#
# Dua laju encoder diuji supaya dapat dipisahkan apakah yang menolong adalah
# fine-tuning itu sendiri atau justru laju tinggi yang dipakai proposal.

Set-Location $PSScriptRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"

$jobs = @(
    # rekayasa penuh + encoder dilatih pada laju konservatif bawaan per model
    @{m = "ast";   b = 32; enc = $null},
    @{m = "wavlm"; b = 16; enc = $null},
    # rekayasa penuh + encoder dilatih pada laju tinggi seperti proposal
    @{m = "ast";   b = 32; enc = "0.001"},
    @{m = "wavlm"; b = 16; enc = "0.001"}
)

foreach ($j in $jobs) {
    $tag = if ($null -eq $j.enc) { "enc-bawaan" } else { "enc-$($j.enc)" }
    Write-Output "##### UNFREEZE $($j.m) / $tag #####"
    $extra = @()
    if ($null -ne $j.enc) { $extra = @("--enc-lr", $j.enc) }
    py -u train.py --model $j.m --split official --augment full --unfreeze `
        --epochs 10 --batch $j.b --workers 4 --seed 42 @extra 2>&1 |
        Select-Object -Last 6
}
Write-Output "##### UNFREEZE SELESAI #####"
