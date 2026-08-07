# Memberi simpangan baku pada tangga ablasi.
#
# Tangga ablasi melaporkan selisih -16,91 poin persentase untuk pembekuan
# encoder, +9,01 untuk early stopping, dan +4,78 untuk augmentasi penuh.
# Ketiganya masih berdiri di atas satu inisialisasi acak, padahal sel AST yang
# sebanding menunjukkan simpangan baku 1,85 poin. Selisih +4,78 karena itu belum
# tentu bertahan, sedangkan -16,91 hampir pasti bertahan. Yang membedakan
# keduanya harus ditunjukkan dengan data, bukan diperkirakan.
#
# Titik tolak L1 dan langkah terakhir L5 sudah dijadwalkan seed tambahannya di
# run_seeds.ps1 dan run_seeds2.ps1, sehingga di sini hanya L2, L3, dan L4.

Set-Location $PSScriptRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"

$langkah = @(
    @{nama = "L2-loudness";  args = @("--augment", "proposal", "--uniform-lr", "0.001",
                                      "--epochs", "20", "--patience", "99")},
    @{nama = "L3-lr-beku";   args = @("--augment", "proposal",
                                      "--epochs", "20", "--patience", "99")},
    @{nama = "L4-earlystop"; args = @("--augment", "proposal", "--epochs", "10")}
)

foreach ($s in @(1337, 2024)) {
    foreach ($l in $langkah) {
        Write-Output "##### SEED3 $($l.nama) / seed $s #####"
        py -u train.py --model ast --split official --batch 32 --workers 4 `
            --seed $s @($l.args) 2>&1 | Select-Object -Last 6
    }
}
Write-Output "##### SEED3 SELESAI #####"
