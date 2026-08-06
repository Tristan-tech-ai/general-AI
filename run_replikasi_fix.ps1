# Run ulang seluruh arm replikasi proposal setelah perbaikan bug encoder beku.
#
# Sebelum perbaikan, bendera --uniform-lr menyetel requires_grad tetapi tidak
# mengubah atribut frozen, sehingga forward encoder tetap berjalan di dalam
# torch.no_grad() dan encoder tidak pernah menerima gradien. Rincian ada di
# KOREKSI_REPLIKASI_PROPOSAL.md.
#
# Hasil lama dipindahkan ke runs_pra_perbaikan/ dan tidak dihapus, supaya
# perbandingan sebelum dan sesudah tetap dapat ditelusuri.

Set-Location $PSScriptRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"

$arsip = Join-Path $PSScriptRoot "runs_pra_perbaikan"
if (-not (Test-Path $arsip)) { New-Item -ItemType Directory $arsip | Out-Null }

# CNN-LSTM dikecualikan. Kelas itu tidak punya atribut frozen sehingga tidak
# pernah terkena bug encoder beku, dan hasil lamanya tetap sahih.
@(Get-ChildItem -Path (Join-Path $PSScriptRoot "runs") -Directory -Filter "*ULR*" |
    Where-Object { $_.Name -notlike "cnnlstm*" }) |
    ForEach-Object {
        $tujuan = Join-Path $arsip $_.Name
        if (Test-Path $tujuan) { Remove-Item -Recurse -Force $tujuan }
        Move-Item $_.FullName $tujuan
        Write-Output "diarsipkan: $($_.Name)"
    }

# Model berbasis SSL saja. CNN-LSTM tidak punya atribut frozen sehingga
# hasil lamanya tetap sahih dan tidak perlu diulang.
$jobs = @(
    @{m = "ast";      b = 32; split = "official"},
    @{m = "ast";      b = 32; split = "random"},
    @{m = "wavlm";    b = 16; split = "official"},
    @{m = "wavlm";    b = 16; split = "random"},
    @{m = "hubert";   b = 32; split = "random"},
    @{m = "wav2vec2"; b = 32; split = "random"}
)

foreach ($j in $jobs) {
    Write-Output "##### REPLIKASI $($j.m) / $($j.split) #####"
    py -u train.py --model $j.m --split $j.split --augment proposal `
        --normalize peak --uniform-lr 0.001 --epochs 20 --patience 99 `
        --batch $j.b --workers 4 --seed 42 2>&1 | Select-Object -Last 6
}

# Ulangi langkah L2 tangga ablasi, yang juga memakai --uniform-lr.
Write-Output "##### REPLIKASI L2-loudness #####"
py -u train.py --model ast --split official --augment proposal `
    --uniform-lr 0.001 --epochs 20 --patience 99 `
    --batch 32 --workers 4 --seed 42 2>&1 | Select-Object -Last 6

Write-Output "##### REPLIKASI SELESAI #####"
