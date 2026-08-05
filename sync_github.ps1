# Sinkronisasi hasil ke GitHub. Dipanggil tiap iterasi loop.
# Hanya mengirim kode, dokumentasi, grafik, dan skor kecil (lihat .gitignore).
# Dataset, bobot model, dan arsip tidak pernah dikirim.

param([string]$Pesan = "")

Set-Location $PSScriptRoot

# regenerasi grafik & laporan agar yang dikirim selalu mutakhir
py make_charts.py 2>&1 | Out-Null
if (Test-Path generations_results.json) { py gen_report.py 2>&1 | Out-Null }
if (Test-Path generations_results.json) { py tradeoff_report.py 2>&1 | Out-Null }

git add -A 2>&1 | Out-Null
$berubah = (git status --porcelain | Measure-Object).Count
if ($berubah -eq 0) {
    Write-Output "tidak ada perubahan"
    exit 0
}

if ([string]::IsNullOrWhiteSpace($Pesan)) {
    $Pesan = "Pembaruan hasil eksperimen otomatis"
}
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
$full = @"
$Pesan

Sinkronisasi otomatis loop $stamp ($berubah berkas berubah).

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
"@

git -c user.name="Tristan-tech-ai" -c user.email="ifys4251@gmail.com" `
    commit -q -m $full 2>&1 | Out-Null
git push -q origin main 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Output "terkirim: $berubah berkas - $(git log --oneline -1)"
} else {
    Write-Output "PUSH GAGAL (exit $LASTEXITCODE)"
}
