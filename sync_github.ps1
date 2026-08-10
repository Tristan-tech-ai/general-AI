# Sinkronisasi hasil ke GitHub. Dipanggil tiap iterasi loop.
# Hanya mengirim kode, dokumentasi, grafik, dan skor kecil (lihat .gitignore).
# Dataset, bobot model, dan arsip tidak pernah dikirim.

param([string]$Pesan = "")

Set-Location $PSScriptRoot

# Pengaman sebelum apa pun dikirim: pastikan tidak ada kelompok run yang memuat
# lebih dari satu konfigurasi pelatihan tanpa alasan tercatat. Penggabungan
# semacam itu melaporkan ragam antar konfigurasi sebagai ragam antar
# inisialisasi acak, dan sudah menyebabkan tujuh kekeliruan dalam penelitian ini.
py cek_konfigurasi.py
if ($LASTEXITCODE -ne 0) {
    Write-Output "DIHENTIKAN: ada kelompok run dengan konfigurasi tercampur."
    Write-Output "Perbaiki atau daftarkan pengecualiannya sebelum menyinkronkan."
    exit 1
}

# Regenerasi gambar, laporan, dan naskah agar yang dikirim selalu mutakhir.
# Urutannya penting: gambar dibuat lebih dahulu, sebab naskah menyisipkannya.
py gambar_paper.py 2>&1 | Out-Null
if (Test-Path generations_results.json) { py gen_report.py 2>&1 | Out-Null }
if (Test-Path generations_results.json) { py tradeoff_report.py 2>&1 | Out-Null }
py naskah.py 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Output "DIHENTIKAN: naskah gagal dibangun, jadi angka pada PDF akan usang."
    exit 1
}

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
