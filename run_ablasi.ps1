# Ablasi bertahap: memecah +37,59 poin menjadi kontribusi tiap perbaikan.
#
# Bertolak dari konfigurasi proposal pada partisi resmi (AST = 51,56 persen),
# lalu menambahkan satu perbaikan pada satu waktu. Selisih tiap langkah adalah
# kontribusi perbaikan tersebut.
#
# Semua dijalankan pada AST, partisi resmi, batch 32, seed 42, sehingga hanya
# satu variabel berubah di tiap langkah.

Set-Location $PSScriptRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"

$langkah = @(
    # 1. dasar: konfigurasi proposal (sudah ada, tidak dijalankan ulang)
    # 2. hanya ganti normalisasi peak -> loudness
    @{nama = "L2-loudness"; args = @("--augment", "proposal", "--uniform-lr", "0.001",
                                     "--epochs", "20", "--patience", "99")},
    # 3. loudness + LR per model dengan encoder beku
    @{nama = "L3-lr-beku"; args = @("--augment", "proposal",
                                    "--epochs", "20", "--patience", "99")},
    # 4. + early stopping pada EER (10 epoch)
    @{nama = "L4-earlystop"; args = @("--augment", "proposal", "--epochs", "10")},
    # 5. + augmentasi penuh menggantikan noise-saja
    @{nama = "L5-augpenuh"; args = @("--augment", "full", "--epochs", "10")}
)

foreach ($s in $langkah) {
    Write-Output "##### ABLASI $($s.nama) #####"
    py -u train.py --model ast --split official --batch 32 --workers 4 --seed 42 `
        @($s.args) 2>&1 | Select-Object -Last 3
}
Write-Output "##### ABLASI SELESAI #####"
