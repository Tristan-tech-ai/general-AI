# Lanjutan sapuan band-gain: menjelajah arah yang ditunjuk sumbu redaman, lalu
# mengonfirmasi titik-titik penting dengan inisialisasi tambahan.
#
# Sapuan pertama menghasilkan tiga pola yang berbeda satu sama lain:
#
#   f_lo      datar. Nilai 2000, 3000, dan 4000 Hz seluruhnya berada di dalam
#             simpangan baku konfigurasi bawaannya sendiri, yaitu 0,37 poin
#             persentase. Parameter ini tidak berpengaruh pada rentang tersebut.
#
#   n_bands   berpuncak. Nilai 3 dan 12 sama-sama lebih buruk sekitar 1,3 poin
#             daripada nilai 6. Bentuk ini sesuai dengan mekanisme yang mendasari
#             band-gain, yaitu pita yang terlalu lebar gagal menetralkan isyarat
#             level di dalamnya sedangkan pita yang terlalu sempit mulai merusak
#             struktur halus tempat artefak vocoder berada.
#
#   db        monoton pada rentang yang diuji. Redaman 6 dB lebih baik daripada
#             12 dB, dan 20 dB paling buruk. Tidak ada tanda puncak, sehingga
#             nilai yang lebih lembut lagi belum tentu memburuk dan perlu diuji.
#
# Karena itu skrip ini menjelajah dahulu ke arah redaman yang lebih lembut,
# lalu memberi inisialisasi tambahan pada titik-titik yang selisihnya sudah
# melampaui derau. Titik f_lo sengaja tidak diulang karena menambah inisialisasi
# di sana tidak akan mengubah kesimpulan bahwa parameternya tidak berpengaruh.

Set-Location $PSScriptRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"

# Tahap satu: menjelajah ujung lembut sumbu redaman pada satu inisialisasi.
foreach ($db in @(3, 1)) {
    Write-Output "##### BANDGAIN2 jelajah db $db / seed 42 #####"
    py -u train.py --model wavlm --split official --augment fullbg `
        --epochs 10 --batch 16 --workers 4 --seed 42 --bg-db $db 2>&1 |
        Select-Object -Last 6
}

# Tahap dua: inisialisasi tambahan untuk titik yang selisihnya melampaui derau.
$titik = @(
    @{nama = "db 6";     args = @("--bg-db", "6")},
    @{nama = "bands 3";  args = @("--bg-bands", "3")},
    @{nama = "bands 12"; args = @("--bg-bands", "12")}
)
foreach ($s in @(1337, 2024)) {
    foreach ($t in $titik) {
        Write-Output "##### BANDGAIN2 $($t.nama) / seed $s #####"
        py -u train.py --model wavlm --split official --augment fullbg `
            --epochs 10 --batch 16 --workers 4 --seed $s @($t.args) 2>&1 |
            Select-Object -Last 6
    }
}
Write-Output "##### BANDGAIN2 SELESAI #####"
