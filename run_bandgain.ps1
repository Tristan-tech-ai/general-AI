# Tuning parameter augmentasi band-gain.
#
# Band-gain adalah usulan orisinal penelitian ini, namun ketiga parameternya
# ditetapkan sekali di awal dan tidak pernah diuji. Nilai bawaannya adalah
# f_lo 3000 Hz, enam pita, dan redaman sampai 12 dB.
#
# Gagasan mekanistiknya menyatakan bahwa yang perlu dinetralkan adalah LEVEL
# energi pita tinggi, sedangkan STRUKTUR HALUS di dalam pita harus tetap utuh.
# Ketiga parameter itu mengendalikan keseimbangan tersebut secara langsung:
#   f_lo    menentukan dari frekuensi berapa penetralan dimulai
#   n_bands menentukan sehalus apa penetralannya, semakin banyak pita semakin
#           menyerupai penyaringan yang juga merusak struktur
#   db      menentukan sekuat apa level diacak
#
# Dijalankan pada WavLM Large dengan encoder dibekukan, yaitu konfigurasi
# terbaik dan paling stabil dalam penelitian ini (simpangan baku 0,64), supaya
# selisih antar parameter tidak tenggelam dalam ragam antar inisialisasi.
#
# Satu variabel diubah pada satu waktu, dengan dua yang lain dipertahankan pada
# nilai bawaannya.

Set-Location $PSScriptRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"

$sapuan = @(
    @{nama = "f_lo 2000"; args = @("--bg-f-lo", "2000")},
    @{nama = "f_lo 4000"; args = @("--bg-f-lo", "4000")},
    @{nama = "bands 3";   args = @("--bg-bands", "3")},
    @{nama = "bands 12";  args = @("--bg-bands", "12")},
    @{nama = "db 6";      args = @("--bg-db", "6")},
    @{nama = "db 20";     args = @("--bg-db", "20")}
)

foreach ($s in $sapuan) {
    Write-Output "##### BANDGAIN $($s.nama) #####"
    py -u train.py --model wavlm --split official --augment fullbg `
        --epochs 10 --batch 16 --workers 4 --seed 42 @($s.args) 2>&1 |
        Select-Object -Last 6
}
Write-Output "##### BANDGAIN SELESAI #####"
