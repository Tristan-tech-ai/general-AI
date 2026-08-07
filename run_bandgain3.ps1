# Inisialisasi tambahan untuk titik optimum band-gain yang baru.
#
# Sapuan redaman menghasilkan kurva berpuncak yang bersih setelah jangkar nol
# dipasang:
#
#   0 dB (tanpa band-gain)  98,36
#   1 dB                    99,08
#   3 dB                    99,45   <- puncak
#   6 dB                    99,08
#   12 dB (bawaan)          98,65
#   20 dB                   97,43
#
# Nilai di kedua sisi puncak hampir sama persis, yaitu 99,08 pada 1 dB dan 99,08
# pada 6 dB, sehingga bentuk puncaknya tidak bergantung pada satu titik tunggal.
#
# Titik konfirmasi pada tahap sebelumnya masih memakai 6 dB karena disusun
# sebelum 3 dB diuji. Skrip ini memberi inisialisasi tambahan pada titik 3 dB
# yang sebenarnya, dan pada 1 dB sebagai pembanding terdekatnya, supaya klaim
# mengenai optimum tidak berdiri di atas satu inisialisasi.

Set-Location $PSScriptRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"

foreach ($s in @(1337, 2024)) {
    foreach ($db in @(3, 1)) {
        Write-Output "##### BANDGAIN3 db $db / seed $s #####"
        py -u train.py --model wavlm --split official --augment fullbg `
            --epochs 10 --batch 16 --workers 4 --seed $s --bg-db $db 2>&1 |
            Select-Object -Last 6
    }
}
Write-Output "##### BANDGAIN3 SELESAI #####"
