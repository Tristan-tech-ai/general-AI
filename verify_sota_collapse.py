"""
Verifikasi temuan: apakah model SOTA runtuh menjadi 'semua palsu' di luar domainnya?

Petunjuk awal: pada FoR-2sec zero-shot, akurasi 38,79% dengan recall-spoof 77,57%.
Bila kelas seimbang 544/544, kombinasi itu hanya mungkin bila hampir semua audio
ASLI juga ditandai palsu. Skrip ini menghitung confusion matrix sebenarnya.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.data import load_manifest, make_splits
from forlib.metrics import full_metrics, compute_eer
from load_sota import SotaNes2Net
from eval_sota_modern import score

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def report(y, p):
    L = []

    def out(s=""):
        print(s)
        L.append(s)

    out("# Verifikasi: Runtuhnya Kalibrasi Model SOTA di Luar Domain\n")
    out("Model: Nes2Net-X + XLS-R, checkpoint resmi, EER **1,49%** pada "
        "ASVspoof 2021 DF.")
    out("Diuji ZERO-SHOT pada FoR-2sec (1.088 berkas seimbang), "
        "tanpa adaptasi apa pun.\n")

    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    out("## Confusion matrix pada ambang 0,5\n")
    out("| | prediksi: asli | prediksi: palsu |")
    out("|---|---|---|")
    out(f"| **asli** (n={int((y == 0).sum())}) | {tn} | **{fp}** |")
    out(f"| **palsu** (n={int((y == 1).sum())}) | {fn} | {tp} |")
    out("")
    out(f"- Akurasi: {(tp + tn) / len(y) * 100:.2f}%")
    out(f"- Recall (spoof): {tp / max(tp + fn, 1) * 100:.2f}%")
    out(f"- Spesifisitas (asli dikenali benar): "
        f"**{tn / max(tn + fp, 1) * 100:.2f}%**")
    out(f"- Proporsi SELURUH berkas ditandai palsu: **{pred.mean() * 100:.2f}%**")
    out("")

    out("## Distribusi skor\n")
    out("| kelas | n | min | p25 | median | p75 | maks |")
    out("|---|---|---|---|---|---|---|")
    for lab, nm in [(0, "asli"), (1, "palsu")]:
        s = p[y == lab]
        out(f"| {nm} | {len(s)} | {s.min():.4f} | {np.percentile(s, 25):.4f} | "
            f"{np.median(s):.4f} | {np.percentile(s, 75):.4f} | {s.max():.4f} |")
    out("")

    m = full_metrics(y, p, 0.5)
    eer, _ = compute_eer(y, p)
    out(f"AUC = **{m['auc']:.4f}**, EER = **{eer * 100:.2f}%**\n")
    if m["auc"] < 0.5:
        out(f"AUC di bawah 0,5 berarti pengurutan skornya **TERBALIK**: model "
            f"memberi skor 'palsu' LEBIH TINGGI kepada audio ASLI daripada audio "
            f"palsu. Bila polaritas dibalik, AUC menjadi **{1 - m['auc']:.4f}** — "
            f"jadi model tetap sangat diskriminatif, hanya arahnya terbalik pada "
            f"korpus ini.\n")

    out("## Tafsir\n")
    frac = float((p >= 0.5).mean())
    if frac > 0.9:
        out(f"Model menandai **{frac * 100:.1f}%** dari seluruh berkas sebagai "
            "palsu. Pada rezim seperti ini, 'detection rate' terhadap TTS modern "
            "kehilangan makna: nilai 98-100% dapat dicapai oleh detektor yang "
            "selalu menjawab 'palsu'. Recall wajib dibaca berpasangan dengan "
            "spesifisitas.\n")
    out("Implikasi untuk klaim kebaruan: literatur anti-spoofing melaporkan EER "
        "*dalam domain*, yang tidak menangkap keruntuhan kalibrasi lintas korpus "
        "seperti ini. Mengukurnya eksplisit — recall DAN spesifisitas pada korpus "
        "asing, bukan hanya EER in-domain — adalah celah evaluasi yang nyata.")

    open(os.path.join(HERE, "HASIL_SOTA_COLLAPSE.md"), "w",
         encoding="utf-8").write("\n".join(L))
    print("\n-> HASIL_SOTA_COLLAPSE.md")


def main():
    rows = load_manifest(os.path.join(HERE, "manifest.csv"))
    _, _, te = make_splits(rows, "official")
    y = np.array([r["label"] for r in te])

    model = SotaNes2Net().to(DEV).eval()
    p = score(model, [r["path"] for r in te])
    np.save(os.path.join(HERE, "sota_for_scores.npy"),
            np.stack([y.astype(float), p]))
    report(y, p)


if __name__ == "__main__":
    main()
