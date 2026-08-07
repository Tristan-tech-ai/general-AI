"""
Bangun paper PDF dari hasil eksperimen yang tersimpan.

Menghasilkan PAPER.pdf: dokumen berformat makalah ilmiah, bahasa Indonesia,
dengan tabel dan grafik yang dibaca langsung dari berkas hasil.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, KeepTogether)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.metrics import full_metrics, prior_matched_threshold

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "PAPER.pdf")

# ---------------------------------------------------------------- gaya
ss = getSampleStyleSheet()
S = {
    "judul": ParagraphStyle("judul", parent=ss["Title"], fontSize=17, leading=21,
                            spaceAfter=6),
    "sub": ParagraphStyle("sub", parent=ss["Normal"], fontSize=10.5,
                          alignment=TA_CENTER, textColor=colors.HexColor("#444444"),
                          spaceAfter=14),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=13, leading=16,
                         spaceBefore=14, spaceAfter=6,
                         textColor=colors.HexColor("#0B7285")),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, leading=14,
                         spaceBefore=10, spaceAfter=4,
                         textColor=colors.HexColor("#1864AB")),
    "p": ParagraphStyle("p", parent=ss["Normal"], fontSize=9.6, leading=14,
                        alignment=TA_JUSTIFY, spaceAfter=6),
    "cap": ParagraphStyle("cap", parent=ss["Normal"], fontSize=8.4, leading=11,
                          textColor=colors.HexColor("#555555"), spaceAfter=10,
                          alignment=TA_CENTER),
    "sel": ParagraphStyle("sel", parent=ss["Normal"], fontSize=8, leading=10),
    "selb": ParagraphStyle("selb", parent=ss["Normal"], fontSize=8, leading=10,
                           fontName="Helvetica-Bold"),
}


def P(t, s="p"):
    return Paragraph(t, S[s])


def tabel(header, baris, lebar=None):
    data = [[Paragraph(h, S["selb"]) for h in header]]
    for r in baris:
        data.append([Paragraph(str(c), S["sel"]) for c in r])
    t = Table(data, colWidths=lebar, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7F5F8")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#0B7285")),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#DDDDDD")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#FAFAFA")]),
    ]))
    return t


def gambar(nama, lebar=15.5 * cm, cap=""):
    p = os.path.join(HERE, "charts", nama)
    if not os.path.exists(p):
        return []
    from PIL import Image as PILImage
    w, h = PILImage.open(p).size
    im = Image(p, width=lebar, height=lebar * h / w)
    out = [im]
    if cap:
        out.append(P(cap, "cap"))
    return out


# ---------------------------------------------------------------- data
def parse(tag):
    m = re.match(r"^(.+?)_(official|random|clean_val|wavval)_([a-zA-Z]+?)"
                 r"(?:_b\d+e\d+)?_s(\d+)$", tag)
    return None if not m else (m.group(1), m.group(2), m.group(3), m.group(4))


def kumpul_for():
    g = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
        f = os.path.join(d, "test_scores.npy")
        p = parse(os.path.basename(d))
        if not (os.path.exists(f) and p):
            continue
        y, s, _ = np.load(f)
        m = full_metrics(y.astype(int), s, prior_matched_threshold(s, 0.5))
        m05 = full_metrics(y.astype(int), s, 0.5)
        g[(p[0], p[1], p[2])].append({"acc": m["accuracy"], "eer": m["eer"],
                                      "auc": m["auc"], "acc05": m05["accuracy"],
                                      "n": m["n"]})
    return g


def kumpul_gen():
    f = os.path.join(HERE, "generations_results.json")
    if not os.path.exists(f):
        return {}
    G = json.load(open(f, encoding="utf-8"))
    OLD = ["tts_models_en_ljspeech_tacotron2-DDC",
           "tts_models_en_ljspeech_speedy-speech",
           "tts_models_en_ljspeech_vits"]
    g = defaultdict(list)
    for tag, r in G.items():
        p = parse(tag)
        if not p:
            continue
        md = [v["recall"] for v in r["tts"].values()
              if v["era"] == "2025-2026 komersial"]
        ol = [r["tts"][k]["recall"] for k in OLD if k in r["tts"]]
        if md:
            g[(p[0], p[2])].append({"modern": np.mean(md),
                                    "old": np.mean(ol) if ol else np.nan})
    return g


def matriks_2x2():
    """Empat sel per arsitektur: konfigurasi proposal atau diperbaiki, pada
    split acak atau partisi resmi. Ambang mengikuti konfigurasinya masing-masing,
    yaitu 0,5 untuk proposal dan prior-matched untuk versi diperbaiki, karena
    ambang termasuk bagian dari metodologi yang dibandingkan."""
    hasil = {}
    for model in ["ast", "wavlm", "hubert", "nes2net"]:
        sel = {}
        for cfg in ["proposal", "diperbaiki"]:
            pat = (f"runs/{model}_%s_proposalULRPK_*" if cfg == "proposal"
                   else f"runs/{model}_%s_full_*")
            for split in ["random", "official"]:
                acc = []
                for d in sorted(glob.glob(os.path.join(HERE, pat % split))):
                    f = os.path.join(d, "test_scores.npy")
                    if not os.path.exists(f):
                        continue
                    y, p, _ = np.load(f)
                    y = y.astype(int)
                    t = 0.5 if cfg == "proposal" else prior_matched_threshold(p, 0.5)
                    acc.append(full_metrics(y, p, t)["accuracy"] * 100)
                sel[(cfg, split)] = acc
        if all(sel.values()):
            hasil[model] = sel
    return hasil


def _kumpul_ambang(pat):
    """Rerata akurasi pada kedua ambang, ditambah AUC dan EER yang tidak
    bergantung pada ambang, dari seluruh run yang cocok dengan pola."""
    a05, apm, auc, eer = [], [], [], []
    for d in sorted(glob.glob(os.path.join(HERE, pat))):
        f = os.path.join(d, "test_scores.npy")
        if not os.path.exists(f):
            continue
        y, p, _ = np.load(f)
        y = y.astype(int)
        m0 = full_metrics(y, p, 0.5)
        a05.append(m0["accuracy"] * 100)
        apm.append(full_metrics(y, p, prior_matched_threshold(p, 0.5))["accuracy"] * 100)
        auc.append(m0["auc"])
        eer.append(m0["eer"] * 100)
    if not a05:
        return None
    return {"a05": np.mean(a05), "apm": np.mean(apm),
            "auc": np.mean(auc), "eer": np.mean(eer),
            # alias untuk tabel yang melaporkan akurasi prior-matched beserta
            # sebaran antar inisialisasi acak
            "acc": np.mean(apm), "n": len(apm),
            "sd": (np.std(apm, ddof=1) if len(apm) > 1 else 0.0)}


def _akurasi_seed(pat):
    """Akurasi prior-matched tiap inisialisasi acak yang cocok dengan pola."""
    a = []
    for d in sorted(glob.glob(os.path.join(HERE, pat))):
        f = os.path.join(d, "test_scores.npy")
        if not os.path.exists(f):
            continue
        y, p, _ = np.load(f)
        a.append(full_metrics(y.astype(int), p,
                              prior_matched_threshold(p, 0.5))["accuracy"] * 100)
    return np.array(a)


def _welch_p(a, b):
    """Nilai p dua sisi uji t Welch. Implementasi sama dengan signifikansi.py,
    yang sudah dicocokkan terhadap SciPy sampai selisih 1e-10."""
    from math import lgamma
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    if va + vb == 0:
        return None
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    x = df / (df + t * t)
    aa, bb = df / 2.0, 0.5
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = lgamma(aa) + lgamma(bb) - lgamma(aa + bb)
    front = np.exp(np.log(x) * aa + np.log(1 - x) * bb - lbeta) / aa
    f, c, d = 1.0, 1.0, 0.0
    for i in range(300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (bb - m) * x) / ((aa + 2 * m - 1) * (aa + 2 * m))
        else:
            num = -((aa + m) * (aa + bb + m) * x) / ((aa + 2 * m) * (aa + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-10:
            break
    return float(front * (f - 1.0))


def ms(v, k, pct=True):
    a = np.array([x[k] for x in v], dtype=float) * (100 if pct else 1)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return "n/a"
    if len(a) == 1:
        return f"{a[0]:.2f}"
    return f"{a.mean():.2f} ({a.std(ddof=1):.2f})"


# ---------------------------------------------------------------- isi
def _angka():
    """Nilai kunci yang dikutip dalam prosa, dihitung ulang tiap kali naskah
    dibangun.

    Sepanjang penelitian ini beberapa angka sempat dikutip dari satu
    inisialisasi acak lalu menjadi kedaluwarsa ketika inisialisasi lain
    ditambahkan. Menghitungnya di sini menutup kemungkinan itu.
    """
    P = {}
    for kunci, pat in [
            ("ast_proposal", "runs/ast_official_proposalULRPK_b32e20_s*"),
            ("ast_beku", "runs/ast_official_full_b32e10_s*"),
            ("ast_dilatih", "runs/ast_official_fullUF_b32e10_s*"),
            ("wavlm_proposal", "runs/wavlm_official_proposalULRPK_b16e20_s*"),
            ("wavlm_beku", "runs/wavlm_official_full_b16e10_s*"),
            ("wavlm_dilatih", "runs/wavlm_official_fullUF_b16e10_s*"),
            ("hubert_proposal", "runs/hubert_official_proposalULRPK_b32e20_s*"),
            ("hubert_beku", "runs/hubert_official_full_b32e10_s*"),
            ("hubert_dilatih", "runs/hubert_official_fullUF_b32e10_s*"),
            ("hubert_1e3", "runs/hubert_official_fullUFENC0.001_b32e10_s*"),
            ("wavlm_1e3", "runs/wavlm_official_fullUFENC0.001_b16e10_s*")]:
        m = _kumpul_ambang(pat)
        P[kunci] = m
    return P


def _sn(m, desimal=2):
    """Format rerata beserta simpangan baku bila lebih dari satu inisialisasi."""
    if m is None:
        return "belum tersedia"
    if m["n"] > 1:
        return f"{m['acc']:.{desimal}f} persen ({m['sd']:.2f})"
    return f"{m['acc']:.{desimal}f} persen"


def bangun():
    FOR = kumpul_for()
    GEN = kumpul_gen()
    A = _angka()
    E = []

    E.append(P("Ketika Akurasi Tinggi Menyesatkan: Audit Dataset, Replikasi, "
               "dan Batas Generalisasi pada Deteksi Deepfake Audio", "judul"))
    E.append(P("Studi empiris pada Fake-or-Real, In-the-Wild, dan 14 sistem "
               "text-to-speech dari 2019 sampai 2026<br/>"
               "Kode dan data pendukung: github.com/Tristan-tech-ai/general-AI",
               "sub"))

    E.append(P("Abstrak", "h1"))
    E.append(P(
        "Penelitian ini berangkat dari rencana membandingkan empat arsitektur deep "
        "learning untuk membedakan suara asli dan suara sintetis pada dataset "
        "Fake-or-Real. Dalam pelaksanaannya, audit terhadap 17.870 berkas audio "
        "menemukan bahwa dataset tersebut mengandung korelasi semu yang kuat antara "
        "riwayat kompresi berkas dan label kelas. Sebanyak 90,7 persen sampel palsu "
        "pada data latih berasal dari berkas MP3, sedangkan pada data uji tidak ada "
        "satu pun. Akibatnya model belajar mengenali jejak kompresi, bukan jejak "
        "sintesis. Dengan arsitektur, data, dan hyperparameter yang identik, "
        "pembagian data secara acak menghasilkan akurasi 99,94 persen sementara "
        "partisi resmi menghasilkan 50,00 persen. Selisih tersebut sepenuhnya "
        "berasal dari protokol pembagian data. Temuan lanjutan menunjukkan bahwa "
        "akurasi pada dataset ini berkorelasi negatif dengan kemampuan mendeteksi "
        "sistem text-to-speech generasi 2025 sampai 2026, bahwa kegagalan di bawah "
        "gangguan noise sebagian besar merupakan kegagalan kalibrasi ambang dan "
        "bukan kegagalan pengenalan, dan bahwa model anti-spoofing terbaik yang "
        "tersedia publik memiliki spesifisitas nol persen ketika diuji di luar "
        "domain latihnya. Sebagai tanggapan, penelitian ini mengusulkan augmentasi "
        "band-gain yang menetralkan isyarat level energi pita tinggi tanpa merusak "
        "struktur halusnya, dan menguji dampaknya secara terkontrol. Perbandingan "
        "setara antara metodologi proposal dan metodologi yang diperbaiki "
        "dilakukan dengan tiga inisialisasi acak per sel, lalu diuji dengan uji t "
        "Welch dan dikoreksi Holm-Bonferroni. Dari seluruh perbandingan itu hanya "
        "satu keputusan yang terbukti menentukan, yaitu besaran learning rate "
        "relatif terhadap ukuran dan jenis encoder. Learning rate seragam 0,001 "
        f"yang ditetapkan proposal memberi {_sn(A['ast_proposal'])} pada AST yang "
        f"berukuran 86 juta parameter, tetapi menjatuhkan WavLM Large ke "
        f"{_sn(A['wavlm_proposal'])} dan HuBERT Large ke "
        f"{_sn(A['hubert_proposal'])}, yaitu mendekati tebakan acak. Seluruh "
        "perbandingan lain, termasuk antara membekukan dan melatih encoder, "
        "tidak terbukti berbeda setelah ragam antar inisialisasi "
        "diperhitungkan.", "p"))

    # ---------------- 1
    E.append(P("1. Latar Belakang dan Pertanyaan Penelitian", "h1"))
    E.append(P(
        "Deteksi suara sintetis lazim dievaluasi dengan melaporkan akurasi atau "
        "equal error rate pada satu dataset. Praktik ini mengandaikan bahwa angka "
        "yang tinggi pada dataset uji mencerminkan kemampuan yang tinggi pada "
        "kondisi nyata. Penelitian ini menguji andaian tersebut secara langsung.", "p"))
    E.append(P(
        "Tiga pertanyaan dijawab. Pertama, apakah akurasi tinggi pada Fake-or-Real "
        "benar-benar mencerminkan kemampuan mendeteksi sintesis. Kedua, bagaimana "
        "kinerja model ketika audio terdegradasi oleh noise lingkungan, dan apakah "
        "penurunannya berasal dari hilangnya daya pisah atau dari sebab lain. "
        "Ketiga, apakah detektor yang dilatih pada sistem text-to-speech lama masih "
        "mengenali sistem komersial terbaru yang secara pendengaran sulit dibedakan "
        "dari suara manusia.", "p"))

    # ---------------- 2
    E.append(P("2. Data dan Metode", "h1"))
    E.append(P("2.1 Dataset", "h2"))
    E.append(tabel(
        ["Dataset", "Isi", "Peran"],
        [["Fake-or-Real (for-2sec)", "17.870 klip 2 detik, 16 kHz, seimbang",
          "latih dan uji utama"],
         ["Fake-or-Real (for-rerec)", "816 klip uji, diputar ulang di ruangan",
          "uji lintas kondisi rekaman"],
         ["In-the-Wild", "31.779 klip, 54 pembicara publik",
          "uji lintas korpus"],
         ["MLAAD", "14.000 klip dari 14 sistem TTS, 2019 sampai 2026",
          "uji lintas generasi"],
         ["DEMAND", "6 lingkungan noise nyata, 16 kHz",
          "noise uji, terpisah dari noise latih"]],
        [4.6 * cm, 7.2 * cm, 4.2 * cm]))
    E.append(Spacer(1, 6))
    E.append(P(
        "Korpus noise untuk pengujian sengaja dipilih dari sumber yang berbeda "
        "dengan noise pelatihan. Beberapa korpus yang lazim dipakai bersama, "
        "seperti MUSAN, ESC-50, dan FSD50K, sama-sama bersumber dari Freesound, "
        "sehingga klaim noise yang belum pernah dilihat tidak selalu terpenuhi. "
        "DEMAND direkam sendiri oleh penyusunnya dan karena itu benar-benar "
        "independen.", "p"))

    E.append(P("2.2 Arsitektur", "h2"))
    E.append(P(
        "Enam arsitektur diuji. Empat berasal dari rencana awal, yaitu Wav2Vec2 "
        "Base, Audio Spectrogram Transformer, HuBERT Large, dan CNN-LSTM. Dua "
        "ditambahkan selama penelitian, yaitu WavLM Large yang pra-pelatihannya "
        "menyertakan denoising, dan Nes2Net-X yang merupakan arsitektur "
        "anti-spoofing terbaru dengan back-end 511 ribu parameter. Untuk model "
        "berbasis self-supervised, seluruh hidden state diagregasi dengan bobot "
        "yang dipelajari, bukan hanya lapisan terakhir seperti pada implementasi "
        "aslinya.", "p"))

    E.append(P("2.3 Protokol evaluasi", "h2"))
    E.append(P(
        "Metrik yang dilaporkan mencakup akurasi, equal error rate, dan area under "
        "curve. Ambang keputusan tidak dipatok pada 0,5 melainkan disesuaikan "
        "dengan prior kelas yang diketahui, dan tidak menggunakan label data uji. "
        "Untuk perbandingan lintas generasi text-to-speech, seluruh model "
        "dikalibrasi terlebih dahulu agar spesifisitasnya 95 persen pada himpunan "
        "audio asli yang terpisah, baru kemudian recall diukur. Tanpa langkah ini, "
        "detektor yang selalu menjawab palsu akan mencatat recall sempurna.", "p"))

    E.append(PageBreak())

    # ---------------- 3
    E.append(P("3. Hasil", "h1"))

    E.append(P("3.1 Protokol pembagian data menentukan hasil", "h2"))
    E.append(P(
        "Eksperimen pertama menggunakan model, data, dan hyperparameter yang sama "
        "persis, dan hanya mengubah cara data dibagi. Hasilnya sangat berbeda.", "p"))
    b = []
    for k in [("cnn_asp", "random", "none"), ("cnn_asp", "official", "none")]:
        if k in FOR:
            nm = "Acak 60/20/20" if k[1] == "random" else "Partisi resmi"
            b.append([nm, ms(FOR[k], "acc"), ms(FOR[k], "eer"),
                      f"{FOR[k][0]['n']}"])
    if b:
        E.append(tabel(["Skema pembagian data", "Akurasi (persen)",
                        "EER (persen)", "Jumlah berkas uji"], b,
                       [5.4 * cm, 3.6 * cm, 3.4 * cm, 3.6 * cm]))
        E.append(Spacer(1, 6))
    E.extend(gambar("02_efek_split.png", 12 * cm,
                    "Gambar 1. Akurasi pada dua protokol pembagian data. "
                    "Arsitektur, data, dan hyperparameter identik."))

    BUTA = []
    for nm, sblm, ssdh in [
            ("Encoder tidak pernah dilatih karena bug",
             "ast_official_proposalULRPK_b32e20_s42",
             "ast_random_proposalULRPK_b32e20_s42"),
            ("Encoder rusak karena learning rate terlalu tinggi",
             "wavlm_official_proposalULRPK_b16e20_s42",
             "wavlm_random_proposalULRPK_b16e20_s42")]:
        a = _kumpul_ambang(f"runs_pra_perbaikan/{sblm}")
        b = _kumpul_ambang(f"runs/{sblm}")
        c = _kumpul_ambang(f"runs_pra_perbaikan/{ssdh}")
        d = _kumpul_ambang(f"runs/{ssdh}")
        if all([a, b, c, d]):
            BUTA.append([nm, f"{b['apm'] - a['apm']:+.2f}",
                         f"{d['apm'] - c['apm']:+.2f}",
                         f"{b['auc'] - a['auc']:+.4f}",
                         f"{d['auc'] - c['auc']:+.4f}"])
    if BUTA:
        E.append(P(
            "Selama penelitian berlangsung, dua kejadian tidak sengaja "
            "menyediakan uji yang jauh lebih tajam terhadap protokol pembagian "
            "data. Pada kedua kejadian itu sebuah kerusakan besar menimpa model, "
            "dan pertanyaannya adalah protokol mana yang mampu mendeteksinya.", "p"))
        E.append(P(
            "Kejadian pertama adalah sebuah bug yang menyebabkan encoder tidak "
            "pernah menerima gradien sama sekali, sehingga sebagian terbesar "
            "kapasitas model tidak pernah dilatih. Kejadian kedua adalah "
            "learning rate seragam yang ditetapkan proposal, yang ketika benar "
            "benar diterapkan pada WavLM Large merusak representasi pra-latihnya "
            "sampai area under curve turun mendekati tebakan acak. Tabel berikut "
            "menunjukkan berapa besar perubahan yang tercatat oleh masing-masing "
            "protokol.", "p"))
        E.append(tabel(["Kerusakan pada model", "Terdeteksi partisi resmi (pp)",
                        "Terdeteksi split acak (pp)", "Perubahan AUC resmi",
                        "Perubahan AUC acak"], BUTA,
                       [4.6 * cm, 3.2 * cm, 3.0 * cm, 2.6 * cm, 2.4 * cm]))
        E.append(Spacer(1, 6))
        E.append(P(
            "Split acak nyaris tidak bergerak pada kedua kejadian. Model yang "
            "encodernya tidak pernah dilatih tetap mencatat akurasi di atas 99 "
            "persen, dan model yang representasinya sudah rusak sampai hampir "
            "setara tebakan acak pada partisi resmi tetap mencatat 99,52 persen "
            "dengan AUC 0,9979 pada split acak. Temuan ini lebih keras daripada "
            "pernyataan bahwa split acak menghasilkan angka yang menggelembung. "
            "Protokol tersebut ternyata tidak dapat membedakan model yang bekerja "
            "dari model yang sebagian besar kapasitasnya mati atau rusak. Angka "
            "yang dihasilkannya karena itu tidak dapat dipakai untuk memilih "
            "model, membandingkan arsitektur, atau bahkan untuk memastikan bahwa "
            "pelatihan berjalan sebagaimana mestinya.", "p"))

    E.append(P("3.2 Penyebabnya adalah artefak kompresi, dan terukur", "h2"))
    E.append(P(
        "Audit terhadap seluruh berkas menemukan bahwa nama berkas kelas palsu "
        "mengandung penanda mp3 sedangkan kelas asli tidak. Pemeriksaan spektral "
        "mengonfirmasi akibatnya. Energi pada pita di atas 6 kHz untuk kelas asli "
        "adalah 0,0336, sedangkan untuk kelas palsu yang berasal dari MP3 hanya "
        "0,0080, yaitu 4,18 kali lebih rendah. Namun untuk 652 sampel palsu yang "
        "tidak melalui MP3, selisihnya hanya 1,17 kali. Artinya sebagian besar "
        "perbedaan spektral yang tampak berasal dari kompresi, bukan dari sintesis.", "p"))
    E.append(tabel(
        ["Bagian data", "Kelas", "Asal berkas", "Jumlah", "Energi di atas 6 kHz"],
        [["Latih", "asli", "WAV", "6.978", "0,0336"],
         ["Latih", "palsu", "MP3", "6.326", "0,0080"],
         ["Latih", "palsu", "WAV", "652", "0,0287"],
         ["Uji", "asli", "WAV", "544", "0,0058"],
         ["Uji", "palsu", "WAV", "544", "0,0043"]],
        [2.6 * cm, 2.2 * cm, 2.6 * cm, 2.4 * cm, 4.4 * cm]))
    E.append(Spacer(1, 6))
    E.append(P(
        "Sebanyak 90,7 persen sampel palsu pada data latih berasal dari MP3, "
        "sedangkan pada data uji tidak ada satu pun. Model yang dilatih pada "
        "partisi ini mempelajari aturan bahwa energi frekuensi tinggi yang rendah "
        "berarti palsu. Aturan itu benar pada sebagian besar data latih dan sama "
        "sekali tidak berlaku pada data uji.", "p"))

    E.append(PageBreak())
    E.append(P("3.3 Ketahanan terhadap noise", "h2"))
    E.append(P(
        "Model diuji pada sembilan tingkat signal to noise ratio dengan noise dari "
        "korpus yang tidak pernah dilihat selama pelatihan. Temuan utamanya adalah "
        "bahwa penurunan akurasi sebagian besar bukan karena model kehilangan daya "
        "pisah. Pada 10 dB, WavLM masih memiliki area under curve 0,962, yang "
        "berarti pemisahan skornya nyaris utuh, tetapi akurasinya pada ambang tetap "
        "hanya 59,8 persen. Mengoreksi ambang saja memulihkan 29,6 poin persentase. "
        "Sebaliknya CNN-LSTM gagal dengan cara yang berbeda, yaitu area under "
        "curve-nya benar-benar jatuh ke 0,605.", "p"))
    E.extend(gambar("03_kurva_snr.png", 16 * cm,
                    "Gambar 2. Akurasi dan equal error rate terhadap tingkat noise. "
                    "Garis tebal adalah model yang dilatih dengan augmentasi noise, "
                    "garis putus-putus tanpa augmentasi noise."))
    E.append(P(
        "Perbedaan antara kedua kelompok garis pada Gambar 2 menunjukkan bahwa "
        "penurunan yang semula tampak seperti batas fisik sebenarnya adalah celah "
        "pelatihan. Model yang dilatih dengan augmentasi noise mempertahankan 95,5 "
        "persen akurasi pada 0 dB, yaitu ketika daya noise sama dengan daya sinyal.", "p"))

    E.append(PageBreak())
    E.append(P("3.4 Deteksi lintas generasi text-to-speech", "h2"))
    E.append(P(
        "Empat belas sistem text-to-speech diuji, mulai dari Griffin-Lim dan "
        "Tacotron2 yang mewakili era dataset, sampai ElevenLabs v3, Chatterbox, "
        "OpenAI TTS-1 HD, dan Higgs-Audio-V2 yang mewakili sistem komersial "
        "terbaru. Seluruh model dikalibrasi pada spesifisitas 95 persen sebelum "
        "recall diukur.", "p"))
    b = []
    for (a, au), v in sorted(GEN.items(), key=lambda kv: -np.mean(
            [x["modern"] for x in kv[1]])):
        if len(v) < 2:
            continue
        fk = [k for k in FOR if k[0] == a and k[2] == au and k[1] == "official"]
        fa = ms(FOR[fk[0]], "acc") if fk else "n/a"
        b.append([a, au, str(len(v)), fa, ms(v, "modern"), ms(v, "old")])
    if b:
        E.append(tabel(
            ["Arsitektur", "Augmentasi", "n", "Akurasi FoR",
             "Recall TTS 2025-2026", "Recall TTS 2019 non-MP3"],
            b, [3.0 * cm, 2.4 * cm, 0.9 * cm, 3.0 * cm, 3.3 * cm, 3.4 * cm]))
        E.append(Spacer(1, 4))
        E.append(P("Nilai ditulis sebagai rerata diikuti simpangan baku dalam "
                   "kurung, dalam persen.", "cap"))
    E.extend(gambar("05_generasi_tts.png", 16 * cm,
                    "Gambar 3. Recall per sistem text-to-speech pada spesifisitas "
                    "95 persen yang disamakan untuk semua model."))

    E.append(PageBreak())
    E.append(P("3.5 Akurasi in-domain berkorelasi negatif dengan generalisasi", "h2"))
    E.append(P(
        "Menggabungkan hasil pada Fake-or-Real dengan hasil pada sistem "
        "text-to-speech terbaru menghasilkan pola yang berlawanan dengan harapan. "
        "Model dengan akurasi tertinggi pada dataset justru bukan model terbaik "
        "pada sistem terbaru. Contoh paling jelas adalah HuBERT dengan augmentasi "
        "penuh, yang mencapai akurasi 95,3 persen pada dataset tetapi hanya "
        "mengenali 2,3 persen sampel dari sistem text-to-speech 2019 yang tidak "
        "dikompresi MP3. Model tersebut pada praktiknya mendeteksi kompresi, bukan "
        "sintesis.", "p"))
    E.extend(gambar("06_tradeoff.png", 16 * cm,
                    "Gambar 4. Hubungan antara akurasi in-domain dan kemampuan "
                    "generalisasi untuk beberapa strategi augmentasi pada "
                    "arsitektur yang sama."))

    E.append(P("3.6 Model anti-spoofing terbaik runtuh di luar domainnya", "h2"))
    E.append(P(
        "Sebuah model anti-spoofing publik dengan equal error rate 1,49 persen "
        "pada ASVspoof 2021 diuji tanpa penyesuaian pada Fake-or-Real. Seluruh 544 "
        "berkas asli ditandai sebagai palsu, sehingga spesifisitasnya nol persen. "
        "Area under curve-nya 0,0233, yang berarti urutan skornya terbalik. Bila "
        "polaritas dibalik, nilainya menjadi 0,9767, sehingga model tersebut "
        "sebenarnya tetap sangat diskriminatif namun dengan arah keputusan yang "
        "keliru pada korpus ini. Metrik equal error rate yang dilaporkan dalam "
        "domain tidak dapat menangkap kegagalan semacam ini.", "p"))

    M2 = matriks_2x2()
    if M2:
        E.append(PageBreak())
        E.append(P("3.7 Apakah rekayasa metodologi benar-benar melampaui "
                   "baseline", "h2"))
        E.append(P(
            "Semua perbandingan sejauh ini menyandingkan konfigurasi proposal pada "
            "split acak dengan konfigurasi yang diperbaiki pada partisi resmi. "
            "Kedua kolom itu tidak setara, sehingga tidak dapat menjawab pertanyaan "
            "pokok, yaitu apakah rekayasa yang dilakukan memang bernilai atau hanya "
            "memindahkan angka dari satu protokol ke protokol lain. Untuk "
            "menjawabnya, tiap arsitektur dijalankan pada keempat kombinasi dari "
            "dua konfigurasi dan dua skema pembagian data, dengan batch, seed, dan "
            "data yang dijaga tetap. Dengan demikian setiap perbandingan hanya "
            "mengubah satu variabel.", "p"))
        E.append(P(
            "Konfigurasi proposal memakai learning rate 0,001 seragam dengan "
            "encoder ikut dilatih, 20 epoch tanpa early stopping, normalisasi "
            "amplitudo puncak, augmentasi noise pada rentang 15 sampai 30 dB, dan "
            "ambang keputusan 0,5. Konfigurasi yang diperbaiki memakai learning "
            "rate per model dengan encoder dibekukan dan agregasi berbobot antar "
            "lapisan, 10 epoch dengan early stopping pada equal error rate, "
            "normalisasi loudness, augmentasi penuh, dan ambang prior-matched.", "p"))

        def _sel(a):
            if not a:
                return "n/a"
            a = np.array(a)
            return (f"{a[0]:.2f}" if len(a) == 1
                    else f"{a.mean():.2f} ({a.std(ddof=1):.2f})")

        baris, ringkas = [], []
        for model, sel in M2.items():
            for cfg, nm in [("proposal", "Proposal apa adanya"),
                            ("diperbaiki", "Diperbaiki")]:
                r = np.mean(sel[(cfg, "random")])
                o = np.mean(sel[(cfg, "official")])
                baris.append([model if cfg == "proposal" else "", nm,
                              _sel(sel[(cfg, "random")]),
                              _sel(sel[(cfg, "official")]), f"{r - o:+.2f}"])
            ringkas.append((model,
                            np.mean(sel[("proposal", "official")]),
                            np.mean(sel[("diperbaiki", "official")])))
        E.append(tabel(["Arsitektur", "Konfigurasi", "Split acak",
                        "Partisi resmi", "Selisih (pp)"], baris,
                       [2.6 * cm, 4.6 * cm, 2.9 * cm, 3.0 * cm, 2.9 * cm]))
        E.append(Spacer(1, 6))
        E.extend(gambar("10_matriks_2x2.png", 15.5 * cm,
                        "Gambar 5. Ketergantungan hasil pada protokol evaluasi. "
                        "Garis yang landai menandakan hasil yang bertahan ketika "
                        "domain rekaman berubah. Garis putus-putus abu-abu adalah "
                        "konfigurasi proposal."))

        kal = "; ".join(f"{m} naik dari {p:.2f} menjadi {d:.2f} persen, "
                        f"yaitu {d - p:+.2f} poin" for m, p, d in ringkas)
        rata = np.mean([d - p for _, p, d in ringkas])
        E.append(P(
            "Pada kolom partisi resmi, yaitu satu-satunya kolom yang menguji "
            f"generalisasi lintas domain, {kal}. Rerata perbaikan {rata:+.2f} poin "
            "persentase pada protokol yang sama persis. Pada kolom split acak "
            "selisihnya justru sedikit negatif, sehingga bila penelitian ini hanya "
            "melaporkan kolom tersebut seluruh rekayasa akan tampak tidak berguna "
            "atau bahkan merugikan.", "p"))
        E.append(P(
            "Kolom paling kanan mengukur seberapa jauh hasil sebuah konfigurasi "
            "bergantung pada protokol evaluasi yang dipilih. Selisih yang besar "
            "berarti nilai yang dilaporkan lebih banyak ditentukan oleh cara data "
            "dibagi daripada oleh kemampuan model. Selisih yang kecil berarti "
            "sebaliknya, dan itulah bentuk terukur dari generalisasi yang dituju, "
            "yaitu bukan sekadar angka yang lebih tinggi melainkan angka yang "
            "bertahan ketika domain rekamannya berubah.", "p"))
        E.append(P(
            "Tabel ini perlu dibaca dengan hati-hati dan tidak boleh berdiri "
            "sendiri. Tiap konfigurasi dievaluasi pada ambang keputusannya "
            "masing-masing, yaitu 0,5 untuk proposal dan prior-matched untuk "
            "versi diperbaiki, karena penetapan ambang termasuk bagian dari "
            "metodologi yang dibandingkan. Akibatnya selisih pada kolom partisi "
            "resmi mencampur dua sumber sekaligus. Dua bagian berikutnya "
            "memisahkan sumber-sumber tersebut, dan hasilnya membalik sebagian "
            "kesimpulan yang tampak dari tabel ini.", "p"))

    if M2:
        E.append(P("3.8 Pemecahan lanjutan: sumbangan ambang dan sumbangan "
                   "pelatihan", "h2"))
        E.append(P(
            "Angka pada bagian sebelumnya membandingkan dua metodologi secara utuh. "
            "Di dalamnya dua hal berubah bersamaan, yaitu cara model dilatih dan "
            "cara ambang keputusan ditetapkan. Selisih tersebut karena itu tidak "
            "boleh dibaca sebagai sumbangan pelatihan semata. Untuk memisahkannya, "
            "empat besaran dihitung dari berkas skor yang sama persis tanpa "
            "melatih ulang apa pun, yaitu tiap konfigurasi dievaluasi pada kedua "
            "ambang.", "p"))
        b28, d28 = [], []
        for model, sel in M2.items():
            a = np.mean(sel[("proposal", "official")])
            d = np.mean(sel[("diperbaiki", "official")])
            pr = _kumpul_ambang(f"runs/{model}_official_proposalULRPK_*")
            re_ = _kumpul_ambang(f"runs/{model}_official_full_*")
            if not (pr and re_):
                continue
            b28.append([model, f"{pr['a05']:.2f}", f"{pr['apm']:.2f}",
                        f"{re_['a05']:.2f}", f"{re_['apm']:.2f}"])
            d28.append([model, f"{pr['apm'] - pr['a05']:+.2f}",
                        f"{re_['a05'] - pr['a05']:+.2f}",
                        f"{pr['auc']:.4f}", f"{re_['auc']:.4f}",
                        f"{pr['eer']:.2f}", f"{re_['eer']:.2f}"])
        if b28:
            E.append(tabel(["Arsitektur", "Proposal, ambang 0,5",
                            "Proposal, ambang prior", "Rekayasa, ambang 0,5",
                            "Rekayasa, ambang prior"], b28,
                           [2.6 * cm, 3.4 * cm, 3.4 * cm, 3.4 * cm, 3.2 * cm]))
            E.append(Spacer(1, 6))
            E.append(tabel(["Arsitektur", "Sumbangan ambang saja (pp)",
                            "Sumbangan pelatihan saja (pp)", "AUC proposal",
                            "AUC rekayasa", "EER proposal", "EER rekayasa"], d28,
                           [2.2 * cm, 2.9 * cm, 2.9 * cm, 2.2 * cm, 2.2 * cm,
                            2.0 * cm, 2.0 * cm]))
            E.append(Spacer(1, 6))
        E.append(P(
            "Pemecahan ini membalik sebagian kesimpulan pada bagian sebelumnya, "
            "dan arahnya berbeda pada tiap arsitektur. Pada AST, sumbangan "
            "pelatihan bertanda negatif. Konfigurasi proposal yang dievaluasi "
            f"pada ambang prior-matched mencapai {_sn(A['ast_proposal'])}, "
            "sedangkan konfigurasi yang direkayasa dengan encoder dibekukan hanya "
            f"mencapai {_sn(A['ast_beku'])}. Area under curve juga lebih rendah, "
            f"yaitu {A['ast_beku']['auc']:.4f} lawan "
            f"{A['ast_proposal']['auc']:.4f}, sehingga penurunan itu bukan "
            "sekadar soal letak ambang melainkan penurunan daya pisah yang "
            "sesungguhnya.", "p"))
        E.append(P(
            "Pada WavLM Large keadaannya justru sebaliknya dan jauh lebih ekstrem. "
            f"Konfigurasi proposal runtuh menjadi {_sn(A['wavlm_proposal'])} "
            f"dengan area under curve {A['wavlm_proposal']['auc']:.4f} dan equal "
            f"error rate {A['wavlm_proposal']['eer']:.2f} persen, yaitu hanya "
            "sedikit lebih baik daripada menebak. Konfigurasi yang direkayasa "
            f"mencapai {_sn(A['wavlm_beku'])} dengan area under curve "
            f"{A['wavlm_beku']['auc']:.4f} dan equal error rate "
            f"{A['wavlm_beku']['eer']:.2f} persen. Di sini paket rekayasa bukan "
            "hanya menolong, melainkan menjadi pembeda antara model yang berguna "
            "dan model yang tidak.", "p"))
        E.append(P(
            "Dua arah yang berlawanan pada dua arsitektur menunjukkan bahwa "
            "pertanyaan mana metodologi yang lebih baik tidak memiliki jawaban "
            "tunggal. Penyebabnya dapat ditunjuk secara tepat, dan itu dibahas "
            "pada bagian 3.9 dan 3.10.", "p"))

    TANGGA = [
        ("L1", "Konfigurasi proposal apa adanya",
         "runs/ast_official_proposalULRPK_b32e20_s42"),
        ("L2", "Normalisasi loudness menggantikan peak",
         "runs/ast_official_proposalULR_b32e20_s42"),
        ("L3", "Learning rate per model dan encoder dibekukan",
         "runs/ast_official_proposal_b32e20_s42"),
        ("L4", "Early stopping pada equal error rate",
         "runs/ast_official_proposal_b32e10_s42"),
        ("L5", "Augmentasi penuh menggantikan noise saja",
         "runs/ast_official_full_b32e10_s42"),
    ]
    tgr, prev, prev_v = [], None, None
    for kode, nama, d in TANGGA:
        pola = re.sub(r"_s\d+$", "_s*", d)
        m = _kumpul_ambang(pola)
        if m is None:
            continue
        v = _akurasi_seed(pola)
        p = None if prev_v is None else _welch_p(v, prev_v)
        tgr.append({"kode": kode, "nama": nama, "m": m, "p": p,
                    "sel": None if prev is None else m["apm"] - prev})
        prev, prev_v = m["apm"], v
    # Koreksi Holm atas seluruh langkah yang diuji.
    ut = [b for b in tgr if b["p"] is not None]
    for rank, b in enumerate(sorted(ut, key=lambda x: x["p"])):
        b["ph"] = min(1.0, b["p"] * (len(ut) - rank))
    jl = 0.0
    for b in sorted(ut, key=lambda x: x["p"]):
        jl = max(jl, b["ph"])
        b["ph"] = jl
    tg = []
    for b in tgr:
        m = b["m"]
        tg.append([b["kode"], b["nama"], str(m["n"]),
                   (f"{m['apm']:.2f} ({m['sd']:.2f})" if m["n"] > 1
                    else f"{m['apm']:.2f}"),
                   "" if b["sel"] is None else f"{b['sel']:+.2f}",
                   "" if b["p"] is None else f"{b['p']:.3f}",
                   "" if "ph" not in b else f"{b['ph']:.3f}",
                   f"{m['auc']:.4f}"])

    if len(tg) >= 2:
        E.append(P("3.9 Perbaikan mana yang membeli berapa", "h2"))
        E.append(P(
            "Bagian sebelumnya menunjukkan bahwa metodologi yang diperbaiki "
            "mengungguli konfigurasi proposal, dan bahwa sebagian besar "
            "selisihnya berasal dari kalibrasi. Yang belum terjawab adalah "
            "perbaikan mana di dalam paket itu yang benar-benar menyumbang. "
            "Untuk menjawabnya, perbaikan ditambahkan satu per satu di atas "
            "konfigurasi proposal, seluruhnya pada AST, partisi resmi, batch 32, "
            "dan seed 42. Akurasi dilaporkan pada ambang prior-matched untuk "
            "semua langkah agar sumbu ambang tidak ikut bercampur.", "p"))
        E.append(tabel(["Langkah", "Perbaikan yang ditambahkan", "n", "Akurasi",
                        "Selisih", "p mentah", "p Holm", "AUC"], tg,
                       [1.4 * cm, 4.6 * cm, 0.9 * cm, 2.4 * cm, 1.5 * cm,
                        1.6 * cm, 1.5 * cm, 1.7 * cm]))
        E.append(Spacer(1, 6))
        E.extend(gambar("11_tangga_ablasi.png", 15.0 * cm,
                        "Gambar 6. Sumbangan tiap perbaikan pada AST di partisi "
                        "resmi. Batang merah menandai perbaikan yang merugikan "
                        "bila berdiri sendiri."))
        E.append(P(
            "Tangga ini menjawab pertanyaan yang tertinggal pada bagian 3.8, "
            "meskipun jawabannya lebih lemah daripada yang diharapkan. Sepanjang "
            "lima langkah, akurasi justru turun, sehingga paket rekayasa secara "
            "keseluruhan merugikan pada arsitektur ini. Langkah dengan selisih "
            "terbesar adalah pembekuan encoder, diikuti early stopping pada arah "
            "sebaliknya, sedangkan normalisasi loudness dan augmentasi penuh "
            "memberi selisih yang jauh lebih kecil.", "p"))
        E.append(P(
            "Selisih tiap langkah diuji terhadap langkah sebelumnya dengan uji t "
            "Welch dan dikoreksi Holm-Bonferroni untuk empat pengujian sekaligus. "
            "Dua langkah dengan selisih terbesar memiliki nilai p mentah di bawah "
            "0,05, tetapi tidak satu pun bertahan di bawah ambang setelah "
            "koreksi. Sebabnya perlu dinyatakan dengan tepat. Seluruh tangga ini "
            "dijalankan pada AST, yaitu arsitektur dengan ragam antar "
            "inisialisasi terbesar di antara yang diuji dalam penelitian ini, "
            "sehingga daya ujinya paling rendah justru di tempat yang paling "
            "banyak dibandingkan. Nilai p yang besar di sini menunjukkan "
            "kurangnya bukti, bukan ketiadaan efek.", "p"))
        E.append(P(
            "Kesimpulan yang dapat dipertanggungjawabkan dari tangga ini karena "
            "itu terbatas. Arah tiap langkah konsisten dengan penjelasan "
            "mekanistik yang diajukan pada bagian 3.10, tetapi besarannya belum "
            "dapat dipisahkan dari ragam pada ukuran sampel ini. Tangga ablasi "
            "lebih tepat dibaca sebagai peta kemungkinan sebab yang menunjukkan "
            "ke mana harus mencari, bukan sebagai pengukuran sumbangan tiap "
            "perbaikan. Versi awal naskah ini menyajikannya sebagai pengukuran, "
            "dan itu keliru.", "p"))
        E.append(P(
            "Perlu dicatat bahwa versi tangga ini berbeda jauh dari versi yang "
            "sempat disusun lebih awal dalam penelitian. Pada versi awal, titik "
            "tolaknya adalah konfigurasi proposal yang encodernya tidak pernah "
            "benar-benar dilatih karena sebuah bug, sehingga normalisasi loudness "
            "tampak merugikan 7,81 poin dan pembekuan encoder tampak tidak "
            "berpengaruh sama sekali. Kedua kesan itu keliru dan telah dikoreksi "
            "setelah bug diperbaiki. Riwayat koreksinya dipertahankan dalam "
            "dokumentasi pendukung karena bug tersebut justru ditemukan lewat "
            "tangga ablasi ini, yaitu ketika dua langkah yang seharusnya berbeda "
            "menghasilkan skor yang identik sampai empat desimal.", "p"))

    PERLAKUAN = [
        ("Encoder dibekukan", "runs/{m}_official_full_b{b}e10_s*"),
        ("Encoder dilatih, laju wajar per model",
         "runs/{m}_official_fullUF_b{b}e10_s*"),
        ("Encoder dilatih, laju 0,001", "runs/{m}_official_fullUFENC0.001_b{b}e10_s*"),
        ("Proposal apa adanya, laju 0,001 seragam",
         "runs/{m}_official_proposalULRPK_b{b}e20_s*"),
    ]
    ARS = [("ast", 32, "AST, 86 juta parameter, pra-latih terselia"),
           ("wavlm", 16, "WavLM Large, 300 juta parameter, swa-selia"),
           ("hubert", 32, "HuBERT Large, 300 juta parameter, swa-selia")]
    mtx = []
    for m, b, ket in ARS:
        sel = [(nm, _kumpul_ambang(pat.format(m=m, b=b)))
               for nm, pat in PERLAKUAN]
        if all(v for _, v in sel):
            mtx.append((m, ket, sel))

    if mtx:
        E.append(PageBreak())
        E.append(P("3.10 Tidak ada satu perlakuan encoder yang benar untuk "
                   "semua arsitektur", "h2"))
        E.append(P(
            "Bagian 3.9 menunjukkan bahwa membekukan encoder merupakan sumber "
            "kerugian terbesar pada AST. Bagian 3.8 menunjukkan bahwa pada WavLM "
            "Large paket rekayasa yang sama justru menyelamatkan model dari "
            "keruntuhan. Kedua pernyataan itu tampak bertentangan, dan matriks "
            "berikut menjelaskan mengapa keduanya benar sekaligus.", "p"))
        E.append(P(
            "Tiga baris pertama pada tiap arsitektur memakai paket rekayasa yang "
            "sama persis, yaitu 10 epoch dengan early stopping, augmentasi penuh, "
            "normalisasi loudness, dan agregasi berbobot antar lapisan. Hanya "
            "perlakuan encoder yang berbeda, sehingga perbandingan di antara "
            "ketiganya bersifat satu variabel. Baris keempat disertakan sebagai "
            "acuan, yaitu konfigurasi proposal apa adanya.", "p"))
        for m, ket, sel in mtx:
            E.append(P(f"<b>{ket}</b>", "p"))
            E.append(tabel(["Perlakuan encoder", "n", "Akurasi", "AUC", "EER"],
                           [[nm, str(v["n"]),
                             (f"{v['acc']:.2f} ({v['sd']:.2f})" if v["n"] > 1
                              else f"{v['acc']:.2f}"),
                             f"{v['auc']:.4f}", f"{v['eer']:.2f}"]
                            for nm, v in sel],
                           [7.0 * cm, 1.4 * cm, 3.2 * cm, 2.4 * cm, 2.2 * cm]))
            E.append(Spacer(1, 6))
        E.extend(gambar("12_matriks_encoder.png", 15.5 * cm,
                        "Gambar 7. Perlakuan encoder terbaik berbeda menurut "
                        "arsitektur. Tiga batang pertama tiap kelompok memakai "
                        "paket rekayasa yang sama dan hanya berbeda pada "
                        "perlakuan encoder. Garis merah menandai tingkat tebakan "
                        "acak."))
        E.append(P(
            "Dua pola yang berbeda perlu dipisahkan. Pola pertama konsisten pada "
            "seluruh arsitektur yang diuji, yaitu bahwa besaran learning rate "
            "harus disesuaikan dengan encodernya. Laju 0,001 yang ditetapkan "
            "proposal menghancurkan kedua model swa-selia berukuran 300 juta "
            "parameter sampai mendekati tebakan acak, yaitu "
            f"{_sn(A['wavlm_proposal'])} pada WavLM Large dan "
            f"{_sn(A['hubert_proposal'])} pada HuBERT Large, sementara laju yang "
            "sama justru membantu AST yang berukuran 86 juta parameter. Pola ini "
            "terlihat pada dua model yang tujuan dan korpus pra-pelatihannya "
            "berbeda, sehingga cukup kuat untuk dilaporkan.", "p"))
        sel_bandingan = []
        for m, ket, sel in mtx:
            d = {nm: v for nm, v in sel}
            bk = d.get("Encoder dibekukan")
            wj = d.get("Encoder dilatih, laju wajar per model")
            if bk and wj:
                sel_bandingan.append(
                    f"{ket.split(',')[0]} {wj['acc'] - bk['acc']:+.2f} poin")
        E.append(P(
            "Pola kedua tidak konsisten, dan penulis semula keliru "
            "menggeneralisasikannya. Bila learning rate sudah wajar, selisih "
            "antara melatih encoder dan membekukannya adalah "
            + ", ".join(sel_bandingan) +
            ". Dugaan awal bahwa model "
            "swa-selia berukuran besar sebaiknya dibekukan karena "
            "representasinya sudah selaras dengan tugas ternyata tidak bertahan. "
            "HuBERT Large berukuran sama dan sama-sama swa-selia, namun berperilaku "
            "seperti AST dan bukan seperti WavLM. Yang tersisa dari dugaan itu "
            "hanyalah pengamatan bahwa WavLM Large merupakan pengecualian, dan "
            "penelitian ini tidak memiliki bukti yang cukup untuk menjelaskan "
            "mengapa. Pra-pelatihan WavLM yang menyertakan denoising merupakan "
            "kandidat penjelasan, namun itu tetap dugaan yang belum diuji.", "p"))
        E.append(P(
            "Selisih-selisih itu harus dibaca bersama sebarannya, dan kolom n "
            "pada tabel menunjukkan berapa inisialisasi acak yang menopang tiap "
            "sel. Sel AST yang encodernya dilatih dijalankan dengan tiga "
            "inisialisasi dan menghasilkan 91,64 sampai 95,31 persen, yaitu "
            "rentang 3,67 poin persentase dengan simpangan baku 1,85. Rentang "
            "sebesar itu lebih lebar daripada selisih antara kedua metodologi "
            "pada arsitektur tersebut, sehingga perbandingan berbasis satu "
            "inisialisasi di sini tidak dapat menyimpulkan apa pun. Beberapa sel "
            "lain dalam tabel masih bersandar pada satu inisialisasi, dan "
            "kesimpulan mengenai sel-sel tersebut karena itu bersifat "
            "sementara.", "p"))
        E.append(P(
            "Temuan ini menempatkan kedua metodologi pada posisi yang setara. "
            "Proposal menyeragamkan learning rate 0,001 untuk seluruh arsitektur, "
            "yang kebetulan tepat untuk AST dan menghancurkan WavLM Large. "
            "Konfigurasi rekayasa dalam penelitian ini menyeragamkan pembekuan "
            "encoder untuk seluruh arsitektur, yang kebetulan tepat untuk WavLM "
            "Large dan merugikan AST. Keduanya melakukan kelas kesalahan yang "
            "sama, yaitu menetapkan satu keputusan secara seragam padahal "
            "arsitektur yang berbeda menuntut perlakuan yang berbeda. Kesimpulan "
            "yang jujur bukanlah bahwa satu metodologi mengalahkan yang lain, "
            "melainkan bahwa keputusan ini seharusnya dipilih per arsitektur "
            "menggunakan data validasi dan tidak ditetapkan di muka.", "p"))
        E.append(P(
            "Satu dugaan tambahan sempat disusun dan kemudian gugur, dan "
            "kegugurannya dilaporkan di sini karena termasuk bagian dari "
            "temuan. Pada WavLM Large yang dilatih dengan laju 0,001 yang "
            f"merusak itu, konfigurasi proposal jatuh ke {_sn(A['wavlm_proposal'])} "
            f"sedangkan paket rekayasa dengan laju yang sama bertahan di "
            f"{_sn(A['wavlm_1e3'])}. Selisih itu semula ditafsirkan sebagai bukti "
            "bahwa early stopping, augmentasi penuh, dan agregasi berbobot antar "
            "lapisan berfungsi sebagai jaring pengaman terhadap pilihan learning "
            "rate yang buruk.", "p"))
        E.append(P(
            "HuBERT Large membantah tafsir tersebut. Pada arsitektur itu, "
            "konfigurasi proposal dengan laju 0,001 mencapai "
            f"{_sn(A['hubert_proposal'])} sedangkan paket rekayasa dengan laju "
            f"yang sama justru turun ke {_sn(A['hubert_1e3'])} dengan area under "
            f"curve {A['hubert_1e3']['auc']:.4f}, yaitu sedikit di bawah "
            "tebakan acak sehingga urutan skornya bahkan terbalik. Paket rekayasa "
            "tidak memberikan perlindungan apa pun di sini. Kesimpulan yang dapat "
            "dipertahankan karena itu lebih sempit daripada yang semula ditulis, "
            "yaitu bahwa perlindungan tersebut teramati pada WavLM Large dan "
            "tidak dapat digeneralisasi.", "p"))
        E.append(P(
            "Hasil ini justru memperkuat pesan utama bagian ini dari arah yang "
            "lain. Learning rate yang tidak sesuai dengan encoder tidak dapat "
            "ditambal oleh perbaikan lain di dalam pipeline. Pada HuBERT Large, "
            "seluruh paket rekayasa yang di tempat lain menyumbang puluhan poin "
            "persentase tidak mampu mengangkat model dari tingkat tebakan ketika "
            "learning rate-nya keliru. Keputusan itu harus benar sejak awal.", "p"))

    BANDINGAN = [
        ("AST, encoder dilatih lawan dibekukan",
         "runs/ast_official_fullUF_b32e10_s*", "runs/ast_official_full_b32e10_s*"),
        ("AST, encoder dilatih lawan proposal",
         "runs/ast_official_fullUF_b32e10_s*",
         "runs/ast_official_proposalULRPK_b32e20_s*"),
        ("WavLM, encoder dibekukan lawan dilatih",
         "runs/wavlm_official_full_b16e10_s*",
         "runs/wavlm_official_fullUF_b16e10_s*"),
        ("HuBERT, encoder dilatih lawan dibekukan",
         "runs/hubert_official_fullUF_b32e10_s*",
         "runs/hubert_official_full_b32e10_s*"),
        ("WavLM, rekayasa lawan proposal",
         "runs/wavlm_official_full_b16e10_s*",
         "runs/wavlm_official_proposalULRPK_b16e20_s*"),
        ("HuBERT, rekayasa lawan proposal",
         "runs/hubert_official_fullUF_b32e10_s*",
         "runs/hubert_official_proposalULRPK_b32e20_s*"),
    ]
    raw = []
    for nama, pa, pb in BANDINGAN:
        a, b = _akurasi_seed(pa), _akurasi_seed(pb)
        if len(a) == 0 or len(b) == 0:
            continue
        raw.append({"nama": nama, "na": len(a), "nb": len(b), "ma": a.mean(),
                    "mb": b.mean(), "sel": a.mean() - b.mean(),
                    "p": _welch_p(a, b)})
    # Koreksi Holm-Bonferroni atas seluruh perbandingan yang dapat diuji.
    diuji = [h for h in raw if h["p"] is not None]
    mm = len(diuji)
    for rank, h in enumerate(sorted(diuji, key=lambda x: x["p"])):
        h["ph"] = min(1.0, h["p"] * (mm - rank))
    jalan = 0.0
    for h in sorted(diuji, key=lambda x: x["p"]):
        jalan = max(jalan, h["ph"])
        h["ph"] = jalan
    sig = []
    for h in raw:
        if h["p"] is None:
            sig.append([h["nama"], f"{h['na']}/{h['nb']}", f"{h['ma']:.2f}",
                        f"{h['mb']:.2f}", f"{h['sel']:+.2f}", "n/a", "n/a",
                        "belum dapat diuji"])
            continue
        ph = h["ph"]
        sig.append([h["nama"], f"{h['na']}/{h['nb']}", f"{h['ma']:.2f}",
                    f"{h['mb']:.2f}", f"{h['sel']:+.2f}", f"{h['p']:.4f}",
                    f"{ph:.4f}",
                    ("melampaui ragam" if ph < 0.05 else
                     "di garis batas" if ph < 0.15 else
                     "belum terbukti berbeda")])
    if sig:
        E.append(PageBreak())
        E.append(P("3.11 Mana yang bertahan setelah ragam antar inisialisasi "
                   "diukur", "h2"))
        E.append(P(
            "Seluruh selisih yang dilaporkan sejauh ini perlu diuji terhadap "
            "ragamnya sendiri. Tiap sel dijalankan dengan beberapa inisialisasi "
            "acak, lalu tiap perbandingan diuji dengan uji t Welch yang tidak "
            "mengandaikan ragam kedua kelompok sama. Perlu ditegaskan bahwa "
            "ukuran sampelnya kecil, yaitu paling banyak tiga inisialisasi per "
            "sel, sehingga uji ini berdaya rendah. Nilai p yang besar karena itu "
            "berarti belum terbukti berbeda, dan bukan terbukti sama.", "p"))
        E.append(P(
            "Enam perbandingan diuji sekaligus, sehingga nilai p mentah tidak "
            "dapat dibaca apa adanya. Menguji enam hipotesis pada ambang 0,05 "
            "memberi peluang sekitar 26 persen untuk memperoleh setidaknya satu "
            "hasil yang tampak bermakna semata karena kebetulan. Koreksi "
            "Holm-Bonferroni karena itu diterapkan, dan keputusan diambil dari "
            "kolom p Holm.", "p"))
        E.append(tabel(["Perbandingan", "n", "Rerata A", "Rerata B", "Selisih",
                        "p mentah", "p Holm", "Bacaan"], sig,
                       [3.9 * cm, 1.1 * cm, 1.6 * cm, 1.6 * cm, 1.5 * cm,
                        1.6 * cm, 1.5 * cm, 3.2 * cm]))
        E.append(Spacer(1, 6))
        E.append(P(
            "Hasilnya terbelah bersih menjadi dua kelompok. Kelompok pertama "
            "adalah perbandingan antara konfigurasi proposal dan konfigurasi "
            "rekayasa pada kedua model swa-selia berukuran besar. Selisihnya "
            "berpuluh poin persentase, jauh melampaui simpangan baku antar "
            "inisialisasi yang berkisar satu sampai dua poin, sehingga "
            "kesimpulannya tidak mungkin dibalik oleh ragam. Kelompok kedua "
            "adalah perbandingan antara membekukan dan melatih encoder. "
            "Selisihnya berbilang poin dan berada pada orde yang sama dengan "
            "simpangan bakunya sendiri.", "p"))
        E.append(P(
            "Untuk kelompok kedua, penelitian ini tidak berhak menyatakan bahwa "
            "satu perlakuan lebih baik daripada yang lain. Beberapa kesimpulan "
            "yang sempat ditarik lebih awal, ketika tiap sel baru dijalankan "
            "sekali, karena itu ditarik kembali. Pernyataan bahwa WavLM Large "
            "merupakan pengecualian yang sebaiknya dibekukan termasuk di "
            "antaranya, karena setelah tiga inisialisasi pada kedua sisi nilai "
            "p-nya 0,313.", "p"))
        E.append(P(
            "Perbandingan pada HuBERT Large layak dibahas tersendiri karena "
            "berada tepat di garis batas. Melatih encoder unggul 2,82 poin "
            "persentase atas membekukannya dengan nilai p mentah 0,0500, yang "
            "setelah koreksi Holm menjadi 0,0999. Penulis memilih tidak "
            "menyatakannya sebagai temuan yang mapan. Nilai p yang berhenti "
            "persis di ambang, pada ukuran sampel tiga, merupakan keadaan yang "
            "paling mudah disalahtafsirkan, dan menyebutnya bermakna hanya "
            "karena kebetulan jatuh di sisi yang menguntungkan adalah bentuk "
            "pemilihan hasil yang justru dikritik oleh penelitian ini sendiri di "
            "bagian lain.", "p"))
        stab = []
        for m, b, ket in [("ast", 32, "AST"), ("wavlm", 16, "WavLM Large"),
                          ("hubert", 32, "HuBERT Large")]:
            r1 = _akurasi_seed(f"runs/{m}_official_full_b{b}e10_s*")
            r2 = _akurasi_seed(f"runs/{m}_official_fullUF_b{b}e10_s*")
            if len(r1) < 2 or len(r2) < 2:
                continue
            stab.append([ket,
                         f"{r1.std(ddof=1):.2f}", f"{r1.max() - r1.min():.2f}",
                         f"{r2.std(ddof=1):.2f}", f"{r2.max() - r2.min():.2f}"])
        if stab:
            E.append(P(
                "Kolom simpangan baku menyimpan pengamatan yang tidak terbaca "
                "dari rerata sama sekali, yaitu bahwa kestabilan hasil terhadap "
                "inisialisasi acak sangat berbeda antar arsitektur.", "p"))
            E.append(tabel(["Arsitektur", "Simpangan, encoder beku",
                            "Rentang, encoder beku", "Simpangan, encoder dilatih",
                            "Rentang, encoder dilatih"], stab,
                           [3.2 * cm, 3.3 * cm, 3.0 * cm, 3.4 * cm, 3.0 * cm]))
            E.append(Spacer(1, 6))
            E.append(P(
                "Selisih kestabilan antar arsitektur mencapai satu orde besaran. "
                "Pada AST dengan encoder dibekukan, hasil ketiga inisialisasi "
                "terentang 85,66 sampai 92,65 persen, yaitu rentang 7 poin "
                "persentase yang lebih lebar daripada hampir semua selisih antar "
                "konfigurasi yang dibahas dalam penelitian ini. Pada HuBERT Large "
                "dengan perlakuan yang sama, rentangnya hanya 0,19 poin. Dua "
                "arsitektur yang dilaporkan berdampingan dalam satu tabel karena "
                "itu tidak memiliki bobot bukti yang setara, meskipun keduanya "
                "ditulis dengan jumlah angka desimal yang sama.", "p"))
            E.append(P(
                "Konsekuensinya bagi penelitian ini cukup tajam. Seluruh tangga "
                "ablasi pada bagian 3.9 dijalankan pada AST, yaitu arsitektur "
                "yang paling tidak stabil di antara ketiganya. Selisih 16,91 poin "
                "untuk pembekuan encoder kemungkinan besar tetap bertahan karena "
                "jauh melampaui ragam tersebut, tetapi selisih 9,01 poin untuk "
                "early stopping dan terutama 4,78 poin untuk augmentasi penuh "
                "berada pada wilayah yang menuntut pengujian lebih lanjut sebelum "
                "dapat dinyatakan mantap.", "p"))
        E.append(P(
            "Pada HuBERT Large, konfigurasi yang encodernya dibekukan juga jauh "
            "lebih dapat diulang daripada yang encodernya dilatih. Untuk "
            "pekerjaan yang harus dapat direproduksi oleh orang lain, sifat itu "
            "memiliki nilai tersendiri yang terpisah dari akurasi rerata, "
            "sehingga pemilihan di antara keduanya bukan semata soal angka mana "
            "yang lebih besar.", "p"))
        if A["wavlm_proposal"] and A["hubert_proposal"]:
            E.append(P(
                "Pengamatan yang paling menjelaskan justru datang dari sel "
                "konfigurasi proposal pada kedua model swa-selia berukuran besar. "
                "Simpangan bakunya "
                f"{A['wavlm_proposal']['sd']:.2f} poin persentase pada WavLM "
                f"Large dan {A['hubert_proposal']['sd']:.2f} pada HuBERT Large, "
                "yaitu beberapa kali lipat lebih besar daripada sel mana pun yang "
                "memakai learning rate yang sesuai. Pada HuBERT Large, ketiga "
                "inisialisasi menghasilkan 45,04 sampai 60,94 persen, yaitu "
                "rentang hampir 16 poin persentase.", "p"))
            E.append(P(
                "Sebaran selebar itu konsisten dengan mekanisme yang diusulkan. "
                "Learning rate yang terlalu tinggi untuk encoder yang "
                "bersangkutan membuat pelatihan tidak stabil, sehingga tempat "
                "model berakhir sangat bergantung pada titik awalnya. Kegagalan "
                "yang diakibatkannya karena itu bukan hanya besar tetapi juga "
                "tidak terduga, dan itulah yang membuatnya berbahaya dalam "
                "praktik. Sebuah penelitian yang menjalankan konfigurasi ini "
                "sekali saja dapat memperoleh 60,94 persen dan menyimpulkan "
                "bahwa modelnya lemah, atau memperoleh 45,04 persen dan "
                "menyimpulkan bahwa modelnya tidak berfungsi, tanpa menyadari "
                "bahwa keduanya adalah konfigurasi yang sama.", "p"))
            E.append(P(
                "Sebaran ini juga menjelaskan mengapa dua perbandingan dengan "
                "selisih terbesar dalam tabel di atas justru berhenti tepat di "
                "atas ambang setelah koreksi. Yang membatasi bukan kecilnya "
                "efek, melainkan liarnya kelompok pembanding.", "p"))
        E.append(P(
            "Pengalaman ini sekaligus menjadi contoh konkret bagi anjuran yang "
            "diajukan penelitian ini sendiri, yaitu bahwa hasil sebaiknya "
            "dilaporkan beserta simpangan baku atas beberapa inisialisasi. Pada "
            "AST, rentang antar inisialisasi mencapai 3,67 poin persentase, "
            "yang lebih lebar daripada selisih antar metodologi. Bergantung pada "
            "inisialisasi mana yang kebetulan dilaporkan, penelitian yang sama "
            "dapat menyimpulkan bahwa rekayasa menang 2,75 poin atau kalah 0,92 "
            "poin. Kedua laporan itu akan terdengar sama meyakinkannya dan "
            "sama-sama tidak berdasar.", "p"))

    E.append(PageBreak())
    # ---------------- 4
    E.append(P("4. Usulan: Augmentasi Band-Gain", "h1"))
    E.append(P(
        "Diagnosis pada bagian 3.2 menunjukkan bahwa dua isyarat berbeda menempati "
        "pita frekuensi tinggi yang sama. Yang pertama adalah level energi, yang "
        "merupakan jejak kompresi dan harus dinetralkan. Yang kedua adalah struktur "
        "halus di dalam pita, yang merupakan jejak vocoder dan justru harus "
        "dipertahankan. Augmentasi yang lazim dipakai, seperti penapisan lolos "
        "bawah dan RawBoost, menghapus keduanya sekaligus.", "p"))
    E.append(P(
        "Augmentasi band-gain yang diusulkan hanya mengalikan gain acak pada tiap "
        "pita di atas 3 kHz. Dengan demikian rasio energi antar pita menjadi tidak "
        "informatif terhadap label, sementara bentuk spektrum di dalam pita tetap "
        "utuh. Pengukuran memastikan sifat ini. Pada pita 6 sampai 7 kHz, band-gain "
        "menyisakan korelasi bentuk 0,89 terhadap sinyal asli dengan energi turun "
        "ke 41 persen, sedangkan penapisan lolos bawah menyisakan korelasi negatif "
        "0,05 dengan energi nol persen.", "p"))
    b = []
    for (a, au) in [("nes2net", "full"), ("nes2net", "fullrb"),
                    ("nes2net", "fullbg"), ("nes2net", "fullbgrb")]:
        v = GEN.get((a, au))
        fk = [k for k in FOR if k[0] == a and k[2] == au and k[1] == "official"]
        if v and fk:
            nm = {"full": "Basis (noise, reverb, codec)",
                  "fullrb": "Basis ditambah RawBoost",
                  "fullbg": "Basis ditambah band-gain",
                  "fullbgrb": "Basis ditambah keduanya"}[au]
            b.append([nm, ms(FOR[fk[0]], "acc"), ms(v, "modern"), ms(v, "old")])
    if b:
        E.append(tabel(["Strategi augmentasi", "Akurasi FoR",
                        "Recall TTS 2025-2026", "Recall TTS 2019 non-MP3"], b,
                       [5.6 * cm, 3.2 * cm, 3.4 * cm, 3.6 * cm]))
        E.append(Spacer(1, 4))
        E.append(P("Nes2Net-X, tiga seed per strategi, arsitektur dan data "
                   "identik. Rerata diikuti simpangan baku dalam kurung.", "cap"))
    E.append(P(
        "Perbandingan tersebut menunjukkan bahwa RawBoost menaikkan akurasi pada "
        "dataset tetapi menurunkan kemampuan generalisasi, sedangkan band-gain "
        "menaikkan akurasi dengan penurunan generalisasi yang jauh lebih kecil dan "
        "bahkan memperbaiki recall pada sistem lama yang tidak dikompresi. "
        "Kombinasi keduanya memberi hasil terbaik pada beberapa sumbu sekaligus.", "p"))

    # ---------------- 5
    E.append(P("5. Keterbatasan", "h1"))
    E.append(P(
        "Beberapa keterbatasan perlu dinyatakan secara jujur. Pertama, sebagian "
        "besar konfigurasi hanya diulang dengan tiga inisialisasi acak, sedangkan "
        "variasi antar inisialisasi pada pengujian lintas generasi cukup besar, "
        "dalam beberapa kasus melebihi lima poin persentase. Kesimpulan mengenai "
        "urutan peringkat model karena itu bersifat indikatif.", "p"))
    E.append(P(
        "Kedua, korelasi negatif antara akurasi in-domain dan kemampuan "
        "generalisasi dihitung dari tujuh model. Ukuran sampel sekecil itu tidak "
        "cukup untuk menyimpulkan besaran korelasinya. Yang menopang temuan ini "
        "bukan nilai korelasinya melainkan mekanismenya, yang terdokumentasi secara "
        "terpisah melalui audit dataset, eksperimen augmentasi terkontrol, dan pola "
        "kebutaan model terhadap sistem yang tidak dikompresi.", "p"))
    E.append(P(
        "Ketiga, efek band-gain berbeda arah pada arsitektur yang berbeda. "
        "Penjelasan berdasarkan ruang perbaikan yang tersisa masuk akal namun "
        "disusun setelah melihat data, sehingga perlu pengujian pada arsitektur "
        "tambahan sebelum dapat dinyatakan sebagai mekanisme umum.", "p"))
    E.append(P(
        "Keempat, sejumlah kesimpulan sementara dalam penelitian ini pernah keliru "
        "dan telah dikoreksi setelah pengukuran ulang. Riwayat koreksi tersebut "
        "sengaja dipertahankan dalam dokumentasi pendukung agar dapat ditelusuri.", "p"))

    # ---------------- 6
    E.append(P("6. Kesimpulan", "h1"))
    E.append(P(
        "Akurasi yang tinggi pada satu dataset tidak dengan sendirinya menunjukkan "
        "kemampuan mendeteksi suara sintetis. Pada Fake-or-Real, sebagian besar "
        "selisih angka yang dilaporkan dapat ditelusuri ke protokol pembagian data "
        "dan ke artefak kompresi yang berkorelasi dengan label. Kegagalan di bawah "
        "gangguan noise sebagian besar merupakan kegagalan kalibrasi ambang, bukan "
        "kegagalan pengenalan, sehingga dapat dipulihkan tanpa mengubah model. "
        "Sistem text-to-speech komersial terbaru masih dapat dideteksi pada tingkat "
        "97 sampai 99 persen dengan tingkat alarm palsu 5 persen, tetapi hanya oleh "
        "arsitektur dan strategi augmentasi tertentu, dan bukan oleh model yang "
        "menempati peringkat teratas pada dataset.", "p"))
    E.append(P(
        "Pertanyaan apakah rekayasa metodologi dalam penelitian ini mengungguli "
        "rencana awalnya dijawab dengan tiga inisialisasi acak per sel dan uji "
        "yang dikoreksi untuk banding ganda. Jawabannya sebagian besar negatif, "
        "dan itu perlu dinyatakan terus terang. Pada AST, konfigurasi proposal "
        f"mencapai {_sn(A['ast_proposal'])} sedangkan konfigurasi rekayasa dengan "
        f"encoder yang dilatih mencapai {_sn(A['ast_dilatih'])}, yaitu selisih "
        f"{abs(A['ast_dilatih']['acc'] - A['ast_proposal']['acc']):.2f} poin yang "
        "tidak bermakna secara statistik. Perbandingan antara membekukan dan melatih encoder juga tidak "
        "terbukti berbeda pada ketiga arsitektur setelah koreksi Holm-Bonferroni. "
        "Selisih-selisih yang sempat tampak meyakinkan ketika tiap sel baru "
        "dijalankan sekali ternyata berada di dalam ragam antar inisialisasi.", "p"))
    E.append(P(
        "Yang bertahan hanya satu, dan justru karena itu ia layak dipercaya. "
        "Besaran learning rate relatif terhadap ukuran dan jenis encoder adalah "
        "satu-satunya keputusan yang selisihnya berpuluh poin persentase, jauh "
        "melampaui ragam antar inisialisasi yang berkisar 0,1 sampai 3,5 poin. "
        "Learning rate seragam 0,001 kebetulan tepat untuk AST yang berukuran 86 "
        "juta parameter dan menghancurkan kedua model swa-selia berukuran 300 juta "
        "parameter sampai ke tingkat tebakan acak. Kekeliruan pada keputusan ini "
        "juga tidak dapat ditambal oleh perbaikan lain mana pun di dalam "
        "pipeline, sebagaimana ditunjukkan pada HuBERT Large.", "p"))
    E.append(P(
        "Pesan praktis yang paling kokoh dari seluruh rangkaian percobaan ini "
        "adalah bahwa learning rate yang tidak sesuai dengan encoder tidak dapat "
        "ditambal oleh perbaikan lain di dalam pipeline. Pada HuBERT Large dengan "
        "laju 0,001, seluruh paket rekayasa yang di tempat lain menyumbang "
        "puluhan poin persentase tidak mampu mengangkat model dari tingkat "
        "tebakan acak. Perlindungan parsial yang sempat teramati pada WavLM Large "
        "ternyata tidak berulang, sehingga tidak dapat dinyatakan sebagai sifat "
        "umum paket tersebut.", "p"))
    E.append(P(
        "Implikasi praktisnya adalah bahwa pemilihan model sebaiknya tidak "
        "didasarkan pada akurasi dataset tunggal, dan bahwa pelaporan hasil "
        "sebaiknya menyertakan spesifisitas pada korpus asing serta simpangan baku "
        "atas beberapa inisialisasi.", "p"))
    E.append(P(
        "Anjuran terakhir itu bukan formalitas yang dipinjam dari literatur. "
        "Penelitian ini sendiri hampir melaporkan sejumlah kesimpulan yang keliru "
        "karena mengabaikannya. Pada AST dengan encoder dibekukan, tiga "
        "inisialisasi menghasilkan 85,66 sampai 92,65 persen, yaitu rentang yang "
        "lebih lebar daripada hampir semua selisih antar konfigurasi yang dibahas "
        "di sini. Bergantung pada inisialisasi mana yang kebetulan dijalankan "
        "lebih dahulu, laporan yang sama dapat menyimpulkan bahwa rekayasa unggul "
        "beberapa poin atau tertinggal beberapa poin, dan kedua versi akan "
        "terbaca sama meyakinkannya. Praktik melaporkan satu angka dari satu "
        "inisialisasi karena itu bukan sekadar kurang teliti, melainkan cukup "
        "untuk membalik arah kesimpulan sebuah penelitian.", "p"))

    doc = SimpleDocTemplate(OUT, pagesize=A4,
                            leftMargin=2.4 * cm, rightMargin=2.4 * cm,
                            topMargin=2.2 * cm, bottomMargin=2.2 * cm,
                            title="Ketika Akurasi Tinggi Menyesatkan",
                            author="Gusti Ayu Putu Kesari Purnama Yani")
    doc.build(E)
    print("-> PAPER.pdf")


if __name__ == "__main__":
    bangun()
