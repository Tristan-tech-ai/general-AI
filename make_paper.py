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


def ms(v, k, pct=True):
    a = np.array([x[k] for x in v], dtype=float) * (100 if pct else 1)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return "n/a"
    if len(a) == 1:
        return f"{a[0]:.2f}"
    return f"{a.mean():.2f} ({a.std(ddof=1):.2f})"


# ---------------------------------------------------------------- isi
def bangun():
    FOR = kumpul_for()
    GEN = kumpul_gen()
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
        "menghasilkan kesimpulan yang tidak tunggal. Pada WavLM Large, "
        "konfigurasi proposal runtuh menjadi 56,99 persen dengan area under curve "
        "0,6569 sedangkan konfigurasi yang diperbaiki mencapai 98,62 persen. Pada "
        "AST arahnya terbalik, yaitu proposal mencapai 92,56 persen sedangkan "
        "konfigurasi yang diperbaiki hanya 89,15 persen. Penelusuran sampai ke "
        "tiap keputusan desain menunjukkan bahwa penyebabnya sama pada kedua "
        "kasus, yaitu satu keputusan mengenai perlakuan encoder yang diseragamkan "
        "lintas arsitektur padahal arah pengaruhnya berlawanan. Kesimpulan yang "
        "dilaporkan karena itu bukan bahwa satu metodologi mengungguli yang lain, "
        "melainkan bahwa keputusan tersebut harus dipilih per arsitektur.", "p"))

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
            "pada ambang prior-matched mencapai 92,56 persen, sedangkan "
            "konfigurasi yang direkayasa hanya mencapai 89,15 persen. Area under "
            "curve juga lebih rendah, yaitu 0,9586 lawan 0,9780, sehingga "
            "penurunan itu bukan sekadar soal letak ambang melainkan penurunan "
            "daya pisah yang sesungguhnya. Pada arsitektur ini, paket rekayasa "
            "merugikan.", "p"))
        E.append(P(
            "Pada WavLM Large keadaannya justru sebaliknya dan jauh lebih ekstrem. "
            "Konfigurasi proposal runtuh menjadi 56,99 persen dengan area under "
            "curve 0,6569 dan equal error rate 43,01 persen, yaitu hanya sedikit "
            "lebih baik daripada menebak. Konfigurasi yang direkayasa mencapai "
            "98,62 persen dengan area under curve 0,9992 dan equal error rate 1,41 "
            "persen. Di sini paket rekayasa bukan hanya menolong, melainkan "
            "menjadi pembeda antara model yang berguna dan model yang tidak.", "p"))
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
    tg, prev = [], None
    for kode, nama, d in TANGGA:
        m = _kumpul_ambang(d)
        if m is None:
            continue
        tg.append([kode, nama, f"{m['apm']:.2f}",
                   "" if prev is None else f"{m['apm'] - prev:+.2f}",
                   f"{m['auc']:.4f}", f"{m['eer']:.2f}"])
        prev = m["apm"]

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
        E.append(tabel(["Langkah", "Perbaikan yang ditambahkan", "Akurasi",
                        "Selisih (pp)", "AUC", "EER"], tg,
                       [1.6 * cm, 6.4 * cm, 2.2 * cm, 2.2 * cm, 2.0 * cm,
                        1.8 * cm]))
        E.append(Spacer(1, 6))
        E.extend(gambar("11_tangga_ablasi.png", 15.0 * cm,
                        "Gambar 6. Sumbangan tiap perbaikan pada AST di partisi "
                        "resmi. Batang merah menandai perbaikan yang merugikan "
                        "bila berdiri sendiri."))
        E.append(P(
            "Tangga ini menjawab pertanyaan yang tertinggal pada bagian 3.8. "
            "Sepanjang lima langkah, akurasi justru turun 3,40 poin persentase, "
            "dari 92,56 menjadi 89,15 persen. Paket rekayasa secara keseluruhan "
            "merugikan pada arsitektur ini, dan penyebabnya terpusat pada satu "
            "langkah saja. Membekukan encoder membuang 16,91 poin persentase, "
            "sementara seluruh perbaikan lain digabung hanya mengembalikan 13,51 "
            "poin. Normalisasi loudness ternyata hampir netral, early stopping "
            "menyumbang 9,01 poin, dan augmentasi penuh menyumbang 4,78 poin. "
            "Ketiganya adalah perbaikan yang sah, namun tidak cukup untuk menutup "
            "kerugian dari satu keputusan yang keliru.", "p"))
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
        E.append(P(
            "Dua pola yang berbeda perlu dipisahkan. Pola pertama konsisten pada "
            "seluruh arsitektur yang diuji, yaitu bahwa besaran learning rate "
            "harus disesuaikan dengan encodernya. Laju 0,001 yang ditetapkan "
            "proposal menghancurkan kedua model swa-selia berukuran 300 juta "
            "parameter sampai mendekati tebakan acak, yaitu 56,99 persen pada "
            "WavLM Large dan 50,46 persen pada HuBERT Large, sementara laju yang "
            "sama justru membantu AST yang berukuran 86 juta parameter. Pola ini "
            "terlihat pada dua model yang tujuan dan korpus pra-pelatihannya "
            "berbeda, sehingga cukup kuat untuk dilaporkan.", "p"))
        E.append(P(
            "Pola kedua tidak konsisten, dan penulis semula keliru "
            "menggeneralisasikannya. Bila learning rate sudah wajar, melatih "
            "encoder lebih baik daripada membekukannya pada AST sebesar 4,04 poin "
            "persentase dan pada HuBERT Large sebesar 4,23 poin, tetapi lebih "
            "buruk pada WavLM Large sebesar 2,48 poin. Dugaan awal bahwa model "
            "swa-selia berukuran besar sebaiknya dibekukan karena "
            "representasinya sudah selaras dengan tugas ternyata tidak bertahan. "
            "HuBERT Large berukuran sama dan sama-sama swa-selia, namun berperilaku "
            "seperti AST dan bukan seperti WavLM. Yang tersisa dari dugaan itu "
            "hanyalah pengamatan bahwa WavLM Large merupakan pengecualian, dan "
            "penelitian ini tidak memiliki bukti yang cukup untuk menjelaskan "
            "mengapa. Pra-pelatihan WavLM yang menyertakan denoising merupakan "
            "kandidat penjelasan, namun itu tetap dugaan yang belum diuji.", "p"))
        E.append(P(
            "Perlu dicatat pula bahwa sel WavLM yang dilatih baru dijalankan "
            "dengan satu inisialisasi acak, sedangkan sel WavLM yang dibekukan "
            "memakai tiga. Selisih 2,48 poin itu memang beberapa kali lipat "
            "simpangan bakunya, namun pengujian dengan lebih banyak inisialisasi "
            "tetap diperlukan sebelum pengecualian ini dapat dinyatakan mantap.", "p"))
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
            "Satu hasil tambahan layak dicatat karena membela nilai paket "
            "rekayasa dari sudut yang berbeda. Ketika WavLM Large dilatih pada "
            "laju 0,001 yang merusak itu, konfigurasi proposal jatuh ke 56,99 "
            "persen sedangkan paket rekayasa dengan laju yang sama tetap "
            "bertahan di 80,06 persen. Selisih 23,07 poin persentase itu "
            "menunjukkan bahwa early stopping, augmentasi penuh, dan agregasi "
            "berbobot antar lapisan berfungsi sebagai jaring pengaman terhadap "
            "pilihan learning rate yang buruk. Paket itu tidak menyelamatkan "
            "model sepenuhnya, tetapi mengubah kegagalan total menjadi kegagalan "
            "yang masih dapat diperbaiki. Nilai semacam ini tidak terlihat bila "
            "yang dilaporkan hanya akurasi puncak.", "p"))

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
        "rencana awalnya tidak memiliki jawaban tunggal, dan itulah temuannya. "
        "Pada WavLM Large rekayasa tersebut menjadi pembeda antara model yang "
        "berguna dan model yang hampir setara tebakan, yaitu 98,62 lawan 56,99 "
        "persen. Pada AST rekayasa yang sama justru merugikan, yaitu 89,15 lawan "
        "92,56 persen. Penelusuran sampai ke tiap keputusan desain menunjukkan "
        "bahwa satu keputusan bertanggung jawab atas hampir seluruh selisih itu, "
        "yaitu apakah encoder dibekukan atau ikut dilatih. Arah pengaruh keputusan "
        "tersebut berlawanan pada kedua arsitektur.", "p"))
    E.append(P(
        "Dari sini muncul simetri yang layak dicatat. Rencana awal menyeragamkan "
        "learning rate untuk seluruh arsitektur, sedangkan penelitian ini "
        "menyeragamkan pembekuan encoder untuk seluruh arsitektur. Keduanya "
        "merupakan kelas kesalahan yang sama, dan masing-masing kebetulan tepat "
        "pada arsitektur yang berbeda. Pelajaran yang dapat diambil bukanlah "
        "bahwa satu pihak lebih benar, melainkan bahwa keputusan semacam ini "
        "sebaiknya dipilih per arsitektur dengan data validasi dan tidak "
        "ditetapkan seragam di muka.", "p"))
    E.append(P(
        "Nilai paket rekayasa juga terlihat pada sumbu yang berbeda dari akurasi "
        "puncak. Ketika learning rate yang merusak tetap dipaksakan pada WavLM "
        "Large, konfigurasi proposal jatuh ke 56,99 persen sedangkan paket "
        "rekayasa bertahan di 80,06 persen. Early stopping, augmentasi penuh, dan "
        "agregasi berbobot antar lapisan berfungsi sebagai jaring pengaman "
        "terhadap pilihan hyperparameter yang buruk. Ketahanan semacam ini tidak "
        "terbaca sama sekali bila yang dilaporkan hanya angka terbaik.", "p"))
    E.append(P(
        "Implikasi praktisnya adalah bahwa pemilihan model sebaiknya tidak "
        "didasarkan pada akurasi dataset tunggal, dan bahwa pelaporan hasil "
        "sebaiknya menyertakan spesifisitas pada korpus asing serta simpangan baku "
        "atas beberapa inisialisasi.", "p"))

    doc = SimpleDocTemplate(OUT, pagesize=A4,
                            leftMargin=2.4 * cm, rightMargin=2.4 * cm,
                            topMargin=2.2 * cm, bottomMargin=2.2 * cm,
                            title="Ketika Akurasi Tinggi Menyesatkan",
                            author="Gusti Ayu Putu Kesari Purnama Yani")
    doc.build(E)
    print("-> PAPER.pdf")


if __name__ == "__main__":
    bangun()
