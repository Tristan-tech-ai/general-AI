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
        "struktur halusnya, dan menguji dampaknya secara terkontrol.", "p"))

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
