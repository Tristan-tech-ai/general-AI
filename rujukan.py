"""
Daftar pustaka naskah, terpisah supaya dapat diperiksa satu per satu.

Tiap butir sudah ditelusuri keberadaannya. Yang tidak dapat ditelusuri tidak
dimasukkan, betapa pun cocoknya ia dengan argumen yang sedang dibangun. Butir
yang isinya hanya diketahui dari abstrak diberi catatan pada medan "catatan",
dan naskah tidak boleh mengutip rincian metodenya.

Kunci dipakai sebagai penanda sitasi di dalam naskah, misalnya [reimao2019].
Nomor urut dalam daftar pustaka dihasilkan otomatis menurut urutan kemunculan
pertama, seperti gaya IEEE.
"""

RUJUKAN = {
    # ---------------------------------------------------------- dataset
    "reimao2019": (
        "R. Reimao dan V. Tzerpos, \"FoR: A Dataset for Synthetic Speech "
        "Detection\", dalam Proc. International Conference on Speech Technology "
        "and Human-Computer Dialogue (SpeD), 2019."),
    "muller2022": (
        "N. M. M&uuml;ller, P. Czempin, F. Dieckmann, A. Froghyar dan K. "
        "B&ouml;ttinger, \"Does Audio Deepfake Detection Generalize?\", dalam "
        "Proc. Interspeech, 2022, hlm. 2783-2787."),
    "muller2024": (
        "N. M. M&uuml;ller dkk., \"Harder or Different? Understanding "
        "Generalization of Audio Deepfake Detection\", dalam Proc. Interspeech, "
        "2024."),
    "yamagishi2021": (
        "J. Yamagishi dkk., \"ASVspoof 2021: Accelerating Progress in Spoofed "
        "and Deepfake Speech Detection\", dalam Proc. ASVspoof Workshop, 2021. "
        "arXiv:2109.00537."),
    "delgado2021": (
        "H. Delgado dkk., \"ASVspoof 2021: Automatic Speaker Verification "
        "Spoofing and Countermeasures Challenge Evaluation Plan\", 2021. "
        "arXiv:2109.00535."),
    "liu2023": (
        "X. Liu dkk., \"ASVspoof 2021: Towards Spoofed and Deepfake Speech "
        "Detection in the Wild\", IEEE/ACM Trans. Audio, Speech, and Language "
        "Processing, 2023. arXiv:2210.02437."),
    "wang2024asv5": (
        "X. Wang dkk., \"ASVspoof 5: Crowdsourced Speech Data, Deepfakes, and "
        "Adversarial Attacks at Scale\", dalam Proc. ASVspoof Workshop, 2024. "
        "arXiv:2408.08739."),
    "wang2025asv5": (
        "X. Wang dkk., \"ASVspoof 5: Design, Collection and Validation of "
        "Resources for Spoofing, Deepfake, and Adversarial Attack Detection "
        "Using Crowdsourced Speech\", 2025. arXiv:2502.08857."),
    "spoofceleb": (
        "J. Jung dkk., \"SpoofCeleb: Speech Deepfake Detection and SASV In The "
        "Wild\", 2024. arXiv:2409.17285."),
    "add2022": (
        "J. Yi dkk., \"ADD 2022: The First Audio Deep Synthesis Detection "
        "Challenge\", dalam Proc. ICASSP, 2022. arXiv:2202.08433."),
    "demand": (
        "J. Thiemann, N. Ito dan E. Vincent, \"The Diverse Environments "
        "Multi-channel Acoustic Noise Database (DEMAND)\", Proceedings of "
        "Meetings on Acoustics, vol. 19, 2013."),

    # ------------------------------------------------------- arsitektur
    "wavlm": (
        "S. Chen dkk., \"WavLM: Large-Scale Self-Supervised Pre-Training for "
        "Full Stack Speech Processing\", IEEE Journal of Selected Topics in "
        "Signal Processing, vol. 16, no. 6, hlm. 1505-1518, 2022. "
        "arXiv:2110.13900."),
    "hubert": (
        "W.-N. Hsu dkk., \"HuBERT: Self-Supervised Speech Representation "
        "Learning by Masked Prediction of Hidden Units\", IEEE/ACM Trans. "
        "Audio, Speech, and Language Processing, vol. 29, hlm. 3451-3460, 2021. "
        "arXiv:2106.07447."),
    "wav2vec2": (
        "A. Baevski, H. Zhou, A. Mohamed dan M. Auli, \"wav2vec 2.0: A "
        "Framework for Self-Supervised Learning of Speech Representations\", "
        "dalam Advances in Neural Information Processing Systems, 2020. "
        "arXiv:2006.11477."),
    "ast": (
        "Y. Gong, Y.-A. Chung dan J. Glass, \"AST: Audio Spectrogram "
        "Transformer\", dalam Proc. Interspeech, 2021. arXiv:2104.01778."),
    "aasist": (
        "J. Jung dkk., \"AASIST: Audio Anti-Spoofing Using Integrated "
        "Spectro-Temporal Graph Attention Networks\", dalam Proc. ICASSP, 2022. "
        "arXiv:2110.01200."),
    "rawnet2": (
        "H. Tak, J. Patino, M. Todisco, A. Nautsch, N. Evans dan A. Larcher, "
        "\"End-to-End Anti-Spoofing with RawNet2\", dalam Proc. ICASSP, 2021. "
        "arXiv:2011.01108."),
    "nes2net": (
        "T. Liu dkk., \"Nes2Net: A Lightweight Nested Architecture for "
        "Foundation Model Driven Speech Anti-Spoofing\", IEEE Trans. "
        "Information Forensics and Security, 2025. arXiv:2504.05657."),
    "asp": (
        "K. Okabe, T. Koshinaka dan K. Shinoda, \"Attentive Statistics Pooling "
        "for Deep Speaker Embedding\", dalam Proc. Interspeech, 2018. "
        "arXiv:1803.10963."),
    "scalable_aasist": (
        "Y. Kwak dkk., \"Towards Scalable AASIST: Refining Graph Attention for "
        "Speech Deepfake Detection\", 2025. arXiv:2507.11777."),
    "wavlm_ensemble": (
        "K. Borodin dkk., \"WavLM Model Ensemble for Audio Deepfake "
        "Detection\", 2024. arXiv:2408.07414."),

    # -------------------------------------------------------- augmentasi
    "tak2022ssl": (
        "H. Tak, M. Todisco, X. Wang, J. Jung, J. Yamagishi dan N. Evans, "
        "\"Automatic Speaker Verification Spoofing and Deepfake Detection Using "
        "wav2vec 2.0 and Data Augmentation\", dalam Proc. Odyssey, 2022. "
        "arXiv:2202.12233."),
    "rawboost": (
        "H. Tak, M. Kamble, J. Patino, M. Todisco dan N. Evans, \"RawBoost: A "
        "Raw Data Boosting and Augmentation Method Applied to Automatic Speaker "
        "Verification Anti-Spoofing\", dalam Proc. ICASSP, 2022."),
    "ssl_compare": (
        "\"A Comparison of SSL-Based Feature Extractors and Back-End "
        "Classifiers for Spoofing Detection: A Multi-Corpus Training and "
        "Cross-Linguistic Analysis\", 2026. arXiv:2606.08669."),

    # ------------------------------------------- metrik dan kalibrasi
    "eer_hides": (
        "\"When EER Hides Deployment Failure: Auditing Threshold Transfer and "
        "Unlabeled Score Calibration for Speech Deepfake Detectors\", 2026. "
        "arXiv:2606.21584."),
    "guo2017": (
        "C. Guo, G. Pleiss, Y. Sun dan K. Q. Weinberger, \"On Calibration of "
        "Modern Neural Networks\", dalam Proc. International Conference on "
        "Machine Learning, 2017. arXiv:1706.04599."),
    "generalize_real": (
        "\"How Well Do Current Speech Deepfake Detection Methods Generalize to "
        "the Real World?\", 2026. arXiv:2603.05852."),
    "deepen": (
        "\"DeePen: Penetration Testing for Audio Deepfake Detection\", 2025. "
        "arXiv:2502.20427."),

    # ---------------------------------------------------- bias dataset
    "beyond_silence": (
        "H. Kim dkk., \"Beyond Silence: Bias Analysis through Loss and "
        "Asymmetric Approach in Audio Anti-Spoofing\", 2024. arXiv:2406.17246."),
    "geirhos2020": (
        "R. Geirhos dkk., \"Shortcut Learning in Deep Neural Networks\", Nature "
        "Machine Intelligence, vol. 2, hlm. 665-673, 2020."),
    "torralba2011": (
        "A. Torralba dan A. A. Efros, \"Unbiased Look at Dataset Bias\", dalam "
        "Proc. IEEE Conference on Computer Vision and Pattern Recognition, "
        "2011."),
    "codecfake": (
        "\"CodecFake+: Codec-Based Resynthesized Data as a Proxy for Detecting "
        "CodecFake Speech\", 2025. arXiv:2501.08238."),
    "linguistic_bias": (
        "\"Linguistic Bias Mitigation for Spoofing Detection via Gradient "
        "Reversal and a Variational Information Bottleneck\", 2026. "
        "arXiv:2606.31411."),
    "survey2024": (
        "M. Li dkk., \"Audio Anti-Spoofing Detection: A Survey\", 2024. "
        "arXiv:2404.13914."),
    "taxonomy": (
        "A. Khan dkk., \"Voice Spoofing Countermeasures: Taxonomy, "
        "State-of-the-Art, Experimental Analysis of Generalizability, Open "
        "Challenges, and the Way Forward\", 2022. arXiv:2210.00417."),

    # --------------------------------------- ragam dan reproduksibilitas
    "bouthillier2021": (
        "X. Bouthillier dkk., \"Accounting for Variance in Machine Learning "
        "Benchmarks\", dalam Proc. Machine Learning and Systems (MLSys), 2021. "
        "arXiv:2103.03098."),
    "holm1979": (
        "S. Holm, \"A Simple Sequentially Rejective Multiple Test Procedure\", "
        "Scandinavian Journal of Statistics, vol. 6, no. 2, hlm. 65-70, 1979."),
    "benjamini1995": (
        "Y. Benjamini dan Y. Hochberg, \"Controlling the False Discovery Rate: "
        "A Practical and Powerful Approach to Multiple Testing\", Journal of "
        "the Royal Statistical Society B, vol. 57, no. 1, hlm. 289-300, 1995."),
    "welch1947": (
        "B. L. Welch, \"The Generalization of Student's Problem when Several "
        "Different Population Variances are Involved\", Biometrika, vol. 34, "
        "no. 1-2, hlm. 28-35, 1947."),
    "demsar2006": (
        "J. Dem&#353;ar, \"Statistical Comparisons of Classifiers over Multiple "
        "Data Sets\", Journal of Machine Learning Research, vol. 7, hlm. 1-30, "
        "2006."),
    "nosek2018": (
        "B. A. Nosek, C. R. Ebersole, A. C. DeHaven dan D. T. Mellor, \"The "
        "Preregistration Revolution\", Proceedings of the National Academy of "
        "Sciences, vol. 115, no. 11, hlm. 2600-2606, 2018."),
    "ioannidis2005": (
        "J. P. A. Ioannidis, \"Why Most Published Research Findings Are "
        "False\", PLoS Medicine, vol. 2, no. 8, e124, 2005."),
    "recht2019": (
        "B. Recht, R. Roelofs, L. Schmidt dan V. Shankar, \"Do ImageNet "
        "Classifiers Generalize to ImageNet?\", dalam Proc. International "
        "Conference on Machine Learning, 2019."),

    # --------------------------------- pembanding langsung pada FoR
    "ahmad2026": (
        "S. Ahmad, M. Ahmed dan S. Imtiaz, \"Classical Machine Learning "
        "Baselines for Deepfake Audio Detection on the Fake-or-Real Dataset\", "
        "Clarkson University, 2026. arXiv:2604.13400."),
    "cnnlstm2025": (
        "\"Hybrid CNN-LSTM Architectures for Deepfake Audio Detection Using "
        "MFCC and Spectrogram Analysis\", American Journal of Mathematical and "
        "Computer Modelling, vol. 10, no. 3, 2025."),
    "mfaan2023": (
        "K. Bhagtani dkk., \"MFAAN: Unveiling Audio Deepfakes with a "
        "Multi-Feature Authenticity Network\", 2023. arXiv:2311.03509."),
}

# Butir yang hanya diketahui dari abstrak. Naskah tidak boleh mengutip rincian
# metode dari butir-butir ini, hanya angka yang tercantum pada abstrak.
HANYA_ABSTRAK = {"mfaan2023"}


class Sitasi:
    """Penomoran gaya IEEE menurut urutan kemunculan pertama di dalam naskah."""

    def __init__(self):
        self.urut = []

    def __call__(self, *kunci):
        no = []
        for k in kunci:
            if k not in RUJUKAN:
                raise KeyError(f"rujukan tidak terdaftar: {k}")
            if k not in self.urut:
                self.urut.append(k)
            no.append(self.urut.index(k) + 1)
        return "[" + ", ".join(str(n) for n in sorted(no)) + "]"

    def daftar(self):
        return [(i + 1, RUJUKAN[k]) for i, k in enumerate(self.urut)]

    def belum_dipakai(self):
        return [k for k in RUJUKAN if k not in self.urut]
