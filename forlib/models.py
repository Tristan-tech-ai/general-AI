"""
Empat arsitektur, diimplementasikan dengan perbaikan atas bug yang gagal secara diam-diam.

Perbaikan yang tertanam di sini (lihat ARSITEKTUR.md §6):
  * LR per model, bukan seragam 1e-3       -> ditangani train.py
  * CNN feature extractor SSL selalu beku  -> freeze_feature_encoder()
  * AST max_length=200 (bukan 1024)        -> + interpolasi positional embedding
  * normalisasi diserahkan ke feature-extractor bawaan tiap model
  * agregasi = attentive statistics pooling, bukan mean / [CLS] / hidden akhir
  * layer-weighting yang dipelajari atas SEMUA hidden state, bukan layer terakhir
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

SR = 16000


# ==========================================================================
# Blok bersama
# ==========================================================================
class ConcreteDropout(nn.Module):
    """
    Dropout dengan laju yang DIPELAJARI, bukan ditetapkan manusia.

    Dropout biasa memakai satu angka tetap, misalnya 0,2, yang dipilih dengan
    menebak lalu mencoba. Angka itu tidak pernah berubah selama pelatihan dan
    tidak pernah tahu apa-apa tentang datanya.

    Di sini lajunya menjadi parameter yang ikut dilatih bersama bobot lainnya.
    Persoalannya, menjatuhkan unit adalah keputusan biner, dan keputusan biner
    tidak bisa diturunkan sehingga gradiennya tidak mengalir ke laju itu.
    Concrete Dropout (Gal, Hron, Kendall, NeurIPS 2017) menyelesaikannya dengan
    mengganti undian Bernoulli oleh relaksasi kontinu yang dapat diturunkan,
    sehingga laju dropout ikut menerima gradien seperti bobot biasa.

    Tanpa penahan, gradien itu akan selalu mendorong laju ke nol, karena model
    yang tidak pernah menjatuhkan apa pun selalu mendapat loss latih lebih
    kecil. Penahannya adalah suku entropi pada `regularisasi()`, yang menarik
    laju kembali ke arah 0,5. Laju akhirnya adalah titik seimbang antara
    keduanya, dan titik itu ditentukan data, bukan oleh yang menyetel.
    """

    def __init__(self, p_awal: float = 0.2, suhu: float = 0.1,
                 reg: float = 1e-4):
        super().__init__()
        p_awal = min(max(float(p_awal), 1e-3), 1.0 - 1e-3)
        self.logit_p = nn.Parameter(
            torch.tensor(math.log(p_awal / (1.0 - p_awal)), dtype=torch.float32))
        self.suhu = suhu
        # Nilai bawaan mendekati 2/N dengan N = 13.956 klip latih, yaitu
        # skala yang disarankan makalah aslinya.
        self.reg = reg
        self.n_unit = 0

    @property
    def p(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_p)

    def forward(self, x):
        self.n_unit = x.shape[-1]
        if not self.training:
            return x
        p = self.p
        eps = 1e-7
        u = torch.rand_like(x, dtype=torch.float32)
        # relaksasi kontinu Concrete: mendekati undian Bernoulli(p) ketika
        # suhunya kecil, tetapi tetap dapat diturunkan terhadap p
        z = torch.sigmoid(
            (torch.log(p + eps) - torch.log(1 - p + eps)
             + torch.log(u + eps) - torch.log(1 - u + eps)) / self.suhu)
        return x * (1 - z).to(x.dtype) / (1 - p + eps).to(x.dtype)

    def regularisasi(self) -> torch.Tensor:
        """Suku entropi yang ditambahkan ke loss. Menahan laju agar tidak ke nol."""
        p = self.p
        eps = 1e-7
        entropi_negatif = (p * torch.log(p + eps)
                           + (1 - p) * torch.log(1 - p + eps))
        return self.reg * max(self.n_unit, 1) * entropi_negatif


def dropout_adaptif_pada(model: nn.Module) -> list[ConcreteDropout]:
    return [m for m in model.modules() if isinstance(m, ConcreteDropout)]


def regularisasi_dropout(model: nn.Module):
    """Jumlah suku entropi seluruh ConcreteDropout, atau None bila tidak ada."""
    modul = dropout_adaptif_pada(model)
    if not modul:
        return None
    total = modul[0].regularisasi()
    for m in modul[1:]:
        total = total + m.regularisasi()
    return total


class AttentiveStatsPool(nn.Module):
    """
    Agregator bukti: rata-rata & simpangan baku berbobot atensi.

    Untuk sinyal kuasi-stasioner sepanjang 2 detik, ini mendekati penjumlah
    bukti optimal - jauh lebih baik daripada hidden-state akhir LSTM yang
    memiliki recency bias (lihat ARSITEKTUR.md §3.2).
    """

    def __init__(self, dim: int, bottleneck: int = 128):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv1d(dim, bottleneck, 1),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck),
            nn.Tanh(),
            nn.Conv1d(bottleneck, dim, 1),
        )

    def forward(self, x):                      # x: (B, D, T)
        w = torch.softmax(self.att(x), dim=2)
        mu = (x * w).sum(dim=2)
        var = (x.pow(2) * w).sum(dim=2) - mu.pow(2)
        sd = torch.sqrt(var.clamp(min=1e-8))
        return torch.cat([mu, sd], dim=1)      # (B, 2D)


class LayerWeighting(nn.Module):
    """Bobot softmax yang dipelajari atas seluruh hidden state Transformer."""

    def __init__(self, n_layers: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))

    def forward(self, hidden_states):          # tuple of (B, T, D)
        a = torch.softmax(self.w, dim=0)
        out = hidden_states[0] * a[0]
        for i in range(1, len(hidden_states)):
            out = out + hidden_states[i] * a[i]
        return out

    def weights(self):
        return torch.softmax(self.w.detach(), dim=0).cpu().numpy()


class Head(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, n_cls: int = 2,
                 p: float = 0.2, dropout_adaptif: bool = False):
        super().__init__()
        # Titik tukar dropout sengaja hanya satu dan sama untuk keempat
        # arsitektur, supaya perbandingan tetap vs adaptif tidak tercampur
        # perbedaan letak.
        do = ConcreteDropout(p_awal=p) if dropout_adaptif else nn.Dropout(p)
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            do,
            nn.Linear(hidden, n_cls),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================================================
# 1-3. Model SSL berbasis waveform: Wav2Vec2 / HuBERT / WavLM
# ==========================================================================
SSL_CKPT = {
    "wav2vec2": "facebook/wav2vec2-base",
    "hubert": "facebook/hubert-large-ll60k",
    "wavlm": "microsoft/wavlm-large",
}


class SSLClassifier(nn.Module):
    """
    Encoder SSL (opsional beku) -> layer weighting -> conv bottleneck -> ASP -> head.

    Encoder dibekukan secara default: data latih hanya 13.956 klip (~7,8 jam),
    sementara wav2vec2-base punya 95 juta parameter dan hubert-large 317 juta.
    Fine-tuning penuh pada rasio itu tidak dapat dibenarkan (ARSITEKTUR.md §3.3).
    """

    def __init__(self, kind: str = "wav2vec2", freeze: bool = True,
                 layer_weighting: bool = True, bottleneck: int = 256,
                 dropout: float = 0.2, dropout_adaptif: bool = False):
        super().__init__()
        from transformers import AutoModel, AutoFeatureExtractor

        ckpt = SSL_CKPT[kind]
        self.kind = kind
        self.fe = AutoFeatureExtractor.from_pretrained(ckpt)
        self.encoder = AutoModel.from_pretrained(ckpt)

        # WAJIB: melatih 7 lapis conv di depan hampir selalu merusak
        if hasattr(self.encoder, "freeze_feature_encoder"):
            self.encoder.freeze_feature_encoder()

        self.frozen = freeze
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        D = self.encoder.config.hidden_size
        L = self.encoder.config.num_hidden_layers + 1     # +1 untuk keluaran embedding
        self.lw = LayerWeighting(L) if layer_weighting else None

        self.bottleneck = nn.Sequential(
            nn.Conv1d(D, bottleneck, kernel_size=5, padding=2),
            nn.BatchNorm1d(bottleneck),
            nn.GELU(),
        )
        self.pool = AttentiveStatsPool(bottleneck)
        self.head = Head(bottleneck * 2, 256, 2, dropout, dropout_adaptif)

        # statistik normalisasi bawaan checkpoint (hindari normalisasi ganda)
        self.do_norm = bool(getattr(self.fe, "do_normalize", False))

    def _prep(self, wav):
        if self.do_norm:
            m = wav.mean(dim=1, keepdim=True)
            s = wav.std(dim=1, keepdim=True).clamp(min=1e-7)
            wav = (wav - m) / s
        return wav

    def forward(self, wav, return_layer_weights: bool = False):
        wav = self._prep(wav)
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.encoder(wav, output_hidden_states=True)
        h = self.lw(out.hidden_states) if self.lw is not None else out.last_hidden_state
        h = h.transpose(1, 2)                    # (B, D, T)
        h = self.bottleneck(h)
        z = self.pool(h)
        logits = self.head(z)
        if return_layer_weights and self.lw is not None:
            return logits, self.lw.weights()
        return logits

    def trainable_groups(self, head_lr: float, enc_lr: float):
        g = [{"params": [p for n, p in self.named_parameters()
                         if not n.startswith("encoder.") and p.requires_grad],
              "lr": head_lr}]
        enc = [p for p in self.encoder.parameters() if p.requires_grad]
        if enc:
            g.append({"params": enc, "lr": enc_lr})
        return g


# ==========================================================================
# 4. AST - dengan perbaikan max_length + interpolasi positional embedding
# ==========================================================================
class ASTClassifier(nn.Module):
    """
    Checkpoint AudioSet default memakai max_length=1024 frame (~10,24 detik).
    Audio kita 2 detik ~ 200 frame. Tanpa perbaikan, ~81% masukan adalah padding
    dan positional embedding terlatih dipetakan ke wilayah kosong.

    Di sini max_length=200 DAN bobot posisi lama diinterpolasi (bukan
    diinisialisasi ulang secara acak seperti yang dilakukan
    ignore_mismatched_sizes=True).
    """

    CKPT = "MIT/ast-finetuned-audioset-10-10-0.4593"

    def __init__(self, max_length: int = 200, freeze: bool = False,
                 layer_weighting: bool = True, dropout: float = 0.2,
                 dropout_adaptif: bool = False):
        super().__init__()
        from transformers import ASTConfig, ASTModel, ASTFeatureExtractor

        self.fe = ASTFeatureExtractor.from_pretrained(self.CKPT, max_length=max_length)
        self.max_length = max_length

        src = ASTModel.from_pretrained(self.CKPT)
        cfg = ASTConfig.from_pretrained(self.CKPT)
        cfg.max_length = max_length
        self.encoder = ASTModel(cfg)

        # muat semua bobot, lalu perbaiki positional embedding secara khusus
        sd_src = src.state_dict()
        sd_dst = self.encoder.state_dict()
        for k in sd_dst:
            if k in sd_src and sd_src[k].shape == sd_dst[k].shape:
                sd_dst[k] = sd_src[k]
        pe_key = "embeddings.position_embeddings"
        if pe_key in sd_src and pe_key in sd_dst:
            sd_dst[pe_key] = self._interp_pos(
                sd_src[pe_key], src.config, cfg)
        self.encoder.load_state_dict(sd_dst)
        del src

        self.frozen = freeze
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        D = cfg.hidden_size
        L = cfg.num_hidden_layers + 1
        self.lw = LayerWeighting(L) if layer_weighting else None
        self.bottleneck = nn.Sequential(
            nn.Conv1d(D, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.GELU(),
        )
        self.pool = AttentiveStatsPool(256)
        self.head = Head(512, 256, 2, dropout, dropout_adaptif)

        self.register_buffer("fe_mean", torch.tensor(float(self.fe.mean)))
        self.register_buffer("fe_std", torch.tensor(float(self.fe.std)))

    @staticmethod
    def _grid(cfg):
        f = (cfg.num_mel_bins - cfg.patch_size) // cfg.frequency_stride + 1
        t = (cfg.max_length - cfg.patch_size) // cfg.time_stride + 1
        return f, t

    @classmethod
    def _interp_pos(cls, pe, cfg_src, cfg_dst):
        """Interpolasi grid posisi (f x t_src) -> (f x t_dst), token khusus dijaga."""
        n_special = pe.shape[1] - (cls._grid(cfg_src)[0] * cls._grid(cfg_src)[1])
        fs, ts = cls._grid(cfg_src)
        fd, td = cls._grid(cfg_dst)
        spec = pe[:, :n_special, :]
        grid = pe[:, n_special:, :]                          # (1, fs*ts, D)
        D = grid.shape[-1]
        grid = grid.reshape(1, fs, ts, D).permute(0, 3, 1, 2)  # (1, D, fs, ts)
        grid = F.interpolate(grid, size=(fd, td), mode="bilinear", align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, fd * td, D)
        return torch.cat([spec, grid], dim=1)

    def _fbank(self, wav):
        """Kaldi fbank seperti ASTFeatureExtractor, tetapi di GPU dan batched."""
        feats = []
        for i in range(wav.shape[0]):
            f = torchaudio.compliance.kaldi.fbank(
                wav[i: i + 1], htk_compat=True, sample_frequency=SR,
                use_energy=False, window_type="hanning",
                num_mel_bins=self.encoder.config.num_mel_bins,
                dither=0.0, frame_shift=10,
            )
            n = f.shape[0]
            if n < self.max_length:
                f = F.pad(f, (0, 0, 0, self.max_length - n))
            else:
                f = f[: self.max_length]
            feats.append(f)
        x = torch.stack(feats)
        return (x - self.fe_mean) / (self.fe_std * 2)

    def forward(self, wav, return_layer_weights: bool = False):
        x = self._fbank(wav)
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.encoder(x, output_hidden_states=True)
        h = self.lw(out.hidden_states) if self.lw is not None else out.last_hidden_state
        h = h.transpose(1, 2)
        h = self.bottleneck(h)
        logits = self.head(self.pool(h))
        if return_layer_weights and self.lw is not None:
            return logits, self.lw.weights()
        return logits

    def trainable_groups(self, head_lr: float, enc_lr: float):
        g = [{"params": [p for n, p in self.named_parameters()
                         if not n.startswith("encoder.") and p.requires_grad],
              "lr": head_lr}]
        enc = [p for p in self.encoder.parameters() if p.requires_grad]
        if enc:
            g.append({"params": enc, "lr": enc_lr})
        return g


# ==========================================================================
# 5. CNN-LSTM - baseline, dengan varian ablation
# ==========================================================================
class CNNLSTMClassifier(nn.Module):
    """
    Baseline konvensional pada log-Mel.

    Dua perbaikan atas rancangan proposal:
      * BiLSTM (proposal: LSTM satu arah -> frame awal terdiskriminasi)
      * agregasi lewat ASP atas SELURUH keluaran LSTM, bukan hidden state akhir
        (rumus 38 hal. 45 memakai h_T, agregator terlemah yang tersedia)

    use_lstm=False menghasilkan varian ablation CNN+ASP untuk menguji apakah
    LSTM benar-benar berkontribusi pada segmen 2 detik (ARSITEKTUR.md §3.2).
    """

    def __init__(self, n_mels: int = 128, use_lstm: bool = True,
                 lstm_hidden: int = 256, lstm_layers: int = 2,
                 dropout: float = 0.2, use_asp: bool = True,
                 dropout_adaptif: bool = False):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=1024, hop_length=160, win_length=400,
            n_mels=n_mels, f_min=20, f_max=8000, power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

        ch = [1, 32, 64, 128]
        blocks = []
        for i in range(3):
            blocks += [
                nn.Conv2d(ch[i], ch[i + 1], 3, padding=1),
                nn.BatchNorm2d(ch[i + 1]),
                nn.GELU(),
                nn.MaxPool2d((2, 2)),
            ]
        self.cnn = nn.Sequential(*blocks)
        feat = ch[-1] * (n_mels // 8)

        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(feat, lstm_hidden, lstm_layers,
                                batch_first=True, bidirectional=True,
                                dropout=dropout if lstm_layers > 1 else 0.0)
            seq_dim = lstm_hidden * 2
        else:
            self.proj = nn.Linear(feat, 512)
            seq_dim = 512

        self.use_asp = use_asp
        self.pool = AttentiveStatsPool(seq_dim) if use_asp else None
        self.head = Head(seq_dim * 2 if use_asp else seq_dim, 256, 2, dropout,
                         dropout_adaptif)

        # SpecAugment (hanya aktif saat training)
        self.fmask = torchaudio.transforms.FrequencyMasking(15)
        self.tmask = torchaudio.transforms.TimeMasking(20)

    def forward(self, wav, return_layer_weights: bool = False):
        m = self.to_db(self.mel(wav))                    # (B, n_mels, T)
        m = (m - m.mean(dim=(1, 2), keepdim=True)) / (m.std(dim=(1, 2), keepdim=True) + 1e-5)
        if self.training:
            m = self.tmask(self.fmask(m))
        x = self.cnn(m.unsqueeze(1))                     # (B, C, F', T')
        B, C, Fq, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Fq)  # (B, T, C*F')

        if self.use_lstm:
            x, _ = self.lstm(x)
        else:
            x = self.proj(x)

        if self.use_asp:
            z = self.pool(x.transpose(1, 2))
        else:
            z = x[:, -1, :]                              # perilaku proposal (untuk ablation)
        return self.head(z)

    def trainable_groups(self, head_lr: float, enc_lr: float):
        return [{"params": [p for p in self.parameters() if p.requires_grad],
                 "lr": head_lr}]


# ==========================================================================
# 6. SSL + Nes2Net-X — back-end SOTA hasil fork, dengan satu perbaikan
# ==========================================================================
class SSLNes2Net(nn.Module):
    """
    Front-end SSL (HuggingFace) + back-end Nes2Net-X.

    Nes2Net-X: Liu et al., IEEE T-IFS 2025. EER 1,49% pada ASVspoof 2021 DF —
    terbaik pada tabel pembanding repo resminya, mengungguli AASIST (2,85),
    SLS (1,92), Mamba (1,88), TCM (2,06). Back-end hanya ~511k parameter.

    PERBAIKAN TERHADAP RANCANGAN ASLI: implementasi resmi memanggil SSL dengan
    `features_only=True`, yaitu HANYA layer terakhir. Pengukuran pada proyek ini
    menunjukkan bobot layer yang dipelajari memuncak konsisten di layer 5 dari 24
    (HuBERT dan WavLM, 3 seed). `layer_weighting=True` di sini mengagregasi
    seluruh hidden state; ablasinya (True vs False) menguji apakah kelemahan itu
    nyata pada data ini.
    """

    def __init__(self, kind: str = "wavlm", freeze: bool = True,
                 layer_weighting: bool = True, nes_ratio=(8, 8), dilation: int = 2):
        super().__init__()
        from transformers import AutoModel, AutoFeatureExtractor
        from .nes2net import Nes2NetX

        ckpt = SSL_CKPT[kind]
        self.kind = kind
        self.fe = AutoFeatureExtractor.from_pretrained(ckpt)
        self.encoder = AutoModel.from_pretrained(ckpt)
        if hasattr(self.encoder, "freeze_feature_encoder"):
            self.encoder.freeze_feature_encoder()
        self.frozen = freeze
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        D = self.encoder.config.hidden_size
        L = self.encoder.config.num_hidden_layers + 1
        self.lw = LayerWeighting(L) if layer_weighting else None
        self.backend = Nes2NetX(nes_ratio=nes_ratio, input_channel=D,
                                n_out=2, dilation=dilation, pool="ASTP")
        self.do_norm = bool(getattr(self.fe, "do_normalize", False))

    def forward(self, wav, return_layer_weights: bool = False):
        if self.do_norm:
            m = wav.mean(dim=1, keepdim=True)
            s = wav.std(dim=1, keepdim=True).clamp(min=1e-7)
            wav = (wav - m) / s
        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            out = self.encoder(wav, output_hidden_states=True)
        h = self.lw(out.hidden_states) if self.lw is not None else out.last_hidden_state
        logits = self.backend(h.transpose(1, 2))       # (B, D, T)
        if return_layer_weights and self.lw is not None:
            return logits, self.lw.weights()
        return logits

    def trainable_groups(self, head_lr: float, enc_lr: float):
        g = [{"params": [p for n, p in self.named_parameters()
                         if not n.startswith("encoder.") and p.requires_grad],
              "lr": head_lr}]
        enc = [p for p in self.encoder.parameters() if p.requires_grad]
        if enc:
            g.append({"params": enc, "lr": enc_lr})
        return g


# ==========================================================================
def build_model(name: str, **kw) -> nn.Module:
    name = name.lower()
    if name in ("wav2vec2", "hubert", "wavlm"):
        return SSLClassifier(kind=name, **kw)
    if name == "ast":
        return ASTClassifier(**kw)
    if name == "cnnlstm":
        return CNNLSTMClassifier(**kw)
    if name == "cnn_asp":                                 # ablation tanpa LSTM
        return CNNLSTMClassifier(use_lstm=False, **kw)
    if name == "cnnlstm_proposal":                        # replikasi persis proposal
        return CNNLSTMClassifier(use_lstm=True, use_asp=False, **kw)
    # back-end SOTA hasil fork
    if name.startswith("nes2net"):
        # nes2net           -> wavlm  + layer weighting (perbaikan kami)
        # nes2net_lastlayer -> wavlm  tanpa layer weighting (rancangan asli)
        # nes2net_hubert    -> hubert + layer weighting
        kw.pop("layer_weighting", None)
        kind = "hubert" if name.endswith("hubert") else "wavlm"
        lw = not name.endswith("lastlayer")
        return SSLNes2Net(kind=kind, layer_weighting=lw, **kw)
    raise ValueError(f"model tidak dikenal: {name}")


# LR yang direkomendasikan per model (menggantikan LR seragam 1e-3 di proposal)
DEFAULT_LR = {
    "wav2vec2": (1e-3, 1e-5),
    "hubert":   (1e-3, 5e-6),
    "wavlm":    (1e-3, 5e-6),
    "ast":      (1e-3, 2e-5),
    "cnnlstm":  (1e-3, 1e-3),
    "cnn_asp":  (1e-3, 1e-3),
    "cnnlstm_proposal": (1e-3, 1e-3),
    "nes2net":           (1e-3, 5e-6),
    "nes2net_lastlayer": (1e-3, 5e-6),
    "nes2net_hubert":    (1e-3, 5e-6),
}
