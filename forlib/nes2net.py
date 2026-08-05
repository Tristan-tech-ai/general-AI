"""
Nes2Net-X back-end — di-port dari repo resmi, bebas fairseq.

Sumber: Liu et al., "Nes2Net: A Lightweight Nested Architecture for Foundation
Model Driven Speech Anti-spoofing", IEEE T-IFS 2025.
Repo: https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW  (fork lokal di forks/)
EER 1,49% pada ASVspoof 2021 DF — terbaik pada tabel pembanding repo tersebut,
mengungguli AASIST (2,85), SLS (1,92), Mamba (1,88), TCM (2,06), Conformer (2,27).

DUA PERUBAHAN TERHADAP RANCANGAN ASLI:

1. Front-end bebas fairseq. Aslinya memuat `xlsr2_300m.pt` lewat
   `fairseq.checkpoint_utils`. Di sini dipakai transformers/HuggingFace, jadi
   tidak perlu fairseq (yang bermasalah di Python 3.14) dan checkpoint apa pun
   (wav2vec2 / WavLM / HuBERT) dapat dipasang.

2. **Agregasi berbobot antar-layer.** Aslinya memanggil SSL dengan
   `features_only=True` yang hanya mengembalikan layer TERAKHIR. Pengukuran di
   proyek ini menunjukkan bobot layer yang dipelajari memuncak konsisten di
   **layer 5 dari 24** (HuBERT & WavLM, 3 seed) — bukan layer terakhir. Karena
   itu layer weighting ditambahkan sebagai opsi (default aktif). Ini perbaikan
   yang dapat diuji, bukan sekadar penyalinan.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SEModule(nn.Module):
    def __init__(self, channels, SE_ratio=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // SE_ratio, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(channels // SE_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class Bottle2neck(nn.Module):
    """Res2Net bottleneck dengan penjumlahan berbobot antar-split (Nes2Net-X)."""

    def __init__(self, inplanes, planes, kernel_size=3, dilation=2, scale=8, SE_ratio=8):
        super().__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs, bns, wsum = [], [], []
        pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=(kernel_size, 1),
                                   dilation=(dilation, 1), padding=(pad, 0)))
            bns.append(nn.BatchNorm2d(width))
            wsum.append(nn.Parameter(torch.ones(1, 1, 1, i + 2) * (1.0 / (i + 2))))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.weighted_sum = nn.ParameterList(wsum)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes, SE_ratio)

    def forward(self, x):
        residual = x
        out = self.bn1(self.relu(self.conv1(x))).unsqueeze(-1)      # (B, C, T, 1)
        spx = torch.split(out, self.width, 1)
        sp = spx[self.nums]
        out = None
        for i in range(self.nums):
            sp = torch.cat((sp, spx[i]), -1)
            sp = self.bns[i](self.relu(self.convs[i](sp)))
            sp_s = torch.sum(sp * self.weighted_sum[i], dim=-1, keepdim=False)
            out = sp_s if i == 0 else torch.cat((out, sp_s), 1)
        out = torch.cat((out, spx[self.nums].squeeze(-1)), 1)
        out = self.bn3(self.relu(self.conv3(out)))
        return self.se(out) + residual


class ASTP(nn.Module):
    """Attentive statistics pooling (ECAPA-TDNN)."""

    def __init__(self, in_dim, bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):                       # (B, F, T)
        alpha = torch.tanh(self.linear1(x))     # ReLU di sini sulit konvergen
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        return torch.cat([mean, torch.sqrt(var.clamp(min=1e-10))], dim=1)


class Nes2NetX(nn.Module):
    """Nested Res2Net TDNN: back-end ~511k parameter."""

    def __init__(self, nes_ratio=(8, 8), input_channel=1024, n_out=2,
                 dilation=2, pool="ASTP", se_ratio=8):
        super().__init__()
        assert input_channel % nes_ratio[0] == 0, \
            f"dim SSL {input_channel} harus habis dibagi {nes_ratio[0]}"
        self.n = nes_ratio[0]
        C = input_channel // nes_ratio[0]
        self.C = C
        self.blocks = nn.ModuleList(
            [Bottle2neck(C, C, 3, dilation, nes_ratio[1], se_ratio)
             for _ in range(self.n - 1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(C) for _ in range(self.n - 1)])
        self.bn = nn.BatchNorm1d(input_channel)
        self.relu = nn.ReLU()
        self.pool_name = pool
        if pool == "ASTP":
            self.pooling = ASTP(input_channel, 128)
            self.fc = nn.Linear(input_channel * 2, n_out)
        else:
            self.fc = nn.Linear(input_channel, n_out)

    def forward(self, x):                       # (B, D, T)
        spx = torch.split(x, self.C, 1)
        sp, out = None, None
        for i in range(self.n - 1):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.bns[i](self.relu(self.blocks[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        out = torch.cat((out, spx[-1]), 1)
        out = self.relu(self.bn(out))
        out = self.pooling(out) if self.pool_name == "ASTP" else torch.mean(out, dim=-1)
        return self.fc(out)
