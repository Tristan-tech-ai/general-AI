"""
Muat checkpoint Nes2Net-X SOTA (EER 1,49% ASVspoof 2021 DF) tanpa fairseq.

Checkpoint resmi menyimpan front-end XLS-R dalam penamaan fairseq. Skrip ini
memetakannya ke penamaan HuggingFace sehingga model dapat dijalankan dengan
transformers saja, lalu memuat back-end Nes2Net ke port lokal.

Tujuannya satu: menjalankan model SOTA-2025 yang dilatih pada ASVspoof secara
ZERO-SHOT pada (a) FoR-2sec dan (b) TTS komersial 2025-2026 (Chatterbox,
ElevenLabs-v3). Ini menguji langsung apakah detektor terbaik yang tersedia
publik masih mengenali suara AI generasi terbaru.
"""
from __future__ import annotations

import os
import re
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.nes2net import Nes2NetX

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt", "nes2net_x_avg5.pth")
XLSR = "facebook/wav2vec2-xls-r-300m"


def map_ssl_key(k: str):
    """fairseq Wav2Vec2Model -> HuggingFace Wav2Vec2Model."""
    k = k.replace("ssl_model.model.", "")

    # feature extractor: conv_layers.N.0.* -> conv_layers.N.conv.*
    #                    conv_layers.N.2.1.* -> conv_layers.N.layer_norm.*
    m = re.match(r"^feature_extractor\.conv_layers\.(\d+)\.0\.(weight|bias)$", k)
    if m:
        return f"feature_extractor.conv_layers.{m.group(1)}.conv.{m.group(2)}"
    m = re.match(r"^feature_extractor\.conv_layers\.(\d+)\.2\.1\.(weight|bias)$", k)
    if m:
        return f"feature_extractor.conv_layers.{m.group(1)}.layer_norm.{m.group(2)}"

    if k.startswith("post_extract_proj."):
        return k.replace("post_extract_proj.", "feature_projection.projection.")
    if k.startswith("layer_norm."):
        return k.replace("layer_norm.", "feature_projection.layer_norm.")

    # positional conv memakai weight-norm. fairseq menyimpan weight_g/weight_v;
    # torch modern di HF memakai parametrizations.weight.original0/original1.
    # Tanpa pemetaan eksplisit ini, konvolusi posisi tetap berbobot acak.
    if k == "encoder.pos_conv.0.weight_g":
        return "encoder.pos_conv_embed.conv.parametrizations.weight.original0"
    if k == "encoder.pos_conv.0.weight_v":
        return "encoder.pos_conv_embed.conv.parametrizations.weight.original1"
    if k.startswith("encoder.pos_conv.0."):
        return k.replace("encoder.pos_conv.0.",
                         "encoder.pos_conv_embed.conv.")
    if k.startswith("encoder.layer_norm."):
        return k

    m = re.match(r"^encoder\.layers\.(\d+)\.(.+)$", k)
    if m:
        i, rest = m.group(1), m.group(2)
        rest = rest.replace("self_attn_layer_norm.", "layer_norm.")
        rest = rest.replace("self_attn.", "attention.")
        rest = rest.replace("fc1.", "feed_forward.intermediate_dense.")
        rest = rest.replace("fc2.", "feed_forward.output_dense.")
        return f"encoder.layers.{i}.{rest}"
    return None


class SotaNes2Net(nn.Module):
    """XLS-R (bobot fine-tuned dari checkpoint) + Nes2Net-X back-end."""

    def __init__(self, ckpt_path=CKPT, device="cuda"):
        super().__init__()
        from transformers import AutoModel

        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.encoder = AutoModel.from_pretrained(XLSR)

        # ---- front-end ----
        hf = self.encoder.state_dict()
        new, miss, bad = {}, 0, []
        for k, v in sd.items():
            if not k.startswith("ssl_model."):
                continue
            nk = map_ssl_key(k)
            if nk is None:
                continue
            if nk in hf:
                if hf[nk].shape == v.shape:
                    new[nk] = v
                else:
                    # pos_conv fairseq memakai weight_g/weight_v (weight-norm)
                    bad.append((nk, tuple(hf[nk].shape), tuple(v.shape)))
            else:
                miss += 1
        loaded = self.encoder.load_state_dict(new, strict=False)
        self.n_ssl_loaded = len(new)
        self.n_ssl_total = len(hf)
        self.ssl_missing = len(loaded.missing_keys)
        self.shape_mismatch = bad

        # ---- back-end ----
        # Konfigurasi checkpoint RILIS berbeda dari default di kode repo:
        # bentuk bobot menunjukkan SE_ratio=1 (se.1.weight = [128,128,1], bukan
        # [16,128,1]) dan pool='mean' (fc = [2,1024], bukan [2,2048] untuk ASTP).
        self.backend = Nes2NetX(nes_ratio=(8, 8), input_channel=1024,
                                n_out=2, dilation=2, pool="mean", se_ratio=1)
        bmap = {}
        for k, v in sd.items():
            if not k.startswith("Nested_Res2Net_TDNN."):
                continue
            nk = k.replace("Nested_Res2Net_TDNN.", "")
            nk = nk.replace("Build_in_Res2Nets.", "blocks.")
            bmap[nk] = v
        r = self.backend.load_state_dict(bmap, strict=False)
        self.n_be_loaded = len(bmap)
        self.be_missing = list(r.missing_keys)
        self.be_unexpected = list(r.unexpected_keys)

        self.eval()

    def forward(self, wav):
        # XLS-R: normalisasi zero-mean unit-var (do_normalize=True)
        m = wav.mean(dim=1, keepdim=True)
        s = wav.std(dim=1, keepdim=True).clamp(min=1e-7)
        wav = (wav - m) / s
        with torch.no_grad():
            h = self.encoder(wav).last_hidden_state          # rancangan asli: layer terakhir
        return self.backend(h.transpose(1, 2))


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SotaNes2Net(device=dev).to(dev)
    print(f"front-end XLS-R : {m.n_ssl_loaded}/{m.n_ssl_total} tensor dimuat, "
          f"{m.ssl_missing} tidak terisi")
    if m.shape_mismatch:
        print("  bentuk tidak cocok:", m.shape_mismatch[:5])
    print(f"back-end Nes2Net: {m.n_be_loaded} tensor dimuat, "
          f"{len(m.be_missing)} hilang, {len(m.be_unexpected)} tak terduga")
    if m.be_missing[:5]:
        print("  hilang:", m.be_missing[:5])
    if m.be_unexpected[:5]:
        print("  tak terduga:", m.be_unexpected[:5])

    x = torch.randn(2, 32000, device=dev) * 0.05
    with torch.no_grad():
        o = m(x)
    print("uji maju OK, keluaran:", tuple(o.shape), "logits:", o.cpu().numpy().round(3).tolist())
