"""Uji cepat: setiap model dapat dibangun, menerima batch 2 detik, dan backward."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from forlib.models import build_model, DEFAULT_LR

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B = 4
wav = torch.randn(B, 32000, device=DEV) * 0.05
y = torch.randint(0, 2, (B,), device=DEV)

MODELS = sys.argv[1:] or ["cnnlstm", "cnn_asp", "cnnlstm_proposal", "wav2vec2", "ast"]

for name in MODELS:
    t0 = time.time()
    try:
        kw = {}
        if name in ("wav2vec2", "hubert", "wavlm", "ast"):
            kw = {"freeze": True, "layer_weighting": True}
        m = build_model(name, **kw).to(DEV)
        n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in m.parameters())

        m.train()
        out = m(wav)
        assert out.shape == (B, 2), f"bentuk keluaran salah: {out.shape}"
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        g = sum(1 for p in m.parameters() if p.requires_grad and p.grad is not None)

        extra = ""
        if name == "ast":
            pe = m.encoder.embeddings.position_embeddings
            fb = m._fbank(wav)
            extra = (f" | pos_emb={tuple(pe.shape)} fbank={tuple(fb.shape)}"
                     f" max_len={m.max_length}")
        if hasattr(m, "lw") and m.lw is not None:
            extra += f" | n_layers={len(m.lw.w)}"

        vram = torch.cuda.max_memory_allocated() / 1024**3 if DEV.type == "cuda" else 0
        print(f"OK   {name:20s} {n_tr/1e6:7.2f}M dilatih /{n_all/1e6:8.2f}M  "
              f"grad_ok={g}  {time.time()-t0:5.1f}s  vram={vram:.2f}GiB{extra}")
        del m
        if DEV.type == "cuda":
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    except Exception as e:
        import traceback
        print(f"GAGAL {name:20s} {type(e).__name__}: {e}")
        traceback.print_exc(limit=6)
