"""
Pelatihan & evaluasi deteksi deepfake audio pada FoR-2sec.

Contoh:
  py train.py --model wav2vec2 --split official  --augment codec --epochs 12
  py train.py --model cnnlstm  --split random    --augment none  --epochs 30
  py train.py --model ast      --split clean_val --augment full  --epochs 12

Perbaikan bug yang tertanam (vs konfigurasi proposal hal. 68):
  * LR per model, bukan seragam 1e-3
  * AdamW + warmup + cosine decay (proposal: tidak ada scheduler)
  * early stopping & pemilihan checkpoint pada val EER (proposal: 20 epoch tetap)
  * mixed precision bf16 (proposal: tidak ada)
  * label smoothing (proposal: tidak ada)
  * ambang keputusan dituning dari VALIDATION, bukan tetap 0,5
"""
from __future__ import annotations

import os
import sys
import json
import time
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forlib.data import (build_manifest, load_manifest, make_splits,
                         FoRDataset, AugmentConfig, collate)
from forlib.models import (build_model, DEFAULT_LR,
                           dropout_adaptif_pada, regularisasi_dropout)
from forlib.metrics import (full_metrics, compute_eer, threshold_from_validation,
                            prior_matched_threshold, TemperatureScaler)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data", "for-2seconds")
MANIFEST = os.path.join(HERE, "manifest.csv")


def get_augment(name: str) -> AugmentConfig:
    if name == "none":
        return AugmentConfig.none()
    if name == "codec":
        # HANYA augmentasi codec: menguji prediksi bahwa menetralkan pintasan
        # MP3 menaikkan akurasi pada test set resmi
        return AugmentConfig(codec=0.6, noise=0.0, reverb=0.0, gain=0.3)
    if name == "noise":
        return AugmentConfig(codec=0.0, noise=0.6, reverb=0.0, gain=0.3,
                             snr_range=(0.0, 30.0))
    if name == "full":
        return AugmentConfig(codec=0.5, noise=0.5, reverb=0.25, gain=0.3,
                             snr_range=(0.0, 30.0))
    if name == "rawboost":
        # hanya RawBoost algo 4 (seri LnL->ISD->SSI), seperti repo Nes2Net
        return AugmentConfig(codec=0.0, noise=0.0, reverb=0.0, gain=0.0,
                             rawboost=0.6, rawboost_algo=4)
    if name == "fullbg":
        # ABLASI BERSIH: identik dengan 'full', HANYA menambah band-gain.
        # Preset 'soft' mengubah empat hal sekaligus sehingga hasilnya tidak
        # dapat diatribusikan; ini memperbaikinya.
        return AugmentConfig(codec=0.5, noise=0.5, reverb=0.25, gain=0.3,
                             snr_range=(0.0, 30.0), band_gain=0.6)
    if name == "proposal":
        # Replikasi persis metodologi proposal (hal. 57): HANYA penambahan noise
        # latar pada SNR 15-30 dB. Tanpa codec, reverb, gain, atau band-gain.
        return AugmentConfig(codec=0.0, noise=1.0, reverb=0.0, gain=0.0,
                             snr_range=(15.0, 30.0))
    if name == "fullbgrb":
        # Kombinasi: band-gain (netralkan level HF) + RawBoost (degradasi kanal).
        # Menguji apakah keunggulan FoR RawBoost dan keunggulan generalisasi
        # band-gain dapat diperoleh sekaligus, atau saling meniadakan.
        return AugmentConfig(codec=0.5, noise=0.5, reverb=0.25, gain=0.3,
                             snr_range=(0.0, 30.0), band_gain=0.6,
                             rawboost=0.4, rawboost_algo=4)
    if name == "soft":
        # USULAN: netralkan pintasan level-HF lewat band-gain acak (bukan low-pass),
        # sehingga struktur halus HF — tempat artefak vocoder modern berada — utuh.
        # Noise & reverb tetap ada tapi lebih ringan agar tidak menutupi detail.
        return AugmentConfig(codec=0.0, noise=0.35, reverb=0.15, gain=0.3,
                             snr_range=(5.0, 30.0), band_gain=0.7)
    if name == "softcodec":
        # band-gain sebagai mekanisme utama, codec low-pass sesekali saja
        return AugmentConfig(codec=0.2, noise=0.35, reverb=0.15, gain=0.3,
                             snr_range=(5.0, 30.0), band_gain=0.6)
    if name == "fullrb":
        # gabungan: degradasi rantai (RawBoost) + lingkungan (noise/reverb/codec)
        return AugmentConfig(codec=0.4, noise=0.4, reverb=0.2, gain=0.3,
                             snr_range=(0.0, 30.0),
                             rawboost=0.5, rawboost_algo=4)
    raise ValueError(name)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    L, Y = [], []
    for b in loader:
        wav = b["wav"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            out = model(wav)
        L.append(out.float().cpu())
        Y.append(b["label"])
    logits = torch.cat(L).numpy()
    y = torch.cat(Y).numpy()
    p = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    return logits, y, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="wav2vec2",
                    choices=["wav2vec2", "hubert", "wavlm", "ast",
                             "cnnlstm", "cnn_asp", "cnnlstm_proposal",
                             "nes2net", "nes2net_lastlayer", "nes2net_hubert"])
    ap.add_argument("--split", default="official",
                    choices=["official", "random", "clean_val", "wavval"])
    ap.add_argument("--augment", default="codec",
                    choices=["none", "codec", "noise", "full", "rawboost", "fullrb",
                             "soft", "softcodec", "fullbg", "fullbgrb",
                             "proposal"])
    ap.add_argument("--uniform-lr", type=float, default=None,
                    help="paksa satu learning rate untuk encoder DAN head "
                         "(replikasi proposal: 0.001 seragam untuk semua model)")
    ap.add_argument("--normalize", default="loudness", choices=["loudness", "peak"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--unfreeze", action="store_true",
                    help="fine-tune encoder SSL (default: beku)")
    ap.add_argument("--bg-f-lo", type=float, default=None,
                    help="batas bawah pita band-gain dalam Hz (bawaan 3000)")
    ap.add_argument("--bg-bands", type=int, default=None,
                    help="jumlah pita band-gain (bawaan 6)")
    ap.add_argument("--bg-db", type=float, default=None,
                    help="besar redaman maksimum band-gain dalam dB, "
                         "rentangnya menjadi minus nilai ini sampai +3 "
                         "(bawaan 12)")
    ap.add_argument("--enc-lr", type=float, default=None,
                    help="timpa learning rate encoder saja, head tetap memakai "
                         "nilai bawaan per model; berguna untuk memisahkan "
                         "pengaruh fine-tuning dari pengaruh laju yang dipakai")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "nadam"],
                    help="adamw adalah bawaan seluruh hasil terdahulu; nadam "
                         "menambahkan momentum Nesterov di atas Adam, dengan "
                         "peluruhan bobot terpisah supaya satu-satunya yang "
                         "berbeda memang suku Nesterov-nya")
    ap.add_argument("--dropout", default="tetap", choices=["tetap", "adaptif"],
                    help="tetap memakai laju dropout 0,2 yang ditetapkan "
                         "manusia; adaptif memakai Concrete Dropout, yang "
                         "mempelajari lajunya sendiri dari data")
    ap.add_argument("--no-layer-weighting", action="store_true")
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--augment-val", action="store_true",
                    help="terapkan augmentasi juga pada validasi (proksi lebih baik)")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # tag menyertakan batch & epoch: tanpa ini, run dengan hyperparameter berbeda
    # saling menimpa dan perbandingan menjadi tidak terkontrol
    tag = (f"{args.model}_{args.split}_{args.augment}"
           f"{'AV' if args.augment_val else ''}"
           f"{'ULR' if args.uniform_lr is not None else ''}"
           f"{'PK' if args.normalize == 'peak' else ''}"
           # tanpa penanda ini, run berencoder beku dan berencoder dilatih
           # memakai tag yang sama dan saling menimpa tanpa peringatan
           f"{'UF' if args.unfreeze else ''}"
           f"{('ENC' + str(args.enc_lr)) if args.enc_lr is not None else ''}"
           # penanda parameter band-gain, tanpa ini tiap kombinasi tuning akan
           # menimpa hasil kombinasi sebelumnya
           f"{('F' + str(int(args.bg_f_lo))) if args.bg_f_lo is not None else ''}"
           f"{('N' + str(args.bg_bands)) if args.bg_bands is not None else ''}"
           f"{('D' + str(int(args.bg_db))) if args.bg_db is not None else ''}"
           # tanpa penanda ini, run Adam dan run NAdam memakai tag yang sama
           # dan yang belakangan menimpa yang duluan tanpa peringatan
           f"{'NAD' if args.optimizer == 'nadam' else ''}"
           f"{'DA' if args.dropout == 'adaptif' else ''}"
           f"_b{args.batch}e{args.epochs}_s{args.seed}")
    outdir = os.path.join(HERE, args.out, tag)
    os.makedirs(outdir, exist_ok=True)
    print(f"== {tag} ==\ndevice={device}  amp={amp_dtype}")

    # ---- data ----
    if not os.path.exists(MANIFEST):
        print("membangun manifest ...")
        build_manifest(DATA_ROOT, MANIFEST)
    rows = load_manifest(MANIFEST)
    tr_rows, va_rows, te_rows = make_splits(rows, args.split, args.seed)
    print(f"split={args.split}  train={len(tr_rows)}  val={len(va_rows)}  test={len(te_rows)}")

    aug = get_augment(args.augment)
    # Parameter bentuk band-gain dapat ditimpa dari baris perintah supaya dapat
    # dituning tanpa menambah preset baru untuk tiap kombinasi.
    if args.bg_f_lo is not None:
        aug.bg_f_lo = args.bg_f_lo
    if args.bg_bands is not None:
        aug.bg_n_bands = args.bg_bands
    if args.bg_db is not None:
        aug.bg_db_range = (-abs(args.bg_db), 3.0)
    ds_tr = FoRDataset(tr_rows, aug, args.normalize, seed=args.seed)
    # Validasi teraugmentasi: menyamakan distribusi validasi dengan distribusi
    # latih teraugmentasi, supaya EER validasi menjadi proksi yang lebih baik
    # untuk EER test. Tanpa ini, akurasi validasi jenuh di ~99,8% apa pun
    # checkpoint-nya sehingga pemilihan checkpoint praktis acak.
    ds_va = FoRDataset(va_rows,
                       aug if args.augment_val else AugmentConfig.none(),
                       args.normalize, seed=args.seed + 777)
    ds_te = FoRDataset(te_rows, AugmentConfig.none(), args.normalize)

    # persistent_workers=False WAJIB untuk loader latih: worker memegang salinan
    # dataset, jadi set_epoch() di proses utama hanya sampai ke worker bila
    # mereka dibuat ulang tiap epoch. Dengan persistent_workers=True, perbaikan
    # bug P0-1 tidak akan berefek sama sekali.
    dl = lambda d, sh: DataLoader(d, batch_size=args.batch, shuffle=sh,
                                  num_workers=args.workers, collate_fn=collate,
                                  pin_memory=True, drop_last=sh,
                                  persistent_workers=(args.workers > 0 and not sh))
    dl_tr, dl_va, dl_te = dl(ds_tr, True), dl(ds_va, False), dl(ds_te, False)

    # ---- model ----
    kw = {}
    if args.model in ("wav2vec2", "hubert", "wavlm", "ast"):
        kw["freeze"] = not args.unfreeze
        kw["layer_weighting"] = not args.no_layer_weighting
    if args.model in ("wav2vec2", "hubert", "wavlm", "ast",
                      "cnnlstm", "cnn_asp", "cnnlstm_proposal"):
        kw["dropout_adaptif"] = args.dropout == "adaptif"
    model = build_model(args.model, **kw).to(device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"parameter: {n_tr/1e6:.2f} M dilatih / {n_all/1e6:.2f} M total")

    head_lr, enc_lr = DEFAULT_LR[args.model]
    if args.enc_lr is not None:
        enc_lr = args.enc_lr
        print(f"learning rate encoder ditimpa menjadi {enc_lr}")
    if args.uniform_lr is not None:
        # Proposal hal. 68 menetapkan satu learning rate untuk seluruh model.
        # Untuk model pra-latih ini berarti encoder ikut dilatih pada LR itu,
        # sehingga encoder tidak lagi dibekukan.
        head_lr = enc_lr = args.uniform_lr
        for p in model.parameters():
            p.requires_grad = True
        # BUG YANG DIPERBAIKI: menyetel requires_grad saja tidak cukup. Ketiga
        # kelas berbasis SSL membungkus forward encoder dengan
        #   ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        # sehingga selama atribut frozen masih True tidak ada gradien yang
        # sampai ke encoder. Optimizer tetap menerima grup parameter encoder,
        # namun grad-nya selalu None sehingga AdamW melewatinya. Akibatnya
        # seluruh run replikasi proposal sebelum perbaikan ini menghasilkan skor
        # yang identik bitwise dengan run berencoder beku, yaitu encoder tidak
        # pernah benar-benar ikut dilatih meskipun log mengaku sebaliknya.
        if hasattr(model, "frozen"):
            model.frozen = False
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"mode LR seragam {args.uniform_lr}: encoder DILATIH, "
              f"{n_tr/1e6:.2f} M parameter")
    grup = model.trainable_groups(head_lr, enc_lr)
    if args.optimizer == "nadam":
        # decoupled_weight_decay menyamakan perlakuan peluruhan bobot dengan
        # AdamW. Tanpa itu NAdam memakai peluruhan gaya L2 yang menyatu ke
        # dalam gradien, sehingga perbandingannya akan mengandung dua
        # perbedaan sekaligus, bukan satu.
        try:
            opt = torch.optim.NAdam(grup, weight_decay=0.01,
                                    decoupled_weight_decay=True)
        except TypeError:
            print("PERINGATAN: versi torch ini belum punya "
                  "decoupled_weight_decay pada NAdam. Peluruhan bobot "
                  "dimatikan agar perbandingannya tetap satu perbedaan.")
            opt = torch.optim.NAdam(grup, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(grup, weight_decay=0.01)
    print(f"optimizer: {args.optimizer}   dropout: {args.dropout}")
    steps = max(1, len(dl_tr)) * args.epochs
    warm = int(0.1 * steps)

    def lr_lambda(s):
        if s < warm:
            return s / max(warm, 1)
        prog = (s - warm) / max(steps - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best = {"eer": 1e9, "epoch": -1}
    hist = []
    ckpt = os.path.join(outdir, "best.pt")
    bad = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        # WAJIB: tanpa ini augmentasi identik di setiap epoch (bug P0-1)
        ds_tr.set_epoch(ep)
        t0, tot, seen = time.time(), 0.0, 0
        for i, b in enumerate(dl_tr):
            wav = b["wav"].to(device, non_blocking=True)
            y = b["label"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                loss = crit(model(wav), y)
            # Suku entropi Concrete Dropout dihitung DI LUAR autocast, dalam
            # float32. Di dalam bf16 ia berupa selisih logaritma bernilai
            # kecil yang mudah hilang ke pembulatan, dan bila hilang lajunya
            # akan meluncur ke nol tanpa penahan.
            reg = regularisasi_dropout(model)
            if reg is not None:
                loss = loss.float() + reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            tot += float(loss.item()) * len(y)
            seen += len(y)
            if i % 50 == 0:
                print(f"  ep{ep} step {i}/{len(dl_tr)} loss={tot/max(seen,1):.4f}",
                      end="\r")

        _, yv, pv = evaluate(model, dl_va, device, amp_dtype)
        eer_v, _ = compute_eer(yv, pv)
        acc_v = float(((pv >= 0.5).astype(int) == yv).mean())
        dt = time.time() - t0
        print(f"  ep{ep:02d}  loss={tot/max(seen,1):.4f}  "
              f"val_acc={acc_v*100:.2f}%  val_EER={eer_v*100:.2f}%  ({dt:.0f}s)")
        hist.append({"epoch": ep, "loss": tot / max(seen, 1),
                     "val_acc": acc_v, "val_eer": eer_v, "sec": dt})

        # pemilihan checkpoint berdasarkan EER, bukan akurasi:
        # pada rezim akurasi tinggi, akurasi validasi sudah jenuh
        if eer_v < best["eer"]:
            best = {"eer": eer_v, "acc": acc_v, "epoch": ep}
            torch.save(model.state_dict(), ckpt)
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stopping (patience {args.patience})")
                break

    # ---- evaluasi akhir dengan checkpoint terbaik ----
    model.load_state_dict(torch.load(ckpt, map_location=device))
    lg_v, yv, pv = evaluate(model, dl_va, device, amp_dtype)
    lg_t, yt, pt = evaluate(model, dl_te, device, amp_dtype)

    # ambang HARUS dipilih dari validation - memilih dari test adalah kebocoran
    thr = threshold_from_validation(yv, pv, "youden")
    cal = TemperatureScaler().fit(lg_v, yv)
    pt_cal = cal.transform(lg_t)[:, 1]
    # ambang prior-matched: transduktif (memakai skor test tanpa label),
    # dilaporkan terpisah dan diberi label jelas
    thr_prior = prior_matched_threshold(pt, 0.5)

    res = {
        "tag": tag, "args": vars(args), "best_epoch": best["epoch"],
        "params_trainable_M": n_tr / 1e6,
        "val": full_metrics(yv, pv, 0.5),
        "test@0.5": full_metrics(yt, pt, 0.5),
        "test@val_threshold": full_metrics(yt, pt, thr),
        "test@prior_matched": full_metrics(yt, pt, thr_prior),
        "test_calibrated": full_metrics(yt, pt_cal, 0.5),
        "temperature": cal.T,
        "history": hist,
    }
    if hasattr(model, "lw") and getattr(model, "lw", None) is not None:
        res["layer_weights"] = [float(x) for x in model.lw.weights()]
    adaptif = dropout_adaptif_pada(model)
    if adaptif:
        res["dropout_dipelajari"] = [float(m.p.detach()) for m in adaptif]
        res["dropout_awal"] = 0.2
        print(f"laju dropout yang dipelajari: "
              f"{[round(float(m.p.detach()), 4) for m in adaptif]}  (mulai dari 0,2)")

    np.save(os.path.join(outdir, "test_scores.npy"),
            np.stack([yt.astype(float), pt, pt_cal]))
    json.dump(res, open(os.path.join(outdir, "results.json"), "w"), indent=2)

    m = res["test@val_threshold"]
    print("\n" + "=" * 62)
    print(f"HASIL  {tag}")
    print("=" * 62)
    print(f"  akurasi test  : {m['accuracy']*100:.2f}%  (Â±{m['ci95_pp']:.2f} pp)")
    print(f"  EER           : {m['eer']*100:.2f}%")
    print(f"  AUC           : {m['auc']:.4f}")
    print(f"  F1            : {m['f1']*100:.2f}%")
    print(f"  presisi/recall: {m['precision']*100:.2f}% / {m['recall']*100:.2f}%")
    print(f"  salah         : {m['n_errors']} dari {m['n']} berkas")
    print(f"  ECE           : {m['ece']:.4f}   (kalibrasi T={cal.T:.3f})")
    mp = res["test@prior_matched"]
    print(f"  --- ambang prior-matched (transduktif, tanpa label) ---")
    print(f"  akurasi test  : {mp['accuracy']*100:.2f}%   F1={mp['f1']*100:.2f}%   "
          f"salah={mp['n_errors']}/{mp['n']}")
    if "layer_weights" in res:
        w = res["layer_weights"]
        print(f"  layer dominan : {int(np.argmax(w))} dari {len(w)-1} "
              f"(bobot {max(w):.3f})")
    print(f"\n-> {outdir}")


if __name__ == "__main__":
    main()

