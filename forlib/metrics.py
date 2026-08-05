"""
Metrik evaluasi.

Menambahkan apa yang hilang dari proposal (hal. 70-72): EER, kurva DET,
kalibrasi (ECE), threshold tuning dari validation, dan uji signifikansi
McNemar berpasangan - yang wajib karena test set hanya 1.088 berkas dan
selisih < 1,7 pp tidak dapat dibedakan tanpa uji berpasangan.

Juga memperbaiki dua kesalahan rumus di proposal:
  hal. 72 menulis Sensitivity = TP/(TP+FP) dan Specificity = TP/(TP+FN).
  Yang benar: Sensitivity = TP/(TP+FN), Specificity = TN/(TN+FP).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


def compute_eer(y_true, y_score):
    """Equal Error Rate + threshold pada titik EER. y_true: 1 = fake (positif)."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[i] + fnr[i]) / 2.0)
    return eer, float(thr[i])


def threshold_from_validation(y_true, y_score, criterion="youden"):
    """Pilih ambang dari VALIDATION (bukan test - itu kebocoran)."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    if criterion == "youden":
        return float(thr[int(np.argmax(tpr - fpr))])
    if criterion == "eer":
        return compute_eer(y_true, y_score)[1]
    if criterion == "f1":
        best, bt = -1.0, 0.5
        for t in thr:
            p = (y_score >= t).astype(int)
            tp = int(((p == 1) & (y_true == 1)).sum())
            fp = int(((p == 1) & (y_true == 0)).sum())
            fn = int(((p == 0) & (y_true == 1)).sum())
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best:
                best, bt = f1, float(t)
        return bt
    raise ValueError(criterion)


def prior_matched_threshold(y_score, positive_rate=0.5):
    """
    Ambang yang menyamakan proporsi prediksi positif dengan prior kelas yang
    diketahui. TIDAK memakai label test sama sekali - hanya distribusi skor
    dan fakta bahwa test set FoR seimbang 544/544 (terverifikasi dari struktur
    direktori, bukan dari label prediksi).

    Ini teknik koreksi prior-shift standar. Berguna ketika model memiliki
    diskriminasi baik (EER rendah) tetapi ambangnya bergeser karena perbedaan
    domain - persis situasi FoR. Wajib dilaporkan terpisah dan diberi label
    jelas sebagai transduktif.
    """
    s = np.asarray(y_score, dtype=float)
    return float(np.quantile(s, 1.0 - positive_rate))


def expected_calibration_error(y_true, y_prob, n_bins=15):
    """ECE - seberapa jujur probabilitas yang dikeluarkan model."""
    conf = np.maximum(y_prob, 1 - y_prob)
    pred = (y_prob >= 0.5).astype(int)
    acc = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() > 0:
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def full_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    n = len(y_true)
    acc = (tp + tn) / n
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)                      # = sensitivity (rumus BENAR)
    spec = tn / max(tn + fp, 1)                     # = specificity (rumus BENAR)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")
    eer, eer_thr = compute_eer(y_true, y_score)

    return {
        "n": int(n), "accuracy": float(acc), "precision": float(prec),
        "recall": float(rec), "specificity": float(spec), "f1": float(f1),
        "auc": auc, "eer": float(eer), "eer_threshold": float(eer_thr),
        "threshold": float(threshold),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n_errors": int(fp + fn),
        "ci95_pp": float(1.96 * np.sqrt(acc * (1 - acc) / n) * 100),
        "ece": expected_calibration_error(y_true, y_score),
    }


def mcnemar(y_true, pred_a, pred_b, exact_threshold=25):
    """
    Uji McNemar berpasangan: apakah model A dan B benar-benar berbeda?
    Wajib untuk membandingkan model pada test set yang sama.
    """
    y_true = np.asarray(y_true).astype(int)
    a = (np.asarray(pred_a).astype(int) == y_true)
    b = (np.asarray(pred_b).astype(int) == y_true)
    n01 = int((a & ~b).sum())          # A benar, B salah
    n10 = int((~a & b).sum())          # A salah, B benar
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0, "test": "degenerate"}

    if n < exact_threshold:
        from math import comb
        k = min(n01, n10)
        p = sum(comb(n, i) for i in range(0, k + 1)) * 2 / (2 ** n)
        return {"n01": n01, "n10": n10, "p_value": float(min(p, 1.0)), "test": "exact"}

    from scipy.stats import chi2
    stat = (abs(n01 - n10) - 1) ** 2 / n           # koreksi kontinuitas
    return {"n01": n01, "n10": n10,
            "p_value": float(1 - chi2.cdf(stat, 1)),
            "chi2": float(stat), "test": "chi2"}


def holm_bonferroni(pvals: dict, alpha: float = 0.05):
    """Koreksi perbandingan ganda untuk 6 pasangan dari 4 model."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = max(prev, min(1.0, (m - i) * p))
        prev = adj
        out[k] = {"p_raw": p, "p_adj": adj, "significant": adj < alpha}
    return out


class TemperatureScaler:
    """Kalibrasi 1 parameter, dioptimasi pada VALIDATION set."""

    def __init__(self):
        self.T = 1.0

    def fit(self, logits, y_true, lr=0.01, steps=300):
        import torch
        lg = torch.tensor(np.asarray(logits), dtype=torch.float32)
        y = torch.tensor(np.asarray(y_true), dtype=torch.long)
        logT = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([logT], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(lg / logT.exp(), y)
            loss.backward()
            opt.step()
        self.T = float(logT.exp().item())
        return self

    def transform(self, logits):
        lg = np.asarray(logits, dtype=float) / self.T
        e = np.exp(lg - lg.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
