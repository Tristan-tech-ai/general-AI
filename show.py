"""Tampilkan ringkas hasil semua run di folder runs/."""
import json
import os
import sys
import glob

pat = sys.argv[1] if len(sys.argv) > 1 else "*"
rows = []
for p in sorted(glob.glob(os.path.join("runs", pat, "results.json"))):
    r = json.load(open(p, encoding="utf-8"))
    rows.append(r)

if not rows:
    print("belum ada hasil")
    sys.exit()

hdr = ("run", "val_acc", "val_EER", "test_acc", "test_EER", "test_AUC", "test_F1", "err")
print("| " + " | ".join(hdr) + " |")
print("|" + "|".join(["---"] * len(hdr)) + "|")
for r in rows:
    v, t = r["val"], r["test@val_threshold"]
    t05 = r["test@0.5"]
    best = max(t["accuracy"], t05["accuracy"])
    print(f"| {r['tag']} | {v['accuracy']*100:.2f}% | {v['eer']*100:.2f}% | "
          f"{best*100:.2f}% | {t['eer']*100:.2f}% | {t['auc']:.4f} | "
          f"{max(t['f1'], t05['f1'])*100:.2f}% | {min(t['n_errors'], t05['n_errors'])} |")

print()
for r in rows:
    print(f"--- {r['tag']}  (epoch terbaik {r['best_epoch']}) ---")
    for k in ["val", "test@0.5", "test@val_threshold", "test_calibrated"]:
        m = r[k]
        print(f"  {k:20s} acc={m['accuracy']*100:6.2f}%  EER={m['eer']*100:5.2f}%  "
              f"AUC={m['auc']:.4f}  F1={m['f1']*100:6.2f}%  "
              f"TP={m['tp']:4d} TN={m['tn']:4d} FP={m['fp']:4d} FN={m['fn']:4d}")
    if "layer_weights" in r:
        w = r["layer_weights"]
        top = sorted(range(len(w)), key=lambda i: -w[i])[:4]
        print("  bobot layer teratas:", [(i, round(w[i], 3)) for i in top])
    print()
