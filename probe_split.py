"""
EKSPERIMEN PENENTU.

Hipotesis: hasil tinggi yang dilaporkan pada FoR (94-99%) sebagian besar
merupakan artefak dari RE-SPLITTING ACAK, yang menghapus pergeseran domain
yang sengaja dibangun pembuat dataset ke dalam partisi resmi.

Proposal tesis (hal. 55) merencanakan split acak 60/20/20 — yang persis
melakukan penghapusan itu.

Uji: classifier IDENTIK, fitur IDENTIK, hanya skema split yang berbeda.
Bila hipotesis benar, split acak akan memberi akurasi jauh lebih tinggi
tanpa model menjadi lebih baik sedikit pun.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "for-2seconds")
NB = 32


def job(a):
    path, sp, cl = a
    x, sr = sf.read(path, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    S = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    tot = S.sum() + 1e-20
    e = np.linspace(0, sr / 2, NB + 1)
    prof = np.array([S[(f >= e[i]) & (f < e[i + 1])].sum() / tot for i in range(NB)])
    ax = np.abs(x)
    extra = np.array([
        float(np.sqrt((x ** 2).mean())), float(ax.max()), float(x.mean()),
        float((ax < 1e-6).mean()), float((np.diff(np.signbit(x)) != 0).mean()),
        float((freqs_centroid := (f * S).sum() / tot)),
    ])
    return sp, cl, np.concatenate([np.log10(prof + 1e-12), extra])


def main():
    items = []
    for sp in ["training", "validation", "testing"]:
        for cl in ["real", "fake"]:
            d = os.path.join(ROOT, sp, cl)
            for fn in sorted(os.listdir(d)):
                if fn.lower().endswith(".wav"):
                    items.append((os.path.join(d, fn), sp, cl))

    print(f"memproses {len(items)} berkas ...")
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 4)) as ex:
        R = list(ex.map(job, items, chunksize=64))

    X = np.nan_to_num(np.array([r[2] for r in R]))
    y = np.array([1 if r[1] == "fake" else 0 for r in R])
    sp_arr = np.array([r[0] for r in R])

    L = []
    def out(s=""):
        print(s); L.append(s)

    out("# Eksperimen Penentu: Split Resmi vs Split Acak\n")
    out("Classifier identik (RandomForest 400 pohon), fitur identik "
        "(38 fitur spektral+statistik global).\n")
    out("**Tidak ada informasi fonetik, tidak ada fase, tidak ada model deep learning.**\n")
    out("Satu-satunya yang berubah: cara data dibagi.\n")

    def fit(tr, te, label):
        clf = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        acc = float((clf.predict(X[te]) == y[te]).mean())
        out(f"| {label} | {len(tr)} | {len(te)} | **{acc*100:.2f}%** |")
        return acc

    out("| skema split | n latih | n uji | akurasi uji |")
    out("|---|---|---|---|")

    # A. Split resmi FoR
    tr_off = np.flatnonzero((sp_arr == "training") | (sp_arr == "validation"))
    te_off = np.flatnonzero(sp_arr == "testing")
    a_off = fit(tr_off, te_off, "**Resmi FoR** (training+val → testing)")

    # B. Split acak 60/20/20 seperti rencana proposal (hal. 55)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    n60 = int(0.6 * len(y)); n80 = int(0.8 * len(y))
    tr_rand = idx[:n60]
    te_rand = idx[n80:]
    a_rand = fit(tr_rand, te_rand, "**Acak 60/20/20** (rencana proposal)")

    # C. Split acak hanya di dalam data training resmi (kontrol)
    pool = np.flatnonzero(sp_arr == "training")
    p = rng.permutation(pool)
    a_ctl = fit(p[:int(0.8 * len(p))], p[int(0.8 * len(p)):],
                "Acak, hanya dalam training resmi (kontrol)")

    out("")
    out("## Kesimpulan\n")
    out(f"Selisih split acak vs split resmi: **{(a_rand - a_off)*100:+.2f} poin persentase**")
    out(f"pada model yang sama persis dan fitur yang sama persis.\n")
    if a_rand - a_off > 0.15:
        out("🔴 **HIPOTESIS TERKONFIRMASI.** Split acak menaikkan akurasi secara dramatis")
        out("tanpa peningkatan kemampuan model sedikit pun. Kenaikan itu sepenuhnya berasal")
        out("dari kebocoran domain: partisi resmi FoR sengaja menempatkan sumber rekaman")
        out("yang berbeda di test set, dan split acak menghancurkan pemisahan tersebut.\n")
        out(f"Artinya: angka {a_rand*100:.1f}% dari fitur spektral sepele saja sudah")
        out("mendekati angka yang dilaporkan penelitian deep learning sebelumnya pada FoR")
        out("(93,50% / 94,47% / 94,7%). Ini indikasi kuat bahwa sebagian besar hasil")
        out("terpublikasi pada FoR memakai split acak dan mengukur kebocoran domain,")
        out("bukan kemampuan deteksi deepfake.\n")
    else:
        out("Hipotesis tidak terkonfirmasi pada fitur ini.\n")

    open("probe_split_report.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n-> probe_split_report.md")


if __name__ == "__main__":
    main()
