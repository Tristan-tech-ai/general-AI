"""
Ekstrak temuan riset dari journal.jsonl workflow ke Markdown.

Sintesis workflow gagal karena batas sesi, tetapi seluruh hasil agent tersimpan
di journal. Skrip ini memanennya sehingga tidak bergantung pada tahap sintesis.
"""
import json
import os
import re
import sys
from collections import defaultdict

JOURNAL = (r"C:\Users\Tristan\.claude\projects\C--Users-Tristan-Downloads"
           r"\2eb65358-beca-4011-b948-1c0d9186e535\subagents\workflows"
           r"\wf_b19d7541-a60\journal.jsonl")

if not os.path.exists(JOURNAL):
    print("journal tidak ditemukan:", JOURNAL)
    sys.exit(1)

records = []
with open(JOURNAL, encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            pass

print(f"journal: {len(records)} baris")
types = defaultdict(int)
for r in records:
    types[r.get("type", "?")] += 1
print("tipe:", dict(types))

# kumpulkan temuan
by_dim = defaultdict(list)
verdicts = []
seen = set()

def harvest(obj, label=""):
    if not isinstance(obj, dict):
        return
    if "findings" in obj and isinstance(obj["findings"], list):
        dim = obj.get("dimension") or label or "?"
        for f in obj["findings"]:
            if not isinstance(f, dict):
                continue
            key = (f.get("title", "")[:90]).strip().lower()
            if key and key in seen:
                continue
            seen.add(key)
            by_dim[dim].append(f)
    if "verdict" in obj and "reason" in obj:
        verdicts.append({"label": label, **obj})

for r in records:
    if r.get("type") != "result":
        continue
    label = r.get("label") or r.get("agentLabel") or ""
    val = r.get("result", r.get("value"))
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except Exception:
            continue
    harvest(val, label)

n = sum(len(v) for v in by_dim.values())
print(f"temuan unik: {n} dari {len(by_dim)} dimensi;  verdict: {len(verdicts)}")

RANK = {"untested-idea": 3, "novel-combination": 3, "emerging": 1, "established": 0}
EFF = {"low": 2, "medium": 1, "high": 0}

flat = []
for dim, fs in by_dim.items():
    for f in fs:
        f["_dim"] = dim
        f["_score"] = RANK.get(f.get("novelty", ""), 0) + EFF.get(f.get("effort", ""), 0)
        flat.append(f)
flat.sort(key=lambda f: -f["_score"])

L = []
def out(s=""):
    L.append(s)

out("# Panen Temuan Riset Multi-Agent\n")
out(f"Diekstrak dari `journal.jsonl` run `wf_b19d7541-a60`: "
    f"**{n} temuan unik** dari **{len(by_dim)} dimensi riset**, "
    f"{len(verdicts)} verdict refutasi.\n")
out("> Tahap sintesis workflow gagal karena batas sesi, jadi dokumen ini adalah "
    "panen mentah yang terurut, bukan sintesis. Label `novelty` dan `expected_gain` "
    "adalah **klaim agent**, belum diverifikasi kecuali ada verdict yang menyertainya.\n")

out("## Temuan berpotensi novel / usaha rendah (prioritas baca)\n")
top = [f for f in flat if f["_score"] >= 3][:40]
for i, f in enumerate(top, 1):
    out(f"### {i}. {f.get('title','(tanpa judul)')}")
    out(f"*dimensi: `{f['_dim']}` · novelty: `{f.get('novelty','?')}` · "
        f"usaha: `{f.get('effort','?')}`*\n")
    out(f"**Apa:** {f.get('what','')}\n")
    out(f"**Mekanisme:** {f.get('mechanism','')}\n")
    out(f"**Penerapan:** {f.get('applicability','')}\n")
    out(f"**Perkiraan dampak:** {f.get('expected_gain','')}\n")
    ev = str(f.get("evidence", ""))
    out(f"**Bukti:** {ev[:600]}{'...' if len(ev) > 600 else ''}\n")
    src = f.get("sources") or []
    if src:
        out("**Sumber:** " + ", ".join(f"<{s}>" for s in src[:5]) + "\n")
    out("---\n")

out("\n## Indeks seluruh temuan per dimensi\n")
for dim in sorted(by_dim):
    out(f"### `{dim}` — {len(by_dim[dim])} temuan\n")
    for f in by_dim[dim]:
        out(f"- **{f.get('title','?')}** "
            f"*({f.get('novelty','?')}, usaha {f.get('effort','?')})* — "
            f"{str(f.get('mechanism',''))[:200]}")
    out("")

if verdicts:
    out("\n## Verdict refutasi\n")
    out("| verdict | klaim | alasan (ringkas) | dampak realistis |")
    out("|---|---|---|---|")
    for v in verdicts[:60]:
        lab = re.sub(r"^refute\d+:", "", v.get("label", ""))[:60]
        reason = str(v.get("reason", "")).replace("\n", " ")[:220]
        real = str(v.get("realistic_gain", "")).replace("\n", " ")[:140]
        out(f"| **{v.get('verdict','?')}** | {lab} | {reason} | {real} |")

open("TEMUAN_RISET.md", "w", encoding="utf-8").write("\n".join(L))
print("-> TEMUAN_RISET.md")
