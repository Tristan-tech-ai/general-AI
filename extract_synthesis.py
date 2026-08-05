"""Ekstrak tiga dokumen sintesis workflow dari output task ke berkas Markdown."""
import json
import os
import re
import sys

SRC = (r"C:\Users\Tristan\AppData\Local\Temp\claude"
       r"\C--Users-Tristan-Downloads\2eb65358-beca-4011-b948-1c0d9186e535"
       r"\tasks\wl7rjgij0.output")

raw = open(SRC, encoding="utf-8", errors="replace").read()
print("output task:", len(raw), "chars")

# berkas ini adalah JSON utuh: {summary, agentCount, logs, result:{...}, ...}
obj = None
try:
    top = json.loads(raw)
    for msg in (top.get("logs") or []):
        print("  log:", msg)
    obj = top.get("result")
except Exception as e:
    print("gagal parse JSON utuh:", e)

if obj is None:
    print("tidak ada objek result")
    sys.exit(1)

print("kunci:", list(obj.keys()))
print(f"dimensi={obj.get('dimensi')} temuan={obj.get('temuan_total')} "
      f"diverifikasi={obj.get('diverifikasi')} bertahan={obj.get('bertahan')} "
      f"direfutasi={obj.get('direfutasi')}")

OUT = {
    "roadmap_novelty": "RISET_PETA_NOVELTY.md",
    "rencana_eksperimen": "RISET_RENCANA_EKSPERIMEN.md",
    "celah_dan_kritik": "RISET_CELAH_DAN_KRITIK.md",
}

for key, fn in OUT.items():
    val = obj.get(key)
    if not val:
        print(f"  {key:22s} KOSONG")
        continue
    open(fn, "w", encoding="utf-8").write(val)
    print(f"  {key:22s} -> {fn}  ({len(val):,} chars)")

# daftar bertahan / direfutasi
L = ["# Ringkasan Verifikasi Adversarial\n"]
L.append(f"Dari **{obj.get('temuan_total')}** temuan, **{obj.get('diverifikasi')}** "
         f"diverifikasi dengan 3 lensa adversarial: "
         f"**{obj.get('bertahan')} bertahan**, **{obj.get('direfutasi')} direfutasi**.\n")
L.append("> Tingkat kelolosan yang sangat rendah ini adalah sinyal penting: ide-ide "
         "yang bersumber dari literatur sebagian besar tidak bertahan saat diuji "
         "terhadap konteks nyata proyek ini. Kekuatan tesis berasal dari data yang "
         "sudah diukur sendiri, bukan dari literatur.\n")

surv = obj.get("daftar_bertahan") or []
if surv:
    L.append("## Bertahan\n")
    for f in surv:
        L.append(f"### {f.get('title','?')}")
        L.append(f"*dimensi `{f.get('dim')}` · novelty `{f.get('novelty')}` · "
                 f"usaha `{f.get('effort')}` · verdict `{f.get('verdicts')}`*\n")
        L.append(f"{f.get('gain','')}\n")
        for s in (f.get("sources") or [])[:4]:
            L.append(f"- <{s}>")
        L.append("")

ref = obj.get("daftar_direfutasi") or []
if ref:
    L.append("\n## Direfutasi — jangan dipakai\n")
    L.append("| klaim | dimensi | alasan utama gugur |")
    L.append("|---|---|---|")
    for f in ref:
        why = f.get("alasan") or []
        why = " ".join(str(w) for w in why).replace("\n", " ")[:300]
        L.append(f"| {str(f.get('title','?'))[:90]} | `{f.get('dim')}` | {why} |")

open("RISET_VERIFIKASI.md", "w", encoding="utf-8").write("\n".join(L))
print("  verifikasi             -> RISET_VERIFIKASI.md")
