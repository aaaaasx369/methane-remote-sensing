#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--dataset-root", required=True)
p.add_argument("--input", default="eval_emit_480m.csv")
p.add_argument("--output", default="eval_emit_480m_abs.csv")
a=p.parse_args()
root=Path(a.dataset_root).expanduser().resolve()
rows=list(csv.DictReader((root/a.input).open(newline="", encoding="utf-8")))
if not rows:
    raise RuntimeError("Input CSV has no rows")
for r in rows:
    for c in ("emit_0_path","emit_90_path","emit_360_path"):
        v=str(r.get(c,"")).strip()
        if v and not Path(v).is_absolute():
            r[c]=str((root/v).resolve())
with (root/a.output).open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(root/a.output)
