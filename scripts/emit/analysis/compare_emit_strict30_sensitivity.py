#!/usr/bin/env python3
"""
Compare MethaneFuse on the EMIT strict ±30-day sensitivity subset:
  real t0 / ~t-90 / ~t-180
vs
  t0 / t0 / t0
using exactly the same rows.
"""

from pathlib import Path
import json

ROOT = Path.home() / "MethaneFuse" / "results" / "eval"

REAL = ROOT / "emit_temporal_strict30_real.json"
REP = ROOT / "emit_temporal_strict30_t0repeated.json"

for p in [REAL, REP]:
    if not p.exists():
        raise SystemExit(f"Missing result JSON: {p}")

with REAL.open() as f:
    real = json.load(f)
with REP.open() as f:
    rep = json.load(f)

ro = real["overall"]
bo = rep["overall"]

print("=" * 82)
print("EMIT STRICT ±30-DAY TEMPORAL SENSITIVITY — MATCHED ROWS")
print("=" * 82)
print("Count real temporal :", real.get("count"))
print("Count t0 repeated   :", rep.get("count"))
print()
print(
    f'{"metric":12s}'
    f'{"real temporal":>17s}'
    f'{"t0 repeated":>17s}'
    f'{"delta(real-base)":>20s}'
)
print("-" * 82)

for m in ["acc", "auroc", "recall", "fpr"]:
    rv = float(ro[m])
    bv = float(bo[m])
    print(f"{m:12s}{rv:17.6f}{bv:17.6f}{rv-bv:20.6f}")

rloss = float(real.get("loss", float("nan")))
bloss = float(rep.get("loss", float("nan")))
print(f'{"loss":12s}{rloss:17.6f}{bloss:17.6f}{rloss-bloss:20.6f}')

dauc = float(ro["auroc"]) - float(bo["auroc"])
dfpr = float(ro["fpr"]) - float(bo["fpr"])

print()
print("Interpretation:")
if dauc > 0.02:
    print(f"- Real temporal context improved AUROC by {dauc:.3f}.")
elif dauc < -0.02:
    print(f"- Real temporal context reduced AUROC by {abs(dauc):.3f}.")
else:
    print(f"- AUROC difference was small ({dauc:+.3f}).")

if dfpr < 0:
    print(f"- Real temporal context reduced FPR by {abs(dfpr):.3f}.")
elif dfpr > 0:
    print(f"- Real temporal context increased FPR by {dfpr:.3f}.")
else:
    print("- FPR was unchanged.")

print()
print(
    "Use this together with the expanded60 matched ablation to determine whether "
    "the expanded temporal-date tolerance explains the AUROC drop."
)
