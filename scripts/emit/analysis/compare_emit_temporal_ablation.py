#!/usr/bin/env python3
"""
Compare matched EMIT real-temporal vs t0-repeated MethaneFuse metrics.
"""

from pathlib import Path
import json

MF = Path.home() / "MethaneFuse"
REAL = MF / "results/eval/emit_temporal60_full.json"
BASE = MF / "results/eval/emit_temporal60_t0repeated_same38.json"

for p in (REAL, BASE):
    if not p.exists():
        raise SystemExit(f"Missing metrics JSON: {p}")

with REAL.open() as f:
    real = json.load(f)
with BASE.open() as f:
    base = json.load(f)

ro = real["overall"]
bo = base["overall"]

metrics = ["acc", "auroc", "recall", "fpr"]

print("=" * 78)
print("EMIT TEMPORAL ABLATION — SAME 38 SAMPLES")
print("=" * 78)
print(f"{'metric':12s} {'real temporal':>15s} {'t0 repeated':>15s} {'delta(real-base)':>18s}")
print("-" * 78)

for m in metrics:
    rv = float(ro[m])
    bv = float(bo[m])
    print(f"{m:12s} {rv:15.6f} {bv:15.6f} {rv-bv:18.6f}")

rloss = float(real.get("loss", float("nan")))
bloss = float(base.get("loss", float("nan")))
print(f"{'loss':12s} {rloss:15.6f} {bloss:15.6f} {rloss-bloss:18.6f}")

print()
print("Interpretation:")
d_auc = float(ro["auroc"]) - float(bo["auroc"])
if d_auc > 0.02:
    print(f"Real historical context improved AUROC by {d_auc:.3f}.")
elif d_auc < -0.02:
    print(f"Real historical context reduced AUROC by {abs(d_auc):.3f}.")
else:
    print(f"AUROC difference is small ({d_auc:+.3f}); temporal context had little effect.")

d_fpr = float(ro["fpr"]) - float(bo["fpr"])
if d_fpr < 0:
    print(f"Real temporal context reduced FPR by {abs(d_fpr):.3f}.")
elif d_fpr > 0:
    print(f"Real temporal context increased FPR by {d_fpr:.3f}.")
else:
    print("FPR was unchanged.")
