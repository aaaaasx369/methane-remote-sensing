#!/bin/bash
#SBATCH --account=def-juliana2
#SBATCH --job-name=mf_s2_506prep
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --exclude=fc30640
#SBATCH --output=logs/mf_s2_506prep_%j.out
#SBATCH --error=logs/mf_s2_506prep_%j.err

set -euo pipefail

cd /project/6002520/yunjung1/MethaneFuse

module purge
module load python/3.11
module load scipy-stack/2025a

source /project/6002520/yunjung1/venvs/methanefuse/bin/activate

echo "===== START FULL STRICT506 PREPARATION ====="
date

python scripts/prepare_strict506_480m_stac.py

echo
echo "===== FINAL VALIDATION ====="

python - <<'PY'
from pathlib import Path
import pandas as pd
import rasterio
import numpy as np

f = Path(
    "data/custom/methaneair_full_strict506_480m_eval.csv"
)
audit_f = Path(
    "data/custom/methaneair_full_strict506_480m_audit.csv"
)

df = pd.read_csv(f)
audit = pd.read_csv(audit_f)

print("Eval rows:", len(df))
print("Unique IDs:", df["id"].nunique())
print("\nLabels:")
print(df["label"].value_counts().sort_index())

assert len(df) == 506
assert df["id"].nunique() == 506
assert (df["label"] == 0).sum() == 41
assert (df["label"] == 1).sum() == 465

print("\nAudit rows:", len(audit))
print("Audit status:")
print(audit["status"].value_counts(dropna=False))

assert len(audit) == 506 * 3
assert not (audit["status"] == "failed").any()

expected_bands = (
    "B01", "B02", "B03", "B04",
    "B05", "B06", "B07", "B08",
    "B8A", "B09", "B11", "B12",
)

bad = []

for i, row in df.iterrows():
    for col in [
        "s2_0_path",
        "s2_90_path",
        "s2_360_path",
    ]:
        p = Path(row[col])

        if not p.exists():
            bad.append((row["id"], col, "missing"))
            continue

        try:
            with rasterio.open(p) as src:
                a = src.read()

                ok = (
                    src.count == 12
                    and src.width == 48
                    and src.height == 48
                    and a.dtype == np.uint16
                    and tuple(src.descriptions) == expected_bands
                    and not np.all(a == 0)
                )

                if not ok:
                    bad.append(
                        (
                            row["id"],
                            col,
                            f"shape={a.shape}, "
                            f"dtype={a.dtype}, "
                            f"bands={src.descriptions}"
                        )
                    )

        except Exception as e:
            bad.append(
                (row["id"], col, repr(e))
            )

print("\nBad TIFFs:", len(bad))

if bad:
    for x in bad[:20]:
        print(x)

assert len(bad) == 0

print("\n===================================")
print("STRICT506 DATASET VALIDATED")
print("506 records")
print("1518 temporal TIFFs")
print("465 positive / 41 negative")
print("===================================")
PY

date
echo "===== PREPARATION COMPLETE ====="
