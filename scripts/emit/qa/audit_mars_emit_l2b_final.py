from pathlib import Path
import re
import pandas as pd
import rasterio


ROOT = Path(
    "/Volumes/engg-leung/dora lin/MARS_MultiSensor_Data/EMIT"
)

D = ROOT / "L2B_CH4ENH"
LOG_DIR = ROOT / "logs"

OUT_CSV = LOG_DIR / "emit_l2b_final_disk_audit.csv"
OUT_SUMMARY = LOG_DIR / "emit_l2b_final_summary.txt"

EXPECTED_GRANULES = 152

PATTERN = re.compile(
    r"^EMIT_L2B_(CH4ENH|CH4SENS|CH4UNCERT)_002_(.+)\.tif$"
)


# ============================================================
# INVENTORY
# ============================================================

files = sorted(D.glob("*.tif"))

records = []

for i, p in enumerate(files, 1):

    print(f"[{i}/{len(files)}] {p.name}")

    m = PATTERN.match(p.name)

    if not m:
        continue

    product = m.group(1)
    granule_key = m.group(2)

    rec = {
        "filename": p.name,
        "granule_key": granule_key,
        "product": product,
        "bytes": p.stat().st_size,
        "readable": False,
        "width": None,
        "height": None,
        "bands": None,
        "dtype": None,
        "crs": None,
        "error": None,
    }

    try:

        with rasterio.open(p) as src:

            rec["readable"] = True
            rec["width"] = src.width
            rec["height"] = src.height
            rec["bands"] = src.count
            rec["dtype"] = ",".join(src.dtypes)
            rec["crs"] = str(src.crs)

    except Exception as e:

        rec["error"] = repr(e)

    records.append(rec)


df = pd.DataFrame(records)

df.to_csv(
    OUT_CSV,
    index=False
)


# ============================================================
# PRODUCT COUNTS
# ============================================================

product_counts = (
    df["product"]
    .value_counts()
    .to_dict()
)

enh = int(product_counts.get("CH4ENH", 0))
sens = int(product_counts.get("CH4SENS", 0))
uncert = int(product_counts.get("CH4UNCERT", 0))

readable = int(df["readable"].sum())


# ============================================================
# TRIAD AUDIT
# ============================================================

triad = (
    df.groupby("granule_key")["product"]
    .agg(lambda x: set(x))
)

required = {
    "CH4ENH",
    "CH4SENS",
    "CH4UNCERT",
}

complete_keys = [
    key
    for key, products in triad.items()
    if required.issubset(products)
]

incomplete_keys = [
    key
    for key, products in triad.items()
    if not required.issubset(products)
]


# Require all three files to also be readable
readable_by_granule = {}

for key, group in df.groupby("granule_key"):

    products = set(group["product"])

    product_readable = {
        row["product"]: bool(row["readable"])
        for _, row in group.iterrows()
    }

    readable_by_granule[key] = (
        required.issubset(products)
        and all(
            product_readable.get(p, False)
            for p in required
        )
    )

complete_readable = sum(
    readable_by_granule.values()
)


# ============================================================
# STRUCTURE CHECKS
# ============================================================

one_band = int(
    (df["bands"] == 1).sum()
)

epsg4326 = int(
    (df["crs"] == "EPSG:4326").sum()
)

nonempty = int(
    (df["bytes"] > 1024).sum()
)


# ============================================================
# SUMMARY
# ============================================================

summary = f"""
==============================================================================
EMIT L2B CH4 V002 FINAL DISK QA
==============================================================================

EXPECTED
Unique granules:             {EXPECTED_GRANULES}
Expected TIFFs:              {EXPECTED_GRANULES * 3}

PRODUCT COUNTS
CH4ENH:                      {enh}/{EXPECTED_GRANULES}
CH4SENS:                     {sens}/{EXPECTED_GRANULES}
CH4UNCERT:                   {uncert}/{EXPECTED_GRANULES}
Total recognized TIFFs:      {len(df)}/{EXPECTED_GRANULES * 3}

INTEGRITY
Readable TIFFs:              {readable}/{len(df)}
Non-empty TIFFs:             {nonempty}/{len(df)}
1-band TIFFs:                {one_band}/{len(df)}
EPSG:4326 TIFFs:             {epsg4326}/{len(df)}

GRANULE TRIADS
Complete 3-product triads:   {len(complete_keys)}/{EXPECTED_GRANULES}
Readable complete triads:    {complete_readable}/{EXPECTED_GRANULES}
Incomplete triads:           {len(incomplete_keys)}

============================================================================== 
""".strip()

if (
    enh == EXPECTED_GRANULES
    and sens == EXPECTED_GRANULES
    and uncert == EXPECTED_GRANULES
    and len(df) == EXPECTED_GRANULES * 3
    and readable == EXPECTED_GRANULES * 3
    and complete_readable == EXPECTED_GRANULES
):

    summary += """

✅ ALL 152 EMIT L2B CH4 GRANULES COMPLETE
✅ 456/456 TIFFS READABLE
✅ 152/152 CH4ENH
✅ 152/152 CH4SENS
✅ 152/152 CH4UNCERT
"""

else:

    summary += """

⚠ EMIT L2B FINAL QA IS NOT COMPLETE
"""

    if incomplete_keys:

        summary += "\nIncomplete granules:\n"

        for key in incomplete_keys:

            summary += f"  {key}\n"


OUT_SUMMARY.write_text(
    summary,
    encoding="utf-8"
)

print()
print(summary)

print()
print("Audit CSV:")
print(OUT_CSV)

print()
print("Summary:")
print(OUT_SUMMARY)
