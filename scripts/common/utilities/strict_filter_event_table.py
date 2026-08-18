import pandas as pd
from pathlib import Path

INPUT = Path("outputs/05_clean_event_table_for_gee.csv")
OUTPUT = Path("outputs/06_strict_event_table_for_gee.csv")
REPORT = Path("outputs/06_source_file_report.csv")

df = pd.read_csv(INPUT, low_memory=False)

df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce", utc=True)
df = df.dropna(subset=["datetime_utc", "lat", "lon"])

# 先做 source file 診斷表，看看哪些檔案產生最多列
report = (
    df.groupby("source_file")
    .agg(
        rows=("source_file", "size"),
        n_unique_time=("datetime_utc", "nunique"),
        n_unique_satellite=("satellite", lambda x: x.astype(str).replace("nan", "").nunique()),
        nonempty_satellite=("satellite", lambda x: (x.astype(str).str.strip() != "").sum()),
        min_time=("datetime_utc", "min"),
        max_time=("datetime_utc", "max"),
    )
    .reset_index()
    .sort_values("rows", ascending=False)
)

report.to_csv(REPORT, index=False)
print(f"Saved source-file report to: {REPORT}")

# ------------------------------------------------------------
# 嚴格條件 1：排除明顯不是 satellite event table 的檔案
# ------------------------------------------------------------
bad_keywords = [
    "Survey Summary",
    "Survey_Summary",
    "survey summary",
    "Raw submissions",
    "raw submissions",
    "raw_submissions",
    "OLD",
    "old",
]

for kw in bad_keywords:
    df = df[~df["source_file"].astype(str).str.contains(kw, case=False, na=False)]

# ------------------------------------------------------------
# 嚴格條件 2：只留看起來像 satellite / matched table 的檔案
# ------------------------------------------------------------
good_keywords = [
    "matched",
    "satellite",
    "Satellite",
    "estimates",
    "Estimates",
    "estimate",
    "Estimate",
    "stage",
    "Stage",
]

good_mask = False
for kw in good_keywords:
    good_mask = good_mask | df["source_file"].astype(str).str.contains(kw, case=False, na=False)

df = df[good_mask].copy()

# ------------------------------------------------------------
# 嚴格條件 3：排除只有日期、沒有時間的列
# satellite overpass 通常不會剛好 00:00:00
# ------------------------------------------------------------
df = df[df["datetime_utc"].dt.strftime("%H:%M:%S") != "00:00:00"].copy()

# ------------------------------------------------------------
# 嚴格條件 4：emission_tph 轉數字
# ------------------------------------------------------------
df["emission_tph"] = pd.to_numeric(df["emission_tph"], errors="coerce")

# ------------------------------------------------------------
# 嚴格條件 5：satellite 欄位清理
# ------------------------------------------------------------
if "satellite" not in df.columns:
    df["satellite"] = ""

df["satellite"] = df["satellite"].astype(str).str.strip()
df.loc[df["satellite"].isin(["nan", "None", "NaN"]), "satellite"] = ""

# 如果 satellite 欄位是空的，嘗試從 source_file 猜
def infer_satellite(row):
    sat = str(row.get("satellite", "")).strip()
    if sat:
        return sat

    s = str(row.get("source_file", "")).lower()

    if "sentinel" in s or "s2" in s:
        return "Sentinel-2"
    if "landsat" in s or "ls8" in s or "ls9" in s:
        return "Landsat"
    if "prisma" in s:
        return "PRISMA"
    if "ghgsat" in s or "gsc" in s:
        return "GHGSat"
    if "worldview" in s or "wv3" in s:
        return "WorldView-3"
    if "enmap" in s:
        return "EnMAP"
    if "gaofen" in s or "gf5" in s:
        return "Gaofen-5"
    if "ziyuan" in s or "zy1" in s:
        return "Ziyuan-1"
    if "huanjing" in s or "hj2" in s:
        return "Huanjing-2"

    return ""

df["satellite"] = df.apply(infer_satellite, axis=1)

# ------------------------------------------------------------
# 嚴格條件 6：只留必要欄位
# ------------------------------------------------------------
keep_cols = [
    "paper",
    "site_name",
    "datetime_utc",
    "lat",
    "lon",
    "satellite",
    "team",
    "true_release",
    "emission_tph",
    "wind_speed",
    "wind_direction",
    "source_file",
    "source_sheet",
]

for c in keep_cols:
    if c not in df.columns:
        df[c] = ""

df = df[keep_cols].copy()

# ------------------------------------------------------------
# 嚴格條件 7：合併重複列
# 同一 paper + time + satellite + emission 只留一列
# ------------------------------------------------------------
df["datetime_utc_str"] = df["datetime_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

dedup_cols = [
    "paper",
    "datetime_utc_str",
    "lat",
    "lon",
    "satellite",
    "team",
    "emission_tph",
]

df = df.drop_duplicates(subset=dedup_cols).copy()

# ------------------------------------------------------------
# 嚴格條件 8：再次只留兩篇的正式測試期間
# ------------------------------------------------------------
mask_2023 = (
    df["paper"].astype(str).str.contains("2023", na=False)
    & (df["datetime_utc"] >= "2021-10-16")
    & (df["datetime_utc"] < "2021-11-04")
)

mask_2024 = (
    df["paper"].astype(str).str.contains("2024", na=False)
    & (df["datetime_utc"] >= "2022-10-10")
    & (df["datetime_utc"] < "2022-12-01")
)

df = df[mask_2023 | mask_2024].copy()

df = df.sort_values(["paper", "datetime_utc", "satellite", "team"])
df["datetime_utc"] = df["datetime_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
df = df.drop(columns=["datetime_utc_str"], errors="ignore")

df.to_csv(OUTPUT, index=False)

print(f"Saved strict event table to: {OUTPUT}")
print(f"Rows after strict filtering: {len(df)}")

print("\nPreview:")
preview_cols = ["paper", "datetime_utc", "lat", "lon", "satellite", "true_release", "emission_tph", "source_file"]
print(df[preview_cols].head(50).to_string(index=False))

print("\nSource-file report top 20:")
print(report.head(20).to_string(index=False))