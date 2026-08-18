import pandas as pd
from pathlib import Path

base = Path("outputs")
csv_path = base / "15_methaneair_s2_landsat_availability.csv"

df = pd.read_csv(csv_path)

# 確保 count 欄位是數字
for col in ["s2_count_pm1day", "l8_count_pm1day", "l9_count_pm1day", "landsat_count_pm1day"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

total = len(df)
s2_available = (df["s2_count_pm1day"] > 0).sum()
landsat_available = (df["landsat_count_pm1day"] > 0).sum()
both_available = ((df["s2_count_pm1day"] > 0) & (df["landsat_count_pm1day"] > 0)).sum()

print("Total MethaneAIR events:", total)
print("Events with Sentinel-2:", s2_available)
print("Events with Landsat:", landsat_available)
print("Events with both S2 and Landsat:", both_available)

# 先挑 Sentinel-2 可用的 positive events
s2_candidates = df[df["s2_count_pm1day"] > 0].copy()

# 排放量大的先排前面，之後比較容易看到訊號
if "emission_kg_hr" in s2_candidates.columns:
    s2_candidates["emission_kg_hr"] = pd.to_numeric(
        s2_candidates["emission_kg_hr"], errors="coerce"
    )
    s2_candidates = s2_candidates.sort_values("emission_kg_hr", ascending=False)

out_path = base / "16_methaneair_s2_candidate_events.csv"
s2_candidates.to_csv(out_path, index=False)

print("Saved:", out_path)
print("S2 candidate events:", len(s2_candidates))
print(s2_candidates.head(10)[[
    "event_id", "datetime_utc", "lat", "lon",
    "emission_kg_hr", "s2_count_pm1day", "landsat_count_pm1day"
]])