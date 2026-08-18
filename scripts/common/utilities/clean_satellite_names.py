import pandas as pd
from pathlib import Path

INPUT = Path("outputs/07_unique_overpass_events.csv")
OUTPUT = Path("outputs/09_final_unique_overpass_events.csv")

df = pd.read_csv(INPUT)

def clean_satellite_name(x):
    s = str(x).strip()

    mapping = {
        "WorldView 3": "WorldView-3",
        "WV3": "WorldView-3",
        "LandSat": "Landsat",
        "Landsat 8": "Landsat",
        "Landsat 9": "Landsat",
        "GHGSat CX": "GHGSat",
        "GHGSat C2": "GHGSat",
        "EnMap": "EnMAP",
        "GF5": "Gaofen-5",
        "ZY1": "Ziyuan-1",
        "HJ2B": "Huanjing-2",
    }

    return mapping.get(s, s)

df["satellite_clean"] = df["satellite_from_paper"].apply(clean_satellite_name)

df.to_csv(OUTPUT, index=False)

print(f"Saved: {OUTPUT}")
print("\nCounts by cleaned satellite:")
print(df["satellite_clean"].value_counts())

print("\nCounts by paper:")
print(df["paper"].value_counts())