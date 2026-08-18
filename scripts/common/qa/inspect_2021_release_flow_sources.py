from pathlib import Path
import re

import numpy as np
import pandas as pd


SCENE_CSV = Path(
    "outputs/43_landsat_unique_clean_features.csv"
)

SEARCH_ROOT = Path(
    "raw_data"
)

SCENE_OUTPUT = Path(
    "outputs/53_2021_landsat_scenes_for_review.csv"
)

FILE_OUTPUT = Path(
    "outputs/54_2021_release_flow_file_inventory.csv"
)


TIME_KEYWORDS = (
    "time",
    "date",
    "datetime",
    "timestamp",
    "utc",
)

FLOW_KEYWORDS = (
    "flow",
    "scfh",
    "slpm",
    "kg",
    "release",
    "methane",
    "ch4",
)


def find_matching_columns(columns, keywords):
    result = []

    for column in columns:
        normalized = str(column).strip().lower()

        if any(
            keyword in normalized
            for keyword in keywords
        ):
            result.append(column)

    return result


def unique_examples(series, count=5):
    values = (
        series.dropna()
        .astype(str)
        .drop_duplicates()
        .head(count)
        .tolist()
    )

    return " | ".join(values)


def estimate_datetime_range(df, columns):
    results = []

    for column in columns:
        parsed = pd.to_datetime(
            df[column],
            errors="coerce",
            utc=True,
        )

        valid_count = int(parsed.notna().sum())

        if valid_count == 0:
            continue

        results.append({
            "column": column,
            "valid_count": valid_count,
            "minimum": parsed.min(),
            "maximum": parsed.max(),
        })

    return results


def estimate_numeric_range(df, columns):
    results = []

    for column in columns:
        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        valid = numeric.dropna()

        if len(valid) == 0:
            continue

        results.append({
            "column": column,
            "valid_count": len(valid),
            "minimum": float(valid.min()),
            "maximum": float(valid.max()),
            "mean": float(valid.mean()),
            "positive_count": int((valid > 0).sum()),
            "zero_count": int((valid == 0).sum()),
        })

    return results


def main():
    if not SCENE_CSV.exists():
        raise FileNotFoundError(
            f"Missing scene file: {SCENE_CSV}"
        )

    scenes = pd.read_csv(SCENE_CSV)

    if "landsat_image_time" not in scenes.columns:
        raise ValueError(
            "landsat_image_time column is missing."
        )

    scene_times = pd.to_datetime(
        scenes["landsat_image_time"],
        errors="coerce",
        utc=True,
    )

    scenes_2021 = scenes[
        scene_times.dt.year == 2021
    ].copy()

    scenes_2021["landsat_time_utc"] = (
        scene_times.loc[scenes_2021.index]
    )

    scene_columns = [
        column
        for column in [
            "raster_group_id",
            "landsat_image_time",
            "landsat_time_utc",
            "label",
            "event_id",
            "site_name",
            "landsat_sensor",
            "datetime_utc",
            "emission_tph_mean",
            "emission_tph_median",
            "emission_tph_max",
        ]
        if column in scenes_2021.columns
    ]

    scenes_2021[
        scene_columns
    ].to_csv(
        SCENE_OUTPUT,
        index=False,
    )

    print("=" * 100)
    print("2021 LANDSAT SCENES")
    print("=" * 100)

    print(
        f"\n2021 unique Landsat scenes: "
        f"{len(scenes_2021)}"
    )

    print(
        scenes_2021[
            scene_columns
        ].to_string(index=False)
    )

    candidate_files = []

    if SEARCH_ROOT.exists():
        for path in SEARCH_ROOT.rglob("*.csv"):
            lower_name = path.name.lower()
            lower_path = str(path).lower()

            likely_2021 = (
                re.search(
                    r"21(0[1-9]|1[0-2])[0-3][0-9]",
                    lower_name,
                )
                is not None
                or "2021" in lower_path
                or "releasedat" in lower_name
                or "satellitetestdata" in lower_path
            )

            if likely_2021:
                candidate_files.append(path)

    candidate_files = sorted(
        set(candidate_files)
    )

    print("\n" + "=" * 100)
    print("2021 RELEASE-FLOW FILE CANDIDATES")
    print("=" * 100)

    print(
        f"\nCandidate CSV files: "
        f"{len(candidate_files)}"
    )

    inventory_rows = []

    for file_number, path in enumerate(
        candidate_files,
        start=1,
    ):
        print("\n" + "-" * 100)
        print(
            f"[{file_number}/{len(candidate_files)}] "
            f"{path}"
        )

        try:
            df = pd.read_csv(
                path,
                nrows=10000,
                low_memory=False,
            )

        except Exception as error:
            print(f"READ ERROR: {error}")

            inventory_rows.append({
                "file_path": str(path),
                "status": "read_error",
                "error": str(error),
            })

            continue

        time_columns = find_matching_columns(
            df.columns,
            TIME_KEYWORDS,
        )

        flow_columns = find_matching_columns(
            df.columns,
            FLOW_KEYWORDS,
        )

        datetime_ranges = estimate_datetime_range(
            df,
            time_columns,
        )

        numeric_ranges = estimate_numeric_range(
            df,
            flow_columns,
        )

        print(f"Rows sampled: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print(f"Time-column candidates: {time_columns}")
        print(f"Flow-column candidates: {flow_columns}")

        print("\nFirst 5 rows:")
        print(df.head(5).to_string(index=False))

        print("\nParsed datetime ranges:")

        if datetime_ranges:
            for item in datetime_ranges:
                print(
                    f"  {item['column']}: "
                    f"valid={item['valid_count']}, "
                    f"min={item['minimum']}, "
                    f"max={item['maximum']}"
                )
        else:
            print("  None")

        print("\nNumeric flow ranges:")

        if numeric_ranges:
            for item in numeric_ranges:
                print(
                    f"  {item['column']}: "
                    f"valid={item['valid_count']}, "
                    f"min={item['minimum']}, "
                    f"max={item['maximum']}, "
                    f"mean={item['mean']}, "
                    f"positive={item['positive_count']}, "
                    f"zero={item['zero_count']}"
                )
        else:
            print("  None")

        inventory_row = {
            "file_path": str(path),
            "status": "success",
            "sampled_rows": len(df),
            "columns": " | ".join(
                map(str, df.columns)
            ),
            "time_columns": " | ".join(
                map(str, time_columns)
            ),
            "flow_columns": " | ".join(
                map(str, flow_columns)
            ),
        }

        for column in time_columns:
            inventory_row[
                f"time_examples__{column}"
            ] = unique_examples(df[column])

        for column in flow_columns:
            inventory_row[
                f"flow_examples__{column}"
            ] = unique_examples(df[column])

        inventory_rows.append(
            inventory_row
        )

    inventory_df = pd.DataFrame(
        inventory_rows
    )

    FILE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory_df.to_csv(
        FILE_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(f"\n2021 scenes saved: {SCENE_OUTPUT}")
    print(f"File inventory saved: {FILE_OUTPUT}")


if __name__ == "__main__":
    main()
