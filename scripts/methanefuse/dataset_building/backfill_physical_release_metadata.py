#!/usr/bin/env python3

from pathlib import Path
import re

import pandas as pd
import rasterio
from rasterio.warp import transform


PROJECT = Path("/project/6002520/yunjung1/MethaneFuse")
DATA_ROOT = PROJECT / "data/methaneair_full"

GT_PATH = DATA_ROOT / "ground_truth_confirmed_audited.csv"

MASTER_CANDIDATES = [
    PROJECT / "outputs/027_unified_methane_master_landsat_recovered.csv",
    PROJECT / "outputs/012_unified_methane_master_fixed.csv",
    PROJECT / "outputs/002_unified_methane_master_dedup.csv",
]


def find_master() -> Path:
    for path in MASTER_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("No unified master table found.")


def missing(value) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip() in {"", "nan", "None", "<NA>", "NaT"}


def split_path_values(value) -> list[str]:
    if missing(value):
        return []

    text = str(value)
    pieces = re.split(r"[|;]", text)

    return [
        piece.strip()
        for piece in pieces
        if piece.strip()
    ]


def resolve_existing_path(value: str) -> Path | None:
    path = Path(value)

    candidates = [
        path,
        PROJECT / path,
        PROJECT.parent / path,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    return None


def candidate_paths_from_row(row: pd.Series) -> list[Path]:
    paths = []

    path_columns = [
        column
        for column in row.index
        if any(
            keyword in column.lower()
            for keyword in [
                "path",
                "tif",
                "image",
                "file",
                "t0",
                "l89",
            ]
        )
    ]

    for column in path_columns:
        for value in split_path_values(row[column]):
            path = resolve_existing_path(value)
            if path is not None:
                paths.append(path)

    # Deduplicate while preserving order.
    unique = []
    seen = set()

    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)

    return unique


def search_tiff_by_source_id(source_id: str) -> list[Path]:
    if not source_id:
        return []

    matches = []

    for root in [
        PROJECT / "data",
        PROJECT / "results",
        PROJECT.parent / "data",
    ]:
        if not root.exists():
            continue

        for path in root.rglob(f"*{source_id}*.tif"):
            if path.is_file():
                matches.append(path.resolve())

    return matches


def raster_center_lon_lat(path: Path):
    try:
        with rasterio.open(path) as src:
            center_x = (src.bounds.left + src.bounds.right) / 2
            center_y = (src.bounds.bottom + src.bounds.top) / 2

            if src.crs is None:
                return None, None

            lon, lat = transform(
                src.crs,
                "EPSG:4326",
                [center_x],
                [center_y],
            )

            return float(lon[0]), float(lat[0])

    except Exception:
        return None, None


def parse_date_from_text(value):
    if missing(value):
        return None

    text = str(value)

    patterns = [
        # Landsat product IDs
        r"L[CEOT]0[89]_L2S[A-Z]_\d{6}_(20\d{6})_",

        # YYYY-MM-DD or YYYY_MM_DD
        r"(?<!\d)(20\d{2})[-_](\d{2})[-_](\d{2})(?!\d)",

        # YYYYMMDD
        r"(?<!\d)(20\d{6})(?!\d)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)

        if not match:
            continue

        raw = "".join(match.groups())

        parsed = pd.to_datetime(
            raw,
            format="%Y%m%d",
            errors="coerce",
            utc=True,
        )

        if pd.notna(parsed):
            return parsed

    return None


def date_from_raster(path: Path):
    parsed = parse_date_from_text(path)
    if parsed is not None:
        return parsed

    try:
        with rasterio.open(path) as src:
            tags = src.tags()

        priority_keys = [
            "DATE_ACQUIRED",
            "ACQUISITION_DATE",
            "SENSING_TIME",
            "DATATAKE_SENSING_START",
            "TIFFTAG_DATETIME",
            "acquisition_time",
            "scene_timestamp",
        ]

        for key in priority_keys:
            if key not in tags:
                continue

            parsed = pd.to_datetime(
                tags[key],
                errors="coerce",
                utc=True,
            )

            if pd.notna(parsed):
                return parsed

        for value in tags.values():
            parsed = parse_date_from_text(value)

            if parsed is not None:
                return parsed

    except Exception:
        pass

    return None


def main():
    gt = pd.read_csv(GT_PATH)
    master_path = find_master()
    master = pd.read_csv(master_path)

    if "master_id" not in master.columns:
        raise KeyError("master_id is missing from unified master.")

    master_lookup = master.set_index("master_id", drop=False)

    audit_rows = []

    for index, row in gt.iterrows():
        has_coordinates = (
            pd.notna(pd.to_numeric(row.get("latitude"), errors="coerce"))
            and pd.notna(pd.to_numeric(row.get("longitude"), errors="coerce"))
        )

        has_time = pd.notna(
            pd.to_datetime(
                row.get("acquisition_time_utc"),
                errors="coerce",
                utc=True,
            )
        )

        if has_coordinates and has_time:
            continue

        record_id = str(row.get("record_id", "")).strip()

        audit = {
            "record_id": record_id,
            "site_id": row.get("site_id"),
            "original_latitude": row.get("latitude"),
            "original_longitude": row.get("longitude"),
            "original_time": row.get("acquisition_time_utc"),
            "recovered_latitude": pd.NA,
            "recovered_longitude": pd.NA,
            "recovered_time": pd.NA,
            "raster_path_used": pd.NA,
            "coordinate_recovered": False,
            "time_recovered": False,
        }

        paths = []

        if record_id in master_lookup.index:
            master_row = master_lookup.loc[record_id]

            if isinstance(master_row, pd.DataFrame):
                master_row = master_row.iloc[0]

            paths.extend(candidate_paths_from_row(master_row))

            # Parse date directly from master values.
            if not has_time:
                for value in master_row.values:
                    parsed = parse_date_from_text(value)

                    if parsed is not None:
                        gt.at[index, "acquisition_time_utc"] = (
                            parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
                        )
                        audit["recovered_time"] = (
                            parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
                        )
                        audit["time_recovered"] = True
                        has_time = True
                        break

            source_id = str(
                master_row.get("source_row_ids", "")
            ).strip()

            paths.extend(search_tiff_by_source_id(source_id))

        # Deduplicate paths.
        unique_paths = []
        seen = set()

        for path in paths:
            key = str(path)

            if key not in seen:
                seen.add(key)
                unique_paths.append(path)

        for path in unique_paths:
            if not has_coordinates:
                lon, lat = raster_center_lon_lat(path)

                if lon is not None and lat is not None:
                    gt.at[index, "longitude"] = lon
                    gt.at[index, "latitude"] = lat

                    audit["recovered_longitude"] = lon
                    audit["recovered_latitude"] = lat
                    audit["coordinate_recovered"] = True
                    audit["raster_path_used"] = str(path)

                    has_coordinates = True

            if not has_time:
                parsed = date_from_raster(path)

                if parsed is not None:
                    timestamp = parsed.strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    )

                    gt.at[index, "acquisition_time_utc"] = timestamp
                    audit["recovered_time"] = timestamp
                    audit["time_recovered"] = True
                    audit["raster_path_used"] = str(path)

                    has_time = True

            if has_coordinates and has_time:
                break

        audit_rows.append(audit)

    gt["latitude"] = pd.to_numeric(
        gt["latitude"],
        errors="coerce",
    )
    gt["longitude"] = pd.to_numeric(
        gt["longitude"],
        errors="coerce",
    )

    parsed_time = pd.to_datetime(
        gt["acquisition_time_utc"],
        errors="coerce",
        utc=True,
    )

    gt["has_coordinates"] = (
        gt["latitude"].notna()
        & gt["longitude"].notna()
    )
    gt["has_time"] = parsed_time.notna()
    gt["search_ready"] = (
        gt["has_coordinates"]
        & gt["has_time"]
    )

    output_all = (
        DATA_ROOT / "ground_truth_confirmed_backfilled.csv"
    )
    output_ready = (
        DATA_ROOT / "ground_truth_confirmed_search_ready_v2.csv"
    )
    output_audit = (
        DATA_ROOT / "ground_truth_backfill_audit.csv"
    )

    gt.to_csv(output_all, index=False)

    gt.loc[gt["search_ready"]].to_csv(
        output_ready,
        index=False,
    )

    pd.DataFrame(audit_rows).to_csv(
        output_audit,
        index=False,
    )

    print("Master used:", master_path)
    print("Confirmed rows:", len(gt))
    print("Search-ready before: 713")
    print("Search-ready after:", int(gt["search_ready"].sum()))

    print("\nSearch-ready labels:")
    print(
        gt.loc[gt["search_ready"], "label"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    print("\nStill not search-ready:")
    columns = [
        "record_id",
        "site_id",
        "label",
        "latitude",
        "longitude",
        "acquisition_time_utc",
        "has_coordinates",
        "has_time",
    ]

    print(
        gt.loc[~gt["search_ready"], columns]
        .to_string(index=False)
    )

    print("\nSaved:")
    print(output_all)
    print(output_ready)
    print(output_audit)


if __name__ == "__main__":
    main()
