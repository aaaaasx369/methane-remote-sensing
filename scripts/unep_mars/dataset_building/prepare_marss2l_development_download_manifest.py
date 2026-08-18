from pathlib import Path

import pandas as pd


INPUT = Path(
    "outputs/254_marss2l_development_download_manifest.csv"
)

OUTPUT = Path(
    "outputs/256_marss2l_development_download_manifest_compatible.csv"
)


EXPECTED_ROWS = 864
EXPECTED_SITES = 65


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    required = [
        "download_id",
        "site_key",
        "development_split",
        "development_role",
        "evaluation_label",
        "lon",
        "lat",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing required columns: {missing}"
        )

    # 相容原本的下載程式。
    df["external_role"] = (
        df["development_role"]
        .astype(str)
        .str.strip()
    )

    if "landsat_tile" not in df.columns:
        if "tile" not in df.columns:
            raise KeyError(
                "Neither landsat_tile nor tile exists."
            )

        df["landsat_tile"] = (
            df["tile"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    if (
        "acquisition_datetime_utc"
        not in df.columns
    ):
        if "tile_date" not in df.columns:
            raise KeyError(
                "Neither acquisition_datetime_utc "
                "nor tile_date exists."
            )

        df["acquisition_datetime_utc"] = (
            pd.to_datetime(
                df["tile_date"],
                errors="coerce",
                utc=True,
            )
        )
    else:
        df["acquisition_datetime_utc"] = (
            pd.to_datetime(
                df["acquisition_datetime_utc"],
                errors="coerce",
                utc=True,
            )
        )

    df["lon"] = pd.to_numeric(
        df["lon"],
        errors="coerce",
    )

    df["lat"] = pd.to_numeric(
        df["lat"],
        errors="coerce",
    )

    missing_location = df[
        df[
            [
                "lon",
                "lat",
                "acquisition_datetime_utc",
            ]
        ].isna().any(axis=1)
    ]

    if not missing_location.empty:
        raise RuntimeError(
            "Rows contain missing location/time:\n"
            + missing_location[
                [
                    "download_id",
                    "site_key",
                    "lon",
                    "lat",
                    "acquisition_datetime_utc",
                ]
            ].head(20).to_string(index=False)
        )

    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} rows, "
            f"found {len(df)}."
        )

    if df["site_key"].nunique() != EXPECTED_SITES:
        raise RuntimeError(
            f"Expected {EXPECTED_SITES} sites, "
            f"found {df['site_key'].nunique()}."
        )

    if df["download_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate download_id found."
        )

    role_counts = (
        df["development_role"]
        .value_counts()
    )

    expected_roles = {
        "calibration_negative": 325,
        "model_negative": 325,
        "model_positive": 214,
    }

    for role, expected in expected_roles.items():
        actual = int(
            role_counts.get(role, 0)
        )

        if actual != expected:
            raise RuntimeError(
                f"{role}: expected {expected}, "
                f"found {actual}."
            )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 105)
    print("MARS-S2L DEVELOPMENT DOWNLOAD MANIFEST")
    print("=" * 105)

    print("\nRows:", len(df))
    print("Sites:", df["site_key"].nunique())

    print("\nSites by split:")
    print(
        df.groupby(
            "development_split"
        )["site_key"].nunique()
    )

    print("\nRows by split and role:")
    print(
        pd.crosstab(
            df["development_split"],
            df["development_role"],
            margins=True,
        )
    )

    print("\nRows by sensor:")
    print(
        df["satellite_normalized"]
        .value_counts()
    )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
