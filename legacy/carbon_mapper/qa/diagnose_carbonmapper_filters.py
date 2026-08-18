from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/200_carbonmapper_plume_catalog_raw.csv"
)

HIGH_EMISSION_OUTPUT = Path(
    "outputs/203_carbonmapper_high_emission_filter_audit.csv"
)


def text_column(dataframe, column):
    if column not in dataframe.columns:
        return pd.Series(
            "",
            index=dataframe.index,
            dtype=str,
        )

    return (
        dataframe[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


def numeric_column(dataframe, column):
    if column not in dataframe.columns:
        return pd.Series(
            np.nan,
            index=dataframe.index,
            dtype=float,
        )

    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    dataframe = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    print("=" * 100)
    print("CARBON MAPPER FILTER DIAGNOSTIC")
    print("=" * 100)

    print("\nRows:", len(dataframe))
    print("\nColumns:")
    print(dataframe.columns.tolist())

    gas = (
        text_column(dataframe, "gas")
        .str.upper()
    )

    quality = (
        text_column(
            dataframe,
            "plume_quality",
        )
        .str.lower()
    )

    emission = numeric_column(
        dataframe,
        "emission_auto",
    )

    latitude = numeric_column(
        dataframe,
        "plume_latitude",
    )

    longitude = numeric_column(
        dataframe,
        "plume_longitude",
    )

    scene_time = pd.to_datetime(
        dataframe.get(
            "scene_datetime_utc",
            dataframe.get(
                "scene_timestamp",
                pd.Series(
                    pd.NaT,
                    index=dataframe.index,
                ),
            ),
        ),
        errors="coerce",
        utc=True,
    )

    product_columns = [
        column
        for column in [
            "plume_tif",
            "con_tif",
            "plume_png",
            "plume_rgb_png",
            "rgb_png",
        ]
        if column in dataframe.columns
    ]

    if product_columns:
        has_product = (
            dataframe[product_columns]
            .notna()
            .any(axis=1)
        )
    else:
        has_product = pd.Series(
            False,
            index=dataframe.index,
        )

    gas_ch4 = gas.isin([
        "CH4",
        "METHANE",
    ])

    quality_good = quality.eq("good")

    emission_available = (
        emission.notna()
    )

    emission_ge_1000 = (
        emission >= 1000
    )

    has_coordinates = (
        latitude.notna()
        & longitude.notna()
    )

    has_time = scene_time.notna()

    masks = {
        "all_rows":
            pd.Series(
                True,
                index=dataframe.index,
            ),
        "gas_is_ch4":
            gas_ch4,
        "emission_available":
            emission_available,
        "emission_ge_1000":
            emission_ge_1000,
        "quality_is_good":
            quality_good,
        "has_coordinates":
            has_coordinates,
        "has_time":
            has_time,
        "has_plume_product":
            has_product,
        "ch4_and_ge_1000":
            gas_ch4
            & emission_ge_1000,
        "ch4_ge_1000_good":
            gas_ch4
            & emission_ge_1000
            & quality_good,
        "ch4_ge_1000_good_coords_time":
            gas_ch4
            & emission_ge_1000
            & quality_good
            & has_coordinates
            & has_time,
        "all_original_conditions":
            gas_ch4
            & emission_ge_1000
            & quality_good
            & has_coordinates
            & has_time
            & has_product,
    }

    print("\n" + "=" * 100)
    print("PASS COUNTS")
    print("=" * 100)

    for name, mask in masks.items():
        print(
            f"{name:35s}: "
            f"{int(mask.sum())}"
        )

    print("\n" + "=" * 100)
    print("GAS VALUES")
    print("=" * 100)

    print(
        gas.replace("", np.nan)
        .value_counts(
            dropna=False
        )
        .head(20)
        .to_string()
    )

    print("\n" + "=" * 100)
    print("PLUME QUALITY VALUES")
    print("=" * 100)

    print(
        quality.replace("", np.nan)
        .value_counts(
            dropna=False
        )
        .head(30)
        .to_string()
    )

    print("\n" + "=" * 100)
    print("EMISSION AUTO")
    print("=" * 100)

    print(
        emission.describe(
            percentiles=[
                0.5,
                0.75,
                0.9,
                0.95,
                0.99,
            ]
        ).to_string()
    )

    print(
        "\nEmission >= 1000:",
        int(emission_ge_1000.sum()),
    )

    print(
        "Maximum emission:",
        emission.max(),
    )

    print("\nProduct non-null counts:")

    for column in product_columns:
        print(
            f"{column:20s}: "
            f"{int(dataframe[column].notna().sum())}"
        )

    audit = dataframe.copy()

    audit[
        "audit_gas_ch4"
    ] = gas_ch4

    audit[
        "audit_quality_good"
    ] = quality_good

    audit[
        "audit_emission_ge_1000"
    ] = emission_ge_1000

    audit[
        "audit_has_coordinates"
    ] = has_coordinates

    audit[
        "audit_has_time"
    ] = has_time

    audit[
        "audit_has_product"
    ] = has_product

    audit = audit[
        emission_ge_1000
    ].copy()

    audit = audit.sort_values(
        "emission_auto",
        ascending=False,
    )

    audit.to_csv(
        HIGH_EMISSION_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 100)
    print("TOP HIGH-EMISSION RECORDS")
    print("=" * 100)

    display_columns = [
        "plume_id",
        "gas",
        "scene_timestamp",
        "instrument",
        "platform",
        "emission_auto",
        "plume_quality",
        "plume_latitude",
        "plume_longitude",
        "plume_tif",
        "plume_png",
        "audit_gas_ch4",
        "audit_quality_good",
        "audit_has_coordinates",
        "audit_has_time",
        "audit_has_product",
    ]

    available = [
        column
        for column in display_columns
        if column in audit.columns
    ]

    print(
        audit[available]
        .head(20)
        .to_string(
            index=False,
            max_colwidth=40,
        )
    )

    print("\nSaved:")
    print(HIGH_EMISSION_OUTPUT)


if __name__ == "__main__":
    main()
