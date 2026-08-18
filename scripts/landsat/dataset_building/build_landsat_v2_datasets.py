from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import rasterio

from extract_landsat_patch_features import extract_features


CORE_BASE = Path(
    "outputs/79_landsat_core_schedule_confirmed_features.csv"
)

EXTENDED_BASE = Path(
    "outputs/80_landsat_extended_schedule_confirmed_features.csv"
)

TARGETED_CASA_GRANDE = Path(
    "outputs/82_landsat_targeted_reviewed_features.csv"
)

EHRENBERG_FEATURES = Path(
    "outputs/93_ehrenberg_priority_landsat_features.csv"
)

CORE_OUTPUT = Path(
    "outputs/95_landsat_strict_core_v2_features.csv"
)

EXTENDED_OUTPUT = Path(
    "outputs/96_landsat_extended_v2_features.csv"
)

MERGE_AUDIT_OUTPUT = Path(
    "outputs/97_landsat_v2_merge_audit.csv"
)

SUMMARY_OUTPUT = Path(
    "outputs/98_landsat_v2_dataset_summary.csv"
)


PATH_COLUMNS = [
    "resolved_patch_path",
    "file_path",
    "patch_path",
    "tif_path",
    "filepath",
    "image_path",
    "output_path",
    "filename",
    "patch_filename",
]


def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
    }:
        return ""

    return text


def first_value(row, columns):
    for column in columns:
        if column not in row.index:
            continue

        value = row[column]

        if pd.notna(value):
            text = clean_text(value)

            if text:
                return value

    return np.nan


def resolve_patch_path(row):
    attempted = []

    for column in PATH_COLUMNS:
        if column not in row.index:
            continue

        value = clean_text(row[column])

        if not value:
            continue

        attempted.append(value)

        candidate = Path(value).expanduser()

        if candidate.exists():
            return candidate.resolve()

        candidate_from_project = (
            Path.cwd() / candidate
        )

        if candidate_from_project.exists():
            return candidate_from_project.resolve()

        filename = candidate.name

        matches = list(
            Path("sample_patches").rglob(
                filename
            )
        )

        if len(matches) == 1:
            return matches[0].resolve()

    raise FileNotFoundError(
        "Could not resolve TIFF path. "
        f"Attempted: {attempted}"
    )


def calculate_pixel_hash(path):
    with rasterio.open(path) as dataset:
        array = np.ascontiguousarray(
            dataset.read()
        )

    digest = hashlib.sha256()

    digest.update(
        str(array.shape).encode("utf-8")
    )

    digest.update(
        str(array.dtype).encode("utf-8")
    )

    digest.update(
        array.tobytes()
    )

    return digest.hexdigest()


def extract_label(row):
    value = first_value(
        row,
        [
            "final_label",
            "final_scene_label",
            "label",
        ],
    )

    numeric = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(numeric):
        raise ValueError(
            "Could not determine scene label."
        )

    label = int(numeric)

    if label not in {0, 1}:
        raise ValueError(
            f"Invalid label: {label}"
        )

    return label


def infer_site_key(row):
    possible_text = " ".join(
        clean_text(
            row.get(column)
        ).lower()
        for column in [
            "site_key",
            "site_name",
            "site",
            "location_name",
            "event_id",
            "source_file",
        ]
        if column in row.index
    )

    if "ehrenberg" in possible_text:
        return "ehrenberg"

    if (
        "casa" in possible_text
        or "grande" in possible_text
    ):
        return "casa_grande"

    lat = pd.to_numeric(
        pd.Series([
            row.get("lat", np.nan)
        ]),
        errors="coerce",
    ).iloc[0]

    lon = pd.to_numeric(
        pd.Series([
            row.get("lon", np.nan)
        ]),
        errors="coerce",
    ).iloc[0]

    if pd.notna(lat) and pd.notna(lon):
        if (
            abs(lat - 33.630645) < 0.1
            and abs(lon + 114.489150) < 0.1
        ):
            return "ehrenberg"

        if (
            abs(lat - 32.821821) < 0.1
            and abs(lon + 111.785773) < 0.1
        ):
            return "casa_grande"

    return "unknown"


def infer_sensor(row):
    existing = first_value(
        row,
        [
            "landsat_sensor",
            "sensor",
            "SPACECRAFT_ID",
            "gee_SPACECRAFT_ID",
        ],
    )

    text = clean_text(existing)

    if text:
        if "8" in text:
            return "Landsat-8"

        if "9" in text:
            return "Landsat-9"

    product_id = clean_text(
        first_value(
            row,
            [
                "LANDSAT_PRODUCT_ID",
                "landsat_product_id",
                "gee_LANDSAT_PRODUCT_ID",
            ],
        )
    )

    if product_id.startswith("LC08"):
        return "Landsat-8"

    if product_id.startswith("LC09"):
        return "Landsat-9"

    return "Unknown"


def infer_product_id(row):
    return clean_text(
        first_value(
            row,
            [
                "LANDSAT_PRODUCT_ID",
                "landsat_product_id",
                "gee_LANDSAT_PRODUCT_ID",
            ],
        )
    )


def infer_acquisition_time(row):
    value = first_value(
        row,
        [
            "landsat_image_time_utc",
            "landsat_image_time",
            "candidate_time_utc",
            "acquisition_time_utc",
            "acquisition_time",
        ],
    )

    parsed = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(parsed):
        return pd.NaT

    return parsed


def infer_scene_key(row, pixel_hash):
    product_id = infer_product_id(row)

    if product_id:
        return product_id

    raster_group = clean_text(
        first_value(
            row,
            [
                "raster_group_id",
                "scene_group_id",
            ],
        )
    )

    if raster_group:
        return raster_group

    overpass_id = clean_text(
        first_value(
            row,
            [
                "overpass_id",
                "event_id",
            ],
        )
    )

    if overpass_id:
        return overpass_id

    return pixel_hash


def load_table(path, source_name):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input table: {path}"
        )

    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )

    dataframe["source_dataset"] = (
        source_name
    )

    dataframe["source_row_index"] = (
        np.arange(len(dataframe))
    )

    return dataframe


def prepare_table(
    dataframe,
    feature_names,
):
    prepared_rows = []
    audit_rows = []

    missing_feature_columns = [
        column
        for column in feature_names
        if column not in dataframe.columns
    ]

    if missing_feature_columns:
        raise KeyError(
            "The table is missing image-feature "
            f"columns: {missing_feature_columns}"
        )

    for _, row in dataframe.iterrows():
        source_name = row[
            "source_dataset"
        ]

        source_row_index = int(
            row["source_row_index"]
        )

        try:
            patch_path = resolve_patch_path(
                row
            )

            pixel_hash = calculate_pixel_hash(
                patch_path
            )

            output_row = row.to_dict()

            output_row.update({
                "label":
                    extract_label(row),
                "site_key_normalized":
                    infer_site_key(row),
                "landsat_sensor":
                    infer_sensor(row),
                "landsat_product_id_normalized":
                    infer_product_id(row),
                "acquisition_time_utc":
                    infer_acquisition_time(row),
                "resolved_patch_path":
                    str(patch_path),
                "canonical_pixel_hash":
                    pixel_hash,
                "scene_key":
                    infer_scene_key(
                        row,
                        pixel_hash,
                    ),
            })

            prepared_rows.append(
                output_row
            )

            audit_rows.append({
                "source_dataset":
                    source_name,
                "source_row_index":
                    source_row_index,
                "status":
                    "success",
                "scene_key":
                    output_row["scene_key"],
                "label":
                    output_row["label"],
                "site_key":
                    output_row[
                        "site_key_normalized"
                    ],
                "landsat_sensor":
                    output_row[
                        "landsat_sensor"
                    ],
                "filename":
                    patch_path.name,
                "canonical_pixel_hash":
                    pixel_hash,
                "error":
                    "",
            })

        except Exception as error:
            audit_rows.append({
                "source_dataset":
                    source_name,
                "source_row_index":
                    source_row_index,
                "status":
                    "error",
                "scene_key":
                    "",
                "label":
                    np.nan,
                "site_key":
                    "",
                "landsat_sensor":
                    "",
                "filename":
                    "",
                "canonical_pixel_hash":
                    "",
                "error":
                    str(error),
            })

    prepared = pd.DataFrame(
        prepared_rows
    )

    audit = pd.DataFrame(
        audit_rows
    )

    failed = audit[
        audit["status"] != "success"
    ]

    if len(failed) > 0:
        raise RuntimeError(
            "One or more rows failed during "
            "dataset preparation:\n"
            + failed.to_string(index=False)
        )

    return prepared, audit


def validate_dataset(
    dataframe,
    dataset_name,
    feature_names,
    expected_rows,
    expected_label_counts,
):
    if len(dataframe) != expected_rows:
        raise ValueError(
            f"{dataset_name}: expected "
            f"{expected_rows} rows, found "
            f"{len(dataframe)}."
        )

    label_counts = (
        dataframe["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if label_counts != expected_label_counts:
        raise ValueError(
            f"{dataset_name}: unexpected label "
            f"counts {label_counts}; expected "
            f"{expected_label_counts}."
        )

    duplicated_hash = dataframe[
        "canonical_pixel_hash"
    ].duplicated(
        keep=False
    )

    if duplicated_hash.any():
        duplicate_rows = dataframe.loc[
            duplicated_hash,
            [
                "scene_key",
                "label",
                "site_key_normalized",
                "canonical_pixel_hash",
            ],
        ]

        raise ValueError(
            f"{dataset_name}: duplicate pixel "
            "rasters found:\n"
            + duplicate_rows.to_string(
                index=False
            )
        )

    duplicated_scene = dataframe[
        "scene_key"
    ].duplicated(
        keep=False
    )

    if duplicated_scene.any():
        duplicate_rows = dataframe.loc[
            duplicated_scene,
            [
                "scene_key",
                "label",
                "site_key_normalized",
            ],
        ]

        raise ValueError(
            f"{dataset_name}: duplicate scene "
            "keys found:\n"
            + duplicate_rows.to_string(
                index=False
            )
        )

    unknown_sites = dataframe[
        dataframe[
            "site_key_normalized"
        ] == "unknown"
    ]

    if len(unknown_sites) > 0:
        raise ValueError(
            f"{dataset_name}: some scenes have "
            "unknown site labels:\n"
            + unknown_sites[
                [
                    "scene_key",
                    "resolved_patch_path",
                ]
            ].to_string(index=False)
        )

    missing_feature_values = int(
        dataframe[
            feature_names
        ].isna().sum().sum()
    )

    return {
        "dataset_name": dataset_name,
        "rows": len(dataframe),
        "label_0":
            int((dataframe["label"] == 0).sum()),
        "label_1":
            int((dataframe["label"] == 1).sum()),
        "unique_pixel_hashes":
            int(
                dataframe[
                    "canonical_pixel_hash"
                ].nunique()
            ),
        "unique_scene_keys":
            int(
                dataframe[
                    "scene_key"
                ].nunique()
            ),
        "feature_count":
            len(feature_names),
        "missing_feature_values":
            missing_feature_values,
    }


def order_columns(
    dataframe,
    feature_names,
):
    metadata_columns = [
        "scene_key",
        "overpass_id",
        "raster_group_id",
        "event_id",
        "label",
        "site_key_normalized",
        "site_name",
        "landsat_sensor",
        "landsat_product_id_normalized",
        "acquisition_time_utc",
        "label_status",
        "final_status",
        "label_confidence",
        "label_source",
        "source_dataset",
        "source_row_index",
        "resolved_patch_path",
        "canonical_pixel_hash",
    ]

    metadata_columns = [
        column
        for column in metadata_columns
        if column in dataframe.columns
    ]

    remaining_columns = [
        column
        for column in dataframe.columns
        if (
            column not in metadata_columns
            and column not in feature_names
        )
    ]

    return dataframe[
        metadata_columns
        + feature_names
        + remaining_columns
    ]


def main():
    print("=" * 100)
    print("BUILD LANDSAT STRICT CORE V2 AND EXTENDED V2")
    print("=" * 100)

    core_base_raw = load_table(
        CORE_BASE,
        "core_base_schedule_confirmed",
    )

    extended_base_raw = load_table(
        EXTENDED_BASE,
        "extended_base_schedule_confirmed",
    )

    targeted_raw = load_table(
        TARGETED_CASA_GRANDE,
        "targeted_casa_grande",
    )

    ehrenberg_raw = load_table(
        EHRENBERG_FEATURES,
        "ehrenberg_priority",
    )

    # 從原本的 extract_features 函式取得
    # 精確的 100 個特徵名稱。
    sample_path = resolve_patch_path(
        targeted_raw.iloc[0]
    )

    sample_features = extract_features(
        sample_path
    )

    feature_names = list(
        sample_features.keys()
    )

    print(
        f"\nImage-feature columns: "
        f"{len(feature_names)}"
    )

    core_base, audit_core = prepare_table(
        core_base_raw,
        feature_names,
    )

    extended_base, audit_extended = (
        prepare_table(
            extended_base_raw,
            feature_names,
        )
    )

    targeted, audit_targeted = prepare_table(
        targeted_raw,
        feature_names,
    )

    ehrenberg, audit_ehrenberg = (
        prepare_table(
            ehrenberg_raw,
            feature_names,
        )
    )

    targeted_ids = set(
        targeted["overpass_id"]
        .astype(str)
    )

    expected_targeted_ids = {
        "OP_012",
        "OP_013",
    }

    if targeted_ids != expected_targeted_ids:
        raise ValueError(
            "Unexpected targeted Casa Grande "
            f"overpasses: {sorted(targeted_ids)}"
        )

    ehrenberg_ids = set(
        ehrenberg["overpass_id"]
        .astype(str)
    )

    expected_ehrenberg_ids = {
        "OP_020",
        "OP_022",
        "OP_023",
        "OP_024",
        "OP_026",
        "OP_028",
    }

    if ehrenberg_ids != expected_ehrenberg_ids:
        raise ValueError(
            "Unexpected Ehrenberg overpasses: "
            f"{sorted(ehrenberg_ids)}"
        )

    # Strict Core：
    # 原本 Core 12 張
    # + OP_013 confirmed positive
    # + Ehrenberg 6 張
    strict_targeted = targeted[
        targeted["overpass_id"]
        .astype(str)
        .eq("OP_013")
    ].copy()

    strict_core = pd.concat(
        [
            core_base,
            strict_targeted,
            ehrenberg,
        ],
        ignore_index=True,
        sort=False,
    )

    # Extended：
    # 原本 Extended 13 張
    # + OP_012、OP_013
    # + Ehrenberg 6 張
    extended_v2 = pd.concat(
        [
            extended_base,
            targeted,
            ehrenberg,
        ],
        ignore_index=True,
        sort=False,
    )

    strict_core[
        "dataset_version"
    ] = "strict_core_v2"

    extended_v2[
        "dataset_version"
    ] = "extended_v2"

    strict_summary = validate_dataset(
        strict_core,
        "strict_core_v2",
        feature_names,
        expected_rows=19,
        expected_label_counts={
            0: 10,
            1: 9,
        },
    )

    extended_summary = validate_dataset(
        extended_v2,
        "extended_v2",
        feature_names,
        expected_rows=21,
        expected_label_counts={
            0: 12,
            1: 9,
        },
    )

    strict_core = order_columns(
        strict_core,
        feature_names,
    )

    extended_v2 = order_columns(
        extended_v2,
        feature_names,
    )

    CORE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    strict_core.to_csv(
        CORE_OUTPUT,
        index=False,
    )

    extended_v2.to_csv(
        EXTENDED_OUTPUT,
        index=False,
    )

    merge_audit = pd.concat(
        [
            audit_core,
            audit_extended,
            audit_targeted,
            audit_ehrenberg,
        ],
        ignore_index=True,
    )

    merge_audit.to_csv(
        MERGE_AUDIT_OUTPUT,
        index=False,
    )

    summary_rows = []

    for dataset_name, dataframe in [
        ("strict_core_v2", strict_core),
        ("extended_v2", extended_v2),
    ]:
        for (
            site_key,
            label,
            sensor,
        ), group in dataframe.groupby(
            [
                "site_key_normalized",
                "label",
                "landsat_sensor",
            ],
            dropna=False,
        ):
            summary_rows.append({
                "dataset_name":
                    dataset_name,
                "site_key":
                    site_key,
                "label":
                    label,
                "landsat_sensor":
                    sensor,
                "scene_count":
                    len(group),
            })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 100)
    print("LANDSAT V2 DATASET SUMMARY")
    print("=" * 100)

    for dataset_name, dataframe in [
        ("STRICT CORE V2", strict_core),
        ("EXTENDED V2", extended_v2),
    ]:
        print("\n" + "-" * 100)
        print(dataset_name)
        print("-" * 100)

        print(f"\nScenes: {len(dataframe)}")

        print("\nLabel counts:")
        print(
            dataframe["label"]
            .value_counts()
            .sort_index()
        )

        print("\nSite counts:")
        print(
            dataframe[
                "site_key_normalized"
            ].value_counts()
        )

        print("\nLabel by site:")
        print(
            pd.crosstab(
                dataframe[
                    "site_key_normalized"
                ],
                dataframe["label"],
                margins=True,
            )
        )

        print("\nSensor counts:")
        print(
            dataframe[
                "landsat_sensor"
            ].value_counts()
        )

        print(
            "\nUnique pixel rasters:",
            dataframe[
                "canonical_pixel_hash"
            ].nunique(),
        )

        print(
            "Missing values among 100 "
            "image features:",
            int(
                dataframe[
                    feature_names
                ].isna().sum().sum()
            ),
        )

    print("\nValidation summaries:")
    print(strict_summary)
    print(extended_summary)

    print("\nSaved:")
    print(CORE_OUTPUT)
    print(EXTENDED_OUTPUT)
    print(MERGE_AUDIT_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
