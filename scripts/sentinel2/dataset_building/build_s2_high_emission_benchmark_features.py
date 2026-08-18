from pathlib import Path

import numpy as np
import pandas as pd

import build_s2_low_emission_pilot_benchmark as base


POSITIVE_MANIFEST = Path(
    "outputs/355_s2_high_emission_positive_manifest_clean_v1.csv"
)

POSITIVE_QA = Path(
    "outputs/362_s2_high_emission_positive_local_qa_v1.csv"
)

NEGATIVE_MANIFEST = Path(
    "outputs/367_s2_high_emission_matched_negative_manifest_v2.csv"
)

NEGATIVE_QA = Path(
    "outputs/370_s2_high_emission_negative_local_qa_v2.csv"
)


BENCHMARK_OUTPUT = Path(
    "outputs/372_s2_high_emission_benchmark_v1.csv"
)

FEATURE_OUTPUT = Path(
    "outputs/373_s2_high_emission_features_v1.csv"
)

ANOMALY_OUTPUT = Path(
    "outputs/374_s2_high_emission_matched_anomaly_features_v1.csv"
)

REPORT_OUTPUT = Path(
    "outputs/375_s2_high_emission_feature_report_v1.txt"
)


EXPECTED_POSITIVES = 7
EXPECTED_NEGATIVES = 28
EXPECTED_TOTAL = 35


def require_columns(
    frame,
    columns,
    table_name,
):
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            f"{table_name} 缺少欄位："
            + ", ".join(missing)
        )


def parse_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin([
            "true",
            "1",
            "yes",
        ])
    )


def load_positive_rows():
    manifest = pd.read_csv(
        POSITIVE_MANIFEST,
        low_memory=False,
    )

    qa = pd.read_csv(
        POSITIVE_QA,
        low_memory=False,
    )

    require_columns(
        manifest,
        [
            "positive_id",
            "scene_id",
            "site",
            "acquisition_time_utc",
            "release_rate_kg_h",
            "lat",
            "lon",
        ],
        "Positive manifest",
    )

    require_columns(
        qa,
        [
            "scene_id",
            "patch_path",
            "qa_pass_preliminary",
        ],
        "Positive QA",
    )

    qa = qa.drop_duplicates(
        subset=["scene_id"],
        keep="first",
    )

    positive = manifest.merge(
        qa[
            [
                "scene_id",
                "patch_path",
                "qa_pass_preliminary",
                "local_valid_fraction",
                "local_cloud_fraction",
                "local_shadow_fraction",
                "local_bad_atmosphere_fraction",
                "local_all_zero_fraction",
            ]
        ],
        on="scene_id",
        how="left",
        validate="one_to_one",
    )

    positive["sample_id"] = (
        positive["positive_id"]
        .astype(str)
    )

    positive["label"] = 1

    positive["dataset_role"] = (
        "strict_high_emission_positive"
    )

    positive["matched_group_id"] = (
        positive["scene_id"]
        .astype(str)
    )

    positive[
        "matched_positive_scene_id"
    ] = positive["scene_id"]

    positive[
        "matched_positive_rate_kg_h"
    ] = positive[
        "release_rate_kg_h"
    ]

    return positive


def load_negative_rows():
    manifest = pd.read_csv(
        NEGATIVE_MANIFEST,
        low_memory=False,
    )

    qa = pd.read_csv(
        NEGATIVE_QA,
        low_memory=False,
    )

    require_columns(
        manifest,
        [
            "negative_id",
            "positive_id",
            "scene_id",
            "site",
            "acquisition_time_utc",
            "lat",
            "lon",
            "matched_positive_scene_id",
            "matched_positive_rate_kg_h",
        ],
        "Negative manifest",
    )

    require_columns(
        qa,
        [
            "scene_id",
            "patch_path",
            "qa_pass_preliminary",
        ],
        "Negative QA",
    )

    qa = qa.drop_duplicates(
        subset=["scene_id"],
        keep="first",
    )

    negative = manifest.merge(
        qa[
            [
                "scene_id",
                "patch_path",
                "qa_pass_preliminary",
                "local_valid_fraction",
                "local_cloud_fraction",
                "local_shadow_fraction",
                "local_bad_atmosphere_fraction",
                "local_all_zero_fraction",
            ]
        ],
        on="scene_id",
        how="left",
        validate="one_to_one",
    )

    negative["sample_id"] = (
        negative["negative_id"]
        .astype(str)
    )

    negative["label"] = 0

    negative["dataset_role"] = (
        "clean_high_emission_matched_negative"
    )

    negative["matched_group_id"] = (
        negative[
            "matched_positive_scene_id"
        ].astype(str)
    )

    negative["release_rate_kg_h"] = 0.0

    return negative


def build_benchmark():
    positive = load_positive_rows()
    negative = load_negative_rows()

    common_columns = sorted(
        set(positive.columns)
        | set(negative.columns)
    )

    positive = positive.reindex(
        columns=common_columns
    )

    negative = negative.reindex(
        columns=common_columns
    )

    benchmark = pd.concat(
        [
            positive,
            negative,
        ],
        ignore_index=True,
        sort=False,
    )

    benchmark[
        "acquisition_time_utc"
    ] = pd.to_datetime(
        benchmark[
            "acquisition_time_utc"
        ],
        errors="coerce",
        utc=True,
    )

    for column in [
        "lat",
        "lon",
        "release_rate_kg_h",
    ]:
        benchmark[column] = pd.to_numeric(
            benchmark[column],
            errors="coerce",
        )

    benchmark[
        "qa_pass_preliminary"
    ] = parse_bool(
        benchmark[
            "qa_pass_preliminary"
        ]
    )

    benchmark["patch_exists"] = (
        benchmark["patch_path"]
        .fillna("")
        .map(
            lambda value:
                Path(str(value)).exists()
                if str(value).strip()
                else False
        )
    )

    benchmark[
        "benchmark_version"
    ] = "s2_high_emission_benchmark_v1"

    benchmark = benchmark.sort_values(
        [
            "matched_group_id",
            "label",
            "acquisition_time_utc",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    if len(benchmark) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"預期 {EXPECTED_TOTAL} 張影像，"
            f"實際為 {len(benchmark)}。"
        )

    label_counts = (
        benchmark["label"]
        .value_counts()
        .to_dict()
    )

    if label_counts != {
        0: EXPECTED_NEGATIVES,
        1: EXPECTED_POSITIVES,
    }:
        raise RuntimeError(
            "標籤數量錯誤："
            + str(label_counts)
        )

    group_summary = (
        benchmark.groupby(
            "matched_group_id"
        )["label"]
        .agg(
            total="size",
            positives="sum",
        )
    )

    invalid_groups = group_summary[
        (group_summary["total"] != 5)
        | (
            group_summary["positives"]
            != 1
        )
    ]

    if not invalid_groups.empty:
        raise RuntimeError(
            "Matched group 結構錯誤：\n"
            + invalid_groups.to_string()
        )

    if not benchmark[
        "qa_pass_preliminary"
    ].all():
        failed = benchmark[
            ~benchmark[
                "qa_pass_preliminary"
            ]
        ]

        raise RuntimeError(
            "仍有影像未通過 QA：\n"
            + failed[
                [
                    "sample_id",
                    "scene_id",
                ]
            ].to_string(index=False)
        )

    if not benchmark[
        "patch_exists"
    ].all():
        missing = benchmark[
            ~benchmark[
                "patch_exists"
            ]
        ]

        raise FileNotFoundError(
            "找不到部分 patch：\n"
            + missing[
                [
                    "sample_id",
                    "patch_path",
                ]
            ].to_string(index=False)
        )

    return benchmark, group_summary


def main():
    benchmark, group_summary = (
        build_benchmark()
    )

    benchmark.to_csv(
        BENCHMARK_OUTPUT,
        index=False,
    )

    print("=" * 110)
    print(
        "SENTINEL-2 HIGH-EMISSION BENCHMARK V1"
    )
    print("=" * 110)

    print(
        "\nBenchmark samples:",
        len(benchmark),
    )

    print("\nLabels:")
    print(
        benchmark[
            "label"
        ].value_counts().sort_index()
    )

    print(
        "\nMatched groups:",
        len(group_summary),
    )

    print(
        "\nSamples per matched group:"
    )

    print(group_summary)

    print("\nExtracting features...")

    feature_rows = []

    for number, row in (
        benchmark.iterrows()
    ):
        print(
            f"[{number + 1}/{len(benchmark)}] "
            f"{row['sample_id']} | "
            f"label={row['label']}",
            flush=True,
        )

        feature_rows.append(
            base.extract_patch_features(
                row
            )
        )

    features = pd.DataFrame(
        feature_rows
    )

    features.to_csv(
        FEATURE_OUTPUT,
        index=False,
    )

    anomaly = (
        base.create_matched_anomalies(
            features
        )
    )

    anomaly.to_csv(
        ANOMALY_OUTPUT,
        index=False,
    )

    numeric_missing = int(
        features.select_dtypes(
            include=[np.number]
        ).isna().sum().sum()
    )

    important_features = [
        "local_B11_median",
        "local_B12_median",
        "local_swir_ratio_B12_B11_median",
        "local_swir_nd_B12_B11_median",
        "local_ratio_B11_B8A_median",
        "local_ratio_B12_B8A_median",
    ]

    important_features = [
        column
        for column in important_features
        if column in anomaly.columns
    ]

    positives = anomaly[
        anomaly["label"].eq(1)
    ].copy()

    display_columns = [
        "sample_id",
        "release_rate_kg_h",
        "matched_group_id",
    ]

    for column in important_features:
        display_columns.extend([
            column,
            f"{column}__neg_mean",
            f"{column}__anomaly",
            f"{column}__z_vs_neg",
        ])

    positive_summary = positives[
        display_columns
    ]

    report_lines = [
        "=" * 110,
        (
            "SENTINEL-2 HIGH-EMISSION "
            "FEATURE REPORT V1"
        ),
        "=" * 110,
        "",
        (
            f"Benchmark samples: "
            f"{len(benchmark)}"
        ),
        (
            f"Positive samples: "
            f"{int((benchmark['label'] == 1).sum())}"
        ),
        (
            f"Negative samples: "
            f"{int((benchmark['label'] == 0).sum())}"
        ),
        (
            f"Matched groups: "
            f"{len(group_summary)}"
        ),
        (
            f"Feature rows: "
            f"{len(features)}"
        ),
        (
            f"Feature columns: "
            f"{len(features.columns)}"
        ),
        (
            f"Missing numeric values: "
            f"{numeric_missing}"
        ),
        "",
        "Samples per matched group:",
        group_summary.to_string(),
        "",
        "Positive matched-negative anomalies:",
        positive_summary.to_string(
            index=False
        ),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 110)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 110)

    print(
        "\nFeature rows:",
        len(features),
    )

    print(
        "Feature columns:",
        len(features.columns),
    )

    print(
        "Missing numeric values:",
        numeric_missing,
    )

    print(
        "\nPositive matched-negative anomalies:"
    )

    print(
        positive_summary.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(BENCHMARK_OUTPUT)
    print(FEATURE_OUTPUT)
    print(ANOMALY_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
