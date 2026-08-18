from pathlib import Path
import re

import pandas as pd

import download_and_qa_s2_low_emission_scenes as base


POSITIVE_INPUT = Path(
    "outputs/355_s2_high_emission_positive_manifest_clean_v1.csv"
)

NEGATIVE_INPUT = Path(
    "outputs/367_s2_high_emission_matched_negative_manifest_v2.csv"
)


POSITIVE_PATCH_DIR = Path(
    "sample_patches/s2_high_emission_positives_v1"
)

NEGATIVE_PATCH_DIR = Path(
    "sample_patches/s2_high_emission_negatives_v1"
)

POSITIVE_PREVIEW_DIR = Path(
    "outputs/s2_high_emission_positive_previews_v1"
)

NEGATIVE_PREVIEW_DIR = Path(
    "outputs/s2_high_emission_negative_previews_v1"
)


POSITIVE_INDEX_OUTPUT = Path(
    "outputs/361_s2_high_emission_positive_patch_index_v1.csv"
)

POSITIVE_QA_OUTPUT = Path(
    "outputs/362_s2_high_emission_positive_local_qa_v1.csv"
)

NEGATIVE_INDEX_OUTPUT = Path(
    "outputs/369_s2_high_emission_negative_patch_index_v2.csv"
)

NEGATIVE_QA_OUTPUT = Path(
    "outputs/370_s2_high_emission_negative_local_qa_v2.csv"
)

REPORT_OUTPUT = Path(
    "outputs/371_s2_high_emission_patch_qa_report_v2.txt"
)


EXPECTED_POSITIVES = 7
EXPECTED_NEGATIVES = 28


def safe_name(value):
    return re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        str(value),
    ).strip("_")


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


def load_positive_manifest():
    frame = pd.read_csv(
        POSITIVE_INPUT,
        low_memory=False,
    )

    require_columns(
        frame,
        [
            "positive_id",
            "scene_id",
            "site",
            "acquisition_time_utc",
            "release_rate_kg_h",
            "lat",
            "lon",
            "label",
        ],
        "Positive manifest",
    )

    frame["sample_id"] = (
        frame["positive_id"]
        .astype(str)
    )

    frame["sample_role"] = "positive"

    frame["matched_group_id"] = (
        frame["scene_id"]
        .astype(str)
    )

    frame[
        "matched_positive_scene_id"
    ] = frame["scene_id"]

    frame[
        "matched_positive_rate_kg_h"
    ] = frame[
        "release_rate_kg_h"
    ]

    return frame


def load_negative_manifest():
    frame = pd.read_csv(
        NEGATIVE_INPUT,
        low_memory=False,
    )

    require_columns(
        frame,
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
            "label",
        ],
        "Negative manifest",
    )

    frame["sample_id"] = (
        frame["negative_id"]
        .astype(str)
    )

    frame["sample_role"] = (
        "negative_control"
    )

    frame["matched_group_id"] = (
        frame[
            "matched_positive_scene_id"
        ].astype(str)
    )

    frame["release_rate_kg_h"] = 0.0

    return frame


def validate_manifests(
    positives,
    negatives,
):
    if len(positives) != EXPECTED_POSITIVES:
        raise RuntimeError(
            "Positive 數量應為 "
            f"{EXPECTED_POSITIVES}，"
            f"實際為 {len(positives)}。"
        )

    if len(negatives) != EXPECTED_NEGATIVES:
        raise RuntimeError(
            "Negative 數量應為 "
            f"{EXPECTED_NEGATIVES}，"
            f"實際為 {len(negatives)}。"
        )

    if (
        positives["scene_id"].nunique()
        != EXPECTED_POSITIVES
    ):
        raise RuntimeError(
            "Positive scene_id 有重複。"
        )

    if (
        negatives["scene_id"].nunique()
        != EXPECTED_NEGATIVES
    ):
        raise RuntimeError(
            "Negative scene_id 有重複。"
        )

    positive_scene_ids = set(
        positives["scene_id"]
        .dropna()
        .astype(str)
    )

    negative_scene_ids = set(
        negatives["scene_id"]
        .dropna()
        .astype(str)
    )

    overlap = (
        positive_scene_ids
        & negative_scene_ids
    )

    if overlap:
        raise RuntimeError(
            "Positive 與 negative 有重複 scene：\n"
            + "\n".join(
                sorted(overlap)
            )
        )

    per_positive = (
        negatives.groupby(
            "matched_positive_scene_id"
        )["scene_id"]
        .nunique()
    )

    invalid = per_positive[
        per_positive.ne(4)
    ]

    if not invalid.empty:
        raise RuntimeError(
            "部分 positive 不是剛好 "
            "4 張 negatives：\n"
            + invalid.to_string()
        )


def prepare_frame(frame):
    frame = frame.copy()

    frame[
        "acquisition_time_utc"
    ] = pd.to_datetime(
        frame[
            "acquisition_time_utc"
        ],
        errors="raise",
        utc=True,
    )

    frame["lat"] = pd.to_numeric(
        frame["lat"],
        errors="raise",
    )

    frame["lon"] = pd.to_numeric(
        frame["lon"],
        errors="raise",
    )

    frame["label"] = pd.to_numeric(
        frame["label"],
        errors="raise",
    ).astype(int)

    return frame


def process_manifest(
    frame,
    patch_dir,
    preview_dir,
    index_output,
    qa_output,
):
    patch_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base.PREVIEW_DIR = preview_dir

    index_rows = []
    qa_rows = []

    for number, row in (
        frame.reset_index(
            drop=True
        ).iterrows()
    ):
        acquisition_time = row[
            "acquisition_time_utc"
        ]

        timestamp_text = (
            acquisition_time.strftime(
                "%Y%m%dT%H%M%S"
            )
        )

        stem = safe_name(
            f"{row['sample_id']}_"
            f"{timestamp_text}_"
            f"{row['site']}"
        )

        patch_path = (
            patch_dir
            / f"{stem}.tif"
        )

        print(
            f"\n[{number + 1}/{len(frame)}] "
            f"{row['sample_id']} | "
            f"label={row['label']} | "
            f"{acquisition_time}",
            flush=True,
        )

        print(
            "  Scene:",
            row["scene_id"],
        )

        if patch_path.exists():
            download_status = "existing"

            print(
                "  Existing patch:",
                patch_path,
            )

        else:
            try:
                base.download_scene(
                    scene_id=
                        row["scene_id"],

                    latitude=
                        float(row["lat"]),

                    longitude=
                        float(row["lon"]),

                    output_path=
                        patch_path,
                )

                download_status = (
                    "downloaded"
                )

                print(
                    "  Downloaded:",
                    patch_path,
                )

            except Exception as error:
                print(
                    "  Download failed:",
                    error,
                )

                index_rows.append({
                    **row.to_dict(),

                    "patch_path":
                        str(patch_path),

                    "download_status":
                        "failed",

                    "download_error":
                        str(error),
                })

                continue

        index_record = {
            **row.to_dict(),

            "patch_path":
                str(patch_path),

            "download_status":
                download_status,

            "download_error":
                "",
        }

        index_rows.append(
            index_record
        )

        try:
            qa_result = (
                base.run_local_qa(
                    tif_path=
                        patch_path,

                    latitude=
                        float(
                            row["lat"]
                        ),

                    longitude=
                        float(
                            row["lon"]
                        ),

                    preview_stem=
                        stem,
                )
            )

            qa_record = {
                **index_record,
                **qa_result,
            }

            qa_rows.append(
                qa_record
            )

            print(
                "  Source inside raster:",
                qa_result.get(
                    "source_center_inside_raster"
                ),
            )

            print(
                "  Local valid fraction:",
                qa_result.get(
                    "local_valid_fraction"
                ),
            )

            print(
                "  Local bad atmosphere:",
                qa_result.get(
                    "local_bad_atmosphere_fraction"
                ),
            )

            print(
                "  Preliminary QA pass:",
                qa_result.get(
                    "qa_pass_preliminary"
                ),
            )

        except Exception as error:
            print(
                "  QA failed:",
                error,
            )

            qa_rows.append({
                **index_record,

                "qa_pass_preliminary":
                    False,

                "qa_error":
                    str(error),
            })

    index = pd.DataFrame(
        index_rows
    )

    qa = pd.DataFrame(
        qa_rows
    )

    index.to_csv(
        index_output,
        index=False,
    )

    qa.to_csv(
        qa_output,
        index=False,
    )

    return index, qa


def qa_pass_mask(frame):
    if (
        frame.empty
        or "qa_pass_preliminary"
        not in frame.columns
    ):
        return pd.Series(
            False,
            index=frame.index,
        )

    return (
        frame[
            "qa_pass_preliminary"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin([
            "true",
            "1",
            "yes",
        ])
    )


def summarize_group(
    name,
    expected_count,
    index,
    qa,
):
    successful_downloads = (
        int(
            index[
                "download_status"
            ].isin([
                "downloaded",
                "existing",
            ]).sum()
        )
        if not index.empty
        else 0
    )

    passed = int(
        qa_pass_mask(qa).sum()
    )

    failed = (
        len(qa) - passed
    )

    lines = [
        f"{name}:",
        (
            f"  Expected scenes: "
            f"{expected_count}"
        ),
        (
            f"  Downloaded/existing: "
            f"{successful_downloads}"
        ),
        (
            f"  QA rows: "
            f"{len(qa)}"
        ),
        (
            f"  Preliminary QA passes: "
            f"{passed}"
        ),
        (
            f"  Preliminary QA failures: "
            f"{failed}"
        ),
    ]

    return (
        successful_downloads,
        passed,
        failed,
        lines,
    )


def main():
    base.initialize_earth_engine()

    positives = prepare_frame(
        load_positive_manifest()
    )

    negatives = prepare_frame(
        load_negative_manifest()
    )

    validate_manifests(
        positives,
        negatives,
    )

    print("=" * 115)
    print(
        "DOWNLOAD AND QA SENTINEL-2 "
        "HIGH-EMISSION BENCHMARK"
    )
    print("=" * 115)

    print(
        "\nPositive scenes:",
        len(positives),
    )

    print(
        "Negative scenes:",
        len(negatives),
    )

    print("\n" + "-" * 115)
    print("DOWNLOADING POSITIVES")
    print("-" * 115)

    (
        positive_index,
        positive_qa,
    ) = process_manifest(
        frame=positives,

        patch_dir=
            POSITIVE_PATCH_DIR,

        preview_dir=
            POSITIVE_PREVIEW_DIR,

        index_output=
            POSITIVE_INDEX_OUTPUT,

        qa_output=
            POSITIVE_QA_OUTPUT,
    )

    print("\n" + "-" * 115)
    print("DOWNLOADING NEGATIVES")
    print("-" * 115)

    (
        negative_index,
        negative_qa,
    ) = process_manifest(
        frame=negatives,

        patch_dir=
            NEGATIVE_PATCH_DIR,

        preview_dir=
            NEGATIVE_PREVIEW_DIR,

        index_output=
            NEGATIVE_INDEX_OUTPUT,

        qa_output=
            NEGATIVE_QA_OUTPUT,
    )

    (
        positive_downloads,
        positive_passes,
        positive_failures,
        positive_lines,
    ) = summarize_group(
        name="Positive scenes",
        expected_count=
            EXPECTED_POSITIVES,
        index=positive_index,
        qa=positive_qa,
    )

    (
        negative_downloads,
        negative_passes,
        negative_failures,
        negative_lines,
    ) = summarize_group(
        name="Negative scenes",
        expected_count=
            EXPECTED_NEGATIVES,
        index=negative_index,
        qa=negative_qa,
    )

    total_downloads = (
        positive_downloads
        + negative_downloads
    )

    total_passes = (
        positive_passes
        + negative_passes
    )

    total_failures = (
        positive_failures
        + negative_failures
    )

    report_lines = [
        "=" * 115,
        (
            "SENTINEL-2 HIGH-EMISSION "
            "PATCH QA REPORT"
        ),
        "=" * 115,
        "",
        *positive_lines,
        "",
        *negative_lines,
        "",
        (
            f"Total downloaded/existing: "
            f"{total_downloads}"
        ),
        (
            f"Total preliminary QA passes: "
            f"{total_passes}"
        ),
        (
            f"Total preliminary QA failures: "
            f"{total_failures}"
        ),
    ]

    if total_failures > 0:
        report_lines.extend([
            "",
            "WARNING:",
            (
                "Scenes that failed QA must not "
                "be included in the benchmark "
                "without review or replacement."
            ),
        ])

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 115)
    print(
        "HIGH-EMISSION PATCH QA SUMMARY"
    )
    print("=" * 115)

    print(
        "\nPositive downloaded/existing:",
        positive_downloads,
        "/",
        EXPECTED_POSITIVES,
    )

    print(
        "Positive preliminary QA passes:",
        positive_passes,
    )

    print(
        "Positive preliminary QA failures:",
        positive_failures,
    )

    print(
        "\nNegative downloaded/existing:",
        negative_downloads,
        "/",
        EXPECTED_NEGATIVES,
    )

    print(
        "Negative preliminary QA passes:",
        negative_passes,
    )

    print(
        "Negative preliminary QA failures:",
        negative_failures,
    )

    print(
        "\nTotal downloaded/existing:",
        total_downloads,
        "/",
        (
            EXPECTED_POSITIVES
            + EXPECTED_NEGATIVES
        ),
    )

    print(
        "Total preliminary QA passes:",
        total_passes,
    )

    print(
        "Total preliminary QA failures:",
        total_failures,
    )

    print("\nSaved:")
    print(POSITIVE_INDEX_OUTPUT)
    print(POSITIVE_QA_OUTPUT)
    print(NEGATIVE_INDEX_OUTPUT)
    print(NEGATIVE_QA_OUTPUT)
    print(REPORT_OUTPUT)
    print(POSITIVE_PREVIEW_DIR)
    print(NEGATIVE_PREVIEW_DIR)


if __name__ == "__main__":
    main()
