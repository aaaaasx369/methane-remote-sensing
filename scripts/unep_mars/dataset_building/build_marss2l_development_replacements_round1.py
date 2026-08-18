from pathlib import Path

import numpy as np
import pandas as pd


CANDIDATE_INPUT = Path(
    "outputs/251_marss2l_development_candidate_images.csv"
)

ACTIVE_MANIFEST_INPUT = Path(
    "outputs/256_marss2l_development_download_manifest_compatible.csv"
)

BAD_INPUT = Path(
    "outputs/258_marss2l_development_bad_downloads_and_low_qa.csv"
)

REPLACEMENT_OUTPUT = Path(
    "outputs/259_marss2l_development_replacements_round1.csv"
)

AMENDED_MANIFEST_OUTPUT = Path(
    "outputs/260_marss2l_development_active_manifest_v1_1.csv"
)

AUDIT_OUTPUT = Path(
    "outputs/261_marss2l_development_replacement_audit_round1.csv"
)

UNRESOLVED_OUTPUT = Path(
    "outputs/262_marss2l_development_unresolved_round1.csv"
)


HIGH_EMISSION_THRESHOLD = 1000.0
SPLIT_VERSION = "marss2l_development_v1_1_replacement_round1"


ROLE_TO_BENCHMARK = {
    "calibration_negative": "no_plume_negative",
    "model_negative": "no_plume_negative",
    "model_positive": "high_emission_positive",
}


def main():
    for path in [
        CANDIDATE_INPUT,
        ACTIVE_MANIFEST_INPUT,
        BAD_INPUT,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    candidates = pd.read_csv(
        CANDIDATE_INPUT,
        low_memory=False,
    )

    manifest = pd.read_csv(
        ACTIVE_MANIFEST_INPUT,
        low_memory=False,
    )

    bad = pd.read_csv(
        BAD_INPUT,
        low_memory=False,
    )

    for dataframe in [
        candidates,
        manifest,
        bad,
    ]:
        if (
            "acquisition_datetime_utc"
            in dataframe.columns
        ):
            dataframe[
                "acquisition_datetime_utc"
            ] = pd.to_datetime(
                dataframe[
                    "acquisition_datetime_utc"
                ],
                errors="coerce",
                utc=True,
            )

    candidates["ch4_fluxrate"] = pd.to_numeric(
        candidates["ch4_fluxrate"],
        errors="coerce",
    )

    candidates["site_key"] = (
        candidates["site_key"]
        .astype(str)
        .str.strip()
    )

    manifest["site_key"] = (
        manifest["site_key"]
        .astype(str)
        .str.strip()
    )

    bad["site_key"] = (
        bad["site_key"]
        .astype(str)
        .str.strip()
    )

    # 排除所有原本已經被指派過的影像，
    # 包含下載失敗或 QA 不合格的影像。
    used_image_keys = set(
        manifest["image_key"]
        .dropna()
        .astype(str)
    )

    replacement_rows = []
    audit_rows = []
    unresolved_rows = []

    replacement_counter = 1

    bad = bad.sort_values(
        [
            "development_split",
            "site_key",
            "development_role",
            "download_id",
        ]
    ).reset_index(drop=True)

    for _, bad_row in bad.iterrows():
        site_key = str(
            bad_row["site_key"]
        )

        development_role = str(
            bad_row["development_role"]
        )

        benchmark_role = (
            ROLE_TO_BENCHMARK.get(
                development_role
            )
        )

        if benchmark_role is None:
            unresolved = bad_row.to_dict()
            unresolved["unresolved_reason"] = (
                "unknown_development_role"
            )

            unresolved_rows.append(
                unresolved
            )

            continue

        original_sensor = str(
            bad_row.get(
                "satellite_normalized",
                bad_row.get(
                    "sensor_code",
                    "",
                ),
            )
        ).strip().upper()

        original_time = pd.to_datetime(
            bad_row.get(
                "acquisition_datetime_utc"
            ),
            errors="coerce",
            utc=True,
        )

        pool = candidates[
            candidates["site_key"].eq(
                site_key
            )
            & candidates[
                "benchmark_role"
            ].eq(
                benchmark_role
            )
            & ~candidates[
                "image_key"
            ].astype(str).isin(
                used_image_keys
            )
        ].copy()

        if benchmark_role == (
            "high_emission_positive"
        ):
            pool = pool[
                pool["ch4_fluxrate"].ge(
                    HIGH_EMISSION_THRESHOLD
                )
            ].copy()

        pool = pool.drop_duplicates(
            subset=["image_key"],
            keep="first",
        )

        pool[
            "same_sensor_as_replaced"
        ] = (
            pool[
                "satellite_normalized"
            ]
            .fillna("")
            .astype(str)
            .str.upper()
            .eq(original_sensor)
        )

        if pd.notna(original_time):
            pool[
                "days_from_replaced_scene"
            ] = (
                pool[
                    "acquisition_datetime_utc"
                ]
                - original_time
            ).abs().dt.total_seconds() / 86400
        else:
            pool[
                "days_from_replaced_scene"
            ] = np.nan

        pool = pool.sort_values(
            [
                "same_sensor_as_replaced",
                "days_from_replaced_scene",
                "acquisition_datetime_utc",
                "image_key",
            ],
            ascending=[
                False,
                True,
                True,
                True,
            ],
            na_position="last",
        ).reset_index(drop=True)

        if pool.empty:
            unresolved = bad_row.to_dict()
            unresolved["unresolved_reason"] = (
                "no_unused_same_site_candidate"
            )

            unresolved_rows.append(
                unresolved
            )

            continue

        selected = pool.iloc[0].copy()

        new_download_id = (
            f"MARS_DEV_REPL1_"
            f"{replacement_counter:04d}"
        )

        replacement_counter += 1

        selected[
            "download_id"
        ] = new_download_id

        selected[
            "development_split"
        ] = bad_row[
            "development_split"
        ]

        selected[
            "development_role"
        ] = development_role

        # 相容既有下載程式。
        selected[
            "external_role"
        ] = development_role

        selected[
            "evaluation_label"
        ] = (
            1
            if development_role
            == "model_positive"
            else 0
        )

        selected[
            "split_version"
        ] = SPLIT_VERSION

        selected[
            "selection_used_model_output"
        ] = False

        selected[
            "amendment_reason"
        ] = (
            "replace_failed_or_low_qa_scene"
        )

        selected[
            "replaced_download_id"
        ] = bad_row[
            "download_id"
        ]

        selected[
            "original_replacement_reason"
        ] = bad_row[
            "replacement_reason"
        ]

        if (
            "landsat_tile"
            not in selected.index
            or pd.isna(
                selected.get(
                    "landsat_tile"
                )
            )
        ):
            selected[
                "landsat_tile"
            ] = selected.get(
                "tile",
                "",
            )

        replacement_rows.append(
            selected.to_dict()
        )

        used_image_keys.add(
            str(
                selected[
                    "image_key"
                ]
            )
        )

        audit_rows.append({
            "replaced_download_id":
                bad_row["download_id"],
            "replacement_download_id":
                new_download_id,
            "site_key":
                site_key,
            "development_split":
                bad_row[
                    "development_split"
                ],
            "development_role":
                development_role,
            "replacement_reason":
                bad_row[
                    "replacement_reason"
                ],
            "old_sensor":
                original_sensor,
            "new_sensor":
                selected[
                    "satellite_normalized"
                ],
            "same_sensor":
                bool(
                    selected[
                        "same_sensor_as_replaced"
                    ]
                ),
            "days_from_replaced_scene":
                selected[
                    "days_from_replaced_scene"
                ],
            "selection_used_model_output":
                False,
        })

    replacements = pd.DataFrame(
        replacement_rows
    )

    unresolved = pd.DataFrame(
        unresolved_rows
    )

    bad_download_ids = set(
        bad["download_id"]
        .astype(str)
    )

    retained_manifest = manifest[
        ~manifest["download_id"]
        .astype(str)
        .isin(
            bad_download_ids
        )
    ].copy()

    amended_manifest = pd.concat(
        [
            retained_manifest,
            replacements,
        ],
        ignore_index=True,
        sort=False,
    )

    if (
        amended_manifest[
            "download_id"
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Duplicate download_id found."
        )

    if (
        amended_manifest[
            ["site_key", "image_key"]
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Duplicate site/image assignment found."
        )

    REPLACEMENT_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    replacements.to_csv(
        REPLACEMENT_OUTPUT,
        index=False,
    )

    amended_manifest.to_csv(
        AMENDED_MANIFEST_OUTPUT,
        index=False,
    )

    pd.DataFrame(
        audit_rows
    ).to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    unresolved.to_csv(
        UNRESOLVED_OUTPUT,
        index=False,
    )

    print("=" * 108)
    print("MARS-S2L DEVELOPMENT REPLACEMENT ROUND 1")
    print("=" * 108)

    print(
        "\nBad images requested:",
        len(bad),
    )

    print(
        "Replacement images selected:",
        len(replacements),
    )

    print(
        "Unresolved images:",
        len(unresolved),
    )

    if not replacements.empty:
        print("\nReplacement roles:")
        print(
            replacements[
                "development_role"
            ].value_counts()
        )

        print("\nReplacement sensors:")
        print(
            replacements[
                "satellite_normalized"
            ].value_counts()
        )

        print(
            "\nSame-sensor replacements:",
            int(
                replacements[
                    "same_sensor_as_replaced"
                ].sum()
            ),
            "/",
            len(replacements),
        )

    if not unresolved.empty:
        print("\nUnresolved by role:")
        print(
            unresolved[
                "development_role"
            ].value_counts()
        )

        print("\nUnresolved sites:")
        print(
            unresolved[
                [
                    "download_id",
                    "site_key",
                    "development_role",
                    "replacement_reason",
                    "unresolved_reason",
                ]
            ].to_string(
                index=False,
                max_colwidth=100,
            )
        )

    print(
        "\nActive manifest rows:",
        len(amended_manifest),
    )

    print("\nSaved:")
    print(REPLACEMENT_OUTPUT)
    print(AMENDED_MANIFEST_OUTPUT)
    print(AUDIT_OUTPUT)
    print(UNRESOLVED_OUTPUT)


if __name__ == "__main__":
    main()
