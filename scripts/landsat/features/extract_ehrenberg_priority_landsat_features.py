from pathlib import Path

import numpy as np
import pandas as pd

from extract_landsat_patch_features import extract_features


INPUT_INDEX = Path(
    "outputs/90_ehrenberg_priority_landsat_patch_index.csv"
)

OLD_FEATURE_TABLE = Path(
    "outputs/35_landsat_patch_features.csv"
)

OUTPUT_FEATURES = Path(
    "outputs/93_ehrenberg_priority_landsat_features.csv"
)

OUTPUT_AUDIT = Path(
    "outputs/94_ehrenberg_priority_landsat_feature_audit.csv"
)


def clean_time(value):
    parsed = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(parsed):
        return np.nan

    return parsed.strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )


def main():
    for path in [
        INPUT_INDEX,
        OLD_FEATURE_TABLE,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing input file: {path}"
            )

    index_df = pd.read_csv(
        INPUT_INDEX,
        low_memory=False,
    )

    old_features = pd.read_csv(
        OLD_FEATURE_TABLE,
        low_memory=False,
    )

    successful = index_df[
        index_df["download_status"].isin([
            "success",
            "success_existing",
        ])
    ].copy()

    if len(successful) != 6:
        raise ValueError(
            "Expected 6 successfully downloaded "
            f"Ehrenberg patches, found {len(successful)}."
        )

    print("=" * 95)
    print("EHRENBERG LANDSAT FEATURE EXTRACTION")
    print("=" * 95)

    print(f"\nInput rows: {len(index_df)}")
    print(f"Successful patches: {len(successful)}")

    output_rows = []
    audit_rows = []

    expected_feature_names = None

    for _, row in successful.iterrows():
        overpass_id = str(
            row["overpass_id"]
        ).strip()

        patch_path = Path(
            str(row["file_path"])
        )

        label = int(
            row.get(
                "final_label",
                row.get("label"),
            )
        )

        print("\n" + "-" * 95)
        print(
            f"{overpass_id} | "
            f"label={label} | "
            f"{patch_path}"
        )

        audit_row = {
            "overpass_id": overpass_id,
            "label": label,
            "filename": patch_path.name,
            "status": "pending",
            "error": "",
        }

        try:
            if not patch_path.exists():
                raise FileNotFoundError(
                    f"TIFF does not exist: {patch_path}"
                )

            features = extract_features(
                patch_path
            )

            current_feature_names = set(
                features.keys()
            )

            if expected_feature_names is None:
                expected_feature_names = (
                    current_feature_names
                )

            elif (
                current_feature_names
                != expected_feature_names
            ):
                raise ValueError(
                    "Different patches produced "
                    "different feature schemas."
                )

            output_row = row.to_dict()

            output_row.update({
                "raster_group_id":
                    f"RG_EH_{overpass_id}",
                "event_id":
                    f"EHRENBERG_{overpass_id}",
                "site_key":
                    "ehrenberg",
                "site_name":
                    "Ehrenberg_AZ_release_stack",
                "label":
                    label,
                "final_scene_label":
                    label,
                "final_label":
                    label,
                "label_status":
                    row.get(
                        "final_status",
                        "",
                    ),
                "final_label_source":
                    row.get(
                        "label_source",
                        "ehrenberg_release_review",
                    ),
                "landsat_image_time":
                    clean_time(
                        row.get(
                            "landsat_image_time_utc",
                            row.get(
                                "candidate_time_utc"
                            ),
                        )
                    ),
                "resolved_patch_path":
                    str(patch_path.resolve()),
                "patch_filename":
                    patch_path.name,
            })

            output_row.update(
                features
            )

            output_rows.append(
                output_row
            )

            missing_feature_values = int(
                pd.Series(features)
                .isna()
                .sum()
            )

            audit_row.update({
                "status": "success",
                "feature_count":
                    len(features),
                "raw_dn_min":
                    features.get(
                        "raw_dn_min"
                    ),
                "raw_dn_max":
                    features.get(
                        "raw_dn_max"
                    ),
                "reflectance_min":
                    features.get(
                        "reflectance_min"
                    ),
                "reflectance_max":
                    features.get(
                        "reflectance_max"
                    ),
                "valid_pixel_fraction":
                    features.get(
                        "valid_pixel_fraction"
                    ),
                "missing_feature_values":
                    missing_feature_values,
            })

            print(
                f"[OK] features={len(features)} | "
                f"DN={features['raw_dn_min']:.0f}–"
                f"{features['raw_dn_max']:.0f} | "
                f"reflectance="
                f"{features['reflectance_min']:.4f}–"
                f"{features['reflectance_max']:.4f} | "
                f"valid="
                f"{features['valid_pixel_fraction']:.4f} | "
                f"missing={missing_feature_values}"
            )

        except Exception as error:
            audit_row.update({
                "status": "error",
                "error": str(error),
            })

            print(
                f"[ERROR] {overpass_id}: "
                f"{error}"
            )

        audit_rows.append(
            audit_row
        )

    feature_df = pd.DataFrame(
        output_rows
    )

    audit_df = pd.DataFrame(
        audit_rows
    )

    if expected_feature_names is None:
        raise RuntimeError(
            "No image features were extracted."
        )

    old_column_set = set(
        old_features.columns
    )

    missing_from_old_table = sorted(
        expected_feature_names
        - old_column_set
    )

    shared_feature_names = sorted(
        expected_feature_names
        & old_column_set
    )

    OUTPUT_FEATURES.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_csv(
        OUTPUT_FEATURES,
        index=False,
    )

    audit_df.to_csv(
        OUTPUT_AUDIT,
        index=False,
    )

    print("\n" + "=" * 95)
    print("FEATURE SCHEMA CHECK")
    print("=" * 95)

    print(
        f"\nExtracted feature columns: "
        f"{len(expected_feature_names)}"
    )

    print(
        f"Features also present in old table: "
        f"{len(shared_feature_names)}"
    )

    print(
        f"Features missing from old table: "
        f"{len(missing_from_old_table)}"
    )

    if missing_from_old_table:
        print("\nMissing columns:")

        for column in missing_from_old_table:
            print(column)

    print("\n" + "=" * 95)
    print("EHRENBERG FEATURE SUMMARY")
    print("=" * 95)

    print(f"\nSuccessful rows: {len(feature_df)}")

    print(
        "Failed rows:",
        int(
            (audit_df["status"] != "success")
            .sum()
        ),
    )

    print("\nLabel counts:")
    print(
        feature_df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nSensor counts:")
    print(
        feature_df["landsat_sensor"]
        .value_counts()
    )

    print("\nFeature audit:")
    print(
        audit_df[
            [
                "overpass_id",
                "label",
                "feature_count",
                "raw_dn_min",
                "raw_dn_max",
                "reflectance_min",
                "reflectance_max",
                "valid_pixel_fraction",
                "missing_feature_values",
                "status",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nTotal output columns: "
        f"{len(feature_df.columns)}"
    )

    print("\nSaved:")
    print(OUTPUT_FEATURES)
    print(OUTPUT_AUDIT)


if __name__ == "__main__":
    main()
