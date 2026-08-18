from pathlib import Path

import numpy as np
import pandas as pd

from extract_landsat_patch_features import extract_features


INPUT_INDEX_CSV = Path(
    "outputs/81_landsat_targeted_reviewed_patch_index.csv"
)

OLD_FEATURE_CSV = Path(
    "outputs/35_landsat_patch_features.csv"
)

OUTPUT_FEATURE_CSV = Path(
    "outputs/82_landsat_targeted_reviewed_features.csv"
)

OUTPUT_AUDIT_CSV = Path(
    "outputs/83_landsat_targeted_reviewed_feature_audit.csv"
)


SITE_NAME = "Casa_Grande_AZ_release_stacks"


def detect_sensor(product_id):
    product_id = str(product_id)

    if product_id.startswith("LC08"):
        return "Landsat-8"

    if product_id.startswith("LC09"):
        return "Landsat-9"

    return "Unknown"


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
    if not INPUT_INDEX_CSV.exists():
        raise FileNotFoundError(
            f"Missing targeted patch index: "
            f"{INPUT_INDEX_CSV}"
        )

    if not OLD_FEATURE_CSV.exists():
        raise FileNotFoundError(
            f"Missing old feature table: "
            f"{OLD_FEATURE_CSV}"
        )

    index_df = pd.read_csv(
        INPUT_INDEX_CSV,
        low_memory=False,
    )

    old_features = pd.read_csv(
        OLD_FEATURE_CSV,
        low_memory=False,
    )

    successful = index_df[
        index_df["download_status"].isin([
            "success",
            "success_existing",
        ])
    ].copy()

    print("=" * 90)
    print("TARGETED LANDSAT FEATURE EXTRACTION")
    print("=" * 90)

    print(f"\nInput index rows: {len(index_df)}")
    print(f"Successful downloaded rows: {len(successful)}")

    if len(successful) != 2:
        raise ValueError(
            "Expected exactly two successfully downloaded "
            f"patches, but found {len(successful)}."
        )

    output_rows = []
    audit_rows = []
    extracted_feature_names = None

    for _, row in successful.iterrows():
        overpass_id = str(
            row["overpass_id"]
        ).strip()

        patch_path = Path(
            str(row["file_path"])
        )

        print("\n" + "-" * 90)
        print(f"Overpass: {overpass_id}")
        print(f"Patch: {patch_path}")

        if not patch_path.exists():
            raise FileNotFoundError(
                f"Patch does not exist: {patch_path}"
            )

        try:
            features = extract_features(
                patch_path
            )

            current_feature_names = set(
                features.keys()
            )

            if extracted_feature_names is None:
                extracted_feature_names = (
                    current_feature_names
                )
            elif (
                current_feature_names
                != extracted_feature_names
            ):
                raise ValueError(
                    "The two patches produced different "
                    "feature schemas."
                )

            product_id = row.get(
                "landsat_product_id",
                row.get(
                    "LANDSAT_PRODUCT_ID",
                    "",
                ),
            )

            label = int(row["label"])

            output_row = row.to_dict()

            # 建立兩張新影像專用、穩定且唯一的 scene ID。
            output_row["raster_group_id"] = (
                f"RG_{overpass_id}"
            )

            output_row["event_id"] = (
                f"TARGETED_REVIEW_{overpass_id}"
            )

            output_row["site_name"] = SITE_NAME

            output_row["landsat_sensor"] = (
                detect_sensor(product_id)
            )

            output_row["landsat_image_time"] = (
                clean_time(
                    row.get(
                        "landsat_image_time"
                    )
                )
            )

            output_row["label"] = label
            output_row["final_scene_label"] = label

            output_row["final_label_source"] = (
                "targeted_release_review"
            )

            output_row["resolved_patch_path"] = (
                str(patch_path.resolve())
            )

            output_row["patch_filename"] = (
                patch_path.name
            )

            output_row.update(features)

            output_rows.append(
                output_row
            )

            audit_rows.append({
                "overpass_id": overpass_id,
                "raster_group_id":
                    output_row["raster_group_id"],
                "label": label,
                "filename": patch_path.name,
                "status": "success",
                "feature_count": len(features),
                "raw_dn_min":
                    features.get("raw_dn_min"),
                "raw_dn_max":
                    features.get("raw_dn_max"),
                "reflectance_min":
                    features.get(
                        "reflectance_min"
                    ),
                "reflectance_max":
                    features.get(
                        "reflectance_max"
                    ),
                "reflectance_fraction_below_0":
                    features.get(
                        "reflectance_fraction_below_0"
                    ),
                "reflectance_fraction_above_1":
                    features.get(
                        "reflectance_fraction_above_1"
                    ),
                "valid_pixel_fraction":
                    features.get(
                        "valid_pixel_fraction"
                    ),
                "missing_feature_values":
                    int(
                        pd.Series(features)
                        .isna()
                        .sum()
                    ),
                "error": "",
            })

            print(
                f"[OK] features={len(features)}, "
                f"DN={features['raw_dn_min']:.0f}"
                f"–{features['raw_dn_max']:.0f}, "
                f"reflectance="
                f"{features['reflectance_min']:.4f}"
                f"–{features['reflectance_max']:.4f}, "
                f"valid="
                f"{features['valid_pixel_fraction']:.4f}"
            )

        except Exception as error:
            audit_rows.append({
                "overpass_id": overpass_id,
                "raster_group_id":
                    f"RG_{overpass_id}",
                "label": row.get("label"),
                "filename": patch_path.name,
                "status": "error",
                "feature_count": np.nan,
                "missing_feature_values":
                    np.nan,
                "error": str(error),
            })

            print(
                f"[ERROR] {overpass_id}: "
                f"{error}"
            )

    feature_df = pd.DataFrame(
        output_rows
    )

    audit_df = pd.DataFrame(
        audit_rows
    )

    OUTPUT_FEATURE_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_csv(
        OUTPUT_FEATURE_CSV,
        index=False,
    )

    audit_df.to_csv(
        OUTPUT_AUDIT_CSV,
        index=False,
    )

    print("\n" + "=" * 90)
    print("FEATURE SCHEMA CHECK")
    print("=" * 90)

    if extracted_feature_names is None:
        raise RuntimeError(
            "No features were successfully extracted."
        )

    old_column_set = set(
        old_features.columns
    )

    missing_from_old_table = sorted(
        extracted_feature_names
        - old_column_set
    )

    present_in_old_table = sorted(
        extracted_feature_names
        & old_column_set
    )

    print(
        f"\nExtracted image-feature columns: "
        f"{len(extracted_feature_names)}"
    )

    print(
        f"Feature columns also found in old table: "
        f"{len(present_in_old_table)}"
    )

    print(
        "Extracted features missing from old table:",
        len(missing_from_old_table),
    )

    if missing_from_old_table:
        print(
            "\nMissing feature names:"
        )

        for column in missing_from_old_table:
            print(column)

    print("\n" + "=" * 90)
    print("TARGETED FEATURE SUMMARY")
    print("=" * 90)

    print(f"\nSuccessful feature rows: {len(feature_df)}")
    print(
        "Failed feature rows:",
        int(
            (audit_df["status"] != "success")
            .sum()
        ),
    )

    if len(feature_df) > 0:
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

        print("\nReflectance audit:")
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
    print(OUTPUT_FEATURE_CSV)
    print(OUTPUT_AUDIT_CSV)


if __name__ == "__main__":
    main()
