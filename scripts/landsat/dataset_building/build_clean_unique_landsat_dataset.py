from pathlib import Path

import numpy as np
import pandas as pd


INPUT_ROWS = Path(
    "outputs/37_landsat_raster_duplicate_rows.csv"
)

ADJUDICATION_CSV = Path(
    "outputs/42_landsat_mixed_group_adjudication.csv"
)

OUTPUT_CLEAN = Path(
    "outputs/43_landsat_unique_clean_features.csv"
)

OUTPUT_PROVENANCE = Path(
    "outputs/44_landsat_unique_group_provenance.csv"
)


def join_unique(series):
    values = (
        series.dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    return " | ".join(values)


def parse_datetime_column(series):
    return pd.to_datetime(
        series,
        errors="coerce",
        utc=True,
    )


def select_representative(group, final_label):
    """
    Select the row whose original label agrees with the final label
    and whose event time is closest to the Landsat acquisition time.
    """

    candidates = group[
        group["label"] == final_label
    ].copy()

    if len(candidates) == 0:
        raise ValueError(
            f"No row in {group['raster_group_id'].iloc[0]} "
            f"has final label {final_label}"
        )

    if (
        "datetime_utc" in candidates.columns
        and "landsat_image_time" in candidates.columns
    ):
        event_times = parse_datetime_column(
            candidates["datetime_utc"]
        )

        landsat_times = parse_datetime_column(
            candidates["landsat_image_time"]
        )

        candidates[
            "_time_difference_seconds"
        ] = (
            event_times - landsat_times
        ).abs().dt.total_seconds()

        valid_time_candidates = candidates[
            candidates["_time_difference_seconds"].notna()
        ]

        if len(valid_time_candidates) > 0:
            selected_index = (
                valid_time_candidates[
                    "_time_difference_seconds"
                ].idxmin()
            )

            return candidates.loc[
                selected_index
            ].copy()

    return candidates.iloc[0].copy()


def main():
    if not INPUT_ROWS.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT_ROWS}"
        )

    if not ADJUDICATION_CSV.exists():
        raise FileNotFoundError(
            f"Missing adjudication file: {ADJUDICATION_CSV}"
        )

    df = pd.read_csv(INPUT_ROWS)
    decisions = pd.read_csv(ADJUDICATION_CSV)

    required_columns = {
        "raster_group_id",
        "pixel_hash",
        "label",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce",
    )

    decisions["final_label"] = pd.to_numeric(
        decisions["final_label"],
        errors="coerce",
    )

    decision_lookup = decisions.set_index(
        "raster_group_id"
    ).to_dict("index")

    clean_rows = []
    provenance_rows = []

    print("=" * 80)
    print("BUILD CLEAN UNIQUE LANDSAT DATASET")
    print("=" * 80)

    for raster_group_id, group in df.groupby(
        "raster_group_id",
        sort=True,
    ):
        original_labels = sorted(
            group["label"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        is_mixed = len(original_labels) > 1

        decision_source = "automatic_nonconflicting"
        confidence = "high"
        reason = (
            "All duplicated rows have the same original label."
        )

        if not is_mixed:
            final_label = original_labels[0]
            decision = "keep"

        else:
            if raster_group_id not in decision_lookup:
                raise ValueError(
                    f"Mixed-label group {raster_group_id} "
                    "has no adjudication decision."
                )

            adjudication = decision_lookup[
                raster_group_id
            ]

            decision = str(
                adjudication["decision"]
            ).strip().lower()

            final_label = adjudication[
                "final_label"
            ]

            confidence = adjudication.get(
                "confidence",
                "",
            )

            reason = adjudication.get(
                "reason",
                "",
            )

            decision_source = "manual_adjudication"

        provenance_row = {
            "raster_group_id": raster_group_id,
            "pixel_hash": group["pixel_hash"].iloc[0],
            "source_row_count": len(group),
            "original_labels": ",".join(
                map(str, original_labels)
            ),
            "mixed_original_labels": is_mixed,
            "decision": decision,
            "final_label": final_label,
            "decision_source": decision_source,
            "decision_confidence": confidence,
            "decision_reason": reason,
        }

        for column in [
            "event_id",
            "filename",
            "site_name",
            "landsat_sensor",
            "datetime_utc",
            "landsat_image_time",
        ]:
            if column in group.columns:
                provenance_row[
                    f"source_{column}s"
                ] = join_unique(group[column])

        provenance_rows.append(
            provenance_row
        )

        if decision == "exclude":
            print(
                f"[EXCLUDE] {raster_group_id} | "
                f"labels={original_labels} | "
                f"{reason}"
            )

            continue

        if decision != "keep":
            raise ValueError(
                f"Unknown decision '{decision}' "
                f"for {raster_group_id}"
            )

        if pd.isna(final_label):
            raise ValueError(
                f"Kept group {raster_group_id} "
                "has no final label."
            )

        final_label = int(final_label)

        if final_label not in (0, 1):
            raise ValueError(
                f"Invalid final label {final_label} "
                f"for {raster_group_id}"
            )

        representative = select_representative(
            group,
            final_label,
        )

        representative["label"] = final_label
        representative["final_scene_label"] = final_label
        representative["original_group_labels"] = (
            ",".join(map(str, original_labels))
        )
        representative["duplicate_source_row_count"] = (
            len(group)
        )
        representative["label_decision_source"] = (
            decision_source
        )
        representative["label_decision_confidence"] = (
            confidence
        )
        representative["label_decision_reason"] = reason

        if "event_id" in group.columns:
            representative[
                "duplicate_source_event_ids"
            ] = join_unique(group["event_id"])

        if "filename" in group.columns:
            representative[
                "duplicate_source_filenames"
            ] = join_unique(group["filename"])

        if (
            "datetime_utc" in representative.index
            and "landsat_image_time" in representative.index
        ):
            event_time = pd.to_datetime(
                representative["datetime_utc"],
                errors="coerce",
                utc=True,
            )

            landsat_time = pd.to_datetime(
                representative["landsat_image_time"],
                errors="coerce",
                utc=True,
            )

            if pd.notna(event_time) and pd.notna(landsat_time):
                representative[
                    "representative_time_difference_seconds"
                ] = abs(
                    (
                        event_time - landsat_time
                    ).total_seconds()
                )
            else:
                representative[
                    "representative_time_difference_seconds"
                ] = np.nan

        clean_rows.append(representative)

        print(
            f"[KEEP] {raster_group_id} | "
            f"original_labels={original_labels} | "
            f"final_label={final_label} | "
            f"source_rows={len(group)}"
        )

    clean_df = pd.DataFrame(clean_rows)
    provenance_df = pd.DataFrame(provenance_rows)

    if len(clean_df) == 0:
        raise ValueError(
            "No rasters remained after cleaning."
        )

    # Final integrity checks
    if clean_df["raster_group_id"].duplicated().any():
        raise ValueError(
            "Duplicate raster_group_id remains."
        )

    if clean_df["pixel_hash"].duplicated().any():
        raise ValueError(
            "Duplicate pixel raster remains."
        )

    if not clean_df["label"].isin([0, 1]).all():
        raise ValueError(
            "Invalid labels remain."
        )

    clean_df = clean_df.sort_values(
        by=[
            "label",
            "raster_group_id",
        ]
    ).reset_index(drop=True)

    provenance_df = provenance_df.sort_values(
        by="raster_group_id"
    ).reset_index(drop=True)

    OUTPUT_CLEAN.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_df.to_csv(
        OUTPUT_CLEAN,
        index=False,
    )

    provenance_df.to_csv(
        OUTPUT_PROVENANCE,
        index=False,
    )

    print("\n" + "=" * 80)
    print("CLEAN DATASET SUMMARY")
    print("=" * 80)

    print(
        f"\nOriginal rows: {len(df)}"
    )

    print(
        f"Original unique rasters: "
        f"{df['raster_group_id'].nunique()}"
    )

    print(
        f"Clean unique rasters: "
        f"{len(clean_df)}"
    )

    print(
        f"Excluded raster groups: "
        f"{int((provenance_df['decision'] == 'exclude').sum())}"
    )

    print("\nFinal label counts:")
    print(
        clean_df["label"]
        .value_counts()
        .sort_index()
    )

    if "landsat_sensor" in clean_df.columns:
        print("\nFinal sensor counts:")
        print(
            clean_df["landsat_sensor"]
            .value_counts()
        )

        print("\nFinal label by sensor:")
        print(
            pd.crosstab(
                clean_df["landsat_sensor"],
                clean_df["label"],
                margins=True,
            )
        )

    print(
        "\nRemaining duplicated pixel hashes:",
        int(clean_df["pixel_hash"].duplicated().sum()),
    )

    print("\nSaved:")
    print(OUTPUT_CLEAN)
    print(OUTPUT_PROVENANCE)


if __name__ == "__main__":
    main()
