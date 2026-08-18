from pathlib import Path

import numpy as np
import pandas as pd


PROVENANCE_CSV = Path(
    "outputs/72_landsat_candidate_overpass_provenance.csv"
)

RELEASE_AUDIT_CSV = Path(
    "outputs/69_casa_grande_candidate_release_audit.csv"
)

DAILY_COVERAGE_CSV = Path(
    "outputs/71_casa_grande_release_log_daily_coverage.csv"
)

OUTPUT_CSV = Path(
    "outputs/75_casa_grande_candidate_decision_table.csv"
)

DOWNLOAD_NOW_CSV = Path(
    "outputs/76_casa_grande_download_now_candidates.csv"
)

SCHEDULE_CHECK_CSV = Path(
    "outputs/77_casa_grande_campaign_schedule_check.csv"
)


ZERO_FLOW_TIME_TOLERANCE_SECONDS = 5.0


def parse_boolean(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
        .fillna(False)
    )


def main():
    for path in [
        PROVENANCE_CSV,
        RELEASE_AUDIT_CSV,
        DAILY_COVERAGE_CSV,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}"
            )

    provenance = pd.read_csv(
        PROVENANCE_CSV,
        low_memory=False,
    )

    audit = pd.read_csv(
        RELEASE_AUDIT_CSV,
        low_memory=False,
    )

    daily = pd.read_csv(
        DAILY_COVERAGE_CSV,
        low_memory=False,
    )

    provenance = provenance[
        provenance["site_key"] == "casa_grande"
    ].copy()

    provenance["candidate_time_utc"] = pd.to_datetime(
        provenance["candidate_time_utc"],
        errors="coerce",
        utc=True,
    )

    audit["candidate_time_utc"] = pd.to_datetime(
        audit["candidate_time_utc"],
        errors="coerce",
        utc=True,
    )

    for column in [
        "nearest_ch4_kgh_mean",
        "seconds_to_nearest_interval",
        "recommended_label",
    ]:
        if column in audit.columns:
            audit[column] = pd.to_numeric(
                audit[column],
                errors="coerce",
            )

    if "existing_excluded_ambiguous" in provenance.columns:
        provenance[
            "existing_excluded_ambiguous"
        ] = parse_boolean(
            provenance[
                "existing_excluded_ambiguous"
            ]
        )
    else:
        provenance[
            "existing_excluded_ambiguous"
        ] = False

    daily["first_interval_start"] = pd.to_datetime(
        daily["first_interval_start"],
        errors="coerce",
        utc=True,
    )

    daily["last_interval_end"] = pd.to_datetime(
        daily["last_interval_end"],
        errors="coerce",
        utc=True,
    )

    log_start = daily[
        "first_interval_start"
    ].min()

    log_end = daily[
        "last_interval_end"
    ].max()

    if pd.isna(log_start) or pd.isna(log_end):
        raise ValueError(
            "Could not determine release-log date range."
        )

    merge_columns = [
        column
        for column in [
            "overpass_id",
            "release_review_status",
            "recommended_label",
            "negative_candidate_eligible",
            "same_day_release_interval_count",
            "overlapping_interval_count",
            "overlap_rate_min",
            "overlap_rate_max",
            "nearest_ch4_kgh_mean",
            "seconds_to_nearest_interval",
            "hours_to_nearest_interval",
            "review_reason",
        ]
        if column in audit.columns
    ]

    combined = provenance.merge(
        audit[merge_columns],
        on="overpass_id",
        how="left",
        validate="one_to_one",
    )

    decision_rows = []

    for _, row in combined.iterrows():
        candidate_time = row["candidate_time_utc"]
        release_status = row.get(
            "release_review_status"
        )

        nearest_rate = row.get(
            "nearest_ch4_kgh_mean"
        )

        seconds_to_interval = row.get(
            "seconds_to_nearest_interval"
        )

        existing_ambiguous = bool(
            row.get(
                "existing_excluded_ambiguous",
                False,
            )
        )

        recommended_label = np.nan
        final_status = ""
        evidence_level = ""
        download_recommendation = ""
        decision_reason = ""

        if existing_ambiguous:
            final_status = (
                "existing_excluded_ambiguous_scene"
            )

            evidence_level = "ambiguous"

            download_recommendation = (
                "do_not_download_already_exists"
            )

            decision_reason = (
                "This Landsat overpass already exists locally "
                "and was previously excluded because its label "
                "could not be confirmed."
            )

        elif release_status == (
            "confirmed_positive_overlap"
        ):
            final_status = "confirmed_new_positive"
            recommended_label = 1
            evidence_level = "high"

            download_recommendation = (
                "download_later_positive"
            )

            decision_reason = (
                "The acquisition overlaps a measured interval "
                "with positive methane flow."
            )

        elif release_status == (
            "confirmed_zero_flow_overlap"
        ):
            final_status = "confirmed_new_negative"
            recommended_label = 0
            evidence_level = "high"

            download_recommendation = (
                "download_now_negative"
            )

            decision_reason = (
                "The acquisition overlaps a measured interval "
                "with zero methane flow."
            )

        elif (
            release_status
            == "same_day_but_no_exact_overlap"
            and pd.notna(nearest_rate)
            and float(nearest_rate) == 0.0
            and pd.notna(seconds_to_interval)
            and float(seconds_to_interval)
            <= ZERO_FLOW_TIME_TOLERANCE_SECONDS
        ):
            final_status = (
                "provisional_negative_zero_flow_tolerance"
            )

            recommended_label = 0
            evidence_level = "medium_high"

            download_recommendation = (
                "download_now_negative"
            )

            decision_reason = (
                "The acquisition is within "
                f"{ZERO_FLOW_TIME_TOLERANCE_SECONDS:.0f} seconds "
                "of a measured zero-flow interval. It is treated "
                "as a provisional negative pending image-quality "
                "inspection."
            )

        elif release_status == "overlap_missing_rate":
            final_status = "ambiguous_overlapping_missing_rate"
            evidence_level = "ambiguous"

            download_recommendation = (
                "do_not_download_for_now"
            )

            decision_reason = (
                "The acquisition overlaps a release interval, "
                "but the methane-flow value is missing."
            )

        elif release_status == (
            "same_day_but_no_exact_overlap"
        ):
            final_status = (
                "ambiguous_same_day_interval_gap"
            )

            evidence_level = "ambiguous"

            download_recommendation = (
                "do_not_download_for_now"
            )

            decision_reason = (
                "A release interval exists on the same date, "
                "but it does not overlap the acquisition closely "
                "enough to assign a reliable label."
            )

        elif release_status == "no_release_log_for_date":
            if candidate_time < log_start:
                final_status = (
                    "pre_log_window_needs_campaign_confirmation"
                )

                decision_reason = (
                    "The acquisition occurred before the first "
                    "available release-log interval. Confirm the "
                    "official campaign start date before assigning "
                    "Label 0."
                )

            elif candidate_time > log_end:
                final_status = (
                    "post_log_window_needs_campaign_confirmation"
                )

                decision_reason = (
                    "The acquisition occurred after the final "
                    "available release-log interval. Confirm the "
                    "official campaign end date before assigning "
                    "Label 0."
                )

            else:
                final_status = (
                    "inside_log_window_no_record_needs_confirmation"
                )

                decision_reason = (
                    "The acquisition falls within the overall "
                    "release-log date range, but no interval exists "
                    "on that date. Campaign schedule evidence is "
                    "required before assigning Label 0."
                )

            evidence_level = "pending_schedule"

            download_recommendation = (
                "wait_for_campaign_schedule_check"
            )

        else:
            final_status = "unresolved"
            evidence_level = "ambiguous"

            download_recommendation = (
                "do_not_download_for_now"
            )

            decision_reason = (
                "The release-log evidence did not match a known "
                "decision rule."
            )

        output_row = row.to_dict()

        output_row.update({
            "release_log_start_utc": log_start,
            "release_log_end_utc": log_end,
            "final_candidate_status": final_status,
            "final_recommended_label": recommended_label,
            "evidence_level": evidence_level,
            "download_recommendation":
                download_recommendation,
            "decision_reason": decision_reason,
        })

        decision_rows.append(output_row)

    decision = pd.DataFrame(
        decision_rows
    )

    decision = decision.sort_values(
        "candidate_time_utc"
    ).reset_index(drop=True)

    download_now = decision[
        decision["download_recommendation"]
        == "download_now_negative"
    ].copy()

    schedule_check = decision[
        decision["download_recommendation"]
        == "wait_for_campaign_schedule_check"
    ].copy()

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    decision.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    download_now.to_csv(
        DOWNLOAD_NOW_CSV,
        index=False,
    )

    schedule_check.to_csv(
        SCHEDULE_CHECK_CSV,
        index=False,
    )

    print("=" * 105)
    print("CASA GRANDE CANDIDATE DECISION TABLE")
    print("=" * 105)

    print(f"\nCasa Grande overpasses: {len(decision)}")
    print(
        f"Release-log range: "
        f"{log_start} to {log_end}"
    )

    print("\nFinal candidate-status counts:")
    print(
        decision[
            "final_candidate_status"
        ].value_counts()
    )

    print("\nDownload recommendations:")
    print(
        decision[
            "download_recommendation"
        ].value_counts()
    )

    print("\nRecommended-label counts:")
    print(
        decision[
            "final_recommended_label"
        ].value_counts(dropna=False)
        .sort_index()
    )

    print("\nDownload-now candidates:")

    if len(download_now) == 0:
        print("None")
    else:
        print(
            download_now[
                [
                    "overpass_id",
                    "candidate_time_utc",
                    "landsat_sensor",
                    "LANDSAT_PRODUCT_ID",
                    "CLOUD_COVER",
                    "nearest_ch4_kgh_mean",
                    "seconds_to_nearest_interval",
                    "final_candidate_status",
                    "final_recommended_label",
                ]
            ].to_string(index=False)
        )

    print("\nCampaign-schedule checks:")

    if len(schedule_check) == 0:
        print("None")
    else:
        print(
            schedule_check[
                [
                    "overpass_id",
                    "candidate_time_utc",
                    "landsat_sensor",
                    "LANDSAT_PRODUCT_ID",
                    "CLOUD_COVER",
                    "final_candidate_status",
                ]
            ].to_string(index=False)
        )

    print("\nSaved:")
    print(OUTPUT_CSV)
    print(DOWNLOAD_NOW_CSV)
    print(SCHEDULE_CHECK_CSV)


if __name__ == "__main__":
    main()
