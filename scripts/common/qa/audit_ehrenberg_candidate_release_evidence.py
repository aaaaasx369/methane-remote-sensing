from pathlib import Path
import re

import numpy as np
import pandas as pd


CANDIDATE_CSV = Path(
    "outputs/67_landsat_unique_candidate_overpasses.csv"
)

SEARCH_ROOTS = [
    Path(
        "raw_data/2023_Controlled_Release_2021"
    ),
    Path(
        "raw_data/2023_SatelliteTesting/OLD/"
        "Controlled_Release_2021_main"
    ),
]

FILE_INVENTORY_CSV = Path(
    "outputs/84_ehrenberg_release_file_inventory.csv"
)

EVIDENCE_CSV = Path(
    "outputs/85_ehrenberg_candidate_release_evidence.csv"
)

SUMMARY_CSV = Path(
    "outputs/86_ehrenberg_candidate_release_summary.csv"
)


TIME_COLUMN_KEYWORDS = [
    "timestamp",
    "datetime",
    "date_time",
    "operator_timestamp",
    "stanford_timestamp",
    "surveytime",
    "time",
    "date",
    "cr_start",
    "starttime",
    "endtime",
]

FLOW_COLUMN_KEYWORDS = [
    "scfh",
    "flow",
    "kgh",
    "kg_h",
    "release_rate",
    "ch4",
    "cr_allmeters",
    "cr_quad",
]


# Landsat 時間附近多少小時內的紀錄要保留。
NEARBY_HOURS = 2

# 多少分鐘內視為非常接近 Landsat 過境。
CLOSE_MINUTES = 10


def normalize_column(column):
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def find_columns(columns, keywords):
    selected = []

    for column in columns:
        normalized = normalize_column(column)

        if any(
            keyword in normalized
            for keyword in keywords
        ):
            selected.append(column)

    return selected


def parse_datetime_best(series):
    """
    嘗試不同日期格式，選出能正確解析最多 2021 年資料的方法。
    """
    attempts = []

    for dayfirst in [False, True]:
        parsed = pd.to_datetime(
            series,
            errors="coerce",
            utc=True,
            dayfirst=dayfirst,
        )

        reasonable = (
            parsed.dt.year.between(
                2020,
                2023,
                inclusive="both",
            )
        )

        score = int(
            reasonable.fillna(False).sum()
        )

        attempts.append(
            (
                score,
                dayfirst,
                parsed,
            )
        )

    attempts.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    score, dayfirst, parsed = attempts[0]

    return parsed, score, dayfirst


def extract_filename_dates(path):
    """
    從檔名找出像 211021、20211021 之類的日期。
    """
    text = str(path)

    found_dates = []

    patterns = [
        r"(?<!\d)(20\d{6})(?!\d)",
        r"(?<!\d)(\d{6})(?!\d)",
    ]

    for pattern in patterns:
        for match in re.findall(
            pattern,
            text,
        ):
            formats = (
                ["%Y%m%d"]
                if len(match) == 8
                else ["%y%m%d"]
            )

            for date_format in formats:
                parsed = pd.to_datetime(
                    match,
                    format=date_format,
                    errors="coerce",
                    utc=True,
                )

                if pd.notna(parsed):
                    found_dates.append(
                        parsed.strftime(
                            "%Y-%m-%d"
                        )
                    )

    return sorted(
        set(found_dates)
    )


def inspect_csv(path):
    try:
        df = pd.read_csv(
            path,
            low_memory=False,
        )

    except Exception as error:
        return {
            "status": "read_error",
            "error": str(error),
            "rows": np.nan,
            "columns": "",
            "time_columns": "",
            "flow_columns": "",
            "parsed_min_time": pd.NaT,
            "parsed_max_time": pd.NaT,
            "filename_dates": " | ".join(
                extract_filename_dates(path)
            ),
            "dataframe": None,
            "parsed_time_data": {},
        }

    time_columns = find_columns(
        df.columns,
        TIME_COLUMN_KEYWORDS,
    )

    flow_columns = find_columns(
        df.columns,
        FLOW_COLUMN_KEYWORDS,
    )

    parsed_time_data = {}

    minimum_times = []
    maximum_times = []

    for column in time_columns:
        parsed, score, dayfirst = (
            parse_datetime_best(
                df[column]
            )
        )

        if score == 0:
            continue

        parsed_time_data[column] = {
            "parsed": parsed,
            "score": score,
            "dayfirst": dayfirst,
        }

        valid = parsed.dropna()

        if len(valid) > 0:
            minimum_times.append(
                valid.min()
            )

            maximum_times.append(
                valid.max()
            )

    return {
        "status": "success",
        "error": "",
        "rows": len(df),
        "columns": " | ".join(
            map(str, df.columns)
        ),
        "time_columns": " | ".join(
            map(str, time_columns)
        ),
        "flow_columns": " | ".join(
            map(str, flow_columns)
        ),
        "parsed_min_time": (
            min(minimum_times)
            if minimum_times
            else pd.NaT
        ),
        "parsed_max_time": (
            max(maximum_times)
            if maximum_times
            else pd.NaT
        ),
        "filename_dates": " | ".join(
            extract_filename_dates(path)
        ),
        "dataframe": df,
        "parsed_time_data": parsed_time_data,
        "flow_column_list": flow_columns,
    }


def summarize_flow_values(row, flow_columns):
    values = {}

    for column in flow_columns:
        numeric = pd.to_numeric(
            pd.Series([row.get(column)]),
            errors="coerce",
        ).iloc[0]

        if pd.notna(numeric):
            values[column] = float(numeric)

    return values


def classify_candidate_evidence(
    candidate_time,
    evidence_rows,
    files_covering_date,
):
    """
    這裡只做保守分類，不會自動把沒有資料的日期當 Label 0。
    """
    if len(evidence_rows) == 0:
        if files_covering_date > 0:
            return {
                "evidence_status":
                    "date_has_files_but_no_nearby_measurement",
                "recommended_label":
                    np.nan,
                "negative_candidate_eligible":
                    False,
                "reason":
                    "One or more source files cover this date, "
                    "but no usable flow measurement was found "
                    "near the Landsat acquisition time.",
            }

        return {
            "evidence_status":
                "no_release_evidence_for_date",
            "recommended_label":
                np.nan,
            "negative_candidate_eligible":
                False,
            "reason":
                "No release-data file or nearby flow record "
                "was found for this date. Campaign schedule "
                "confirmation is still required.",
        }

    positive_close = False
    zero_before = False
    zero_after = False
    any_close_flow = False

    for evidence in evidence_rows:
        seconds = evidence.get(
            "seconds_from_landsat"
        )

        if pd.isna(seconds):
            continue

        close = (
            abs(float(seconds))
            <= CLOSE_MINUTES * 60
        )

        if not close:
            continue

        flow_values = evidence.get(
            "flow_values_dictionary",
            {},
        )

        if not flow_values:
            continue

        any_close_flow = True

        values = list(
            flow_values.values()
        )

        if any(
            value > 0
            for value in values
        ):
            positive_close = True

        if all(
            value == 0
            for value in values
        ):
            if seconds <= 0:
                zero_before = True

            if seconds >= 0:
                zero_after = True

    if positive_close:
        return {
            "evidence_status":
                "confirmed_positive_near_overpass",
            "recommended_label": 1,
            "negative_candidate_eligible":
                False,
            "reason":
                "A positive methane-flow measurement exists "
                "within 10 minutes of the Landsat acquisition.",
        }

    if zero_before and zero_after:
        return {
            "evidence_status":
                "confirmed_zero_flow_bracket",
            "recommended_label": 0,
            "negative_candidate_eligible":
                True,
            "reason":
                "Zero-flow measurements bracket the Landsat "
                "acquisition within 10 minutes.",
        }

    if any_close_flow:
        return {
            "evidence_status":
                "nearby_flow_evidence_not_conclusive",
            "recommended_label": np.nan,
            "negative_candidate_eligible":
                False,
            "reason":
                "Flow measurements exist near the Landsat "
                "acquisition, but they do not provide a clear "
                "positive or zero-flow bracket.",
        }

    return {
        "evidence_status":
            "date_has_release_records_but_not_near_overpass",
        "recommended_label": np.nan,
        "negative_candidate_eligible":
            False,
        "reason":
            "Release records exist for the date, but none is "
            "close enough to the Landsat acquisition time.",
    }


def main():
    if not CANDIDATE_CSV.exists():
        raise FileNotFoundError(
            f"Candidate file not found: "
            f"{CANDIDATE_CSV}"
        )

    candidates = pd.read_csv(
        CANDIDATE_CSV,
        low_memory=False,
    )

    candidates[
        "candidate_time_utc"
    ] = pd.to_datetime(
        candidates["candidate_time_utc"],
        errors="coerce",
        utc=True,
    )

    ehrenberg = candidates[
        candidates["site_key"]
        == "ehrenberg"
    ].copy()

    ehrenberg = ehrenberg.sort_values(
        "candidate_time_utc"
    ).reset_index(drop=True)

    print("=" * 105)
    print("EHRENBERG CANDIDATE RELEASE-EVIDENCE AUDIT")
    print("=" * 105)

    print(
        f"\nEhrenberg candidate overpasses: "
        f"{len(ehrenberg)}"
    )

    print("\nCandidate dates:")

    print(
        ehrenberg[
            [
                "overpass_id",
                "candidate_time_utc",
                "landsat_sensor",
                "LANDSAT_PRODUCT_ID",
                "CLOUD_COVER",
            ]
        ].to_string(index=False)
    )

    csv_files = []

    for root in SEARCH_ROOTS:
        if root.exists():
            csv_files.extend(
                root.rglob("*.csv")
            )

    csv_files = sorted(
        set(csv_files)
    )

    print(
        f"\nCSV files scanned: "
        f"{len(csv_files)}"
    )

    inventory_rows = []
    inspected_files = []

    for file_number, path in enumerate(
        csv_files,
        start=1,
    ):
        result = inspect_csv(path)

        inventory_rows.append({
            "file_path": str(path),
            "status": result["status"],
            "rows": result["rows"],
            "columns": result["columns"],
            "time_columns":
                result["time_columns"],
            "flow_columns":
                result["flow_columns"],
            "parsed_min_time":
                result["parsed_min_time"],
            "parsed_max_time":
                result["parsed_max_time"],
            "filename_dates":
                result["filename_dates"],
            "error": result["error"],
        })

        if (
            result["status"] == "success"
            and result["dataframe"] is not None
            and len(
                result["parsed_time_data"]
            ) > 0
        ):
            inspected_files.append(
                (
                    path,
                    result,
                )
            )

    inventory_df = pd.DataFrame(
        inventory_rows
    )

    inventory_df.to_csv(
        FILE_INVENTORY_CSV,
        index=False,
    )

    evidence_output_rows = []
    summary_rows = []

    for _, candidate in ehrenberg.iterrows():
        candidate_time = candidate[
            "candidate_time_utc"
        ]

        candidate_date = (
            candidate_time.strftime(
                "%Y-%m-%d"
            )
        )

        candidate_evidence = []
        files_covering_date = 0

        print("\n" + "-" * 105)

        print(
            f"{candidate['overpass_id']} | "
            f"{candidate_time}"
        )

        for path, result in inspected_files:
            df = result["dataframe"]
            flow_columns = result[
                "flow_column_list"
            ]

            file_has_candidate_date = False

            for (
                time_column,
                time_information,
            ) in result[
                "parsed_time_data"
            ].items():
                parsed_times = time_information[
                    "parsed"
                ]

                same_date_mask = (
                    parsed_times.dt.strftime(
                        "%Y-%m-%d"
                    )
                    == candidate_date
                )

                if same_date_mask.any():
                    file_has_candidate_date = True

                seconds_from_landsat = (
                    parsed_times
                    - candidate_time
                ).dt.total_seconds()

                nearby_mask = (
                    seconds_from_landsat.abs()
                    <= NEARBY_HOURS * 3600
                )

                nearby_indices = df.index[
                    nearby_mask.fillna(False)
                ]

                for row_index in nearby_indices:
                    source_row = df.loc[
                        row_index
                    ]

                    parsed_time = (
                        parsed_times.loc[
                            row_index
                        ]
                    )

                    seconds_difference = (
                        seconds_from_landsat.loc[
                            row_index
                        ]
                    )

                    flow_values = (
                        summarize_flow_values(
                            source_row,
                            flow_columns,
                        )
                    )

                    evidence_row = {
                        "overpass_id":
                            candidate[
                                "overpass_id"
                            ],
                        "candidate_time_utc":
                            candidate_time,
                        "candidate_date_utc":
                            candidate_date,
                        "landsat_sensor":
                            candidate[
                                "landsat_sensor"
                            ],
                        "LANDSAT_PRODUCT_ID":
                            candidate[
                                "LANDSAT_PRODUCT_ID"
                            ],
                        "source_file":
                            str(path),
                        "source_row_index":
                            row_index,
                        "time_column":
                            time_column,
                        "parsed_source_time_utc":
                            parsed_time,
                        "seconds_from_landsat":
                            seconds_difference,
                        "minutes_from_landsat":
                            (
                                seconds_difference
                                / 60
                            ),
                        "flow_column_count":
                            len(flow_values),
                        "flow_values":
                            " | ".join(
                                f"{key}={value}"
                                for key, value
                                in flow_values.items()
                            ),
                        "flow_values_dictionary":
                            flow_values,
                    }

                    candidate_evidence.append(
                        evidence_row
                    )

            if file_has_candidate_date:
                files_covering_date += 1

        candidate_evidence = sorted(
            candidate_evidence,
            key=lambda row: abs(
                row[
                    "seconds_from_landsat"
                ]
            ),
        )

        decision = classify_candidate_evidence(
            candidate_time,
            candidate_evidence,
            files_covering_date,
        )

        summary_rows.append({
            "overpass_id":
                candidate["overpass_id"],
            "candidate_time_utc":
                candidate_time,
            "candidate_date_utc":
                candidate_date,
            "landsat_sensor":
                candidate[
                    "landsat_sensor"
                ],
            "LANDSAT_PRODUCT_ID":
                candidate[
                    "LANDSAT_PRODUCT_ID"
                ],
            "CLOUD_COVER":
                candidate.get(
                    "CLOUD_COVER"
                ),
            "files_covering_date":
                files_covering_date,
            "nearby_evidence_rows":
                len(candidate_evidence),
            **decision,
        })

        for evidence in candidate_evidence:
            output_row = dict(
                evidence
            )

            # dictionary 不能直接方便存進 CSV，
            # 所以輸出前移除。
            output_row.pop(
                "flow_values_dictionary",
                None,
            )

            evidence_output_rows.append(
                output_row
            )

        print(
            f"Files covering date: "
            f"{files_covering_date}"
        )

        print(
            f"Nearby evidence rows: "
            f"{len(candidate_evidence)}"
        )

        print(
            f"Status: "
            f"{decision['evidence_status']}"
        )

        if len(candidate_evidence) > 0:
            print(
                "\nNearest evidence:"
            )

            nearest_rows = pd.DataFrame(
                [
                    {
                        key: value
                        for key, value
                        in evidence.items()
                        if key
                        != "flow_values_dictionary"
                    }
                    for evidence
                    in candidate_evidence[:10]
                ]
            )

            print(
                nearest_rows[
                    [
                        "source_file",
                        "time_column",
                        "parsed_source_time_utc",
                        "minutes_from_landsat",
                        "flow_values",
                    ]
                ].to_string(
                    index=False
                )
            )

    evidence_df = pd.DataFrame(
        evidence_output_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    evidence_df.to_csv(
        EVIDENCE_CSV,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_CSV,
        index=False,
    )

    print("\n" + "=" * 105)
    print("EHRENBERG RELEASE-EVIDENCE SUMMARY")
    print("=" * 105)

    print("\nEvidence-status counts:")

    print(
        summary_df[
            "evidence_status"
        ].value_counts()
    )

    print("\nRecommended-label counts:")

    print(
        summary_df[
            "recommended_label"
        ].value_counts(
            dropna=False
        ).sort_index()
    )

    print(
        "\nNegative candidate eligibility:"
    )

    print(
        summary_df[
            "negative_candidate_eligible"
        ].value_counts()
    )

    print("\nComplete candidate summary:")

    print(
        summary_df[
            [
                "overpass_id",
                "candidate_time_utc",
                "landsat_sensor",
                "CLOUD_COVER",
                "files_covering_date",
                "nearby_evidence_rows",
                "evidence_status",
                "recommended_label",
                "negative_candidate_eligible",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(FILE_INVENTORY_CSV)
    print(EVIDENCE_CSV)
    print(SUMMARY_CSV)


if __name__ == "__main__":
    main()
