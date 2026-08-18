from pathlib import Path
import re

import pandas as pd


SEARCH_ROOTS = [
    Path("raw_data"),
    Path("outputs"),
]

OUTPUT_CSV = Path(
    "outputs/50_release_interval_source_candidates.csv"
)

SUPPORTED_SUFFIXES = {
    ".csv",
    ".xlsx",
    ".xls",
    ".parquet",
}


START_PATTERNS = [
    r"release.*start",
    r"start.*release",
    r"start.*time",
    r"start.*date",
    r"begin.*time",
    r"begin.*date",
    r"datetime.*start",
    r"time.*start",
]

END_PATTERNS = [
    r"release.*end",
    r"end.*release",
    r"end.*time",
    r"end.*date",
    r"stop.*time",
    r"stop.*date",
    r"datetime.*end",
    r"time.*end",
]

TIME_PATTERNS = [
    r"datetime",
    r"timestamp",
    r"time",
    r"date",
    r"utc",
]

RELEASE_PATTERNS = [
    r"release",
    r"emission",
    r"flow",
    r"methane",
    r"ch4",
]

SITE_PATTERNS = [
    r"site",
    r"location",
    r"latitude",
    r"longitude",
    r"\blat\b",
    r"\blon\b",
]


def normalize_column_name(column):
    return str(column).strip().lower()


def matches_any(column, patterns):
    normalized = normalize_column_name(column)

    return any(
        re.search(pattern, normalized)
        for pattern in patterns
    )


def detect_columns(columns):
    columns = list(columns)

    start_columns = [
        column
        for column in columns
        if matches_any(column, START_PATTERNS)
    ]

    end_columns = [
        column
        for column in columns
        if matches_any(column, END_PATTERNS)
    ]

    time_columns = [
        column
        for column in columns
        if matches_any(column, TIME_PATTERNS)
    ]

    release_columns = [
        column
        for column in columns
        if matches_any(column, RELEASE_PATTERNS)
    ]

    site_columns = [
        column
        for column in columns
        if matches_any(column, SITE_PATTERNS)
    ]

    return {
        "start_columns": start_columns,
        "end_columns": end_columns,
        "time_columns": time_columns,
        "release_columns": release_columns,
        "site_columns": site_columns,
    }


def example_values(df, columns, max_values=3):
    examples = {}

    for column in columns:
        values = (
            df[column]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(max_values)
            .tolist()
        )

        examples[column] = " | ".join(values)

    return examples


def inspect_dataframe(
    df,
    file_path,
    sheet_name="",
):
    detected = detect_columns(df.columns)

    # 至少要出現時間、釋放或場址相關欄位才記錄
    relevant = (
        detected["start_columns"]
        or detected["end_columns"]
        or detected["time_columns"]
        or detected["release_columns"]
    )

    if not relevant:
        return None

    candidate_columns = list(dict.fromkeys(
        detected["start_columns"]
        + detected["end_columns"]
        + detected["time_columns"]
        + detected["release_columns"]
        + detected["site_columns"]
    ))

    examples = example_values(
        df,
        candidate_columns,
    )

    has_start_and_end = bool(
        detected["start_columns"]
        and detected["end_columns"]
    )

    score = 0

    if detected["start_columns"]:
        score += 4

    if detected["end_columns"]:
        score += 4

    if detected["release_columns"]:
        score += 2

    if detected["site_columns"]:
        score += 1

    if has_start_and_end:
        score += 5

    return {
        "file_path": str(file_path),
        "sheet_name": sheet_name,
        "sampled_rows": len(df),
        "total_columns": len(df.columns),
        "candidate_score": score,
        "has_start_and_end": has_start_and_end,
        "start_columns": " | ".join(
            map(str, detected["start_columns"])
        ),
        "end_columns": " | ".join(
            map(str, detected["end_columns"])
        ),
        "time_columns": " | ".join(
            map(str, detected["time_columns"])
        ),
        "release_columns": " | ".join(
            map(str, detected["release_columns"])
        ),
        "site_columns": " | ".join(
            map(str, detected["site_columns"])
        ),
        "example_values": " || ".join(
            f"{column}: {value}"
            for column, value in examples.items()
        ),
    }


def inspect_file(path):
    rows = []

    try:
        suffix = path.suffix.lower()

        if suffix == ".csv":
            # 只讀前 500 列，先確認欄位與格式
            df = pd.read_csv(
                path,
                nrows=500,
                low_memory=False,
            )

            result = inspect_dataframe(
                df,
                path,
            )

            if result:
                rows.append(result)

        elif suffix in {".xlsx", ".xls"}:
            excel_file = pd.ExcelFile(path)

            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(
                        path,
                        sheet_name=sheet_name,
                        nrows=500,
                    )

                    result = inspect_dataframe(
                        df,
                        path,
                        sheet_name=sheet_name,
                    )

                    if result:
                        rows.append(result)

                except Exception as error:
                    print(
                        f"[SHEET ERROR] {path} | "
                        f"{sheet_name} | {error}"
                    )

        elif suffix == ".parquet":
            df = pd.read_parquet(path).head(500)

            result = inspect_dataframe(
                df,
                path,
            )

            if result:
                rows.append(result)

    except Exception as error:
        print(
            f"[FILE ERROR] {path} | {error}"
        )

    return rows


def main():
    print("=" * 90)
    print("SEARCH FOR CONTROLLED-RELEASE TIME INTERVAL SOURCES")
    print("=" * 90)

    files = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_SUFFIXES
            ):
                files.append(path)

    print(f"\nData files found: {len(files)}")

    result_rows = []

    for index, path in enumerate(
        sorted(files),
        start=1,
    ):
        print(
            f"[SCAN] {index:04d}/{len(files):04d} "
            f"{path}"
        )

        result_rows.extend(
            inspect_file(path)
        )

    result_df = pd.DataFrame(
        result_rows
    )

    if len(result_df) == 0:
        print(
            "\nNo candidate files were found."
        )

        return

    result_df = result_df.sort_values(
        by=[
            "has_start_and_end",
            "candidate_score",
            "file_path",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\n" + "=" * 90)
    print("TOP RELEASE-TIME SOURCE CANDIDATES")
    print("=" * 90)

    display_columns = [
        "candidate_score",
        "has_start_and_end",
        "file_path",
        "sheet_name",
        "start_columns",
        "end_columns",
        "time_columns",
        "release_columns",
        "site_columns",
    ]

    print(
        result_df[
            display_columns
        ].head(30).to_string(
            index=False
        )
    )

    print("\nFiles containing BOTH start and end candidates:")

    interval_df = result_df[
        result_df["has_start_and_end"]
    ]

    if len(interval_df) == 0:
        print("None found.")
    else:
        print(
            interval_df[
                [
                    "file_path",
                    "sheet_name",
                    "start_columns",
                    "end_columns",
                    "release_columns",
                    "site_columns",
                ]
            ].to_string(index=False)
        )

    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
