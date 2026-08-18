from pathlib import Path
import os
import json

import pandas as pd


ROOT = Path(
    "/Users/happydoraaa/methane_release_project/"
    "external_data/methaneair_controlled_release"
)

OUTPUT = (
    Path("/Users/happydoraaa/methane_release_project/")
    / "outputs/56_downloaded_ground_truth_inventory.txt"
)

KEYWORDS = [
    "ground",
    "truth",
    "release",
    "controlled",
    "flow",
    "rate",
    "emission",
    "meter",
    "source",
    "on",
    "off",
    "start",
    "end",
    "time",
    "utc",
    "kg",
    "latitude",
    "longitude",
    "lat",
    "lon",
    "segment",
]


def human_size(size):
    size = float(size)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} PB"


def is_candidate_name(name):
    lower = str(name).lower()

    return any(
        keyword in lower
        for keyword in KEYWORDS
    )


lines = []


def write(text=""):
    text = str(text)
    print(text)
    lines.append(text)


write("=" * 120)
write("DOWNLOADED METHANEAIR / CONTROLLED-RELEASE DATA INVENTORY")
write("=" * 120)
write(f"Root: {ROOT}")
write()

if not ROOT.exists():
    raise SystemExit(f"Folder does not exist: {ROOT}")


# ============================================================================
# 1. 全部檔案清單
# ============================================================================

all_files = sorted(
    path
    for path in ROOT.rglob("*")
    if path.is_file()
)

write("ALL FILES")
write("-" * 120)
write(f"Total files: {len(all_files)}")

for path in all_files:
    relative = path.relative_to(ROOT)

    try:
        size = human_size(path.stat().st_size)
    except Exception:
        size = "UNKNOWN"

    write(f"{size:>12}  {relative}")

write()


# ============================================================================
# 2. CSV 檢查
# ============================================================================

csv_files = sorted(ROOT.rglob("*.csv"))

write("=" * 120)
write("CSV TABLES")
write("=" * 120)
write(f"CSV files: {len(csv_files)}")

for path in csv_files:
    write()
    write("#" * 120)
    write(f"CSV FILE: {path.relative_to(ROOT)}")
    write("#" * 120)

    try:
        try:
            df = pd.read_csv(
                path,
                low_memory=False,
            )
        except Exception:
            df = pd.read_csv(
                path,
                sep=None,
                engine="python",
            )

        columns = [str(column) for column in df.columns]

        candidate_columns = [
            column
            for column in columns
            if is_candidate_name(column)
        ]

        write(f"Rows: {len(df)}")
        write(f"Columns: {len(columns)}")

        write()
        write("ALL COLUMN NAMES:")
        for index, column in enumerate(columns, start=1):
            write(f"{index:3d}. {column}")

        write()
        write("POSSIBLE GROUND-TRUTH / TIME / LOCATION COLUMNS:")

        if candidate_columns:
            for column in candidate_columns:
                series = df[column]

                write()
                write(f"  COLUMN: {column}")
                write(f"  dtype: {series.dtype}")
                write(f"  non-null: {series.notna().sum()} / {len(series)}")
                write(f"  unique: {series.nunique(dropna=True)}")

                samples = (
                    series.dropna()
                    .astype(str)
                    .drop_duplicates()
                    .head(12)
                    .tolist()
                )

                write(f"  examples: {samples}")
        else:
            write("  NONE FOUND BY COLUMN NAME")

        write()
        write("FIRST 10 ROWS:")

        if len(df) == 0:
            write("EMPTY TABLE")
        else:
            display_columns = (
                candidate_columns
                if candidate_columns
                else columns[:15]
            )

            # 防止一行太寬，最多顯示 20 欄
            display_columns = display_columns[:20]

            write(
                df[display_columns]
                .head(10)
                .to_string(index=False)
            )

        # 搜尋字串欄位內容
        content_hits = []

        for column in columns:
            series = df[column]

            if series.dtype != object:
                continue

            text = (
                series.dropna()
                .astype(str)
                .head(10000)
                .str.lower()
            )

            pattern = (
                r"ground.?truth|"
                r"controlled.?release|"
                r"release.?rate|"
                r"flow.?rate|"
                r"metered|"
                r"emission.?rate|"
                r"source.?on|"
                r"source.?off|"
                r"kg.?/?h"
            )

            if text.str.contains(
                pattern,
                regex=True,
                na=False,
            ).any():
                content_hits.append(column)

        write()
        write("COLUMNS WHOSE CONTENT CONTAINS GROUND-TRUTH KEYWORDS:")
        write(content_hits if content_hits else "NONE")

    except Exception as exc:
        write(
            f"ERROR: {type(exc).__name__}: {exc}"
        )


# ============================================================================
# 3. Excel 檢查
# ============================================================================

excel_files = sorted(
    list(ROOT.rglob("*.xlsx"))
    + list(ROOT.rglob("*.xls"))
)

write()
write("=" * 120)
write("EXCEL TABLES")
write("=" * 120)
write(f"Excel files: {len(excel_files)}")

for path in excel_files:
    write()
    write("#" * 120)
    write(f"EXCEL FILE: {path.relative_to(ROOT)}")
    write("#" * 120)

    try:
        xls = pd.ExcelFile(path)

        write(f"Sheets: {xls.sheet_names}")

        for sheet in xls.sheet_names:
            df = pd.read_excel(
                path,
                sheet_name=sheet,
            )

            columns = [str(column) for column in df.columns]

            candidate_columns = [
                column
                for column in columns
                if is_candidate_name(column)
            ]

            write()
            write(f"SHEET: {sheet}")
            write(f"Rows: {len(df)}")
            write(f"Columns: {columns}")
            write(
                "Candidate columns: "
                + str(candidate_columns)
            )

            display_columns = (
                candidate_columns
                if candidate_columns
                else columns[:15]
            )

            if display_columns:
                write(
                    df[display_columns]
                    .head(10)
                    .to_string(index=False)
                )

    except Exception as exc:
        write(
            f"ERROR: {type(exc).__name__}: {exc}"
        )


# ============================================================================
# 4. NetCDF metadata 檢查，不讀取大型陣列
# ============================================================================

nc_files = sorted(ROOT.rglob("*.nc"))

write()
write("=" * 120)
write("NETCDF FILES")
write("=" * 120)
write(f"NetCDF files: {len(nc_files)}")

try:
    from netCDF4 import Dataset
except ImportError:
    write(
        "netCDF4 is not installed. Run: "
        "python -m pip install netCDF4"
    )
    Dataset = None

if Dataset is not None:
    for path in nc_files:
        write()
        write("#" * 120)
        write(f"NETCDF FILE: {path.relative_to(ROOT)}")
        write(f"Size: {human_size(path.stat().st_size)}")
        write("#" * 120)

        try:
            ds = Dataset(path, mode="r")

            write("GLOBAL ATTRIBUTES:")

            for attr in ds.ncattrs():
                try:
                    value = ds.getncattr(attr)
                except Exception:
                    value = "<unreadable>"

                write(f"  {attr}: {value}")

            write()
            write("DIMENSIONS:")

            for name, dimension in ds.dimensions.items():
                write(
                    f"  {name}: {len(dimension)}"
                )

            write()
            write("VARIABLES:")

            candidate_variables = []

            for name, variable in ds.variables.items():
                attrs = {}

                for attr in variable.ncattrs():
                    try:
                        attrs[attr] = variable.getncattr(attr)
                    except Exception:
                        attrs[attr] = "<unreadable>"

                write()
                write(f"  VARIABLE: {name}")
                write(f"  dimensions: {variable.dimensions}")
                write(f"  shape: {variable.shape}")
                write(f"  dtype: {variable.dtype}")

                if attrs:
                    write(
                        "  attributes: "
                        + json.dumps(
                            attrs,
                            ensure_ascii=False,
                            default=str,
                        )
                    )

                combined_text = (
                    name
                    + " "
                    + " ".join(
                        f"{key} {value}"
                        for key, value in attrs.items()
                    )
                ).lower()

                if any(
                    keyword in combined_text
                    for keyword in KEYWORDS
                ):
                    candidate_variables.append(name)

            write()
            write(
                "POSSIBLE GROUND-TRUTH / TIME / LOCATION VARIABLES:"
            )
            write(
                candidate_variables
                if candidate_variables
                else "NONE"
            )

            ds.close()

        except Exception as exc:
            write(
                f"ERROR: {type(exc).__name__}: {exc}"
            )


# ============================================================================
# 5. 儲存
# ============================================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT.write_text(
    "\n".join(lines),
    encoding="utf-8",
)

write()
write("=" * 120)
write("SAVED REPORT")
write("=" * 120)
write(OUTPUT)
