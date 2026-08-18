from pathlib import Path
import pandas as pd


PROJECT = Path(".")

SEARCH_TERMS = [
    "cal_temporal_z_source_p95_percentile",
    "calibrated_source_p95_percentile",
    "prediction_score",
    "0.559805",
    "site_calibrated",
]

TRAINING_OUTPUT = Path(
    "outputs/121_landsat_site_calibrated_loso_predictions.csv"
)

EXTERNAL_INDEX = Path(
    "outputs/234_marss2l_external_patch_index_v1_2.csv"
)


def search_python_files():
    matches = {}

    for path in PROJECT.rglob("*.py"):
        if any(
            part in {
                ".venv",
                "__pycache__",
                ".git",
            }
            for part in path.parts
        ):
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        found = [
            term
            for term in SEARCH_TERMS
            if term in text
        ]

        if found:
            matches[str(path)] = found

    return matches


def find_model_artifacts():
    extensions = {
        ".joblib",
        ".pkl",
        ".pickle",
        ".json",
    }

    artifacts = []

    for path in PROJECT.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower()
            in extensions
        ):
            if ".venv" not in path.parts:
                artifacts.append(path)

    return sorted(artifacts)


def inspect_csv(path):
    print("\n" + "=" * 110)
    print(path)
    print("=" * 110)

    if not path.exists():
        print("FILE NOT FOUND")
        return

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nColumn names:")
    for column in df.columns:
        print(column)

    print("\nFirst two rows:")
    print(
        df.head(2).to_string(
            index=False,
            max_colwidth=80,
        )
    )


def inspect_external_dataset():
    if not EXTERNAL_INDEX.exists():
        return

    df = pd.read_csv(
        EXTERNAL_INDEX,
        low_memory=False,
    )

    success = df[
        df["download_status"] == "success"
    ].copy()

    print("\n" + "=" * 110)
    print("MARS-S2L EXTERNAL DATASET CONTRACT")
    print("=" * 110)

    print("\nSuccessful rows:", len(success))
    print("Unique sites:", success["site_key"].nunique())

    print("\nRoles:")
    print(
        success["external_role"]
        .value_counts()
    )

    evaluation = success[
        success["external_role"].isin([
            "high_emission_positive",
            "test_negative",
        ])
    ]

    print("\nExternal evaluation rows:", len(evaluation))

    print("\nEvaluation labels:")
    print(
        evaluation["evaluation_label"]
        .value_counts()
        .sort_index()
    )

    print("\nSensors in evaluation set:")
    print(
        pd.crosstab(
            evaluation["external_role"],
            evaluation["sensor_code"],
            margins=True,
        )
    )


def main():
    print("=" * 110)
    print("FROZEN LANDSAT PIPELINE DISCOVERY")
    print("=" * 110)

    matches = search_python_files()

    print("\nPython files containing frozen-model terms:")

    if not matches:
        print("No matching Python files found.")
    else:
        for path, terms in matches.items():
            print(f"\n{path}")
            for term in terms:
                print(f"  - {term}")

    artifacts = find_model_artifacts()

    print("\n" + "=" * 110)
    print("POSSIBLE SAVED MODEL ARTIFACTS")
    print("=" * 110)

    if artifacts:
        for path in artifacts:
            print(path)
    else:
        print("No .joblib/.pkl/.pickle/.json model artifact found.")

    inspect_csv(TRAINING_OUTPUT)
    inspect_external_dataset()


if __name__ == "__main__":
    main()
