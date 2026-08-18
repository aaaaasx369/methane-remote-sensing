from pathlib import Path
import json

import pandas as pd


INPUT = Path(
    "outputs/129_evanston_landsat_release_candidates.csv"
)

OVERPASS_OUTPUT = Path(
    "outputs/131_evanston_unique_landsat_overpasses.csv"
)

GEE_JS_OUTPUT = Path(
    "outputs/132_search_evanston_landsat_scenes.js"
)


COLLECTIONS = {
    "Landsat-8": "LANDSAT/LC08/C02/T1_L2",
    "Landsat-9": "LANDSAT/LC09/C02/T1_L2",
}


def javascript_string(value):
    return json.dumps(
        str(value),
        ensure_ascii=False,
    )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(
        INPUT,
        low_memory=False,
    )

    required = [
        "release_ID",
        "landsat_sensor",
        "acquisition_date",
        "acquisition_time_utc",
        "lat",
        "lon",
        "ch4_kgh_mean",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing columns: {missing}"
        )

    df["ch4_kgh_mean"] = pd.to_numeric(
        df["ch4_kgh_mean"],
        errors="coerce",
    )

    df["lat"] = pd.to_numeric(
        df["lat"],
        errors="coerce",
    )

    df["lon"] = pd.to_numeric(
        df["lon"],
        errors="coerce",
    )

    df["acquisition_time_utc"] = pd.to_datetime(
        df["acquisition_time_utc"],
        errors="coerce",
        utc=True,
    )

    df = df.dropna(
        subset=[
            "acquisition_date",
            "landsat_sensor",
            "lat",
            "lon",
            "ch4_kgh_mean",
        ]
    ).copy()

    group_columns = [
        "acquisition_date",
        "landsat_sensor",
    ]

    rows = []

    for group_values, group in df.groupby(
        group_columns,
        dropna=False,
    ):
        acquisition_date, sensor = group_values

        group = group.sort_values(
            "ch4_kgh_mean",
            ascending=False,
        )

        representative = group.iloc[0]

        release_ids = "|".join(
            sorted(
                group["release_ID"]
                .astype(str)
                .unique()
            )
        )

        rows.append({
            "overpass_id": "",
            "site_key": "evanston",
            "acquisition_date":
                acquisition_date,
            "landsat_sensor":
                sensor,
            "collection_id":
                COLLECTIONS.get(
                    sensor,
                    "",
                ),
            "latitude":
                float(
                    group["lat"].median()
                ),
            "longitude":
                float(
                    group["lon"].median()
                ),
            "release_rows":
                len(group),
            "release_time_min_utc":
                group[
                    "acquisition_time_utc"
                ].min(),
            "release_time_max_utc":
                group[
                    "acquisition_time_utc"
                ].max(),
            "flow_min_kg_h":
                float(
                    group[
                        "ch4_kgh_mean"
                    ].min()
                ),
            "flow_median_kg_h":
                float(
                    group[
                        "ch4_kgh_mean"
                    ].median()
                ),
            "flow_max_kg_h":
                float(
                    group[
                        "ch4_kgh_mean"
                    ].max()
                ),
            "representative_release_id":
                representative[
                    "release_ID"
                ],
            "all_release_ids":
                release_ids,
        })

    overpasses = pd.DataFrame(rows)

    overpasses = overpasses.sort_values(
        [
            "acquisition_date",
            "landsat_sensor",
        ]
    ).reset_index(drop=True)

    overpasses["overpass_id"] = [
        f"EV_OP_{number:03d}"
        for number in range(
            1,
            len(overpasses) + 1,
        )
    ]

    overpasses.to_csv(
        OVERPASS_OUTPUT,
        index=False,
    )

    javascript_lines = [
        "// Auto-generated Evanston Landsat scene search",
        "// Paste this file into the Google Earth Engine Code Editor.",
        "",
        "var allResults = ee.FeatureCollection([]);",
        "",
    ]

    for _, row in overpasses.iterrows():
        if not row["collection_id"]:
            continue

        overpass_id = str(
            row["overpass_id"]
        )

        variable_name = (
            overpass_id.lower()
        )

        date_string = str(
            row["acquisition_date"]
        )

        longitude = float(
            row["longitude"]
        )

        latitude = float(
            row["latitude"]
        )

        collection_id = str(
            row["collection_id"]
        )

        javascript_lines.extend([
            f"// {overpass_id}",
            (
                f"var point_{variable_name} = "
                f"ee.Geometry.Point("
                f"[{longitude}, {latitude}]);"
            ),
            (
                f"var start_{variable_name} = "
                f"ee.Date('{date_string}')"
                f".advance(-1, 'day');"
            ),
            (
                f"var end_{variable_name} = "
                f"ee.Date('{date_string}')"
                f".advance(2, 'day');"
            ),
            (
                f"var collection_{variable_name} = "
                f"ee.ImageCollection("
                f"'{collection_id}')"
            ),
            (
                f"  .filterBounds("
                f"point_{variable_name})"
            ),
            (
                f"  .filterDate("
                f"start_{variable_name}, "
                f"end_{variable_name});"
            ),
            "",
            (
                f"var features_{variable_name} = "
                f"collection_{variable_name}"
                f".map(function(image) {{"
            ),
            "  return ee.Feature(null, {",
            (
                "    overpass_id: "
                f"{javascript_string(overpass_id)},"
            ),
            (
                "    release_date: "
                f"{javascript_string(date_string)},"
            ),
            (
                "    expected_sensor: "
                f"{javascript_string(row['landsat_sensor'])},"
            ),
            (
                "    flow_max_kg_h: "
                f"{float(row['flow_max_kg_h'])},"
            ),
            (
                "    release_rows: "
                f"{int(row['release_rows'])},"
            ),
            (
                "    representative_release_id: "
                f"{javascript_string(row['representative_release_id'])},"
            ),
            "    system_index: image.get('system:index'),",
            "    landsat_product_id: image.get('LANDSAT_PRODUCT_ID'),",
            "    spacecraft_id: image.get('SPACECRAFT_ID'),",
            "    acquisition_time_utc: ee.Date(",
            "      image.get('system:time_start')",
            "    ).format('YYYY-MM-dd HH:mm:ss'),",
            "    cloud_cover: image.get('CLOUD_COVER'),",
            "    wrs_path: image.get('WRS_PATH'),",
            "    wrs_row: image.get('WRS_ROW'),",
            "    collection_category: image.get('COLLECTION_CATEGORY'),",
            "    collection_number: image.get('COLLECTION_NUMBER')",
            "  });",
            "});",
            "",
            (
                "allResults = allResults.merge("
                f"features_{variable_name});"
            ),
            "",
        ])

    javascript_lines.extend([
        "print('Evanston Landsat scene candidates', allResults);",
        "print('Candidate count', allResults.size());",
        "",
        "Export.table.toDrive({",
        "  collection: allResults,",
        "  description: 'evanston_landsat_scene_candidates',",
        "  fileNamePrefix: 'evanston_landsat_scene_candidates',",
        "  fileFormat: 'CSV'",
        "});",
        "",
    ])

    GEE_JS_OUTPUT.write_text(
        "\n".join(javascript_lines),
        encoding="utf-8",
    )

    print("=" * 100)
    print("EVANSTON UNIQUE LANDSAT OVERPASSES")
    print("=" * 100)

    print("\nUnique overpasses:", len(overpasses))

    print("\nBy sensor:")
    print(
        overpasses[
            "landsat_sensor"
        ].value_counts()
    )

    print("\nOverpasses:")
    print(
        overpasses[
            [
                "overpass_id",
                "acquisition_date",
                "landsat_sensor",
                "release_rows",
                "flow_max_kg_h",
                "representative_release_id",
            ]
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.2f}",
        )
    )

    print("\nSaved:")
    print(OVERPASS_OUTPUT)
    print(GEE_JS_OUTPUT)


if __name__ == "__main__":
    main()
