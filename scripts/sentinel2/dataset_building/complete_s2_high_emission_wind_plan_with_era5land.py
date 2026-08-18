from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import json
import math
import time

import numpy as np
import pandas as pd


POSITIVE_INPUT = Path(
    "outputs/355_s2_high_emission_positive_manifest_clean_v1.csv"
)

INTERNAL_PLAN_INPUT = Path(
    "outputs/377_s2_high_emission_wind_plan_v1.csv"
)

EXTERNAL_EVIDENCE_OUTPUT = Path(
    "outputs/382_s2_casa_grande_era5_wind_evidence_v1.csv"
)

COMPLETE_PLAN_OUTPUT = Path(
    "outputs/383_s2_high_emission_wind_plan_complete_v2.csv"
)

REPORT_OUTPUT = Path(
    "outputs/384_s2_high_emission_wind_plan_complete_report_v2.txt"
)

RAW_OUTPUT_DIR = Path(
    "outputs/s2_high_emission_era5_raw_v1"
)


API_ENDPOINT = (
    "https://archive-api.open-meteo.com/v1/archive"
)


def fetch_era5(
    latitude,
    longitude,
    target_time,
):
    target_time = pd.Timestamp(
        target_time
    ).tz_convert("UTC")

    # 前後各多抓一天，確保能取得包住拍攝時間的兩個整點。
    start_date = (
        target_time
        - pd.Timedelta(days=1)
    ).date().isoformat()

    end_date = (
        target_time
        + pd.Timedelta(days=1)
    ).date().isoformat()

    parameters = {
        "latitude":
            float(latitude),

        "longitude":
            float(longitude),

        "start_date":
            start_date,

        "end_date":
            end_date,

        "hourly":
            (
                "wind_speed_10m,"
                "wind_direction_10m"
            ),

        "wind_speed_unit":
            "ms",

        "timezone":
            "GMT",

        "models":
            "era5",

        "cell_selection":
            "nearest",
    }

    url = (
        API_ENDPOINT
        + "?"
        + urlencode(parameters)
    )

    last_error = None

    for attempt in range(1, 4):
        try:
            with urlopen(
                url,
                timeout=90,
            ) as response:
                payload = json.loads(
                    response.read().decode(
                        "utf-8"
                    )
                )

            return payload, url

        except Exception as error:
            last_error = error

            if attempt < 3:
                time.sleep(
                    3 * attempt
                )

    raise RuntimeError(
        "ERA5 request failed after "
        f"3 attempts: {last_error}"
    )


def build_hourly_table(payload):
    if "hourly" not in payload:
        raise KeyError(
            "API response does not contain hourly data."
        )

    hourly = payload["hourly"]

    required = [
        "time",
        "wind_speed_10m",
        "wind_direction_10m",
    ]

    missing = [
        name
        for name in required
        if name not in hourly
    ]

    if missing:
        raise KeyError(
            "Hourly response missing: "
            + ", ".join(missing)
        )

    table = pd.DataFrame({
        "time_utc":
            pd.to_datetime(
                hourly["time"],
                errors="coerce",
                utc=True,
            ),

        "wind_speed_m_s":
            pd.to_numeric(
                hourly[
                    "wind_speed_10m"
                ],
                errors="coerce",
            ),

        "wind_from_degrees":
            pd.to_numeric(
                hourly[
                    "wind_direction_10m"
                ],
                errors="coerce",
            ),
    })

    table = table.dropna(
        subset=[
            "time_utc",
            "wind_speed_m_s",
            "wind_from_degrees",
        ]
    ).sort_values(
        "time_utc"
    ).reset_index(drop=True)

    if table.empty:
        raise RuntimeError(
            "ERA5 returned no valid wind rows."
        )

    return table


def meteorological_to_vector(
    speed,
    wind_from_degrees,
):
    """
    Meteorological direction indicates where wind
    comes from.

    Convert to the toward/downwind vector:
      u: eastward component
      v: northward component
    """
    radians = math.radians(
        float(wind_from_degrees)
    )

    u_toward = (
        -float(speed)
        * math.sin(radians)
    )

    v_toward = (
        -float(speed)
        * math.cos(radians)
    )

    return u_toward, v_toward


def vector_to_downwind(
    u_toward,
    v_toward,
):
    speed = math.sqrt(
        u_toward ** 2
        + v_toward ** 2
    )

    downwind = (
        math.degrees(
            math.atan2(
                u_toward,
                v_toward,
            )
        )
        + 360.0
    ) % 360.0

    wind_from = (
        downwind
        + 180.0
    ) % 360.0

    return (
        speed,
        downwind,
        wind_from,
    )


def interpolate_wind(
    table,
    target_time,
):
    target_time = pd.Timestamp(
        target_time
    ).tz_convert("UTC")

    before = table[
        table[
            "time_utc"
        ].le(target_time)
    ].tail(1)

    after = table[
        table[
            "time_utc"
        ].ge(target_time)
    ].head(1)

    if before.empty or after.empty:
        nearest_index = (
            (
                table["time_utc"]
                - target_time
            )
            .abs()
            .idxmin()
        )

        nearest = table.loc[
            nearest_index
        ]

        u, v = meteorological_to_vector(
            nearest["wind_speed_m_s"],
            nearest[
                "wind_from_degrees"
            ],
        )

        (
            interpolated_speed,
            downwind,
            wind_from,
        ) = vector_to_downwind(
            u,
            v,
        )

        return {
            "before_time_utc":
                nearest["time_utc"],

            "after_time_utc":
                nearest["time_utc"],

            "before_speed_m_s":
                nearest[
                    "wind_speed_m_s"
                ],

            "after_speed_m_s":
                nearest[
                    "wind_speed_m_s"
                ],

            "before_wind_from_degrees":
                nearest[
                    "wind_from_degrees"
                ],

            "after_wind_from_degrees":
                nearest[
                    "wind_from_degrees"
                ],

            "interpolation_fraction":
                0.0,

            "interpolated_u_toward_m_s":
                u,

            "interpolated_v_toward_m_s":
                v,

            "interpolated_speed_m_s":
                interpolated_speed,

            "interpolated_wind_from_degrees":
                wind_from,

            "fixed_downwind_direction_degrees":
                downwind,

            "interpolation_method":
                "nearest_hour_fallback",
        }

    row_before = before.iloc[0]
    row_after = after.iloc[0]

    t0 = row_before[
        "time_utc"
    ]

    t1 = row_after[
        "time_utc"
    ]

    if t1 == t0:
        fraction = 0.0
    else:
        fraction = (
            (
                target_time
                - t0
            ).total_seconds()
            /
            (
                t1
                - t0
            ).total_seconds()
        )

    u0, v0 = meteorological_to_vector(
        row_before[
            "wind_speed_m_s"
        ],
        row_before[
            "wind_from_degrees"
        ],
    )

    u1, v1 = meteorological_to_vector(
        row_after[
            "wind_speed_m_s"
        ],
        row_after[
            "wind_from_degrees"
        ],
    )

    interpolated_u = (
        u0
        + fraction
        * (
            u1 - u0
        )
    )

    interpolated_v = (
        v0
        + fraction
        * (
            v1 - v0
        )
    )

    (
        interpolated_speed,
        downwind,
        wind_from,
    ) = vector_to_downwind(
        interpolated_u,
        interpolated_v,
    )

    return {
        "before_time_utc":
            t0,

        "after_time_utc":
            t1,

        "before_speed_m_s":
            row_before[
                "wind_speed_m_s"
            ],

        "after_speed_m_s":
            row_after[
                "wind_speed_m_s"
            ],

        "before_wind_from_degrees":
            row_before[
                "wind_from_degrees"
            ],

        "after_wind_from_degrees":
            row_after[
                "wind_from_degrees"
            ],

        "interpolation_fraction":
            fraction,

        "interpolated_u_toward_m_s":
            interpolated_u,

        "interpolated_v_toward_m_s":
            interpolated_v,

        "interpolated_speed_m_s":
            interpolated_speed,

        "interpolated_wind_from_degrees":
            wind_from,

        "fixed_downwind_direction_degrees":
            downwind,

        "interpolation_method":
            "linear_vector_interpolation",
    }


def main():
    RAW_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    positives = pd.read_csv(
        POSITIVE_INPUT,
        low_memory=False,
    )

    plan = pd.read_csv(
        INTERNAL_PLAN_INPUT,
        low_memory=False,
    )

    positives[
        "acquisition_time_utc"
    ] = pd.to_datetime(
        positives[
            "acquisition_time_utc"
        ],
        errors="raise",
        utc=True,
    )

    for column in [
        "lat",
        "lon",
        "release_rate_kg_h",
    ]:
        positives[column] = (
            pd.to_numeric(
                positives[column],
                errors="raise",
            )
        )

    missing_ids = set(
        plan.loc[
            plan[
                "wind_plan_status"
            ].astype(str).eq(
                "missing_short_term_direction"
            ),
            "positive_id",
        ].astype(str)
    )

    missing = positives[
        positives[
            "positive_id"
        ].astype(str).isin(
            missing_ids
        )
    ].copy()

    if len(missing) != 3:
        raise RuntimeError(
            "Expected 3 missing wind scenes, "
            f"found {len(missing)}."
        )

    evidence_rows = []

    print("=" * 115)
    print(
        "COMPLETE HIGH-EMISSION WIND PLAN "
        "WITH ERA5-LAND"
    )
    print("=" * 115)

    for number, row in (
        missing.iterrows()
    ):
        positive_id = str(
            row["positive_id"]
        )

        target_time = row[
            "acquisition_time_utc"
        ]

        print(
            f"\n[{len(evidence_rows) + 1}/3] "
            f"{positive_id} | "
            f"{target_time}",
            flush=True,
        )

        payload, request_url = (
            fetch_era5(
                latitude=
                    row["lat"],

                longitude=
                    row["lon"],

                target_time=
                    target_time,
            )
        )

        raw_path = (
            RAW_OUTPUT_DIR
            / f"{positive_id}.json"
        )

        raw_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        hourly = build_hourly_table(
            payload
        )

        result = interpolate_wind(
            hourly,
            target_time,
        )

        record = {
            "positive_id":
                positive_id,

            "event_id":
                row["event_id"],

            "scene_id":
                row["scene_id"],

            "site":
                row["site"],

            "acquisition_time_utc":
                target_time,

            "release_rate_kg_h":
                row[
                    "release_rate_kg_h"
                ],

            "lat":
                row["lat"],

            "lon":
                row["lon"],

            **result,

            "fixed_upwind_direction_degrees":
                (
                    result[
                        "fixed_downwind_direction_degrees"
                    ]
                    + 180.0
                ) % 360.0,

            "wind_source_name":
                "ERA5 via Open-Meteo",

            "wind_source_type":
                "external_reanalysis",

            "wind_evidence_tier":
                "secondary",

            "nominal_temporal_resolution":
                "hourly",

            "nominal_spatial_resolution":
                "0.25_degree",

            "raw_response_path":
                str(raw_path),

            "request_url":
                request_url,
        }

        evidence_rows.append(
            record
        )

        print(
            "  Before:",
            result[
                "before_time_utc"
            ],
            "| from",
            f"{result['before_wind_from_degrees']:.2f}°",
            "|",
            f"{result['before_speed_m_s']:.2f} m/s",
        )

        print(
            "  After:",
            result[
                "after_time_utc"
            ],
            "| from",
            f"{result['after_wind_from_degrees']:.2f}°",
            "|",
            f"{result['after_speed_m_s']:.2f} m/s",
        )

        print(
            "  Interpolated speed:",
            f"{result['interpolated_speed_m_s']:.2f} m/s",
        )

        print(
            "  Fixed downwind:",
            (
                f"{result['fixed_downwind_direction_degrees']:.2f}°"
            ),
        )

    evidence = pd.DataFrame(
        evidence_rows
    )

    evidence.to_csv(
        EXTERNAL_EVIDENCE_OUTPUT,
        index=False,
    )

    # 加入完整來源與證據等級欄位。
    plan["wind_source_name"] = (
        "controlled_release_ground_truth"
    )

    plan["wind_source_type"] = (
        "experiment_or_ground_truth_record"
    )

    plan["wind_evidence_tier"] = (
        "primary"
    )

    plan[
        "wind_temporal_resolution"
    ] = "30_to_300_second_windows"

    plan[
        "wind_spatial_resolution"
    ] = "site_or_experiment_record"

    plan[
        "wind_interpolation_method"
    ] = "short_term_circular_mean"

    for _, external in (
        evidence.iterrows()
    ):
        mask = (
            plan["positive_id"]
            .astype(str)
            .eq(
                str(
                    external[
                        "positive_id"
                    ]
                )
            )
        )

        if mask.sum() != 1:
            raise RuntimeError(
                "Unable to locate unique plan row for "
                f"{external['positive_id']}."
            )

        plan.loc[
            mask,
            "fixed_downwind_direction_degrees",
        ] = external[
            "fixed_downwind_direction_degrees"
        ]

        plan.loc[
            mask,
            "fixed_upwind_direction_degrees",
        ] = external[
            "fixed_upwind_direction_degrees"
        ]

        plan.loc[
            mask,
            "wind_plan_status",
        ] = (
            "locked_external_era5_hourly"
        )

        plan.loc[
            mask,
            "wind_source_name",
        ] = (
            "ERA5 via Open-Meteo"
        )

        plan.loc[
            mask,
            "wind_source_type",
        ] = "external_reanalysis"

        plan.loc[
            mask,
            "wind_evidence_tier",
        ] = "secondary"

        plan.loc[
            mask,
            "wind_temporal_resolution",
        ] = "hourly"

        plan.loc[
            mask,
            "wind_spatial_resolution",
        ] = "0.25_degree"

        plan.loc[
            mask,
            "wind_interpolation_method",
        ] = external[
            "interpolation_method"
        ]

        plan.loc[
            mask,
            "valid_short_term_window_count",
        ] = 0

        plan.loc[
            mask,
            "available_short_term_windows",
        ] = "external_hourly_bracket"

        plan.loc[
            mask,
            "maximum_short_term_deviation_degrees",
        ] = np.nan

    plan[
        "wind_plan_version"
    ] = (
        "s2_high_emission_wind_plan_complete_v2"
    )

    plan[
        "analysis_interpretation_rule"
    ] = np.where(
        plan[
            "wind_evidence_tier"
        ].eq("primary"),
        (
            "primary wind-aligned analysis"
        ),
        (
            "secondary sensitivity analysis; "
            "do not interpret as site-measured wind"
        ),
    )

    unresolved = (
        plan[
            "fixed_downwind_direction_degrees"
        ].isna()
        | plan[
            "fixed_upwind_direction_degrees"
        ].isna()
    )

    if unresolved.any():
        raise RuntimeError(
            "Complete plan still contains missing "
            "wind directions:\n"
            + plan.loc[
                unresolved,
                [
                    "positive_id",
                    "wind_plan_status",
                ]
            ].to_string(index=False)
        )

    plan.to_csv(
        COMPLETE_PLAN_OUTPUT,
        index=False,
    )

    source_counts = (
        plan[
            "wind_source_type"
        ].value_counts()
    )

    tier_counts = (
        plan[
            "wind_evidence_tier"
        ].value_counts()
    )

    report_lines = [
        "=" * 115,
        (
            "SENTINEL-2 HIGH-EMISSION "
            "COMPLETE WIND PLAN V2"
        ),
        "=" * 115,
        "",
        f"Complete plan rows: {len(plan)}",
        (
            "Rows with valid fixed direction: "
            f"{int((~unresolved).sum())}"
        ),
        "",
        "Wind source types:",
        source_counts.to_string(),
        "",
        "Evidence tiers:",
        tier_counts.to_string(),
        "",
        "External ERA5 evidence:",
        evidence[
            [
                "positive_id",
                "acquisition_time_utc",
                "release_rate_kg_h",
                "before_time_utc",
                "after_time_utc",
                "interpolated_speed_m_s",
                "interpolated_wind_from_degrees",
                "fixed_downwind_direction_degrees",
            ]
        ].to_string(index=False),
        "",
        "Complete wind plan:",
        plan[
            [
                "positive_id",
                "release_rate_kg_h",
                "fixed_downwind_direction_degrees",
                "fixed_upwind_direction_degrees",
                "wind_plan_status",
                "wind_source_type",
                "wind_evidence_tier",
                "analysis_interpretation_rule",
            ]
        ].to_string(index=False),
    ]

    REPORT_OUTPUT.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 115)
    print("COMPLETE WIND PLAN SUMMARY")
    print("=" * 115)

    print(
        "\nComplete plan rows:",
        len(plan),
    )

    print(
        "Rows with valid direction:",
        int(
            (~unresolved).sum()
        ),
    )

    print("\nWind source types:")
    print(source_counts)

    print("\nEvidence tiers:")
    print(tier_counts)

    print("\nComplete wind plan:")
    print(
        plan[
            [
                "positive_id",
                "release_rate_kg_h",
                "fixed_downwind_direction_degrees",
                "fixed_upwind_direction_degrees",
                "wind_plan_status",
                "wind_source_type",
                "wind_evidence_tier",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(EXTERNAL_EVIDENCE_OUTPUT)
    print(COMPLETE_PLAN_OUTPUT)
    print(REPORT_OUTPUT)
    print(RAW_OUTPUT_DIR)


if __name__ == "__main__":
    main()
