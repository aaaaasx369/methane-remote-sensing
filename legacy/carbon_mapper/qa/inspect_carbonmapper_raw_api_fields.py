from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


API_URL = (
    "https://api.carbonmapper.org/"
    "api/v1/catalog/plumes/annotated"
)

OUTPUT_JSON = Path(
    "outputs/204_carbonmapper_raw_api_sample.json"
)

OUTPUT_PATHS = Path(
    "outputs/205_carbonmapper_relevant_field_paths.txt"
)

TOKENS = {
    "quality",
    "qa",
    "qc",
    "confidence",
    "review",
    "status",
    "publish",
    "time",
    "date",
    "timestamp",
    "datetime",
    "emission",
    "gas",
    "instrument",
    "platform",
    "plume",
    "scene",
}


def walk_paths(
    value: Any,
    prefix: str = "",
):
    rows = []

    if isinstance(value, dict):
        for key, child in value.items():
            path = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            rows.append(
                (
                    path,
                    type(child).__name__,
                    child,
                )
            )

            rows.extend(
                walk_paths(
                    child,
                    path,
                )
            )

    elif isinstance(value, list):
        for index, child in enumerate(
            value[:3]
        ):
            path = (
                f"{prefix}[{index}]"
            )

            rows.extend(
                walk_paths(
                    child,
                    path,
                )
            )

    return rows


def short_value(value):
    if isinstance(
        value,
        (dict, list),
    ):
        text = json.dumps(
            value,
            ensure_ascii=False,
        )
    else:
        text = repr(value)

    if len(text) > 300:
        return text[:300] + "..."

    return text


def main():
    response = requests.get(
        API_URL,
        params={
            "limit": 10,
            "offset": 0,
            "sort": "desc",
        },
        timeout=(30, 120),
        headers={
            "Accept": "application/json",
            "User-Agent":
                "methane-release-research/1.0",
        },
    )

    response.raise_for_status()

    payload = response.json()

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_JSON.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    items = payload.get(
        "items",
        [],
    )

    print("=" * 110)
    print("CARBON MAPPER RAW API FIELD INSPECTION")
    print("=" * 110)

    print("\nItems returned:", len(items))
    print("Top-level payload keys:")
    print(sorted(payload.keys()))

    all_relevant_paths = {}

    for item_number, item in enumerate(
        items,
        start=1,
    ):
        print("\n" + "=" * 110)
        print(f"ITEM {item_number}")
        print("=" * 110)

        print("\nTop-level item keys:")
        print(sorted(item.keys()))

        paths = walk_paths(item)

        relevant = []

        for path, value_type, value in paths:
            path_lower = path.lower()

            if any(
                token in path_lower
                for token in TOKENS
            ):
                relevant.append(
                    (
                        path,
                        value_type,
                        value,
                    )
                )

                all_relevant_paths[
                    path
                ] = value_type

        print("\nRelevant field paths and values:")

        for path, value_type, value in relevant:
            print(
                f"{path:60s} "
                f"[{value_type}] "
                f"{short_value(value)}"
            )

    lines = [
        f"{path}\t{value_type}"
        for path, value_type
        in sorted(
            all_relevant_paths.items()
        )
    ]

    OUTPUT_PATHS.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\nSaved:")
    print(OUTPUT_JSON)
    print(OUTPUT_PATHS)


if __name__ == "__main__":
    main()
