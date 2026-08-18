#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.multisensor import TriSensorTemporalCsvDataset


def shape(value):
    if hasattr(value, "shape"):
        return list(value.shape)
    return type(value).__name__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path.resolve()}"
        )

    dataset = TriSensorTemporalCsvDataset(
        csv_path=str(csv_path),
        wv3_chn_ids=None,
        pad_to_multiple=14,
    )

    sensor_samples, label = dataset[args.index]

    result = {
        "csv": str(csv_path),
        "index": args.index,
        "label": int(label),
        "sensors": [],
    }

    for sensor_name, frames in sensor_samples:
        result["sensors"].append(
            {
                "sensor": sensor_name,
                "frames": len(frames),
                "imgs": [
                    shape(frame["imgs"])
                    for frame in frames
                ],
                "chn_ids": [
                    shape(frame["chn_ids"])
                    for frame in frames
                ],
            }
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
