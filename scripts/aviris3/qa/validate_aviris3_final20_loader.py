from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.multisensor import (
    TriSensorTemporalCsvDataset,
    custom_collate_fn,
)

from src.data.sensor_transforms import (
    DEFAULT_WV3_BANDS,
    load_wv3_channel_ids_from_srf,
)


# ============================================================
# PATHS — FROM METHANEFUSE REPO ROOT
# ============================================================

DATA_ROOT = Path(
    "data/custom/"
    "aviris3_methanefuse_final20_exact"
)

CSV = (
    DATA_ROOT
    / "aviris3_methanefuse_test.csv"
)

SRF = (
    DATA_ROOT
    / "WV3_VNIR_SWIR_response.csv"
)


# ============================================================
# BASIC CSV QA
# ============================================================

print("=" * 75)
print("CSV QA")
print("=" * 75)

if not CSV.exists():
    raise FileNotFoundError(
        CSV.resolve()
    )

if not SRF.exists():
    raise FileNotFoundError(
        SRF.resolve()
    )

df = pd.read_csv(
    CSV
)

print("CSV:", CSV.resolve())
print("Rows:", len(df))

print("\nLabels:")
print(
    df["label"]
    .value_counts()
    .sort_index()
    .to_string()
)

required = [
    "id",
    "label",
    "emit_0_path",
    "emit_90_path",
    "emit_360_path",
]

missing_cols = [
    c for c in required
    if c not in df.columns
]

if missing_cols:
    raise RuntimeError(
        f"Missing columns: {missing_cols}"
    )

if len(df) != 20:
    raise RuntimeError(
        f"Expected 20 rows, got {len(df)}"
    )

counts = (
    df["label"]
    .value_counts()
    .to_dict()
)

if counts != {0: 10, 1: 10}:
    raise RuntimeError(
        f"Expected labels 10/10; "
        f"got {counts}"
    )


# ============================================================
# VERIFY PORTABLE PATHS
# ============================================================

print("\n" + "=" * 75)
print("PORTABLE PATH QA")
print("=" * 75)

path_errors = []

for _, row in df.iterrows():

    for col in [
        "emit_0_path",
        "emit_90_path",
        "emit_360_path",
    ]:

        p = Path(
            str(row[col])
        )

        if not p.exists():

            path_errors.append(
                (
                    row["id"],
                    col,
                    str(p)
                )
            )

print(
    "Referenced TIFFs:",
    len(df) * 3
)

print(
    "Path errors:",
    len(path_errors)
)

if path_errors:

    for x in path_errors[:10]:
        print("ERROR:", x)

    raise RuntimeError(
        "Portable CSV contains "
        "missing paths"
    )


# ============================================================
# BUILD EXACT WV3 CHANNEL IDS
#
# This is the same function used by
# evaluate_classification.py
# ============================================================

wv3_chn_ids = (
    load_wv3_channel_ids_from_srf(
        str(SRF),
        DEFAULT_WV3_BANDS,
    )
    .unsqueeze(-1)
)

print("\n" + "=" * 75)
print("WV3 CHANNEL IDS")
print("=" * 75)

print(
    "Bands:",
    len(DEFAULT_WV3_BANDS)
)

print(
    "chn_ids shape:",
    tuple(
        wv3_chn_ids.shape
    )
)

print(
    "chn_ids:",
    wv3_chn_ids
    .squeeze(-1)
    .tolist()
)

if tuple(
    wv3_chn_ids.shape
) != (16, 1):

    raise RuntimeError(
        "WV3 channel ids are not 16x1"
    )


# ============================================================
# CREATE THE REAL METHANEFUSE DATASET
# ============================================================

print("\n" + "=" * 75)
print("CREATE TriSensorTemporalCsvDataset")
print("=" * 75)

ds = TriSensorTemporalCsvDataset(
    csv_path=str(CSV),

    wv3_chn_ids=
        wv3_chn_ids,

    pad_to_multiple=14,

    # Do not silently substitute another
    # sample if something is invalid.
    max_retries=1,
)

print(
    "Dataset length:",
    len(ds)
)

print(
    "Recognized sensors:",
    ds._wide_sensor_columns
)

if len(ds) != 20:
    raise RuntimeError(
        f"Dataset has {len(ds)} rows"
    )

if "wv3" not in (
    ds._wide_sensor_columns
):
    raise RuntimeError(
        "MethaneFuse did not map "
        "emit_* columns to wv3"
    )

expected_cols = (
    "emit_0_path",
    "emit_90_path",
    "emit_360_path",
)

if tuple(
    ds._wide_sensor_columns[
        "wv3"
    ]
) != expected_cols:

    raise RuntimeError(
        "Wrong WV3 temporal columns: "
        f"{ds._wide_sensor_columns}"
    )


# ============================================================
# LOAD ALL 20 ROWS INDIVIDUALLY
# ============================================================

print("\n" + "=" * 75)
print("LOAD ALL 20 ROWS")
print("=" * 75)

labels_seen = []

global_min = float("inf")
global_max = float("-inf")

for i in range(
    len(ds)
):

    sensor_samples, label = (
        ds[i]
    )

    qid = str(
        df.iloc[i]["id"]
    )

    labels_seen.append(
        int(label)
    )

    if len(sensor_samples) != 1:

        raise RuntimeError(
            f"{qid}: expected exactly "
            f"1 sensor, got "
            f"{len(sensor_samples)}"
        )

    sensor, frames = (
        sensor_samples[0]
    )

    if sensor != "wv3":

        raise RuntimeError(
            f"{qid}: sensor={sensor}, "
            "expected wv3"
        )

    if len(frames) != 3:

        raise RuntimeError(
            f"{qid}: expected 3 temporal "
            f"frames, got {len(frames)}"
        )

    for temporal_idx, x in enumerate(
        frames
    ):

        img = x["imgs"]
        chn = x["chn_ids"]

        if tuple(
            img.shape
        ) != (
            16,
            518,
            518,
        ):

            raise RuntimeError(
                f"{qid} temporal "
                f"{temporal_idx}: "
                f"bad imgs shape "
                f"{tuple(img.shape)}"
            )

        if tuple(
            chn.shape
        ) != (
            16,
            1,
        ):

            raise RuntimeError(
                f"{qid} temporal "
                f"{temporal_idx}: "
                f"bad chn_ids shape "
                f"{tuple(chn.shape)}"
            )

        if not torch.isfinite(
            img
        ).all():

            raise RuntimeError(
                f"{qid} temporal "
                f"{temporal_idx}: "
                "NaN/Inf"
            )

        global_min = min(
            global_min,
            float(
                img.min()
            )
        )

        global_max = max(
            global_max,
            float(
                img.max()
            )
        )

    print(
        f"[{i+1:02d}/20] "
        f"{qid} "
        f"label={label} "
        f"sensor={sensor} "
        f"frames=3 "
        f"frame_shape="
        f"{tuple(frames[0]['imgs'].shape)}"
    )


# ============================================================
# LABEL CHECK
# ============================================================

seen_counts = Counter(
    labels_seen
)

print("\nLabels actually loaded:")
print(
    dict(
        sorted(
            seen_counts.items()
        )
    )
)

if seen_counts != Counter(
    {
        0: 10,
        1: 10,
    }
):
    raise RuntimeError(
        f"Loaded labels wrong: "
        f"{seen_counts}"
    )


# ============================================================
# DATALOADER / COLLATE TEST
# ============================================================

print("\n" + "=" * 75)
print("DATALOADER / COLLATE")
print("=" * 75)

loader = DataLoader(
    ds,
    batch_size=5,
    shuffle=False,

    # Keep at zero for a simple
    # deterministic Mac smoke test.
    num_workers=0,

    collate_fn=
        custom_collate_fn,
)


rows_seen = 0
batch_count = 0
batch_labels = []

for (
    x_dict,
    labels,
    sensors,
    sample_to_row
) in loader:

    batch_count += 1

    imgs = x_dict[
        "imgs"
    ]

    chn_ids = x_dict[
        "chn_ids"
    ]

    print(
        f"Batch {batch_count}:"
    )

    print(
        "  imgs:",
        tuple(
            imgs.shape
        )
    )

    print(
        "  chn_ids:",
        tuple(
            chn_ids.shape
        )
    )

    print(
        "  labels:",
        tuple(
            labels.shape
        ),
        labels.tolist()
    )

    print(
        "  sensors:",
        sensors
    )

    print(
        "  sample_to_row:",
        sample_to_row.tolist()
    )

    # 3 temporal x 16 bands = 48
    if imgs.shape[1:] != (
        48,
        518,
        518,
    ):

        raise RuntimeError(
            "Unexpected collated image "
            f"shape: {tuple(imgs.shape)}"
        )

    if chn_ids.shape[1:] != (
        48,
        1,
    ):

        raise RuntimeError(
            "Unexpected collated chn_ids "
            f"shape: {tuple(chn_ids.shape)}"
        )

    if not all(
        x == "wv3"
        for x in sensors
    ):

        raise RuntimeError(
            f"Unexpected sensors: "
            f"{sensors}"
        )

    expected_mapping = list(
        range(
            len(labels)
        )
    )

    if (
        sample_to_row.tolist()
        != expected_mapping
    ):

        raise RuntimeError(
            "sample_to_row mismatch: "
            f"{sample_to_row.tolist()}"
        )

    rows_seen += int(
        labels.numel()
    )

    batch_labels.extend(
        labels.tolist()
    )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 75)
print("METHANEFUSE REPO LOADER VALIDATION")
print("=" * 75)

print(
    "Dataset rows:",
    len(ds)
)

print(
    "Rows loaded individually:",
    len(labels_seen)
)

print(
    "Rows loaded through DataLoader:",
    rows_seen
)

print(
    "Batches:",
    batch_count
)

print(
    "Sensor:",
    "wv3"
)

print(
    "Temporal frames per row:",
    3
)

print(
    "Channels per frame:",
    16
)

print(
    "Channels after temporal concat:",
    48
)

print(
    "Spatial size:",
    "518 x 518"
)

print(
    "Loaded label counts:",
    dict(
        sorted(
            Counter(
                batch_labels
            ).items()
        )
    )
)

print(
    "Post-loader value range:",
    round(
        global_min,
        6
    ),
    "to",
    round(
        global_max,
        6
    )
)

print(
    "\nALL 20 ROWS LOADED "
    "SUCCESSFULLY"
)

print(
    "CUSTOM COLLATE PASS"
)

print(
    "REPO COMPATIBILITY PASS"
)
