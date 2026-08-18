#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re

import numpy as np
import pandas as pd
import rasterio

ROOT = Path('/Users/happydoraaa/methane_release_project')
MASTER_PATH = ROOT / 'outputs/36_multisite_s2_master_table.csv'
PATCH_INDEX_PATH = ROOT / 'outputs/20_controlled_release_s2_patch_index.csv'
OUTPUT_DIR = ROOT / 'methanefuse_input/s2_12band_smoke'
OUTPUT_CSV = ROOT / 'outputs/52_methanefuse_smoke_test.csv'
REPORT_CSV = ROOT / 'outputs/52_methanefuse_smoke_conversion_report.csv'
RESOLUTION_CSV = ROOT / 'outputs/53_methanefuse_source_path_resolution.csv'

CONTROLLED_SITES = {
    'Casa_Grande_AZ_release_stacks',
    'Ehrenberg_AZ_release_stack',
}
MAX_PATCH_TIME_DIFFERENCE_HOURS = 1.0

BAND_NAMES = [
    'B01', 'B02', 'B03', 'B04',
    'B05', 'B06', 'B07', 'B08',
    'B8A', 'B09', 'B11', 'B12',
]

# Neutral fill values for a pipeline smoke test only.
S2_MEANS = np.array(
    [
        786.1282, 1025.8877, 1593.7307, 2315.2612,
        2710.4630, 3115.9010, 3289.0830, 3465.5364,
        3495.5798, 3517.7960, 4180.2856, 3567.8670,
    ],
    dtype=np.float32,
)

# Supported local input layouts. Target indices use the MethaneFuse
# 12-band order: B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12.
#
# 6-band MethaneAIR layout: B02,B03,B04,B08,B11,B12.
SOURCE_TO_TARGET_6 = {
    0: 1,   # B02
    1: 2,   # B03
    2: 3,   # B04
    3: 7,   # B08
    4: 10,  # B11
    5: 11,  # B12
}

# 8-band controlled-release layout inferred and validated from the local
# pixel distributions: B02,B03,B04,B08,B8A,B11,B12,SCL.
# The final SCL band is used only for validation and is not copied into the
# 12-band spectral tensor.
SOURCE_TO_TARGET_8 = {
    0: 1,   # B02
    1: 2,   # B03
    2: 3,   # B04
    3: 7,   # B08
    4: 8,   # B8A
    5: 10,  # B11
    6: 11,  # B12
}

PATH_COLUMNS = [
    'image_path',
    'resolved_patch_path',
    'patch_path',
    'patch_path_raw',
    'relative_path',
    'file_path',
    'filepath',
    'filename',
]


def clean_text(value):
    if pd.isna(value):
        return ''
    text = str(value).strip()
    if text.lower() in {'', 'nan', 'none', '<na>'}:
        return ''
    return text


def safe_name(value):
    return ''.join(
        char if char.isalnum() or char in '-_' else '_'
        for char in clean_text(value)
    )


def normalize_site(value):
    text = clean_text(value).lower()
    text = re.sub(r'[^a-z0-9]+', ' ', text)
    removable = {
        'release', 'releases', 'stack', 'stacks', 'site',
        'facility', 'controlled', 'methane', 'source', 'test',
    }
    return ' '.join(token for token in text.split() if token not in removable)


def direct_candidates(value):
    text = clean_text(value)
    if not text:
        return []

    path = Path(text).expanduser()
    if path.is_absolute():
        return [path]

    return [
        ROOT / path,
        ROOT / 'outputs' / path,
        ROOT / 'patches' / path,
        ROOT / 'data' / path,
        ROOT / 'downloads' / path,
    ]


if not MASTER_PATH.exists():
    raise SystemExit(f'找不到 master table：{MASTER_PATH}')

print('Indexing local TIFF files...')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tif_index = {}
for path in ROOT.rglob('*.tif'):
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path

    if OUTPUT_DIR in resolved.parents:
        continue

    tif_index.setdefault(path.name, []).append(resolved)

print('Indexed TIFF basenames:', len(tif_index))


def lookup_by_basename(value):
    text = clean_text(value)
    if not text:
        return []

    basename = Path(text).name
    options = [basename]
    if not basename.lower().endswith('.tif'):
        options.append(basename + '.tif')

    matches = []
    for option in options:
        matches.extend(tif_index.get(option, []))
    return sorted(set(matches))


def resolve_value_to_existing_tif(value):
    for candidate in direct_candidates(value):
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    matches = lookup_by_basename(value)
    if matches:
        return matches[0]

    return None


master = pd.read_csv(MASTER_PATH, low_memory=False)
master['_row_id'] = np.arange(len(master))
master['_site_normalized'] = master['site_id'].map(normalize_site)
master['_label_numeric'] = pd.to_numeric(master['label'], errors='coerce')
master['_acquisition_time'] = pd.to_datetime(
    master['acquisition_time_utc'], errors='coerce', utc=True
)

print('Rows:', len(master))
print(
    'Available path columns:',
    [column for column in PATH_COLUMNS if column in master.columns],
)

resolution = {}

# 1. Resolve paths directly from the master row where possible.
for _, row in master.iterrows():
    row_id = int(row['_row_id'])

    for column in PATH_COLUMNS:
        if column not in master.columns:
            continue

        value = clean_text(row.get(column, ''))
        if not value:
            continue

        path = resolve_value_to_existing_tif(value)
        if path is not None:
            resolution[row_id] = {
                'source_path': str(path),
                'resolution_method': f'master_{column}',
                'patch_index_filename': '',
                'patch_index_time_difference_hours': np.nan,
            }
            break

# 2. Use the controlled-release patch index for unresolved controlled rows.
if PATCH_INDEX_PATH.exists():
    patch_index = pd.read_csv(PATCH_INDEX_PATH, low_memory=False)

    required = {'filename', 'site', 'label', 's2_image_time'}
    missing = sorted(required - set(patch_index.columns))
    if missing:
        print('Patch index missing required columns:', missing)
    else:
        patch_index = patch_index.copy()
        patch_index['_patch_id'] = np.arange(len(patch_index))
        patch_index['_site_normalized'] = patch_index['site'].map(normalize_site)
        patch_index['_label_numeric'] = pd.to_numeric(
            patch_index['label'], errors='coerce'
        )
        patch_index['_s2_time'] = pd.to_datetime(
            patch_index['s2_image_time'], errors='coerce', utc=True
        )

        def patch_path_for_row(idx_row):
            for column in ['relative_path', 'filename']:
                if column in patch_index.columns:
                    path = resolve_value_to_existing_tif(idx_row.get(column, ''))
                    if path is not None:
                        return path
            return None

        patch_index['_existing_tif_path'] = patch_index.apply(
            patch_path_for_row, axis=1
        )

        unresolved_controlled = master[
            master['site_id'].isin(CONTROLLED_SITES)
            & ~master['_row_id'].isin(resolution.keys())
        ].copy()

        candidate_pairs = []

        for _, mrow in unresolved_controlled.iterrows():
            candidates = patch_index[
                patch_index['_site_normalized'].eq(mrow['_site_normalized'])
                & patch_index['_label_numeric'].eq(mrow['_label_numeric'])
                & patch_index['_s2_time'].notna()
                & patch_index['_existing_tif_path'].notna()
            ].copy()

            if pd.isna(mrow['_acquisition_time']):
                continue

            candidates['_time_diff_hours'] = (
                candidates['_s2_time'] - mrow['_acquisition_time']
            ).abs().dt.total_seconds() / 3600.0

            candidates = candidates[
                candidates['_time_diff_hours']
                <= MAX_PATCH_TIME_DIFFERENCE_HOURS
            ]

            for _, prow in candidates.iterrows():
                candidate_pairs.append(
                    {
                        'master_row_id': int(mrow['_row_id']),
                        'patch_id': int(prow['_patch_id']),
                        'time_diff_hours': float(prow['_time_diff_hours']),
                        'patch_filename': clean_text(prow['filename']),
                        'source_path': str(prow['_existing_tif_path']),
                    }
                )

        # Greedy one-to-one assignment, globally sorted by the smallest time difference.
        used_master = set()
        used_patch = set()

        for pair in sorted(
            candidate_pairs,
            key=lambda item: (
                item['time_diff_hours'],
                item['master_row_id'],
                item['patch_id'],
            ),
        ):
            if pair['master_row_id'] in used_master:
                continue
            if pair['patch_id'] in used_patch:
                continue

            resolution[pair['master_row_id']] = {
                'source_path': pair['source_path'],
                'resolution_method': 'controlled_patch_index_site_label_time',
                'patch_index_filename': pair['patch_filename'],
                'patch_index_time_difference_hours': pair['time_diff_hours'],
            }
            used_master.add(pair['master_row_id'])
            used_patch.add(pair['patch_id'])

else:
    print('Patch index not found:', PATCH_INDEX_PATH)

resolution_rows = []
for _, row in master.iterrows():
    row_id = int(row['_row_id'])
    item = resolution.get(
        row_id,
        {
            'source_path': '',
            'resolution_method': 'unresolved',
            'patch_index_filename': '',
            'patch_index_time_difference_hours': np.nan,
        },
    )

    resolution_rows.append(
        {
            'row_id': row_id,
            'sample_id': row.get('sample_id', ''),
            'scene_id': row.get('scene_id', ''),
            'site_id': row.get('site_id', ''),
            'label': row.get('label', ''),
            'acquisition_time_utc': row.get('acquisition_time_utc', ''),
            **item,
        }
    )

resolution_df = pd.DataFrame(resolution_rows)
resolution_df.to_csv(RESOLUTION_CSV, index=False)

print()
print('Resolved source paths:', int(resolution_df['source_path'].ne('').sum()), '/', len(resolution_df))
print('Resolution methods:')
print(resolution_df['resolution_method'].value_counts().to_string())

manifest_rows = []
report_rows = []

for _, row in master.iterrows():
    row_id = int(row['_row_id'])
    resolved = resolution_df.loc[resolution_df['row_id'].eq(row_id)].iloc[0]

    sample_id = clean_text(row.get('sample_id', ''))
    if not sample_id:
        sample_id = clean_text(row.get('scene_id', ''))
    if not sample_id:
        sample_id = f'sample_{row_id:04d}'

    output_id = f'{row_id:04d}_{safe_name(sample_id)}'
    output_path = OUTPUT_DIR / f'{output_id}_S2_12band_smoke.tif'

    source_path_text = clean_text(resolved['source_path'])
    source_path = Path(source_path_text) if source_path_text else None

    report = {
        'row_index': row_id,
        'sample_id': sample_id,
        'site_id': row.get('site_id', ''),
        'label': row.get('label', ''),
        'source_path': source_path_text,
        'resolution_method': resolved['resolution_method'],
        'patch_index_filename': resolved['patch_index_filename'],
        'patch_index_time_difference_hours': resolved[
            'patch_index_time_difference_hours'
        ],
        'output_path': str(output_path),
        'source_band_count': np.nan,
        'input_band_layout': '',
        'scl_unique_values': '',
        'success': False,
        'error': '',
    }

    try:
        if source_path is None or not source_path.exists():
            raise FileNotFoundError('No resolved local TIFF path.')

        with rasterio.open(source_path) as src:
            source = src.read()
            profile = src.profile.copy()

        band_count = int(source.shape[0])
        report['source_band_count'] = band_count

        if band_count == 6:
            source_to_target = SOURCE_TO_TARGET_6
            input_band_layout = 'B02,B03,B04,B08,B11,B12'
            scl_values = ''

        elif band_count == 8:
            source_to_target = SOURCE_TO_TARGET_8
            input_band_layout = 'B02,B03,B04,B08,B8A,B11,B12,SCL'

            # Validate the eighth layer before treating it as SCL. Sentinel-2
            # SCL codes are integer classes in the inclusive range 0-11.
            scl = source[7].astype(np.float64)
            finite_scl = scl[np.isfinite(scl)]

            if finite_scl.size == 0:
                raise ValueError(
                    '8-band input has an empty final layer; expected SCL.'
                )

            if not np.all(np.isclose(finite_scl, np.round(finite_scl))):
                raise ValueError(
                    '8-band input final layer is not integer-like; '
                    'cannot verify it as SCL.'
                )

            scl_min = float(np.min(finite_scl))
            scl_max = float(np.max(finite_scl))

            if scl_min < 0 or scl_max > 11:
                raise ValueError(
                    '8-band input final layer is outside the expected '
                    f'SCL range 0-11: min={scl_min}, max={scl_max}'
                )

            scl_values = ','.join(
                str(int(value))
                for value in np.unique(finite_scl)
            )

        else:
            raise ValueError(
                'Expected either 6-band '
                '(B02,B03,B04,B08,B11,B12) or 8-band '
                '(B02,B03,B04,B08,B8A,B11,B12,SCL) input; '
                f'found {band_count}'
            )

        report['input_band_layout'] = input_band_layout
        report['scl_unique_values'] = scl_values

        height, width = source.shape[1:]
        target = np.empty((12, height, width), dtype=np.float32)

        for target_index, mean_value in enumerate(S2_MEANS):
            target[target_index, :, :] = mean_value

        for source_index, target_index in source_to_target.items():
            source_band = source[source_index].astype(np.float32)
            finite = np.isfinite(source_band)
            target[target_index][finite] = source_band[finite]

        target = np.nan_to_num(
            target, nan=0.0, posinf=65535.0, neginf=0.0
        )
        target = np.clip(np.rint(target), 0, 65535).astype(np.uint16)

        profile.update(
            count=12,
            dtype='uint16',
            nodata=None,
            compress='deflate',
        )

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(target)
            for band_number, band_name in enumerate(BAND_NAMES, start=1):
                dst.set_band_description(band_number, band_name)

        with rasterio.open(output_path) as check:
            if check.count != 12:
                raise RuntimeError(
                    f'Converted output contains {check.count} bands.'
                )

        label = int(float(row['label']))

        manifest_rows.append(
            {
                'id': sample_id,
                'label': label,
                's2_0_path': str(output_path.resolve()),
                's2_90_path': str(output_path.resolve()),
                's2_360_path': str(output_path.resolve()),
                'site_id': row.get('site_id', ''),
                'source_scene_id': row.get('scene_id', ''),
                'source_acquisition_time_utc': row.get(
                    'acquisition_time_utc', ''
                ),
                'source_tiff_path': str(source_path),
                'source_path_resolution_method': resolved['resolution_method'],
                'smoke_test_only': True,
            }
        )
        report['success'] = True

    except Exception as exc:
        report['error'] = type(exc).__name__ + ': ' + str(exc)

    report_rows.append(report)

manifest = pd.DataFrame(manifest_rows)
report = pd.DataFrame(report_rows)

manifest.to_csv(OUTPUT_CSV, index=False)
report.to_csv(REPORT_CSV, index=False)

print()
print('Conversion success:', len(manifest), '/', len(master))
print('Failed:', int((~report['success'].astype(bool)).sum()))

if not manifest.empty:
    print()
    print('Label counts:')
    print(manifest['label'].value_counts().sort_index().to_string())

print()
print('Success by site:')
print(
    report.groupby('site_id')['success']
    .agg(rows='size', success='sum')
    .to_string()
)

failed = report[~report['success'].astype(bool)]
if not failed.empty:
    print()
    print('Failed rows:')
    print(
        failed[
            [
                'sample_id',
                'site_id',
                'resolution_method',
                'patch_index_filename',
                'patch_index_time_difference_hours',
                'source_band_count',
                'input_band_layout',
                'scl_unique_values',
                'error',
            ]
        ].head(50).to_string(index=False)
    )

print()
print('Successful conversions by source layout:')
if report['success'].astype(bool).any():
    print(
        report[report['success'].astype(bool)]
        .groupby(['site_id', 'source_band_count', 'input_band_layout'])
        .size()
        .reset_index(name='rows')
        .to_string(index=False)
    )
else:
    print('No successful conversions.')

print()
print('Created:')
print(OUTPUT_CSV)
print(REPORT_CSV)
print(RESOLUTION_CSV)
print()
print(
    'Important: s2_0_path, s2_90_path, and s2_360_path point '
    'to the same converted image. This is only a pipeline smoke test; synthetic missing bands and repeated temporal inputs are not valid for a formal accuracy benchmark.'
)
