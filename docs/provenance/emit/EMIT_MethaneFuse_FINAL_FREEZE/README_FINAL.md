# EMIT MethaneFuse External Benchmark — FINAL FREEZE

## Status

**FREEZE / COMPLETE**

This folder is the canonical frozen endpoint for the EMIT external MethaneFuse
evaluation.

## Canonical dataset

Primary real-temporal benchmark:

- 19 complete matched POS/NEG pairs
- 38 samples total
- 19 positive / 19 candidate negative
- 114 physical TIFFs
- each sample contains:
  - real `t0`
  - real EMIT acquisition near `t-90`
  - real EMIT acquisition near `t-180`
- temporal target tolerance for the primary set: ±60 days
- all retained frames passed pixel QA
- each model TIFF is 16 bands × 518 × 518, uint16

Positive `t0` samples are anchored to published EMIT CH4PLM plume evidence.

Negative `t0` samples are **same-site different-time candidate negatives**
without a published CH4PLM source-scene detection. They are not independently
confirmed zero-emission observations.

## MethaneFuse compatibility

The released generic MethaneFuse loader expects:

- `emit_0_path`
- `emit_90_path`
- `emit_360_path`

For this EMIT benchmark:

- `emit_0_path` = actual `t0`
- `emit_90_path` = actual `~t-90`
- `emit_360_path` = actual `~t-180`

The third column retains the loader's `360` name only for compatibility.

EMIT L2A surface reflectance was converted to the 16-band simulated
WorldView-3 representation used by the released interface.

## Four canonical experiments

### 1. Expanded60 — real temporal
Pairs: 19  
Samples: 38

- Accuracy: 0.578947
- AUROC: 0.540166
- Recall: 0.894737
- FPR: 0.736842
- Loss: 1.124318

### 2. Expanded60 — matched t0-repeated baseline
Same 38 samples and labels; only temporal context is replaced by `t0/t0/t0`.

- Accuracy: 0.552632
- AUROC: 0.617729
- Recall: 0.894737
- FPR: 0.789474
- Loss: 1.272010

Matched real-minus-baseline effects:

- ΔAccuracy: +0.026316
- ΔAUROC: -0.077562
- ΔRecall: +0.000000
- ΔFPR: -0.052632
- ΔLoss: -0.147692

### 3. Strict30 sensitivity — real temporal
Pairs: 11  
Samples: 22

- Accuracy: 0.590909
- AUROC: 0.495868
- Recall: 0.909091
- FPR: 0.727273
- Loss: 1.254381

### 4. Strict30 sensitivity — matched t0-repeated baseline
Same 22 strict rows; temporal context replaced by `t0/t0/t0`.

- Accuracy: 0.500000
- AUROC: 0.619835
- Recall: 0.909091
- FPR: 0.909091
- Loss: 1.392758

Matched real-minus-baseline effects:

- ΔAccuracy: +0.090909
- ΔAUROC: -0.123967
- ΔRecall: +0.000000
- ΔFPR: -0.181818
- ΔLoss: -0.138377

## Final interpretation

For the 19-pair primary real-temporal benchmark, MethaneFuse achieved AUROC
0.540. On the exact same 38 observations with t0 repeated
across all temporal slots, AUROC was 0.618
(ΔAUROC = -0.078 for real temporal context).

The same direction persisted under the stricter ±30-day temporal sensitivity:
real-temporal AUROC was 0.496, compared with
0.620 for the matched t0-repeated baseline
(ΔAUROC = -0.124).

Therefore:

> **In this EMIT external benchmark, real historical temporal context did not
> improve overall AUROC discrimination, and this finding was not explained
> simply by the wider ±60-day temporal matching tolerance.**

At the fixed classification threshold, real historical context did reduce the
false-positive rate in both matched analyses, so the result should not be
described as uniform degradation across every metric.

## Limitations

1. The benchmark is small: 19 pairs in the primary analysis and 11 pairs in
   the strict sensitivity subset.
2. Candidate negatives are no-published-detection controls, not independently
   verified methane-free observations.
3. This is an external-domain EMIT evaluation through the released
   MethaneFuse-compatible simulated WV3 representation.
4. These differences are descriptive; this freeze does not claim statistical
   significance.
5. Results should not be generalized to claim that historical imagery is
   universally harmful. The conclusion is specific to this EMIT benchmark.

## Folder contents

- `dataset/`
  - portable manifests for all four canonical experiments
- `samples/`
  - the 114 frozen real-temporal model TIFFs
- `metrics/`
  - original four MethaneFuse result JSONs
- `FINAL_METRICS.json`
  - canonical machine-readable summary
- `RESULTS_TABLE.csv`
  - one-row-per-experiment result table with confusion counts
- `provenance/`
  - temporal search, pair audit, QA, and SRF records where available
- `scripts/`
  - preprocessing and ablation scripts where available
- `ENVIRONMENT.txt`
  - MethaneFuse git state and checkpoint fingerprint
- `FREEZE_INVENTORY.csv`
  - file sizes and SHA256 digests
- `SHA256SUMS.txt`
  - integrity checksums

## Canonical scientific status

**EMIT analysis: FREEZE / COMPLETE.**

Further temporal-tolerance expansion is not part of the canonical analysis.
