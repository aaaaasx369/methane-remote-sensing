# MethaneAIR-Sentinel-2 Same-Parent Benchmark — FINAL FREEZE

Status: FREEZE / COMPLETE

## Research question

Can MethaneFuse distinguish MethaneAIR-observed methane-positive events
from same-parent Sentinel-2 temporal no-detection controls when site and
parent-source differences are controlled?

## Final benchmark

Main benchmark:
- 201 matched pairs
- 402 samples
- 201 positives
- 201 temporal no-detection controls
- positive Sentinel-2 t0 aligned within ±72 h of MethaneAIR parent observation
- positive and negative t0 observations are distinct
- no pair shares the same t0 raster
- no pair shares a Sentinel-2 t0 overpass within 20 minutes

Negative evidence:
- B1 strong high-resolution no-L4-detection controls: 193 pairs
- B2 background-weak high-resolution no-L4-detection controls: 8 pairs

Site distribution:
- Permian (Delaware): 169 pairs
- SW Marcellus: 28 pairs
- NE Marcellus: 3 pairs
- Haynesville: 1 pair

## Main ±72 h result

TP = 193
FN = 8
TN = 12
FP = 189

AUROC = 0.4560
Balanced accuracy = 0.5100
Recall = 0.9602
Specificity = 0.0597
FPR = 0.9403

Median P(positive) for positives = 0.8251
Median P(positive) for negatives = 0.8448

Pairwise positive > matched-negative fraction = 0.4279
Tie fraction = 0
Median paired score difference = -0.00761
Mean paired score difference = -0.01156

## ±24 h sensitivity

119 matched pairs / 238 samples

AUROC = 0.4670
Balanced accuracy = 0.5042
Recall = 0.9496
Specificity = 0.0588
FPR = 0.9412
Pairwise win fraction = 0.4706

The stricter ±24 h temporal-alignment requirement did not materially
improve discrimination.

## Scientific conclusion

MethaneFuse showed a strong positive prediction bias and near-chance
discrimination on the balanced same-parent MethaneAIR/Sentinel-2 benchmark.

After removing all pairs that shared the same Sentinel-2 t0 raster or
acquisition overpass, the final 201-pair benchmark achieved AUROC 0.456,
balanced accuracy 0.510, and FPR 94.0%.

Only 42.8% of positive observations received a higher positive-class score
than their own matched temporal no-detection control. The median positive
score was slightly higher for matched negatives than for positives.

Restricting the analysis to 119 pairs aligned within 24 h produced nearly
identical performance (AUROC 0.467, FPR 94.1%).

The poor discrimination therefore cannot be explained solely by:
- classification threshold choice,
- duplicated t0 imagery,
- loose ±72 h positive temporal alignment,
- B2 background-weak controls,
- or cross-site composition alone.

## Important limitation

The negative class consists of high-resolution MethaneAIR-supported
no-detection temporal controls. These should NOT be described as confirmed
zero-emission observations or confirmed no-release events.

## Interpretation boundary

This benchmark supports a conclusion of poor external discrimination /
strong positive prediction bias under this dataset construction.

It should NOT be phrased as proving that MethaneFuse cannot detect methane
under all conditions.

## Canonical files

dataset/paired_72h_disjoint_t0_eval.csv
    Final primary benchmark.

dataset/paired_24h_disjoint_t0_eval.csv
    Strict temporal sensitivity subset.

predictions/
    Per-sample MethaneFuse prediction outputs.

metrics/
    Original evaluator JSON outputs.

audit/disjoint_t0_repair_audit.csv
    Complete distinct-t0 repair provenance.

audit/disjoint_t0_repair_summary.txt
    Dataset-repair summary.

FINAL_METRICS.json
    Canonical frozen numerical results.
