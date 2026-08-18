# MethaneSAT Same-Site Temporal Confounding Audit — Final Results

**Status: COMPLETE / FREEZE**

## Research question
Can MethaneSAT L3 XCH4 imagery distinguish L4-detected observations from far-time no-detection observations at the same physical location, and is the separability driven by localized plume structure or by a broad patch-level XCH4 shift?

## Frozen benchmark
- 116 matched pairs = 116 L4-detected positive observations + 116 same-site, far-time temporal no-detection controls.
- Exact same location within each pair; 480 m × 480 m L3 XCH4 crops; temporal separation ≥ 90 days.
- Fixed image QA threshold: valid fraction ≥ 0.50.
- 232 total analysis images.
- Controls are no-detection controls, **not externally confirmed zero-emission states**.

## Key results

| Feature / representation | AUROC | 95% CI | Median paired Δ | Positive > control |
|---|---:|---:|---:|---:|
| Raw p95 | 0.7985 | 0.7417–0.8536 | +37.20 | 82.8% |
| Raw p99 | 0.7984 | 0.7416–0.8532 | +39.05 | 83.6% |
| Raw mean | 0.7886 | 0.7287–0.8467 | +24.84 | 80.2% |
| Raw center−ring | 0.6102 | 0.5439–0.6759 | +5.17 | 61.2% |
| Background-centered p99 | 0.6463 | 0.5783–0.7136 | +13.30 | 71.6% |
| Background-centered p95 | 0.6442 | 0.5777–0.7098 | +8.96 | 67.2% |
| Background-z p95 | 0.6032 | 0.5324–0.6713 | +0.224 | 61.2% |
| Background-z median | 0.5088 | 0.4360–0.5824 | +0.0065 | 53.4% |

## Spatial localization
- Outer-ring baseline shift vs whole-patch raw shift: **r = 0.9762**.
- Raw radial difference stays nearly flat from the center to ~240 m.
- Raw center-minus-outer mean = **+2.89**, 95% CI **−1.09 to +6.79**, sign-test **p = 0.077**.
- Background-centered center difference = **+4.75**, 95% CI **+1.12 to +8.42**.

## Final interpretation
MethaneSAT L3 XCH4 imagery is strongly distinguishable between L4-detected observations and same-location far-time no-detection controls, but most of the separability is explained by a **broad patch-level XCH4 elevation** rather than a strongly localized source-centered plume morphology. Background normalization reduces the best AUROC from about **0.80 → 0.65 → 0.60**, leaving only a modest residual relative/high-XCH4 signal.

## MethaneFuse decision
The released MethaneFuse pipeline does **not** natively support MethaneSAT. Do not relabel MethaneSAT as Sentinel-5P to force released-checkpoint inference. Any future neural-network work would need to be explicitly described as a new MethaneSAT-specific baseline or newly trained/fine-tuned adapter.

## Limitations
- Controls are no-detection observations, not independently confirmed zero-emission states.
- Same-site pairing reduces site/surface confounding but does not remove temporal atmospheric, seasonal, retrieval, or operational differences.
- The 224×224 arrays are standardized grids derived from 480 m crops; they are not new native-resolution observations.
- Do not describe the 116 paired observations as 116 independent sources unless source-level independence is established separately.

## Professor-ready wording
> I built a same-location, far-time MethaneSAT benchmark to reduce site confounding. Across 116 pairs, raw L3 XCH4 statistics strongly separated L4-detected observations from no-detection controls, with p95 AUROC ≈ 0.80. However, the spatial audit showed that the positive-control difference was broad across the 480 m patch: the outer-ring baseline shift was almost perfectly correlated with the whole-patch shift (r = 0.976). After local-background centering, the best AUROC dropped to ≈ 0.65, and after background z-normalization it dropped to ≈ 0.60. This suggests that most of the discrimination is driven by broad XCH4 elevation, with only a modest residual localized/relative signal.

## Recommended endpoint
**Freeze this MethaneSAT analysis.** Do not lower QA thresholds, add more hand-crafted features, or force MethaneSAT through the Sentinel-5P branch. The next work should be communication: poster/result panel, professor update, and integration with the broader multi-sensor project.
