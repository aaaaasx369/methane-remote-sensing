# MethaneSAT × MethaneFuse — FINAL FREEZE

Status: **COMPLETE / FREEZE**

## Research question

Can MethaneFuse distinguish MethaneSAT observations with a retained
L4 methane point-source detection from temporally different observations
of the **same exact location**, thereby reducing geographic/site/background
confounding?

## Pairing design

Each pair uses:

- the same latitude / longitude;
- the same physical 480 m × 480 m footprint;
- MethaneSAT L3 Band 1 XCH4;
- a positive acquisition with an L4 retained point-source detection;
- a different acquisition within 90 days as a temporal control.

Positive:
L4-detected observation → label = 1

Temporal weak negative:
same location, different date, no retained local L4 point-source detection
→ label = 0

Temporal controls are **weak negatives / no-detection controls**.
They are not confirmed methane-free observations.

---

## Model representation

Each model sample is an NPZ containing:

    ch4.shape = (3, 224, 224)
    dtype     = float32

    ch4[0] = MethaneSAT L3 XCH4 480 m crop
    ch4[1] = NaN
    ch4[2] = NaN

MethaneSAT is evaluated through the MethaneFuse S5P-style input branch as
an experimental cross-sensor zero-shot adaptation.

This is **not native MethaneSAT support in MethaneFuse**.

---

# STRICT DATASET

Definition:

- Tier A temporal negatives only;
- candidate L4 collection has zero retained point-source detections;
- |Δt| <= 90 days;
- exact same location;
- 480 m crop QA pass;
- XCH4 missing ratio <= 50%.

Pairs: **56**

Samples: **112**

Positive: **56**

Negative: **56**

## MethaneFuse zero-shot result

- AUROC: **0.6327**
- Accuracy: **0.5982**
- Recall: **0.5714**
- FPR: **0.3750**
- Loss: **1.0049**

Confusion matrix:

| | Pred negative | Pred positive |
|---|---:|---:|
| Actual negative | 35 | 21 |
| Actual positive | 24 | 32 |

---

# EXPANDED DATASET

Definition:

Includes Tier A plus Tier B temporal controls.

Tier B allows other retained L4 detections elsewhere in the candidate
collection, but requires no retained L4 point-source detection within
10 km of the paired location.

All candidates remain within 90 days and use the same exact location
and 480 m footprint.

Pairs: **71**

Samples: **142**

Positive: **71**

Negative: **71**

## MethaneFuse zero-shot result

- AUROC: **0.6358**
- Accuracy: **0.5915**
- Recall: **0.6056**
- FPR: **0.4225**
- Loss: **0.9799**

Confusion matrix:

| | Pred negative | Pred positive |
|---|---:|---:|
| Actual negative | 41 | 30 |
| Actual positive | 28 | 43 |

---

# Main finding

MethaneFuse retains **modest but consistent zero-shot separability**
on MethaneSAT after substantially reducing geographic and site/background
confounding through same-location temporal pairing.

STRICT AUROC = **0.633**

EXPANDED AUROC = **0.636**

The difference between the two AUROCs is only:

**0.0031**

The near-identical performance under the stricter and expanded negative
definitions indicates that the observed ~0.63 AUROC is not driven solely
by the Tier B temporal controls.

The remaining signal is real enough to produce above-random ranking,
but performance is modest rather than strong.

---

# Professor / poster-ready conclusion

> After controlling for geographic and site/background confounding using
> same-location temporal no-detection controls, MethaneFuse retained modest
> but consistent zero-shot discrimination on MethaneSAT. Performance was
> nearly identical for the strict zero-detection control set
> (AUROC = 0.633, 56 pairs) and the expanded
> no-local-detection set (AUROC = 0.636, 71
> pairs), suggesting limited but reproducible cross-sensor separability.

---

# Important limitations

1. Temporal negatives are no-detection controls, not confirmed methane-free
   ground truth.

2. MethaneSAT is not a native released MethaneFuse sensor branch.

3. MethaneSAT XCH4 is passed through an S5P-style model input representation,
   so this should be described as experimental cross-sensor zero-shot
   evaluation.

4. Positive and negative acquisitions occur at different times, so temporal
   atmospheric and environmental variation is not fully eliminated.

---

# Freeze decision

No additional MethaneSAT downloads, negative mining, threshold tuning,
or dataset expansion are required for this analysis.

**STATUS: COMPLETE / FREEZE**
