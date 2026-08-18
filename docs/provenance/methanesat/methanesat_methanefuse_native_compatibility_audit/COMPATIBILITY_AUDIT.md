# MethaneSAT → MethaneFuse native compatibility audit

**Decision: NO_NATIVE_METHANESAT_SUPPORT**

- Model default sensors: `['s2', 'l89', 's5p', 'wv3']`
- Loader prefixes: `['emit', 'l89', 's2', 's5p', 'wv3', 'emit->wv3']`
- Python files mentioning MethaneSAT: `[]`
- S5P repeats single-channel input to expected channels: `True`
- S5P normalization applied: `True`
- S5P stats channel count: `3`
- Example MethaneSAT pair shape: `[224, 224]`

## Decision

Do **not** run the released MethaneFuse checkpoint on MethaneSAT by relabeling MethaneSAT as S5P.

The S5P loader may mechanically accept a single-channel NPZ and repeat channels, but that uses S5P-specific preprocessing/sensor identity and is not a MethaneSAT-native evaluation.
