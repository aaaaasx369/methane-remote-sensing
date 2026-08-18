Dataset: Five-site multisource methane evaluation dataset
Total rows: 75
Positive labels: 15
Negative labels: 60

Label definition:
- label = 1: known methane release or positive plume observation
- label = 0: matched negative or no-known-plume reference

Source composition:
- Casa Grande controlled release: 3 positive, 12 negative
- Ehrenberg controlled release: 6 positive, 24 negative
- MethaneAIR sites: 6 positive, 24 negative

Important limitations:
1. MethaneAIR negative samples are no-known-plume references, not confirmed zero-emission observations.
2. Only a subset of controlled-release Sentinel-2 acquisitions falls strictly inside the release interval.
3. The converted TIFFs are for pipeline smoke testing:
   - missing Sentinel-2 bands were filled with neutral mean values;
   - s2_0_path, s2_90_path, and s2_360_path currently point to the same image.
4. Model accuracy from this package must not be presented as a formal MethaneFuse benchmark.
