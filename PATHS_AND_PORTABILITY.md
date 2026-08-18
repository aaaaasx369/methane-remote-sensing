# Paths and portability

Many research scripts were originally executed directly on one of three environments:

- Mac (`~/methane_release_project`)
- laboratory SMB storage
- FIR / Alliance HPC

The curation step intentionally did **not** bulk-edit hard-coded paths inside scripts, because doing so without rerunning every workflow could silently change reproducibility. `docs/curation/PROVENANCE.csv` records where every retained file came from.

For future cleanup, migrate paths into command-line arguments or environment variables one workflow at a time and test each workflow before replacing the archived canonical script.
