# Methane Remote-Sensing Research Code

Curated research-code snapshot consolidated from the Mac workstation, laboratory SMB server, and FIR/Alliance HPC on **2026-08-17**.

## What was cleaned

- Harvested source candidates from all three environments.
- Removed obvious Python environments, Google Cloud SDK/vendor files, caches, and unmodified upstream MethaneFuse code.
- Removed exact SHA256 duplicates.
- Collapsed explicit version chains and backup/before-fix copies to their latest canonical versions.
- Kept project-specific MethaneFuse changes under `methanefuse_overrides/` instead of vendoring the full upstream repository.
- Excluded external controlled-release code/notebooks whose ownership is separate from this project.
- Ran a high-confidence credential scan before packaging.
- Ran `py_compile` syntax checks on all retained Python files.

**Retained files:** 397

**Python syntax failures:** 0

## Repository layout

```text
scripts/
  aviris3/
  common/
  emit/
  enmap/
  landsat/
  methaneair/
  methanesat/
  methanefuse/
  multisensor/
  sentinel2/
  stanford_controlled_release/
  unep_mars/

hpc/
  ... SLURM / SBATCH launchers ...

methanefuse_overrides/
  ... project-modified upstream internals ...

legacy/carbon_mapper/
  ... retained for historical reproducibility, not as the current main validation source ...

docs/provenance/
  ... final-freeze and active-workflow documentation ...

docs/curation/
  PROVENANCE.csv
  CANONICAL_VERSIONS.csv
  EXCLUDED_FILES.csv
  LAB_ZERO_BYTE_AUDIT.csv
  PYTHON_SYNTAX_AUDIT.csv
```

## Canonical-version policy

When several files were clearly successive versions of the same task, only the latest working branch was retained. Examples include:

- EnMAP primary downloader → `download_enmap_primary721_v4_safe_partials.py`
- EnMAP all-methane matcher → `match_all_methane_to_enmap_v4_timestamp_repaired.py`
- Stanford EnMAP downloader → `download_stanford_enmap_v4.py`
- Stanford Landsat downloader → `download_stanford_landsat_v3.py`
- EMIT adapter → `prepare_emit_for_methanefuse_v4_resume_rflonly.py`
- MethaneAIR/S2 corrected QA → `methaneair_s2_corrected_qa_v7_1.py`
- MethaneAIR same-parent benchmark builder → `build_methaneair_sameparent_paired_benchmark_v11_2.py`
- MethaneFuse LoRA-MoE adapter → final `finetune_loramoe_adapter.py`, with `before_*` copies removed

The complete decision table is in `docs/curation/CANONICAL_VERSIONS.csv`.

## Important limitation: lab zero-byte entries

The lab harvest reported **319 zero-byte files**. Most are package/environment debris or have a non-zero copy on Mac/FIR. After reviewing the filenames, the two clearly project-specific zero-byte scripts for which no non-zero same-name copy was found are **`build_mars_download_lists.py`** and **`build_mars_priority_sites.py`**. A zero-byte file contains no recoverable source, so these are documented rather than reconstructed or invented. The full audit is in `docs/curation/LAB_ZERO_BYTE_AUDIT.csv`.

## Before pushing to GitHub

1. Create the repository as **private** first.
2. Review `docs/curation/EXCLUDED_FILES.csv` and `PROVENANCE.csv`.
3. Confirm the desired license with the supervisor/institution.
4. Do not add raw imagery, NetCDFs, checkpoints, credentials, or result directories; `.gitignore` already excludes common cases.
5. Upstream MethaneFuse should remain a separate dependency/fork; apply or copy only the project-specific overrides as needed.

## Reproducibility note

Scripts retain their original canonical filenames and, in many cases, their original environment-specific paths. They were organized without mass-refactoring internals so that the archived logic remains faithful to the versions actually used during the project. See `PATHS_AND_PORTABILITY.md`.
