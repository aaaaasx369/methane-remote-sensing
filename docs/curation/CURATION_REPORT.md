# Curation report — 2026-08-17

## Harvested inputs

- Mac raw harvested code-like files: 25319
- Lab raw harvested code-like files: 341
- FIR harvested files: 269

After filtering obvious environments/vendor trees and lab zero-byte entries, there were **677 meaningful non-zero candidates**.

After excluding external/unmodified upstream/generated-report material: **501** candidates.

After semantic version cleanup and SHA256 deduplication: **397 retained files**.

## Exclusion counts

- `upstream_methanefuse_clone`: 59
- `upstream_methanefuse_unmodified_clone`: 56
- `exact_sha256_duplicate`: 33
- `generated_result_summary`: 19
- `third_party_methaneair_controlled_release_code`: 18
- `third_party_stanford_notebooks`: 10
- `generated_output_log`: 6
- `intermediate_handoff_readme`: 6
- `superseded:finetune_loramoe_adapter`: 5
- `superseded:build_unified_methane_master`: 4
- `superseded:find_and_download_s2_reference_negatives`: 4
- `superseded:prepare_emit_for_methanefuse`: 4
- `superseded:audit_sentinel2_downloads`: 3
- `superseded:build_emit_real_temporal`: 3
- `superseded:download_enmap_primary721`: 3
- `superseded:download_sentinel2_matches_best_qa`: 3
- `superseded:evaluate_classification`: 3
- `superseded:export_methaneair_gee`: 3
- `superseded:integrate_aviris3_into_professor_master`: 3
- `superseded:make_methanefuse_smoke_dataset`: 3
- `superseded:match_all_methane_to_enmap`: 3
- `superseded:build_methaneair_sameparent_paired_benchmark`: 2
- `superseded:download_enmap_primary72h_nominal`: 2
- `superseded:download_stanford_enmap`: 2
- `superseded:repair_enmap_corrupt_partials_bulk`: 2
- `intermediate_dataset_readme`: 1
- `superseded:audit_landsat_matched_negative_pairing`: 1
- `superseded:build_final_confirmed_landsat_dataset`: 1
- `superseded:complete_s2_high_emission_wind_plan_with_era5land`: 1
- `superseded:download_and_qa_s2_high_emission_benchmark`: 1
- `superseded:download_sentinel2_matches_resume`: 1
- `superseded:download_stanford_landsat`: 1
- `superseded:evaluate_classification_with_predictions_copy`: 1
- `superseded:freeze_build_methaneair_368_methanefuse`: 1
- `superseded:methane_metadata_scanner_multi`: 1
- `superseded:methaneair_s2_corrected_qa`: 1
- `superseded:prepare_selected_methaneair_sites`: 1
- `superseded:run_methanefuse_two_negative_group_cv`: 1
- `superseded:search_landsat_matched_negative_candidates`: 1
- `superseded:search_methaneair_s2_scenes_gee`: 1
- `superseded:train_baseline_classifier`: 1
- `superseded:validate_methanesat_temporal_negatives_against_inventory`: 1
- `superseded:validate_s2_methaneair_overlap`: 1
- `superseded:visualize_wrong_predictions`: 1
- `superseded:visualize_wrong_predictions_swir`: 1
- `third_party_external_data`: 1

## Validation

- High-confidence secret scan flags remaining in packaged repo: 0 (flagged files, if any, were excluded)
- Python files syntax-checked: 348
- Python syntax failures: 0

## What was not done automatically

- No source files on Mac, lab SMB, or FIR were deleted or modified.
- No bulk rewrite of absolute paths was performed.
- No open-source license was selected automatically.
- The repository was not pushed directly to a GitHub account because no authenticated GitHub connector was available in this session.
