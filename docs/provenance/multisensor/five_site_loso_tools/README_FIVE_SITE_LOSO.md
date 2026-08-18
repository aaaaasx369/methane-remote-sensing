# MethaneFuse five-site leave-one-site-out fine-tuning

This switches the experiment from checkpoint-only evaluation to five-fold
leave-one-site-out Stage B adaptation.

For each fold:

- Four sites provide training data.
- A validation subset is selected only from those four sites.
- The fifth site remains completely held out until final evaluation.
- The released 480 m classification checkpoint initializes the model.
- The base encoder and patch embeddings remain frozen.
- LoRA-MoE adapters and classification/fusion heads are updated.

## Scope limitation

The 75 rows do not use one uniform physical ground-truth definition. Casa Grande
and Ehrenberg use controlled-release-derived labels, while MethaneAIR sites use
positive plume observations and no-known-plume references. Current S2 temporal
slots may also repeat one acquisition and use synthetic 12-band conversion.

The correct experiment name is:

**Exploratory five-site heterogeneous-label LOSO MethaneFuse adaptation**

## Prepare on the Mac

```bash
python prepare_methanefuse_five_site_loso.py \
  --input /Users/happydoraaa/methane_release_project/outputs/52_methanefuse_smoke_test.csv \
  --output-root /Users/happydoraaa/MethaneFuse/data/custom/five_site_loso \
  --bundle-images
```

Audit:

```bash
cat /Users/happydoraaa/MethaneFuse/data/custom/five_site_loso/folds.csv
cat /Users/happydoraaa/MethaneFuse/data/custom/five_site_loso/split_audit.csv
```

## Submit on an Alliance cluster

Place the Slurm script and aggregation script in the MethaneFuse repository root.
Edit or export `REPO_ROOT`, `DATA_ROOT`, `VENV_PATH`, `INITIAL_CHECKPOINT`,
`PANOPTICON_WEIGHTS`, and `WV3_SRF` when your cluster paths differ.

```bash
mkdir -p logs
sbatch run_methanefuse_five_site_loso.slurm
squeue -u "$USER"
```

## Aggregate

```bash
python aggregate_methanefuse_loso_results.py \
  --data-root data/custom/five_site_loso \
  --results-root results/five_site_loso
```

Primary output:

```text
results/five_site_loso/five_site_loso_summary.csv
```
