#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import tifffile
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score

BANDS = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]
IDX = {b:i for i,b in enumerate(BANDS)}

def load_12band(path):
    a = tifffile.imread(path)
    if a.ndim != 3:
        raise ValueError(f"{path}: expected 3D TIFF, got {a.shape}")
    if a.shape[0] == 12:
        chw = a
    elif a.shape[-1] == 12:
        chw = np.moveaxis(a, -1, 0)
    else:
        raise ValueError(f"{path}: cannot find 12-band axis in {a.shape}")
    return chw.astype(np.float64)

def valid_pair(x, y):
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    return m

def corr(x, y):
    m = valid_pair(x, y)
    if m.sum() < 50 or np.nanstd(x[m]) == 0 or np.nanstd(y[m]) == 0:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0,1])

def ndvi(a):
    b8, b4 = a[IDX["B8"]], a[IDX["B4"]]
    return (b8 - b4) / (b8 + b4 + 1e-9)

def swir_proxy(a):
    # Diagnostic only: not a quantitative CH4 retrieval.
    # Log B11/B12 ratio is used as a stable methane-sensitive SWIR contrast proxy.
    b11, b12 = a[IDX["B11"]], a[IDX["B12"]]
    m = np.isfinite(b11) & np.isfinite(b12) & (b11 > 0) & (b12 > 0)
    z = np.full(b11.shape, np.nan, dtype=float)
    z[m] = np.log((b11[m] + 1e-9) / (b12[m] + 1e-9))
    return z

def summarize_delta(t0, ref):
    z0, zr = swir_proxy(t0), swir_proxy(ref)
    d = z0 - zr
    v = d[np.isfinite(d)]
    if v.size == 0:
        return {k: np.nan for k in ["mean","median","std","p90","p95","abs_mean"]}
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
        "p90": float(np.percentile(v,90)),
        "p95": float(np.percentile(v,95)),
        "abs_mean": float(np.mean(np.abs(v))),
    }

def resolve_path(p, project_root):
    p = str(p)
    path = Path(p)
    if path.exists():
        return path
    # The CSV normally stores absolute Mac paths. If moved, try rebuilding below project_root
    marker = "/methanefuse_input/"
    if marker in p:
        candidate = project_root / "methanefuse_input" / p.split(marker,1)[1]
        if candidate.exists():
            return candidate
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval-csv",
        default="/Users/happydoraaa/MethaneFuse/data/custom/five_site_zero_shot_eval.csv",
    )
    ap.add_argument(
        "--project-root",
        default="/Users/happydoraaa/methane_release_project",
    )
    ap.add_argument(
        "--output-dir",
        default="/Users/happydoraaa/methane_release_project/outputs/reference_similarity_test",
    )
    args = ap.parse_args()

    project_root = Path(args.project_root)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.eval_csv)
    rows = []

    for i, r in df.iterrows():
        try:
            p0 = resolve_path(r["s2_0_path"], project_root)
            p90 = resolve_path(r["s2_90_path"], project_root)
            p360 = resolve_path(r["s2_360_path"], project_root)

            t0, t90, t360 = map(load_12band, [p0,p90,p360])

            c90 = corr(t0[IDX["B4"]], t90[IDX["B4"]])
            c360 = corr(t0[IDX["B4"]], t360[IDX["B4"]])

            if np.isnan(c90) and np.isnan(c360):
                best_name, best_corr, ref = "none", np.nan, t90
            elif np.isnan(c360) or (not np.isnan(c90) and c90 >= c360):
                best_name, best_corr, ref = "t90", c90, t90
            else:
                best_name, best_corr, ref = "t360", c360, t360

            d90 = summarize_delta(t0, t90)
            d360 = summarize_delta(t0, t360)
            dbest = summarize_delta(t0, ref)

            n0 = ndvi(t0)
            nr = ndvi(ref)
            ndvi0 = float(np.nanmedian(n0))
            ndvir = float(np.nanmedian(nr))

            rows.append({
                "id": r["id"],
                "site": r["site"],
                "label": int(r["label"]),
                "minimum_scl_clear_fraction": r.get("minimum_scl_clear_fraction", np.nan),
                "corr_B4_t0_t90": c90,
                "corr_B4_t0_t360": c360,
                "best_reference": best_name,
                "best_B4_corr": best_corr,
                "reference_corr_ge_0p7": bool(best_corr >= 0.7) if np.isfinite(best_corr) else False,
                "ndvi_t0_median": ndvi0,
                "ndvi_best_ref_median": ndvir,
                "abs_ndvi_change": abs(ndvi0-ndvir),
                "swir_delta_best_mean": dbest["mean"],
                "swir_delta_best_median": dbest["median"],
                "swir_delta_best_std": dbest["std"],
                "swir_delta_best_p90": dbest["p90"],
                "swir_delta_best_p95": dbest["p95"],
                "swir_delta_best_abs_mean": dbest["abs_mean"],
                "swir_delta_t90_abs_mean": d90["abs_mean"],
                "swir_delta_t360_abs_mean": d360["abs_mean"],
                "error": "",
            })
        except Exception as e:
            rows.append({
                "id": r.get("id", i),
                "site": r.get("site", ""),
                "label": r.get("label", np.nan),
                "error": repr(e),
            })

    result = pd.DataFrame(rows)
    result.to_csv(outdir / "reference_similarity_per_sample.csv", index=False)

    good = result[result["error"].fillna("").eq("")].copy()
    print("\n=== REFERENCE SIMILARITY TEST ===")
    print("rows:", len(result), "successful:", len(good), "errors:", len(result)-len(good))

    if len(good):
        print("\nBest-reference choice:")
        print(good["best_reference"].value_counts(dropna=False).to_string())
        print("\nB4 correlation >= 0.7:", int(good["reference_corr_ge_0p7"].sum()), "/", len(good))

        print("\nMedian best B4 correlation by label:")
        print(good.groupby("label")["best_B4_corr"].median().to_string())

        print("\nMedian SWIR |temporal difference| by label:")
        print(good.groupby("label")["swir_delta_best_abs_mean"].median().to_string())

        print("\nWithin-site summary:")
        print(
            good.groupby(["site","label"])
            .agg(
                n=("id","size"),
                median_best_corr=("best_B4_corr","median"),
                median_abs_ndvi_change=("abs_ndvi_change","median"),
                median_swir_abs=("swir_delta_best_abs_mean","median"),
            )
            .to_string()
        )

        # Purely diagnostic univariate AUROCs. Flip if lower values indicate positive more strongly.
        if good["label"].nunique() == 2:
            for col in ["best_B4_corr","abs_ndvi_change","swir_delta_best_abs_mean"]:
                d = good[["label",col]].dropna()
                if d["label"].nunique() == 2:
                    auc = roc_auc_score(d["label"], d[col])
                    print(f"\nAUROC(label vs {col}): {auc:.3f} (orientation-sensitive)")

    print("\nOutput:", outdir / "reference_similarity_per_sample.csv")

if __name__ == "__main__":
    main()
