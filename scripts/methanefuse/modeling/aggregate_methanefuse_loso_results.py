#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/custom/five_site_loso")
    parser.add_argument("--results-root", default="results/five_site_loso")
    parser.add_argument(
        "--output",
        default="results/five_site_loso/five_site_loso_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    results_root = Path(args.results_root)
    output = Path(args.output)

    folds = pd.read_csv(data_root / "folds.csv")
    rows = []

    for _, fold in folds.iterrows():
        json_path = results_root / f"{fold['fold_slug']}_heldout.json"
        if not json_path.exists():
            rows.append(
                {
                    "fold_index": fold["fold_index"],
                    "held_out_site": fold["held_out_site"],
                    "status": "missing_result",
                    "result_json": str(json_path),
                }
            )
            continue

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        overall = payload.get("overall", {})
        rows.append(
            {
                "fold_index": int(fold["fold_index"]),
                "fold_slug": fold["fold_slug"],
                "held_out_site": fold["held_out_site"],
                "train_rows": int(fold["train_rows"]),
                "val_rows": int(fold["val_rows"]),
                "test_rows": int(fold["test_rows"]),
                "status": "complete",
                "checkpoint": payload.get("checkpoint"),
                "count": payload.get("count", overall.get("count")),
                "loss": payload.get("loss"),
                "acc": overall.get("acc"),
                "auroc": overall.get("auroc"),
                "fpr": overall.get("fpr"),
                "recall": overall.get("recall"),
                "result_json": str(json_path),
            }
        )

    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    print("\nPer-site LOSO results:")
    print(result.to_string(index=False))

    complete = result[result["status"] == "complete"].copy()
    if len(complete):
        print("\nUnweighted mean across held-out sites:")
        for metric in ["acc", "auroc", "fpr", "recall"]:
            values = pd.to_numeric(complete[metric], errors="coerce")
            print(
                f"{metric}: mean={values.mean():.6f}, "
                f"std={values.std(ddof=1):.6f}, valid_folds={values.notna().sum()}"
            )

    print("\nCreated:")
    print(output.resolve())


if __name__ == "__main__":
    main()
