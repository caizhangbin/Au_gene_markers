#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def parse_args():
    p = argparse.ArgumentParser(
        description="Elastic Net (glmnet-style) regression to predict continuous SCC from OG presence/absence using repeated CV."
    )
    p.add_argument("--input", required=True, help="OG presence/absence table (TSV). Must contain sample id column + OG columns.")
    p.add_argument("--meta", required=True, help="Metadata TSV containing sample id + SCC numeric value.")
    p.add_argument("--id_col", required=True, help="Sample ID column name (shared between input and metadata).")
    p.add_argument("--scc_col", required=True, help="Column in metadata containing continuous SCC (numeric).")
    p.add_argument("--outdir", required=True, help="Output directory.")

    # CV settings
    p.add_argument("--outer_splits", type=int, default=5, help="Outer CV folds (default 5).")
    p.add_argument("--outer_repeats", type=int, default=20, help="Outer CV repeats (default 20).")
    p.add_argument("--inner_cv", type=int, default=5, help="Inner CV folds for ElasticNetCV (default 5).")

    # Elastic Net settings
    p.add_argument("--log1p", action="store_true", help="Apply log1p transform to SCC (recommended if SCC is skewed).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    p.add_argument("--max_iter", type=int, default=30000, help="Max iterations for solver (default 30000).")
    p.add_argument("--n_alphas", type=int, default=100, help="Number of alphas in ElasticNetCV (default 100).")

    return p.parse_args()


def to01_series(x: pd.Series) -> pd.Series:
    """Convert mixed-type vector to strict 0/1 int with NA -> 0."""
    x = x.astype(str)
    x = np.where(np.isin(x, ["1", "TRUE", "T", "True", "true"]), 1,
                 np.where(np.isin(x, ["0", "FALSE", "F", "False", "false"]), 0, np.nan))
    x = pd.Series(x).astype("float")
    x = x.fillna(0).astype(int)
    return x


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # -----------------------------
    # Read input tables
    # -----------------------------
    X_df = pd.read_csv(args.input, sep="\t")
    meta = pd.read_csv(args.meta, sep="\t")

    if args.id_col not in X_df.columns:
        raise ValueError(f"--id_col '{args.id_col}' not found in --input.")
    if args.id_col not in meta.columns:
        raise ValueError(f"--id_col '{args.id_col}' not found in --meta.")
    if args.scc_col not in meta.columns:
        raise ValueError(f"--scc_col '{args.scc_col}' not found in --meta.")

    df = X_df.merge(meta[[args.id_col, args.scc_col]], on=args.id_col, how="inner")
    if df.shape[0] == 0:
        raise ValueError("No samples matched after merging input and metadata. Check sample IDs.")

    y = pd.to_numeric(df[args.scc_col], errors="coerce")
    if y.isna().any():
        bad = df.loc[y.isna(), args.id_col].tolist()[:10]
        raise ValueError(f"SCC column '{args.scc_col}' contains NA/non-numeric. Example bad samples: {bad}")

    if args.log1p:
        y = np.log1p(y.to_numpy())
    else:
        y = y.to_numpy()

    # OG columns
    og_cols = [c for c in df.columns if c.startswith("OG")]
    if len(og_cols) == 0:
        raise ValueError("No OG columns found (expected columns starting with 'OG').")

    X = df[og_cols].copy()
    for c in og_cols:
        X[c] = to01_series(X[c])

    sample_ids = df[args.id_col].astype(str).to_numpy()

    # -----------------------------
    # Repeated outer CV
    # -----------------------------
    outer_cv = RepeatedKFold(
        n_splits=args.outer_splits,
        n_repeats=args.outer_repeats,
        random_state=args.seed
    )

    # ElasticNetCV settings
    l1_grid = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]

    fold_rows = []
    pred_rows = []
    selected_counts = pd.Series(0, index=og_cols, dtype=int)

    fold_idx = 0
    for train_idx, test_idx in outer_cv.split(X):
        fold_idx += 1

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("enet", ElasticNetCV(
                l1_ratio=l1_grid,
                n_alphas=args.n_alphas,
                cv=args.inner_cv,
                random_state=args.seed,
                max_iter=args.max_iter
            ))
        ])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        enet = model.named_steps["enet"]

        # record fold metrics + params
        fold_rows.append({
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "best_alpha": float(enet.alpha_),
            "best_l1_ratio": float(enet.l1_ratio_),
            "R2": float(r2),
            "MAE": float(mae),
            "RMSE": float(rmse)
        })

        # record predictions per sample
        for sid, yt, yp in zip(sample_ids[test_idx], y_test, y_pred):
            pred_rows.append({
                args.id_col: sid,
                "y_true": float(yt),
                "y_pred": float(yp),
                "fold": fold_idx
            })

        # selected markers (non-zero coefficients)
        coefs = pd.Series(enet.coef_, index=og_cols)
        selected = coefs[coefs != 0].index
        selected_counts.loc[selected] += 1

    folds_df = pd.DataFrame(fold_rows)
    preds_df = pd.DataFrame(pred_rows)

    # Some samples appear multiple times in predictions across folds/repeats.
    # Summarize per sample by averaging predictions:
    preds_summary = preds_df.groupby(args.id_col, as_index=False).agg(
        y_true=("y_true", "mean"),
        y_pred_mean=("y_pred", "mean"),
        y_pred_sd=("y_pred", "std"),
        n_predictions=("y_pred", "count")
    )

    # Overall CV performance based on per-sample averaged predictions
    overall_r2 = r2_score(preds_summary["y_true"], preds_summary["y_pred_mean"])
    overall_mae = mean_absolute_error(preds_summary["y_true"], preds_summary["y_pred_mean"])
    overall_rmse = np.sqrt(mean_squared_error(preds_summary["y_true"], preds_summary["y_pred_mean"]))

    # marker stability table
    total_fits = args.outer_splits * args.outer_repeats
    stability = (
        selected_counts.to_frame("n_selected")
        .assign(selection_rate=lambda d: d["n_selected"] / total_fits)
        .sort_values(["selection_rate", "n_selected"], ascending=False)
        .reset_index()
        .rename(columns={"index": "Orthogroup"})
    )

    # -----------------------------
    # Save outputs
    # -----------------------------
    folds_path = os.path.join(args.outdir, "outerCV_fold_metrics.tsv")
    preds_path = os.path.join(args.outdir, "outerCV_predictions_all.tsv")
    preds_summary_path = os.path.join(args.outdir, "outerCV_predictions_summary.tsv")
    stability_path = os.path.join(args.outdir, "elastic_net_marker_stability.tsv")

    folds_df.to_csv(folds_path, sep="\t", index=False)
    preds_df.to_csv(preds_path, sep="\t", index=False)
    preds_summary.to_csv(preds_summary_path, sep="\t", index=False)
    stability.to_csv(stability_path, sep="\t", index=False)

    # Summary report
    summary_path = os.path.join(args.outdir, "elastic_net_repeatedCV_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Elastic Net SCC regression (glmnet-style) with repeated outer CV\n")
        f.write("==============================================================\n\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Metadata: {args.meta}\n")
        f.write(f"ID column: {args.id_col}\n")
        f.write(f"SCC column: {args.scc_col}\n")
        f.write(f"log1p SCC: {args.log1p}\n\n")
        f.write(f"N samples: {len(sample_ids)}\n")
        f.write(f"N OG features: {len(og_cols)}\n\n")
        f.write(f"Outer CV: {args.outer_splits}-fold x {args.outer_repeats} repeats = {total_fits} fits\n")
        f.write(f"Inner CV (ElasticNetCV): {args.inner_cv}-fold\n\n")
        f.write("Fold-level performance (mean ± SD across fits):\n")
        f.write(f"  R2   : {folds_df['R2'].mean():.4f} ± {folds_df['R2'].std():.4f}\n")
        f.write(f"  MAE  : {folds_df['MAE'].mean():.4f} ± {folds_df['MAE'].std():.4f}\n")
        f.write(f"  RMSE : {folds_df['RMSE'].mean():.4f} ± {folds_df['RMSE'].std():.4f}\n\n")
        f.write("Overall performance (per-sample average predictions):\n")
        f.write(f"  R2   : {overall_r2:.4f}\n")
        f.write(f"  MAE  : {overall_mae:.4f}\n")
        f.write(f"  RMSE : {overall_rmse:.4f}\n\n")
        f.write("Top marker stability (first 20 rows):\n")
        f.write(stability.head(20).to_string(index=False))
        f.write("\n")

    # Print quick summary
    print(f"[Done] Saved results to: {args.outdir}")
    print(f"Fold-level R2 mean ± SD: {folds_df['R2'].mean():.4f} ± {folds_df['R2'].std():.4f}")
    print(f"Overall (avg preds) R2: {overall_r2:.4f}")
    print(f"Outputs:")
    print(f"  {summary_path}")
    print(f"  {folds_path}")
    print(f"  {preds_summary_path}")
    print(f"  {stability_path}")


if __name__ == "__main__":
    main()
