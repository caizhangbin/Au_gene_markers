#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def parse_args():
    p = argparse.ArgumentParser(
        description="Elastic Net (glmnet-style) regression to predict continuous SCC from OG presence/absence."
    )
    p.add_argument("--input", required=True, help="OG presence/absence table (TSV). Must contain sample id column + OG columns.")
    p.add_argument("--meta", required=True, help="Metadata TSV containing sample id + SCC numeric value.")
    p.add_argument("--id_col", required=True, help="Sample ID column name (shared between input and metadata).")
    p.add_argument("--scc_col", required=True, help="Column in metadata containing continuous SCC (numeric).")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--test_size", type=float, default=0.2, help="Test fraction (default 0.2).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    p.add_argument("--log1p", action="store_true", help="Apply log1p transform to SCC (recommended if SCC is skewed).")
    p.add_argument("--max_iter", type=int, default=20000, help="Max iterations for solver (default 20000).")
    p.add_argument("--n_alphas", type=int, default=100, help="Number of alphas in CV (default 100).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # -----------------------------
    # Read input tables
    # -----------------------------
    X_df = pd.read_csv(args.input, sep="\t")
    meta = pd.read_csv(args.meta, sep="\t")

    if args.id_col not in X_df.columns:
        raise ValueError(f"--id_col '{args.id_col}' not found in --input columns.")
    if args.id_col not in meta.columns:
        raise ValueError(f"--id_col '{args.id_col}' not found in --meta columns.")
    if args.scc_col not in meta.columns:
        raise ValueError(f"--scc_col '{args.scc_col}' not found in --meta columns.")

    # Merge
    df = X_df.merge(meta[[args.id_col, args.scc_col]], on=args.id_col, how="inner")

    if df.shape[0] == 0:
        raise ValueError("No samples matched between --input and --meta after merging. Check sample IDs.")

    # Target reminder
    y = pd.to_numeric(df[args.scc_col], errors="coerce")
    if y.isna().any():
        bad = df.loc[y.isna(), args.id_col].tolist()[:10]
        raise ValueError(
            f"SCC column '{args.scc_col}' contains non-numeric/NA values. Example bad samples: {bad}"
        )

    if args.log1p:
        y = np.log1p(y)

    # Identify OG columns safely: columns that start with "OG"
    og_cols = [c for c in df.columns if c.startswith("OG")]

    if len(og_cols) == 0:
        raise ValueError("No OG columns found (expected columns starting with 'OG').")

    X = df[og_cols].copy()

    # Force to numeric 0/1 (safe)
    for c in og_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(int)

    # -----------------------------
    # Train/test split
    # -----------------------------
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, df[args.id_col],
        test_size=args.test_size,
        random_state=args.seed
    )

    # -----------------------------
    # Elastic Net CV model
    # -----------------------------
    # l1_ratio controls mixing:
    # 1.0 = Lasso, 0.0 = Ridge, in between = Elastic Net
    l1_grid = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95]

    model = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("enet", ElasticNetCV(
            l1_ratio=l1_grid,
            n_alphas=args.n_alphas,
            cv=5,
            random_state=args.seed,
            max_iter=args.max_iter
        ))
    ])

    model.fit(X_train, y_train)

    # Predictions
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Metrics
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    enet = model.named_steps["enet"]

    # -----------------------------
    # Output: model summary
    # -----------------------------
    summary_path = os.path.join(args.outdir, "elastic_net_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Elastic Net SCC regression (glmnet-style)\n")
        f.write("=====================================\n\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Metadata: {args.meta}\n")
        f.write(f"ID column: {args.id_col}\n")
        f.write(f"SCC column: {args.scc_col}\n")
        f.write(f"log1p SCC: {args.log1p}\n\n")
        f.write(f"N samples: {df.shape[0]}\n")
        f.write(f"N OG features: {len(og_cols)}\n")
        f.write(f"Train size: {X_train.shape[0]}\n")
        f.write(f"Test size: {X_test.shape[0]}\n\n")
        f.write(f"Best alpha (lambda): {enet.alpha_}\n")
        f.write(f"Best l1_ratio: {enet.l1_ratio_}\n\n")
        f.write(f"Train R^2: {r2_train:.4f}\n")
        f.write(f"Test  R^2: {r2_test:.4f}\n")
        f.write(f"Test  MAE: {mae_test:.4f}\n")
        f.write(f"Test RMSE: {rmse_test:.4f}\n")

    # -----------------------------
    # Output: coefficients (selected markers)
    # -----------------------------
    coefs = pd.Series(enet.coef_, index=og_cols)

    coef_table = (
        coefs.to_frame("coef")
        .assign(abs_coef=lambda d: d["coef"].abs())
        .sort_values("abs_coef", ascending=False)
    )

    coef_table_path = os.path.join(args.outdir, "elastic_net_coefficients_all.tsv")
    coef_table.to_csv(coef_table_path, sep="\t")

    # Non-zero coefficients = selected genes
    selected = coef_table[coef_table["coef"] != 0].copy()
    selected_path = os.path.join(args.outdir, "elastic_net_selected_markers.tsv")
    selected.to_csv(selected_path, sep="\t")

    # -----------------------------
    # Output: predictions
    # -----------------------------
    pred_df = pd.DataFrame({
        args.id_col: id_test.values,
        "y_true": y_test,
        "y_pred": y_pred_test
    })

    pred_path = os.path.join(args.outdir, "elastic_net_predictions_test.tsv")
    pred_df.to_csv(pred_path, sep="\t", index=False)

    # Print quick summary
    print(f"[Done] Saved results to: {args.outdir}")
    print(f"Best alpha: {enet.alpha_}")
    print(f"Best l1_ratio: {enet.l1_ratio_}")
    print(f"Test R^2: {r2_test:.4f} | MAE: {mae_test:.4f} | RMSE: {rmse_test:.4f}")
    print(f"Selected markers (non-zero coef): {selected.shape[0]}")
    print(f"Summary: {summary_path}")
    print(f"All coefs: {coef_table_path}")
    print(f"Selected: {selected_path}")
    print(f"Predictions: {pred_path}")


if __name__ == "__main__":
    main()
