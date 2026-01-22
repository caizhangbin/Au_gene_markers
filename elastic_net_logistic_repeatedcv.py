#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Elastic Net Logistic Regression (glmnet-style) for marker genes using nested repeated CV."
    )
    p.add_argument("--input", required=True,
                   help="TSV file: sample + OG presence/absence + target column (e.g., biomarkers_three_classes.tsv)")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--id_col", required=True, help="Sample ID column name (e.g., sample)")
    p.add_argument("--targets_col", required=True, help="Target/category column name (e.g., target_column)")
    p.add_argument("--toi", required=True,
                   help="Target of interest label (positive class), e.g. level_1 or level_2 or level_3")

    # Outer CV settings
    p.add_argument("--outer_splits", type=int, default=5, help="Outer folds (default 5)")
    p.add_argument("--outer_repeats", type=int, default=20, help="Outer repeats (default 20)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Inner CV settings for LogisticRegressionCV
    p.add_argument("--inner_cv", type=int, default=5, help="Inner folds (default 5)")
    p.add_argument("--Cs", type=int, default=20,
                   help="Number of inverse-regularization values (C grid size) (default 20)")

    # Optional filtering
    p.add_argument("--drop_class", action="append", default=[],
                   help="Drop a class label entirely (can be used multiple times). "
                        "Example: --drop_class level_3 to remove Middle SCC.")

    return p.parse_args()


def to01_series(x: pd.Series) -> pd.Series:
    """Convert mixed input to strict 0/1 integers; NA -> 0."""
    x = x.astype(str)
    x = np.where(np.isin(x, ["1", "TRUE", "T", "True", "true"]), 1,
                 np.where(np.isin(x, ["0", "FALSE", "F", "False", "false"]), 0, np.nan))
    x = pd.Series(x).astype("float").fillna(0).astype(int)
    return x


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # -----------------------------
    # Read input
    # -----------------------------
    df = pd.read_csv(args.input, sep="\t")

    for col in [args.id_col, args.targets_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in input file.")

    # Optional: drop one or more classes (e.g. remove Middle SCC)
    if args.drop_class:
        df = df[~df[args.targets_col].isin(args.drop_class)].copy()

    # Binary target: TOI vs rest
    y = (df[args.targets_col].astype(str) == str(args.toi)).astype(int).to_numpy()

    n_pos = int(y.sum())
    n_total = len(y)
    n_neg = n_total - n_pos
    if n_pos < 5 or n_neg < 5:
        raise ValueError(f"Too few samples for classification: positives={n_pos}, negatives={n_neg}.")

    # OG columns
    og_cols = [c for c in df.columns if c.startswith("OG")]
    if len(og_cols) == 0:
        raise ValueError("No OG columns found (expected columns starting with 'OG').")

    X = df[og_cols].copy()
    for c in og_cols:
        X[c] = to01_series(X[c])

    sample_ids = df[args.id_col].astype(str).to_numpy()

    # -----------------------------
    # Outer repeated stratified CV
    # -----------------------------
    outer_cv = RepeatedStratifiedKFold(
        n_splits=args.outer_splits,
        n_repeats=args.outer_repeats,
        random_state=args.seed
    )

    # Marker stability counter
    selected_counts = pd.Series(0, index=og_cols, dtype=int)

    fold_rows = []
    pred_rows = []

    fold_idx = 0
    for train_idx, test_idx in outer_cv.split(X, y):
        fold_idx += 1

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # -----------------------------
        # Inner CV for hyperparameters (glmnet-style)
        # LogisticRegressionCV chooses best C and l1_ratio
        # -----------------------------
        # NOTE: saga solver is required for elasticnet penalty
        model = Pipeline(steps=[
            # Scaling is fine even for 0/1 features; helps optimizer
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegressionCV(
                penalty="elasticnet",
                solver="saga",
                scoring="roc_auc",
                cv=args.inner_cv,
                Cs=args.Cs,
                l1_ratios=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95],
                max_iter=20000,
                n_jobs=1,             # safer on HPC nodes; increase if needed
                refit=True,
                class_weight="balanced",
                random_state=args.seed
            ))
        ])

        model.fit(X_train, y_train)

        clf = model.named_steps["clf"]

        # predicted probabilities for AUC
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Metrics
        auc = roc_auc_score(y_test, y_prob)
        balacc = balanced_accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        # Best params chosen inside inner CV
        # C_ and l1_ratio_ can be arrays; extract first value
        best_C = float(np.array(clf.C_).ravel()[0])
        best_l1 = float(np.array(clf.l1_ratio_).ravel()[0])

        fold_rows.append({
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "positives_train": int(y_train.sum()),
            "positives_test": int(y_test.sum()),
            "best_C": best_C,
            "best_l1_ratio": best_l1,
            "AUC": float(auc),
            "balanced_accuracy": float(balacc),
            "MCC": float(mcc),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
        })

        # Save predictions for this fold
        for sid, yt, yp, pr in zip(sample_ids[test_idx], y_test, y_pred, y_prob):
            pred_rows.append({
                args.id_col: sid,
                "y_true": int(yt),
                "y_pred": int(yp),
                "y_prob": float(pr),
                "fold": fold_idx
            })

        # Marker selection: non-zero coefficients
        coef = pd.Series(clf.coef_.ravel(), index=og_cols)
        selected = coef[coef != 0].index
        selected_counts.loc[selected] += 1

    folds_df = pd.DataFrame(fold_rows)
    preds_df = pd.DataFrame(pred_rows)

    # Average per-sample probability (a stable summary across repeats)
    preds_summary = preds_df.groupby(args.id_col, as_index=False).agg(
        y_true=("y_true", "mean"),
        y_prob_mean=("y_prob", "mean"),
        y_prob_sd=("y_prob", "std"),
        n_predictions=("y_prob", "count"),
    )
    preds_summary["y_pred_mean"] = (preds_summary["y_prob_mean"] >= 0.5).astype(int)

    # Overall performance using per-sample mean probabilities
    overall_auc = roc_auc_score(preds_summary["y_true"], preds_summary["y_prob_mean"])
    overall_balacc = balanced_accuracy_score(preds_summary["y_true"], preds_summary["y_pred_mean"])
    overall_mcc = matthews_corrcoef(preds_summary["y_true"], preds_summary["y_pred_mean"])

    # Marker stability table
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
    stability_path = os.path.join(args.outdir, "elastic_net_logistic_marker_stability.tsv")

    folds_df.to_csv(folds_path, sep="\t", index=False)
    preds_df.to_csv(preds_path, sep="\t", index=False)
    preds_summary.to_csv(preds_summary_path, sep="\t", index=False)
    stability.to_csv(stability_path, sep="\t", index=False)

    # Human-readable summary
    summary_path = os.path.join(args.outdir, "elastic_net_logistic_repeatedCV_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Elastic Net Logistic Regression (glmnet-style) with repeated nested CV\n")
        f.write("===============================================================\n\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"TOI (positive class): {args.toi}\n")
        f.write(f"Dropped classes: {args.drop_class}\n\n")
        f.write(f"N samples: {n_total} (pos={n_pos}, neg={n_neg})\n")
        f.write(f"N OG features: {len(og_cols)}\n\n")
        f.write(f"Outer CV: {args.outer_splits}-fold x {args.outer_repeats} repeats = {total_fits} fits\n")
        f.write(f"Inner CV: {args.inner_cv}-fold (AUC scoring)\n\n")

        f.write("Fold-level performance (mean ± SD):\n")
        f.write(f"  AUC : {folds_df['AUC'].mean():.4f} ± {folds_df['AUC'].std():.4f}\n")
        f.write(f"  BalAcc : {folds_df['balanced_accuracy'].mean():.4f} ± {folds_df['balanced_accuracy'].std():.4f}\n")
        f.write(f"  MCC : {folds_df['MCC'].mean():.4f} ± {folds_df['MCC'].std():.4f}\n\n")

        f.write("Overall performance (per-sample mean probability):\n")
        f.write(f"  AUC : {overall_auc:.4f}\n")
        f.write(f"  BalAcc : {overall_balacc:.4f}\n")
        f.write(f"  MCC : {overall_mcc:.4f}\n\n")

        f.write("Top stable markers (first 20 OGs):\n")
        f.write(stability.head(20).to_string(index=False))
        f.write("\n")

    print(f"[Done] Saved outputs to: {args.outdir}")
    print(f"Fold-level AUC mean ± SD: {folds_df['AUC'].mean():.4f} ± {folds_df['AUC'].std():.4f}")
    print(f"Overall AUC (avg probs): {overall_auc:.4f}")
    print(f"Summary: {summary_path}")
    print(f"Marker stability: {stability_path}")


if __name__ == "__main__":
    main()
