#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import inspect

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Multinomial Elastic Net Logistic Regression (3-class) with nested repeated CV + marker stability + optional TOI markers."
    )
    p.add_argument("--input", required=True, help="Input TSV (sample + OGs + target column).")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--id_col", required=True, help="Sample ID column name (e.g. sample).")
    p.add_argument("--targets_col", required=True, help="Target label column (e.g. target_column).")

    # Optional: focus a specific class for marker export
    p.add_argument("--toi", default=None,
                   help="Target-of-interest class label (e.g., level_2). If provided, outputs TOI marker table with direction.")

    p.add_argument("--outer_splits", type=int, default=5)
    p.add_argument("--outer_repeats", type=int, default=20)
    p.add_argument("--inner_cv", type=int, default=5)
    p.add_argument("--Cs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    # Optional: restrict to subset of classes
    p.add_argument("--keep_class", action="append", default=[],
                   help="Keep only these labels (can be repeated). Example: --keep_class level_1 --keep_class level_2 --keep_class level_3")

    return p.parse_args()


def to01_series(x: pd.Series) -> pd.Series:
    """Convert mixed OG presence/absence to strict 0/1; unknown/NA -> 0."""
    x = x.astype(str)
    x = np.where(np.isin(x, ["1", "TRUE", "T", "True", "true"]), 1,
                 np.where(np.isin(x, ["0", "FALSE", "F", "False", "false"]), 0, np.nan))
    return pd.Series(x).astype(float).fillna(0).astype(int)


def build_logregcv_kwargs(args):
    """
    Build kwargs for LogisticRegressionCV, only including parameters supported by the installed sklearn version.
    This avoids crashes due to API differences (like multi_class removed).
    """
    sig = inspect.signature(LogisticRegressionCV)
    supported = set(sig.parameters.keys())

    base_kwargs = dict(
        penalty="elasticnet",
        solver="saga",
        scoring="neg_log_loss",      # good for multiclass
        cv=args.inner_cv,
        Cs=args.Cs,
        l1_ratios=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95],
        max_iter=30000,
        n_jobs=1,
        refit=True,
        class_weight="balanced",
        random_state=args.seed
    )

    # Only add multi_class if it exists in this sklearn version
    # If not present, sklearn will automatically handle multiclass using appropriate strategy.
    if "multi_class" in supported:
        base_kwargs["multi_class"] = "multinomial"

    # Some versions support use_legacy_attributes; we do NOT need to set it
    return {k: v for k, v in base_kwargs.items() if k in supported}


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input, sep="\t")

    for col in [args.id_col, args.targets_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in input.")

    # Optionally filter classes
    if args.keep_class:
        df = df[df[args.targets_col].isin(args.keep_class)].copy()

    # Encode targets
    y_raw = df[args.targets_col].astype(str).to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 3:
        raise ValueError(f"Need 3 classes for multinomial run, found {len(classes)}: {list(le.classes_)}")

    # Resolve TOI index (optional)
    toi_idx = None
    toi_label = None
    if args.toi is not None:
        toi_label = str(args.toi)
        if toi_label not in set(le.classes_):
            raise ValueError(f"--toi '{toi_label}' not found in classes: {list(le.classes_)}")
        toi_idx = int(le.transform([toi_label])[0])

    # Extract OG features
    og_cols = [c for c in df.columns if c.startswith("OG")]
    if len(og_cols) == 0:
        raise ValueError("No OG columns found (must start with 'OG').")

    X = df[og_cols].copy()
    for c in og_cols:
        X[c] = to01_series(X[c])

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    sample_ids = df[args.id_col].astype(str).to_numpy()

    outer_cv = RepeatedStratifiedKFold(
        n_splits=args.outer_splits,
        n_repeats=args.outer_repeats,
        random_state=args.seed
    )

    selected_counts_any = pd.Series(0, index=og_cols, dtype=int)

    # TOI-specific aggregation
    selected_counts_toi = pd.Series(0, index=og_cols, dtype=int)
    coef_sum_toi = pd.Series(0.0, index=og_cols)
    coef_n_toi = pd.Series(0, index=og_cols, dtype=int)

    fold_rows = []
    pred_rows = []

    logreg_kwargs = build_logregcv_kwargs(args)

    fold_idx = 0
    for train_idx, test_idx in outer_cv.split(X, y):
        fold_idx += 1

        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test  = y[test_idx]

        model = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegressionCV(**logreg_kwargs))
        ])

        model.fit(X_train, y_train)
        clf = model.named_steps["clf"]

        proba = model.predict_proba(X_test)   # shape (n_test, n_classes)
        y_pred = np.argmax(proba, axis=1)

        balacc = balanced_accuracy_score(y_test, y_pred)
        f1mac  = f1_score(y_test, y_pred, average="macro")

        # Macro AUC (OVR) only valid if fold contains all classes
        auc_macro = np.nan
        if len(np.unique(y_test)) == len(le.classes_):
            auc_macro = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")

        fold_rows.append({
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "balanced_accuracy": float(balacc),
            "macro_F1": float(f1mac),
            "macro_AUC_ovr": float(auc_macro) if not np.isnan(auc_macro) else np.nan,
            "best_C": float(np.array(clf.C_).ravel()[0]),
            "best_l1_ratio": float(np.array(clf.l1_ratio_).ravel()[0]) if hasattr(clf, "l1_ratio_") else np.nan,
        })

        # Save fold predictions
        for sid, yt, yp, pr in zip(sample_ids[test_idx], y_test, y_pred, proba):
            row = {args.id_col: sid, "y_true": int(yt), "y_pred": int(yp), "fold": fold_idx}
            for i, lab in enumerate(le.classes_):
                row[f"prob_{lab}"] = float(pr[i])
            pred_rows.append(row)

        # Marker stability across ANY class
        coef_df = pd.DataFrame(clf.coef_, columns=og_cols)  # rows=classes, cols=features
        selected_any = coef_df.columns[(coef_df != 0).any(axis=0)]
        selected_counts_any.loc[selected_any] += 1

        # TOI marker stability + direction
        if toi_idx is not None:
            toi_coef = coef_df.iloc[toi_idx, :]  # coefficients for TOI class
            selected_toi = toi_coef.index[toi_coef != 0]
            selected_counts_toi.loc[selected_toi] += 1
            coef_sum_toi.loc[selected_toi] += toi_coef.loc[selected_toi].astype(float)
            coef_n_toi.loc[selected_toi] += 1

    folds_df = pd.DataFrame(fold_rows)
    preds_df = pd.DataFrame(pred_rows)

    total_fits = args.outer_splits * args.outer_repeats

    stability_any = (
        selected_counts_any.to_frame("n_selected_any")
        .assign(selection_rate_any=lambda d: d["n_selected_any"] / total_fits)
        .sort_values(["selection_rate_any", "n_selected_any"], ascending=False)
        .reset_index()
        .rename(columns={"index": "Orthogroup"})
    )

    # TOI table
    toi_markers_path = None
    if toi_idx is not None:
        mean_coef_selected = pd.Series(np.nan, index=og_cols)
        mask = coef_n_toi > 0
        mean_coef_selected.loc[mask] = coef_sum_toi.loc[mask] / coef_n_toi.loc[mask]

        stability_toi = pd.DataFrame({
            "Orthogroup": og_cols,
            "n_selected_toi": selected_counts_toi.values,
            "selection_rate_toi": (selected_counts_toi.values / total_fits),
            "mean_coef_toi_selectedFits": mean_coef_selected.values
        }).sort_values(["selection_rate_toi", "n_selected_toi"], ascending=False)

        stability_toi["direction_in_TOI"] = np.where(
            stability_toi["mean_coef_toi_selectedFits"] > 0, "enriched_in_TOI",
            np.where(stability_toi["mean_coef_toi_selectedFits"] < 0, "depleted_in_TOI", "NA")
        )

    # Save outputs
    folds_path = os.path.join(args.outdir, "outerCV_fold_metrics_multiclass.tsv")
    preds_path = os.path.join(args.outdir, "outerCV_predictions_all_multiclass.tsv")
    stability_any_path = os.path.join(args.outdir, "elastic_net_multiclass_marker_stability_ANY.tsv")
    class_map_path = os.path.join(args.outdir, "class_label_mapping.tsv")
    summary_path = os.path.join(args.outdir, "elastic_net_multiclass_summary.txt")

    folds_df.to_csv(folds_path, sep="\t", index=False)
    preds_df.to_csv(preds_path, sep="\t", index=False)
    stability_any.to_csv(stability_any_path, sep="\t", index=False)

    map_df = pd.DataFrame({"encoded_class": np.arange(len(le.classes_)), "label": le.classes_})
    map_df.to_csv(class_map_path, sep="\t", index=False)

    if toi_idx is not None:
        toi_markers_path = os.path.join(args.outdir, f"elastic_net_multiclass_marker_stability_TOI_{toi_label}.tsv")
        stability_toi.to_csv(toi_markers_path, sep="\t", index=False)

    # Summary
    with open(summary_path, "w") as f:
        f.write("Multiclass Elastic Net Logistic Regression (3-class) with nested repeated CV\n")
        f.write("====================================================================\n\n")
        f.write(f"Input: {args.input}\n")
        f.write(f"Targets col: {args.targets_col}\n")
        f.write(f"Classes: {list(le.classes_)}\n")
        f.write(f"Class counts: {dict(zip(le.classes_, counts))}\n\n")
        if toi_idx is not None:
            f.write(f"TOI focus enabled: {toi_label} (encoded={toi_idx})\n\n")

        f.write(f"Outer CV: {args.outer_splits}-fold x {args.outer_repeats} repeats = {total_fits} fits\n")
        f.write(f"Inner CV: {args.inner_cv}-fold, scoring=neg_log_loss\n\n")

        f.write("Fold-level metrics (mean ± SD):\n")
        f.write(f"  Balanced Accuracy: {folds_df['balanced_accuracy'].mean():.4f} ± {folds_df['balanced_accuracy'].std():.4f}\n")
        f.write(f"  Macro F1:          {folds_df['macro_F1'].mean():.4f} ± {folds_df['macro_F1'].std():.4f}\n")

        valid_auc = folds_df['macro_AUC_ovr'].dropna()
        if len(valid_auc) > 0:
            f.write(f"  Macro AUC (OVR):   {valid_auc.mean():.4f} ± {valid_auc.std():.4f}  (only folds containing all 3 classes)\n")
        else:
            f.write("  Macro AUC (OVR):   NA (some folds missing a class; use BalAcc/MacroF1)\n")

        f.write("\nTop stable markers (ANY class, first 20):\n")
        f.write(stability_any.head(20).to_string(index=False))
        f.write("\n")

        if toi_idx is not None:
            f.write(f"\nTop stable markers for TOI={toi_label} (first 20):\n")
            f.write(stability_toi.head(20).to_string(index=False))
            f.write("\n")

    print(f"[Done] Saved multiclass outputs to: {args.outdir}")
    print(f"Summary: {summary_path}")
    print(f"Marker stability (any class): {stability_any_path}")
    if toi_markers_path is not None:
        print(f"Marker stability (TOI={toi_label}): {toi_markers_path}")


if __name__ == "__main__":
    main()
