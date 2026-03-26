#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Patch for biomARkers multiclass RF supplementary exports.

Purpose
-------
This patch keeps the biomARkers method unchanged and only fixes the export logic
for supplementary RF tables so that:
  - RF_target_important_features.tsv
  - RF_nonTarget_important_features.tsv

match the exact feature lists that were actually used by the original get_rules()
workflow and reported in the old log output.

How it works
------------
1. Wrap rfcls.get_rules() so the exact OG lists used internally by the original
   workflow are captured and stored as:
      rfcls._exact_target_important_features
      rfcls._exact_non_target_important_features

2. Build a summary table with p values, prevalence, rule membership, and optional
   annotation columns.

3. Export the target and nonTarget supplementary tables by subsetting on the
   exact stored OG lists, NOT by recalculating booleans from p values.

Important
---------
You must call patch_get_rules_capture_exact_lists(rfcls) BEFORE rfcls.get_rules(...)
and call write_rf_summary_tables(...) AFTER rfcls.get_rules(...).

This file is designed to be copied into your project and imported, or the
functions can be pasted directly into your rewritten biomARkers script.
"""

from pathlib import Path
from typing import List, Iterable, Optional
import pandas as pd


def write_log(write_mode, log_file, message):
    """
    Safe fallback logger.
    If your original script already has write_log(), you can delete this helper
    and keep using your existing version.
    """
    if isinstance(message, list):
        message = "".join(str(x) for x in message)
    else:
        message = str(message)

    try:
        if log_file:
            with open(log_file, "a", encoding="utf-8") as handle:
                handle.write(message)
        else:
            print(message, end="")
    except Exception:
        print(message, end="")


def _safe_rule_ogs(df: Optional[pd.DataFrame]) -> List[str]:
    """
    Extract OG IDs from a rules dataframe with a column named 'orthogroups'.

    Expected format in 'orthogroups' column:
      OG0001234,OG0005678
    """
    if df is None or not hasattr(df, "columns") or "orthogroups" not in df.columns:
        return []

    vals = []
    for x in df["orthogroups"].dropna():
        if isinstance(x, str):
            parts = [p.strip() for p in x.split(",") if p.strip()]
            vals.extend(parts)
    return sorted(set(vals))


def patch_get_rules_capture_exact_lists(rfcls, pval: float = 0.01):
    """
    Monkey patch rfcls.get_rules() so the exact important feature lists used by
    the original workflow are saved for later export.

    Stored attributes after rfcls.get_rules(...) runs:
      rfcls._exact_target_important_features
      rfcls._exact_non_target_important_features

    This does NOT change the method. It only records the exact lists that the
    original get_rules() logic uses internally.
    """
    original_get_rules = rfcls.get_rules

    def wrapped_get_rules(*args, **kwargs):
        # Run original method first
        result = original_get_rules(*args, **kwargs)

        try:
            # These objects are expected from the original biomARkers RF class
            pval_df = getattr(getattr(rfcls, "fgc", None), "p_value_of_features_per_cluster", None)
            ranked = getattr(getattr(rfcls, "fgc", None), "data_clustering_ranked", None)

            if pval_df is None:
                raise ValueError("rfcls.fgc.p_value_of_features_per_cluster not found")
            if ranked is None:
                raise ValueError("rfcls.fgc.data_clustering_ranked not found")

            if not hasattr(rfcls, "clust") or len(rfcls.clust) < 2:
                raise ValueError("rfcls.clust missing or malformed")
            cluster_of_interest = rfcls.clust[1]

            if "cluster" not in ranked.columns:
                raise ValueError("'cluster' column not found in data_clustering_ranked")
            if "target" not in ranked.columns:
                raise ValueError("'target' column not found in data_clustering_ranked")
            if not hasattr(rfcls, "toi"):
                raise ValueError("rfcls.toi not found")

            # Feature columns are all columns except metadata columns
            non_feature_cols = {"cluster", "target"}
            feature_cols = [c for c in ranked.columns if c not in non_feature_cols]

            # Target important features
            # Exact old logic: p <= threshold in the cluster of interest
            target_important = []
            if cluster_of_interest in pval_df.columns:
                target_important = pval_df.index[pval_df[cluster_of_interest].astype(float) <= pval].tolist()

            # nonTarget important features
            # Exact old logic: significant in any non target cluster
            non_target_important = []
            non_target_cols = [c for c in pval_df.columns if c != cluster_of_interest]
            if len(non_target_cols) > 0:
                non_target_important = pval_df.index[
                    pval_df[non_target_cols].astype(float).le(pval).any(axis=1)
                ].tolist()

            # Store exact lists for later export
            rfcls._exact_target_important_features = list(target_important)
            rfcls._exact_non_target_important_features = list(non_target_important)

        except Exception as e:
            # Still allow pipeline to run. Export function can fallback later.
            rfcls._exact_target_important_features = []
            rfcls._exact_non_target_important_features = []
            write_log(
                getattr(rfcls, "write", "default"),
                getattr(rfcls, "log_file", ""),
                f"\nWarning: could not capture exact important feature lists from get_rules(): {e}\n",
            )

        return result

    rfcls.get_rules = wrapped_get_rules
    return rfcls


def write_rf_summary_tables(rfcls, outdir, fileID="", annot_file=None, pval: float = 0.01):
    """
    Write RF summary and supplementary tables.

    Output files
    ------------
    1. RF_feature_summary.tsv
    2. RF_target_important_features.tsv
    3. RF_nonTarget_important_features.tsv
    4. RF_final_target_biomarkers.tsv

    Critical behavior
    -----------------
    The target and nonTarget supplementary tables are exported using the exact
    feature lists captured from the original get_rules() workflow, so the row
    membership should match the old logged output.
    """
    try:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # collect core objects safely
        # -----------------------------
        pval_df = getattr(getattr(rfcls, "fgc", None), "p_value_of_features_per_cluster", None)
        ranked = getattr(getattr(rfcls, "fgc", None), "data_clustering_ranked", None)

        if pval_df is None:
            raise ValueError("rfcls.fgc.p_value_of_features_per_cluster not found")
        if ranked is None:
            raise ValueError("rfcls.fgc.data_clustering_ranked not found")

        merged = ranked.copy()

        # standardize feature list
        non_feature_cols = {"cluster", "target"}
        feature_cols = [c for c in merged.columns if c not in non_feature_cols]

        # make sure Orthogroup name is the index name of pval_df
        pval_df = pval_df.copy()
        pval_df.index.name = "Orthogroup"

        # -----------------------------
        # exact old workflow feature lists
        # -----------------------------
        exact_target = list(getattr(rfcls, "_exact_target_important_features", []) or [])
        exact_non_target = list(getattr(rfcls, "_exact_non_target_important_features", []) or [])

        # fallback only if exact lists are missing
        coi = getattr(rfcls, "clust", [None, None])[1] if hasattr(rfcls, "clust") else None
        if not exact_target and coi is not None and coi in pval_df.columns:
            exact_target = pval_df.index[pval_df[coi].astype(float) <= pval].tolist()

        if not exact_non_target and coi is not None and coi in pval_df.columns:
            non_df = pval_df.drop(columns=coi)
            if non_df.shape[1] > 0:
                exact_non_target = non_df.index[non_df.astype(float).le(pval).any(axis=1)].tolist()

        exact_target_set = set(exact_target)
        exact_non_target_set = set(exact_non_target)

        # -----------------------------
        # helper sets for rule membership
        # -----------------------------
        target_rule_ogs = set(_safe_rule_ogs(getattr(rfcls, "target_rules", None)))
        non_target_rule_ogs = set(_safe_rule_ogs(getattr(rfcls, "non_target_rules", None)))
        final_target_biomarkers = set(getattr(rfcls, "final_feat", []) or [])

        # -----------------------------
        # build base summary
        # -----------------------------
        summary = pd.DataFrame({
            "Orthogroup": sorted(set(feature_cols) | set(pval_df.index))
        })

        if coi is not None:
            summary["cluster_of_interest"] = coi
            summary["is_target_important"] = summary["Orthogroup"].isin(exact_target_set)
            summary["is_non_target_important"] = summary["Orthogroup"].isin(exact_non_target_set)
            summary["in_target_rules"] = summary["Orthogroup"].isin(target_rule_ogs)
            summary["in_non_target_rules"] = summary["Orthogroup"].isin(non_target_rule_ogs)
            summary["final_target_biomarker"] = summary["Orthogroup"].isin(final_target_biomarkers)

        # -----------------------------
        # merge cluster p values
        # -----------------------------
        pval_cols = []
        tmp_p = pval_df.reset_index()
        for col in pval_df.columns:
            new_col = f"rf_cluster_{col}_pvalue"
            pval_cols.append(new_col)
            summary = summary.merge(
                tmp_p[["Orthogroup", col]].rename(columns={col: new_col}),
                on="Orthogroup",
                how="left"
            )

        # -----------------------------
        # prevalence by target level
        # -----------------------------
        if "target" in merged.columns:
            for level in sorted(pd.unique(merged["target"])):
                sub = merged[merged["target"] == level]
                n = sub.shape[0]
                if n == 0:
                    continue
                counts = sub[feature_cols].sum(axis=0)
                prev = counts / n
                tmp = pd.DataFrame({
                    "Orthogroup": feature_cols,
                    f"count_target_{level}": counts.values,
                    f"prevalence_target_{level}": prev.values,
                })
                summary = summary.merge(tmp, on="Orthogroup", how="left")

        # -----------------------------
        # prevalence by RF cluster
        # -----------------------------
        if "cluster" in merged.columns:
            for cluster in sorted(pd.unique(merged["cluster"])):
                sub = merged[merged["cluster"] == cluster]
                n = sub.shape[0]
                if n == 0:
                    continue
                counts = sub[feature_cols].sum(axis=0)
                prev = counts / n
                tmp = pd.DataFrame({
                    "Orthogroup": feature_cols,
                    f"count_rfcluster_{cluster}": counts.values,
                    f"prevalence_rfcluster_{cluster}": prev.values,
                })
                summary = summary.merge(tmp, on="Orthogroup", how="left")

        # -----------------------------
        # prevalence in rule mining subsets
        # -----------------------------
        if coi is not None and "cluster" in merged.columns and "target" in merged.columns and hasattr(rfcls, "toi"):
            subset_target = merged[(merged["cluster"] == coi) & (merged["target"] == rfcls.toi)]
            subset_nontarget = merged[(merged["cluster"] != coi) & (merged["target"] != rfcls.toi)]

            for label, sub in [
                ("target_subset", subset_target),
                ("non_target_subset", subset_nontarget),
            ]:
                n = sub.shape[0]
                if n == 0:
                    continue
                counts = sub[feature_cols].sum(axis=0)
                prev = counts / n
                tmp = pd.DataFrame({
                    "Orthogroup": feature_cols,
                    f"count_{label}": counts.values,
                    f"prevalence_{label}": prev.values,
                })
                summary = summary.merge(tmp, on="Orthogroup", how="left")

        # -----------------------------
        # optional annotation merge
        # -----------------------------
        if annot_file and Path(annot_file).is_file():
            try:
                annot_df = pd.read_csv(annot_file, sep="\t")
                if "Orthogroup" in annot_df.columns:
                    keep = [
                        c for c in ["Orthogroup", "Annotation(s)", "Annotation", "Gene", "Description", "Label"]
                        if c in annot_df.columns
                    ]
                    if keep:
                        annot_df = annot_df[keep].drop_duplicates()
                        summary = summary.merge(annot_df, on="Orthogroup", how="left")
            except Exception as e:
                write_log(
                    getattr(rfcls, "write", "default"),
                    getattr(rfcls, "log_file", ""),
                    f"\nWarning: annotation merge failed: {e}\n",
                )

        # -----------------------------
        # column order
        # -----------------------------
        first_cols = ["Orthogroup"]
        preferred = [
            "cluster_of_interest",
            "is_target_important",
            "is_non_target_important",
            "in_target_rules",
            "in_non_target_rules",
            "final_target_biomarker",
        ]
        first_cols.extend([c for c in preferred if c in summary.columns])
        first_cols.extend([c for c in pval_cols if c in summary.columns])
        remaining = [c for c in summary.columns if c not in first_cols]
        summary = summary[first_cols + remaining]

        # -----------------------------
        # write outputs
        # -----------------------------
        summary.to_csv(outdir / f"{fileID}RF_feature_summary.tsv", sep="\t", index=False)

        # CRITICAL FIX
        # Export exact OG lists used by original workflow
        summary[summary["Orthogroup"].isin(exact_target)].to_csv(
            outdir / f"{fileID}RF_target_important_features.tsv",
            sep="\t",
            index=False,
        )

        summary[summary["Orthogroup"].isin(exact_non_target)].to_csv(
            outdir / f"{fileID}RF_nonTarget_important_features.tsv",
            sep="\t",
            index=False,
        )

        summary[summary.get("final_target_biomarker", False) == True].to_csv(
            outdir / f"{fileID}RF_final_target_biomarkers.tsv",
            sep="\t",
            index=False,
        )

        write_log(
            getattr(rfcls, "write", "default"),
            getattr(rfcls, "log_file", ""),
            [
                "\nRF summary tables written using exact important feature lists from get_rules().\n",
                f"Target important features exported: {len(exact_target)}\n",
                f"nonTarget important features exported: {len(exact_non_target)}\n",
            ],
        )

    except Exception as e:
        write_log(
            getattr(rfcls, "write", "default"),
            getattr(rfcls, "log_file", ""),
            f"\nWarning: could not write RF summary tables: {e}\n",
        )


def run_after_rf_pipeline(rfcls, outdir, fileID="", annot_file=None, pval: float = 0.01):
    """
    Convenience helper showing intended usage after the RF workflow has already
    created and fitted rfcls.

    Example usage
    -------------
    rfcls = patch_get_rules_capture_exact_lists(rfcls, pval=0.01)

    # run original method exactly as before
    rfcls.get_rules(...)

    # then export tables
    run_after_rf_pipeline(
        rfcls,
        outdir=args.outdir,
        fileID="",
        annot_file="OG_to_label.tsv",
        pval=0.01
    )
    """
    write_rf_summary_tables(
        rfcls=rfcls,
        outdir=outdir,
        fileID=fileID,
        annot_file=annot_file,
        pval=pval,
    )


if __name__ == "__main__":
    print(
        "This file provides helper functions only.\n"
        "Import it into your biomARkers rewrite and use:\n"
        "  1) patch_get_rules_capture_exact_lists(rfcls)\n"
        "  2) rfcls.get_rules(...)\n"
        "  3) write_rf_summary_tables(rfcls, outdir, ...)\n"
    )