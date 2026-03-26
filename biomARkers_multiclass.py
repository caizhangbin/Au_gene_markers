import argparse
from pathlib import Path
import sys
import traceback
import joblib
from time import localtime, strftime
import pandas as pd 

import ast
import numpy as np


def _safe_rule_ogs(df):
    """Return a sorted list of orthogroups present in a rules dataframe."""
    if df is None or getattr(df, 'empty', True) or 'orthogroups' not in df.columns:
        return []
    vals = []
    for item in df['orthogroups'].tolist():
        if isinstance(item, list):
            vals.extend(item)
        elif isinstance(item, str):
            try:
                parsed = ast.literal_eval(item)
                if isinstance(parsed, list):
                    vals.extend(parsed)
            except Exception:
                continue
    return sorted(set(vals))


def export_rf_summary_reports(rfcls, outdir, fileID, pval, annot_file=None):
    """Write manuscript-friendly RF summary tables for downstream supplement use."""
    try:
        if not hasattr(rfcls, 'fgc') or not hasattr(rfcls.fgc, 'p_value_of_features_per_cluster'):
            return

        outdir = Path(outdir)
        pval_df = rfcls.fgc.p_value_of_features_per_cluster.copy()
        pval_df.index.name = 'Orthogroup'
        pval_wide = pval_df.reset_index()
        pval_wide.to_csv(outdir / f'{fileID}RF_cluster_pvalues.tsv', sep='	', index=False)

        feature_cols = [c for c in rfcls.predictors if c in rfcls.data.columns]
        base = rfcls.data[feature_cols].copy()
        base.index.name = rfcls.id_col
        base_reset = base.reset_index()

        meta = rfcls.isolate_clust.copy()
        meta.columns = [rfcls.id_col, 'target', 'rf_cluster']
        merged = meta.merge(base_reset, on=rfcls.id_col, how='left')

        coi = rfcls.clust[1] if hasattr(rfcls, 'clust') else None
        target_important = set()
        non_target_important = set()
        if coi is not None and coi in pval_df.columns:
            target_important = set(pval_df.index[pval_df[coi] <= pval].tolist())
            non_df = pval_df.drop(columns=coi)
            if non_df.shape[1] > 0:
                non_target_important = set(non_df.index[non_df.astype(float).le(pval).any(axis=1)].tolist())

        target_rule_ogs = set(_safe_rule_ogs(getattr(rfcls, 'target_rules', None)))
        non_target_rule_ogs = set(_safe_rule_ogs(getattr(rfcls, 'non_target_rules', None)))
        final_target_biomarkers = set(getattr(rfcls, 'final_feat', []) or [])

        summary = pd.DataFrame({'Orthogroup': sorted(set(feature_cols) | set(pval_df.index))})
        if coi is not None:
            summary['cluster_of_interest'] = coi
            summary['is_target_important'] = summary['Orthogroup'].isin(target_important)
            summary['is_non_target_important'] = summary['Orthogroup'].isin(non_target_important)
            summary['in_target_rules'] = summary['Orthogroup'].isin(target_rule_ogs)
            summary['in_non_target_rules'] = summary['Orthogroup'].isin(non_target_rule_ogs)
            summary['final_target_biomarker'] = summary['Orthogroup'].isin(final_target_biomarkers)

        # merge cluster p values
        pval_cols = []
        for col in pval_df.columns:
            new_col = f'rf_cluster_{col}_pvalue'
            pval_cols.append(new_col)
            tmp = pval_df[[col]].rename(columns={col: new_col}).reset_index()
            summary = summary.merge(tmp, on='Orthogroup', how='left')

        # prevalence by target group
        target_levels = list(pd.unique(merged['target']))
        for level in target_levels:
            sub = merged[merged['target'] == level]
            n = sub.shape[0]
            if n == 0:
                continue
            counts = sub[feature_cols].sum(axis=0)
            prev = counts / n
            tmp = pd.DataFrame({
                'Orthogroup': feature_cols,
                f'count_target_{level}': counts.values,
                f'prevalence_target_{level}': prev.values,
            })
            summary = summary.merge(tmp, on='Orthogroup', how='left')

        # prevalence by RF cluster
        for cluster in sorted(pd.unique(merged['rf_cluster'])):
            sub = merged[merged['rf_cluster'] == cluster]
            n = sub.shape[0]
            if n == 0:
                continue
            counts = sub[feature_cols].sum(axis=0)
            prev = counts / n
            tmp = pd.DataFrame({
                'Orthogroup': feature_cols,
                f'count_rfcluster_{cluster}': counts.values,
                f'prevalence_rfcluster_{cluster}': prev.values,
            })
            summary = summary.merge(tmp, on='Orthogroup', how='left')

        # prevalence in target subset used for target-rule mining
        if coi is not None:
            subset_target = merged[(merged['rf_cluster'] == coi) & (merged['target'] == rfcls.toi)]
            subset_nontarget = merged[(merged['rf_cluster'] != coi) & (merged['target'] != rfcls.toi)]
            for label, sub in [('target_subset', subset_target), ('non_target_subset', subset_nontarget)]:
                n = sub.shape[0]
                if n == 0:
                    continue
                counts = sub[feature_cols].sum(axis=0)
                prev = counts / n
                tmp = pd.DataFrame({
                    'Orthogroup': feature_cols,
                    f'count_{label}': counts.values,
                    f'prevalence_{label}': prev.values,
                })
                summary = summary.merge(tmp, on='Orthogroup', how='left')

        # optional annotation merge
        if annot_file and Path(annot_file).is_file():
            try:
                annot_df = pd.read_csv(annot_file, sep='	')
                if 'Orthogroup' in annot_df.columns:
                    keep = [c for c in ['Orthogroup', 'Annotation(s)', 'Annotation', 'Gene', 'Description'] if c in annot_df.columns]
                    annot_df = annot_df[keep].drop_duplicates()
                    summary = summary.merge(annot_df, on='Orthogroup', how='left')
            except Exception:
                pass

        # Order columns for easier supplement use
        first_cols = ['Orthogroup']
        preferred = [
            'cluster_of_interest', 'is_target_important', 'is_non_target_important',
            'in_target_rules', 'in_non_target_rules', 'final_target_biomarker'
        ]
        first_cols.extend([c for c in preferred if c in summary.columns])
        first_cols.extend([c for c in pval_cols if c in summary.columns])
        remaining = [c for c in summary.columns if c not in first_cols]
        summary = summary[first_cols + remaining]

        summary.to_csv(outdir / f'{fileID}RF_feature_summary.tsv', sep='	', index=False)

        # concise manuscript-ready subset tables
        summary[summary.get('is_target_important', False) == True].to_csv(
            outdir / f'{fileID}RF_target_important_features.tsv', sep='	', index=False
        )
        summary[summary.get('is_non_target_important', False) == True].to_csv(
            outdir / f'{fileID}RF_nonTarget_important_features.tsv', sep='	', index=False
        )
        summary[summary.get('final_target_biomarker', False) == True].to_csv(
            outdir / f'{fileID}RF_final_target_biomarkers.tsv', sep='	', index=False
        )
    except Exception as e:
        try:
            write_log(getattr(rfcls, 'write', 'default'), getattr(rfcls, 'log_file', ''),
                      f'\nWarning: could not write RF summary tables: {e}\n')
        except Exception:
            pass


# Support importing either the original rfbiomarker module name or the updated
# multiclass-compatible version (keeps CLI behavior unchanged).
try:
    from rfbiomarker import RFBiomarkers, write_log, write_params, timefmt, save_models
except ModuleNotFoundError:
    from rfbiomarker_multiclass import RFBiomarkers, write_log, write_params, timefmt, save_models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='%(prog)s [-h] -i INPUT -o DIR -c COLUMN -t TARGET [-d -f -r -p -v -w --min --max --test_size --seeds --force --unsup]', 
                                     description='Uses random forest classifier and clustering to identify biomarkers')
    parser.add_argument('-i', '--input', metavar='', type=argparse.FileType('r'), nargs='?', 
                        help='Input tab-separated data with column names')
    parser.add_argument('-o', '--outdir', metavar='', type=str, required=False, 
                        help='Directory to save output files (default is current directory)')
    parser.add_argument( "--annot_file", type=str, required=False, default=None,
                        help="Optional annotation file (not required for biomarker identification).") 
    parser.add_argument('-c', '--targets_col', metavar='', type=str, required=True, 
                        help='Name of column containing target values')
    parser.add_argument('-d', '--id_col', metavar='', type=str, required=False, 
                        help='Name of column containing sample IDs')    
    parser.add_argument('-t', '--toi', metavar='', type=str, required=False, 
                        help='Target value of interest')
    parser.add_argument('-f', '--fileid', metavar='', type=str, required=False, 
                        help='Optional name for output files')
    parser.add_argument('-p', '--predictors', metavar='', nargs='+', required=False, 
                        help='List of columns to use as predictors (space delim, by default uses all columns except for specified target column and sample IDs column)')
    parser.add_argument('-r', '--remove', metavar='', nargs='+', required=False, 
                        help='List of columns to not use as predictors (space delim, opposite of --predictors, i.e. will use all columns in data except for those specified and the target/sample ID columns)')
    parser.add_argument('--min', metavar='', type=int, required=False, 
                        help='Minimum number of samples a feature must be present in (default is 5%% of total)')
    parser.add_argument('--max', metavar='', type=int, required=False, 
                        help='Maximum number of samples a feature must be present in (default is 95%% of total)')
    parser.add_argument('--test_size', metavar='', type=float, default=0.2, 
                        help='Test size used to train model (default test size is 0.2, i.e. will use 80%% of data to train model and 20%% to test)')
    parser.add_argument('--max_biomk', metavar='', type=float, default=4, 
                        help='Maximum number of biomarkers per group (default is 4)')
    parser.add_argument('--pval', metavar='', type=float, default=0.001, 
                        help='p-value threshhold for filtering important features (default is 0.001)')
    parser.add_argument('--lift', metavar='', type=float, default=1.5, 
                        help='Minimum lift for filtering association rules (default is 1.5)')
    parser.add_argument('-s', '--seeds', metavar='', type=int, nargs=2, required=False, 
                        help='Seed/random state values to use for subsampling training data and running model (by default will calculate best seeds)')
    parser.add_argument('-w','--write', metavar='', type=str, required=False, choices=['none', 'all'], 
                        help='Model data to write to parameters.txt. Options are: "none" or "all" (default behavior writes pertinent information)')
    parser.add_argument('--force', action='store_true', required=False, 
                        help='Overwrite previous output files')
    parser.add_argument('--unsup', action='store_true', required=False, 
                        help='Not recommended! Run unsupervised RF (by default will train RF on 80%% of data, use --test_size to change)')
    parser.add_argument('--save_models', action='store_true', required=False, 
                        help='Save RF and FGC class objects')
    parser.add_argument('--fgc', metavar='', type=str, required=False, 
                        help='Pre-generated FGC class object')
    parser.add_argument('--rf', metavar='', type=str, required=False, 
                        help='Pre-generated RF class object')
    parser.add_argument('--fpg_file', action='store_true', required=False, 
                        help='Use pre-generated FPG output files')
    parser.add_argument('--rules_file', action='store_true', required=False, 
                        help='Use pre-generated association rules files')
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    args = parser.parse_args()
    try:
        if not args.input:
            raise ValueError(f'-i/--input: No input data. Please provide tab-separated data file')
        if not args.outdir:
            outdir = Path.cwd()
        elif Path(args.outdir).exists():
            outdir = Path(args.outdir) 
        else:
            try:
                Path(args.outdir).mkdir()
            except PermissionError:
                err_str = f'-o/--outdir: unable to create "{args.outdir}/", permission denied'
                raise PermissionError(err_str)
            outdir = Path(args.outdir)
        [fileID, write_str] = [f'{args.fileid}_', f'\nID name for files (fileID): {args.fileid}'] if args.fileid else ['', '']
        write = args.write if args.write else "default"
        log_file = f'{outdir}/{fileID}log.txt'
        params_file = f'{outdir}/{fileID}parameters.tsv'
        wrt_str = []

        if Path(params_file).is_file(): ## !! add checks for file existing 
            wa = "a"
            wrt_str = ['\n\n# ---\n', '# RERUN WITH PRE']
            if args.rules_file:
                wrt_str.append('VIOUSLY GENERATED ASSOCIATION RULES\n')
            elif args.fpg_file: 
                wrt_str.append('VIOUSLY GENERATED PATTERN FILES\n')
            elif args.fgc:
                wrt_str.append('-SAVED FGC MODEL\n')
            elif args.rf:
                wrt_str.append('-SAVED RF MODEL\n')
            elif not args.force:
                err_str = ''.join([f'-f/--fileID: analysis output files in {str(outdir.resolve())} ', 
                                     '"'.join(['with file ID' , args.fileid, '" ']) if args.fileid else '', 
                                     'already exist. Provide a unique file ID or ', 
                                     'use --force to ignore this error and overwrite previous output files'])
                raise FileExistsError(err_str)
            else:
                wrt_str = []
                wa = "w"
        else:
            wa = "w"
        
        wrt_str.append(f'biomARkers version 0.1\nCurrent working directory is {Path.cwd()}\n')
        write_log(write, log_file, wrt_str, wa)
        cmd_calls = []
        for k in vars(args).keys():
            if k == 'input':
                cmd_in = f'[{timefmt()}] Command line parameters: \n\t--input'
                cmd_opt = f'{cmd_in} {args.input.name}' if args.input else f'{cmd_in} -'
                cmd_calls.append(cmd_opt)
            elif k in ['save_models', 'force', 'unsup']:
                if vars(args)[k]:
                    cmd_calls.append(f'\n\t--{k}')
            elif vars(args)[k]:
                cmd_calls.append(f'\n\t--{k} {str(vars(args)[k]) if not isinstance(vars(args)[k], list) else " ".join([str(v) for v in vars(args)[k]])}')
        wrt_str = f'{" ".join(cmd_calls)}\n\nWriting files to (outdir): {outdir.resolve()}{write_str}\n\n'
        write_log(write, log_file, wrt_str)
        prm_str = [f'# [{timefmt()}] RFbiomarkers version 0.1\n',
                   f'outdir\t{outdir.resolve()}\n'
                   f'fileID\t{args.fileid}\n']
        write_params(params_file, prm_str, wa="w")
        print(f'\nSaving output to {outdir.resolve()}\n')

        data = pd.read_csv(args.input, sep='\t')
        
        if args.id_col in data.columns:
            data.set_index(args.id_col, inplace=True)
            id_col = args.id_col
        else:
            id_col = 'index'

        if args.targets_col in data.columns:
            targets_col = args.targets_col
        else:
            err_str = ' '.join([f'-c/--targets_col: "{args.targets_col}"', 
                                  'is not one of the columns in the given data.', 
                                  'Check your spelling and make sure your input data is tab-separated'])
            raise ValueError(err_str)
        if (args.toi is None) or (args.toi in data[targets_col].to_list()): 
            toi = args.toi
        else:
            err_str = f'-t/--toi: "{args.toi}" is not one of the values in the target column. Check your spelling'
            raise ValueError(err_str)
        if args.predictors:
            p_miss = []
            for p in args.predictors:
                if p in data.columns:
                    continue
                p_miss.append(p)
            if p_miss:
                err_str = f'-p/--predictors: "{", ".join(p_miss)}" were not found in the column names of the given data'
                raise ValueError(err_str)
            predictors = [c for c in args.predictors if c != targets_col]
        elif args.remove:
            predictors = [c for c in data.columns if c != targets_col and c not in args.remove]
        else:
            predictors = [c for c in data.columns if c != targets_col]
        
        min_thresh = args.min 
        max_thresh = args.max 
        max_biomk = args.max_biomk
        if (0 < args.test_size < 1):
            test_size = args.test_size 
        else:
            err_str = f'--test_size: must be a number between 0 and 1'
            raise ValueError(err_str)
        if args.seeds:
            seeds = tuple(args.seeds)
            best_seeds = False
        else:
            seeds = (None, None)
            best_seeds = True
        if args.unsup:
            train = False
        else:
            train = True

        rfcls = RFBiomarkers(data, 
                             id_col,
                             predictors,
                             targets_col, 
                             toi,
                             max_biomk,
                             outdir, 
                             fileID, 
                             write, 
                             log_file, 
                             params_file,
                             min_thresh, 
                             max_thresh)
        wrt_str = [f'Name of target column (targets): {rfcls.__dict__["targets"]}\n', f'Target value of interest (toi): {rfcls.__dict__["toi"]}\n']
        write_log(write, log_file, wrt_str)
        
        if write == 'all':
            wrt_str = f'Predictor/feature column names used (predictors): {", ".join(rfcls.__dict__["predictors"])}\n'
            write_log(write, log_file, wrt_str)
        if args.remove and [r for r in args.remove if r not in data.columns]:
            wrt_str = [f'Warning: command line option -r/--remove: none of "', 
                       ", ".join([r for r in args.remove if r not in data.columns]), 
                       '" were found in the column names of the given data and could not be removed as predictor(s)\n']
            write_log(write, log_file, wrt_str)

        wrt_str = [f'\nMinimum number of samples with feature present needed to be included in model (min_thresh): {rfcls.__dict__["min_thresh"]}\n', 
                   f'Maximum number of samples with feature present needed to be included in model (max_thresh): {rfcls.__dict__["max_thresh"]}\n', 
                   f'Number of features used in the model (num_feat): {rfcls.__dict__["X"].shape[1]-1}\n']
        write_log(write, log_file, wrt_str)
        prm_str = [f'targets\t{rfcls.__dict__["targets"]}\n', 
                   f'toi\t{rfcls.__dict__["toi"]}\n', 
                   f'min_thresh\t{rfcls.__dict__["min_thresh"]}\n', 
                   f'max_thresh\t{rfcls.__dict__["max_thresh"]}\n', 
                   f'max_biomk\t{rfcls.__dict__["max_biomk"]}\n', 
                   f'num_feat\t{rfcls.__dict__["X"].shape[1]-1}\n']
        write_params(params_file, prm_str)
        if write == 'all':
            prm_str = f'predictors\t{rfcls.__dict__["predictors"]}\n'
            write_params(params_file, prm_str)

        rffile = None
        fgcfile = None

        fpg_file = True if args.fpg_file else None 
        rules_file = True if args.rules_file else None
        if not rules_file:
            rfcls.generate_RF(best_seeds=best_seeds, 
                            seeds=seeds, 
                            train=train, 
                            test_size=test_size,
                            rf_file=rffile, 
                            fgc_file=fgcfile)
            if args.save_models:
                if not Path(str(rffile)).is_file():
                    wrt_str = save_models(rfcls, 'rf') 
                    write_log(write, log_file, wrt_str)

            rfcls.generate_RFclusters(plot=False, 
                                      fgc_file=fgcfile) 
            if args.save_models:
                if not Path(str(fgcfile)).is_file():
                    wrt_str = save_models(rfcls, 'fgc') 
                    write_log(write, log_file, wrt_str)

            lift = args.lift
            pval = args.pval
            rfcls.get_rules(fpg_file=fpg_file, pvalue=pval, min_lift=lift, non_lift=lift-0.3)
        rfcls.get_biomarkers(rules_file=rules_file) 

        if args.annot_file:
            annot_file = args.annot_file
            if Path(annot_file).is_file():
                annots_df = rfcls.get_annotions(annot_file)
                annots_df.to_csv(f'{Path(outdir, Path(annot_file).stem)}.orthog_gene_names.tsv', 
                                 sep='\t', index=False)

        export_rf_summary_reports(rfcls, outdir, fileID, pval=args.pval, annot_file=args.annot_file)

        if args.save_models:
            wrt_str = save_models(rfcls, 'rfbio')
            write_log(write, log_file, wrt_str)

        write_log(write, log_file, f'\nEnd time: {timefmt()}')
    
    except KeyboardInterrupt:
        write_str = 'Abort by user interrupt. Analysis not finished'
        with open(f'{outdir}/{fileID}parameters.tsv', "a") as p:
            p.write(f'# {write_str}')
        print(write_str)
        sys.exit(1)
    except Exception as exc:
        filename, lineno, funcname, text = traceback.extract_tb(exc.__traceback__)[-1]
        funcname = f'{funcname}():' if funcname != "<module>" else "command line input"
        write_str = f'{type(exc).__name__}: {filename}: line {lineno}, {funcname} {exc}'
        with open(f'{outdir}/{fileID}parameters.tsv', "a") as p:
            p.write(f'# {write_str}\n# WARNING: Analysis not completed')
        print(f'\n{write_str}\n')
        sys.exit(1)