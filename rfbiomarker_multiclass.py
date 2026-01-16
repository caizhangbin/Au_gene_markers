import pandas as pd 
import numpy as np
import re
from ast import literal_eval
from collections import Counter, defaultdict
from itertools import combinations

from pathlib import Path
import glob
from time import localtime, strftime
import joblib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.metrics import f1_score as f1
from sklearn.model_selection import train_test_split
from fgclustering import FgClustering ## requires python >=3.9, <3.12
from importlib.metadata import version
from fgclustering.statistics import calculate_local_feature_importance
import fgclustering.utils as utils

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, fpgrowth

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

def write_log(write, file, wrt_str,  wa="a"):
    if write != 'none': 
        with open(file, wa) as f:
            f.writelines(wrt_str)
    return

def write_params(file, wrt_str, wa="a"):
    with open(file, wa) as f:
        f.writelines(wrt_str)
    return

def timefmt():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())

def save_models(clsobj, model_type):
    """
    Save random forest/clustering model to output directory
    """
    mdl_names = ['RFBiomarkers', 'RandomForestClassifier', 'FgClustering']
    mdl_types = ["rfbio", "rf", "fgc"]
    if  model_type not in mdl_types:
        write_str = f'"{model_type}" is not a valid model type, must be one of: {", ".join(mdl_types)}'
        raise ValueError(write_str)
    od = clsobj.__dict__["outdir"]
    fi = clsobj.__dict__["fileID"]
    if model_type == 'rfbio':
        mdl = clsobj
    else:
        mdl = clsobj.__dict__[model_type]
    if type(mdl).__name__ in mdl_names:
        model_name = type(mdl).__name__
        file = Path(f'{od}/{fi}model.{model_type}.gz')
        joblib.dump(mdl, file)
        return f'{model_name} class object saved to {str(file)}\n' 
    else:
        write_str = f'"{model_name}" is not a valid model type name, must be one of: {", ".join(mdl_names)}'
        raise ValueError(write_str)

class RFBiomarkers():
    def __init__(self, data, id_col, predictors, targets_col, toi, max_biomk, outdir, fileID, 
                 write, log_file, params_file, min_thresh=None, max_thresh=None, RF_type='classifier'):
        self.data = data ## dataframe
        self.id_col = id_col
        self.predictors = predictors ## columns to use as predictors
        self.targets = targets_col ## column to use as target
        self.toi = toi ## specific target of interest
        self.max_biomk = int(max_biomk) ## maximum number of biomarkers per group
        self.RF_type = RF_type
        self.outdir = outdir
        self.fileID = fileID
        self.write = write
        self.log_file = log_file
        self.params_file = params_file

        self.y = self.data[self.targets]
        # NOTE: the original implementation assumed exactly two target classes.
        # We now support true multi-class classification by tracking all levels.
        self.levels = list(pd.unique(self.y))
        # Keep legacy attributes for backwards compatibility (binary use-cases)
        self.level1 = self.levels[0] if len(self.levels) > 0 else None
        self.level2 = self.levels[1] if len(self.levels) > 1 else None
        self.X = self.data.drop(
            columns=[c for c  in self.data.columns.to_list() if c not in self.predictors])
        # filter out genes that are always present/absent in the same isolates (only keep one)
        self.X = self.X.T.drop_duplicates().T
        if not min_thresh:
            min_thresh=(round(self.X.shape[0]*0.05))
        self.min_thresh = min_thresh
        if not max_thresh:
            max_thresh=(round(self.X.shape[0]*0.95))
        self.max_thresh = max_thresh
        # filter out genes that are in too many or too few isolates (i.e. core/cloud genomes)
        while True: # !! currently only works if all predictor values are 1 or 0
            prev_shape = self.X.shape
            cols_sum = self.X.sum(axis=0, numeric_only=True)
            drop_cols = cols_sum[cols_sum < self.min_thresh].index
            self.X.drop(columns=drop_cols, inplace=True)
            drop_cols = cols_sum[cols_sum > self.max_thresh].index
            self.X.drop(columns=drop_cols, inplace=True)
            if self.X.shape == prev_shape:
                break
        self.X[self.targets] = self.y

    def generate_RF(self, best_seeds=True, seeds=(None, None), 
                    train=True, test_size=0.2, plot=True, n=50,
                    rf_file=None, fgc_file=None):
        """
        Generate sample clusters from random forest classifier and ranks features by importance for predicting correct cluster.
        plot: bool, plot feature importance for top n features (default=False)
        n: int, number of features to plot (default=50)
        Adds attributes to class object: 
            rf: RandomForestClassifier object
            seeds: tuple (i, j) with random state values for selecting training data (i) and running model (j)
            X_train, X_test, y_train, y_test: pandas DataFrames containing training and test data 
            y_pred: numpy array of predicted values
        """
        if rf_file or fgc_file: ## check if model file exists
            if fgc_file and not rf_file:
                rf_file = fgc_file.replace('_model.fgc.gz', '_model.rf.gz')
            if Path(rf_file).is_file():
                wrt_str = f'\n[{timefmt()}] Loading RF model from {str(rf_file)}...'
                write_log(self.write, self.log_file, wrt_str)
                self.rf = joblib.load(rf_file)
                self.seeds = seeds 

                wrt_str = [f"\nSeed used for training data selection (dataselect_seed): {self.seeds[0]}\n", 
                            f"Seed used for RF random state (model_seed): {self.seeds[1]}\n", 
                            f'Train model (train): {train}\n',
                            ]
                write_log(self.write, self.log_file, wrt_str)
            else:
                wrt_str = [f"\nWarning: Cannot find saved RF model {rf_file}\n"]
                write_log(self.write, self.log_file, wrt_str)
        
        if not Path(str(rf_file)).is_file():
            if self.RF_type == 'regressor':
                wrt_str = 'Regression model not implemented yet, sorry!\n'
                write_log(self.write, self.log_file, wrt_str)
                raise NotImplementedError(wrt_str)
            elif self.RF_type == 'classifier':
                if self.y.apply(isinstance, args = [float]).any(): 
                    if len(set([x for x in self.y])) > 10:
                        wrt_str = 'Warning: target values are numeric with more than 10 categorical values. Consider using regression model (not implemented yet, sorry!)'
                        print(wrt_str)
                        write_log(self.write, self.log_file, f'{wrt_str}\n')
            
            def _get_best_seeds(range1=50, range2=50): 
                # Track best (accuracy, f1_metric, precision_metric, dataselect_seed, model_seed)
                max_accuracy=(0, 0, 0, 0, 0)
                for i in range(range1): 
                    X_train, X_test, y_train, y_test = train_test_split(self.X.drop(self.targets, axis=1, inplace=False), 
                                                                        self.y, stratify=self.y, test_size=test_size, random_state=i)
                    for j in range(range2):
                        rf = RandomForestClassifier(random_state=j)
                        rf.fit(X_train, y_train)
                        y_pred = rf.predict(X_test)
                        # For multi-class problems, use macro-averaged metrics.
                        # For binary problems, this is also safe and avoids pos_label issues.
                        acc = accuracy_score(y_test, y_pred)
                        f1_metric = f1(y_test, y_pred, average='macro')
                        prec_metric = precision_score(y_test, y_pred, average='macro', zero_division=0)

                        if acc >= max_accuracy[0]:
                            if f1_metric >= max_accuracy[1]:
                                if prec_metric >= max_accuracy[2]:
                                    max_accuracy = (acc, f1_metric, prec_metric, i, j)
                                    if self.write == 'all':
                                        wrt_str = [f'\ntraining seed: {i}; RF seed: {j}\n', classification_report(y_test, y_pred)]
                                        write_log(self.write, self.log_file, wrt_str)
                return max_accuracy
            
            if best_seeds:
                wrt_str = f'\n[{timefmt()}] Looking for best seeds...'
                write_log(self.write, self.log_file, wrt_str)
                max_accuracy = _get_best_seeds()
                self.seeds = (max_accuracy[3], max_accuracy[4])
            else: 
                self.seeds = seeds
            if not train:
                # tuples are immutable; preserve original intent
                self.seeds = (None, self.seeds[1])

            wrt_str = [f"\nSeed used for training data selection (dataselect_seed): {self.seeds[0]}\n", 
                        f"Seed used for RF random state (model_seed): {self.seeds[1]}\n", 
                        f'Train model (train): {train}\n']
            write_log(self.write, self.log_file, wrt_str)
            
            with open(self.params_file, "a") as p:
                p.writelines(f"dataselect_seed\t{self.seeds[0]}\n")
                p.writelines(f"model_seed\t{self.seeds[1]}\n")
                p.writelines(f"train\t{train}\n")
                
            if train:
                wrt_str = f'Splitting data: {round((1-test_size)*100)}% training, {round((test_size)*100)}% testing (test_size)\n'
                write_log(self.write, self.log_file, wrt_str)
                with open(self.params_file, "a") as p:
                    p.writelines(f"test_size\t{test_size}\n")
                X_train, X_test, y_train, y_test = train_test_split(
                    self.X.drop(self.targets, axis=1, inplace=False), 
                    self.y, stratify=self.y, test_size=test_size, random_state=self.seeds[0])
                
                wrt_str =f'\n[{timefmt()}] Generating random forest {self.RF_type}, fitting model...\n'
                write_log(self.write, self.log_file, wrt_str)
                if self.RF_type == 'classifier':
                    self.rf = RandomForestClassifier(random_state=self.seeds[1])
                elif self.RF_type == 'regressor':
                    self.rf = RandomForestRegressor(random_state=self.seeds[1])
                self.rf.fit(X_train, y_train)

                wrt_str = f'\n[{timefmt()}] Testing model...\n'
                write_log(self.write, self.log_file, wrt_str)

                self.rf.y_pred = self.rf.predict(X_test)
                self.rf.y_test = y_test
                self.rf.train_seed = self.seeds[0]
                wrt_str = f"Model scores:\n{classification_report(self.rf.y_test, self.rf.y_pred)}"
                write_log(self.write, self.log_file, wrt_str)
            if plot:
                wrt_str = f'\n[{timefmt()}] Plotting random forest feature importance...\n'
                write_log(self.write, self.log_file, wrt_str)
                plt.figure(figsize=(15,15))
                sns.set_theme(font_scale=0.8)
                feature_importances = pd.Series(self.rf.feature_importances_, 
                                                index=X_train.columns
                                                ).sort_values(ascending=False)
                feature_importances[0:n].plot.bar() 
                plt.suptitle(
                    f"Important features before random forest clustering",
                    fontsize=20)
                plt.savefig(Path(f'{self.outdir}/{self.fileID}important_feat_noclust.png'))
                plt.savefig(Path(f'{self.outdir}/{self.fileID}important_feat_noclust.pdf'))  
                plt.close()
        return
    

    def plot_RFtrees(self, n=6):
        """
        Generate plots of decision trees from random forest
            n: int, number of plots to generate (default=6)
        """ 
        wrt_str = f'\n[{timefmt()}] Plotting first {n} decision trees in random forest...\n'
        write_log(self.write, self.log_file, wrt_str)
        from sklearn import tree
        fn=self.X.drop(self.targets, axis=1, inplace=False).columns.to_list()
        cn=self.y.unique()
        for i in range(n):
            fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
            tree.plot_tree(self.rf.estimators_[i],
                        feature_names = fn, 
                        class_names=cn,
                        filled = True)
            plt.savefig(Path(f'{self.outdir}/{self.fileID}dectree_{i}.png'))
            plt.savefig(Path(f'{self.outdir}/{self.fileID}dectree_{i}.pdf'))   
        return

    
    def generate_RFclusters(self, plot=True, n=100, fgc_file=None):
        """
        Generate sample clusters from random forest classifier and ranks features 
        by importance for predicting correct cluster.
            plot: bool, plot feature importance for top n features (default=False)
            n: int, number of features to plot (default=100)
        Adds attributes to class object: 
            fgc: FgClustering object
            importance_local: pandas DataFrame with Importance score for each feature for each cluster 
            clust: tuple with number of targets in cluster of interest and cluster ID
        """
        if fgc_file:
            if Path(fgc_file).is_file():
                wrt_str = f'\n[{timefmt()}] Loading FGC model from {str(fgc_file)}...'
                write_log(self.write, self.log_file, wrt_str)
                self.fgc = joblib.load(fgc_file)
            else:
                wrt_str = [f"\nWarning: Cannot find saved FGC model {fgc_file}\n"]
                write_log(self.write, self.log_file, wrt_str)
        
        if not Path(str(fgc_file)).is_file():
            wrt_str = f'\n[{timefmt()}] Clustering random forest...\n'
            write_log(self.write, self.log_file, wrt_str)
            self.fgc = FgClustering(model=self.rf, data=self.X, 
                                    target_column=self.targets, random_state=self.seeds[1])
            if version('fgclustering') >= '1.1.1':                                                                                                                                                                                       ## !! change number of clusters to be equal to number of target values
                self.fgc.run(k=2)
            else:
                self.fgc.run(number_of_clusters=2) 

            self.fgc.calculate_statistics(data=self.X, target_column=self.targets)
            calculate_local_feature_importance(self.fgc.data_clustering_ranked, 1000) ## p-values
        if version('fgclustering') >= '1.1.1': 
            importance_dict = {"Feature": self.fgc.p_value_of_features_per_cluster.index}
            for cluster in self.fgc.p_value_of_features_per_cluster.columns: 
                importance_dict[f"cluster{cluster}_importance"] = utils.log_transform(
                    self.fgc.p_value_of_features_per_cluster[cluster].to_list())
            self.importance_local = pd.DataFrame(importance_dict)
        ## cluster of interest size and ID 
        self.isolate_clust = self.data.reset_index()[[self.id_col, self.targets]].assign(cluster = self.fgc.data_clustering_ranked['cluster'])                                                                                         ###!!!! need to test and integrate into other fns
        self.isolate_clust.columns = [self.id_col, 'target', 'cluster']
               # Save RF cluster labels for each isolate
        try:
            clust_out = self.isolate_clust.copy()
            clust_out.to_csv(
                Path(self.outdir) / f"{self.fileID}RF_clusters.tsv",
                sep="\t",
                index=False
            )
        except Exception as e:
            wrt_str = [f"\nWarning: could not write RF cluster labels: {e}\n"]
            write_log(self.write, self.log_file, wrt_str)

        clust_comp = self.isolate_clust[["cluster", "target"]].groupby(                                                                                                                                                                     ## changed to isolate_clust from fgc.data_clustering_ranked
            ["cluster", "target"], as_index=False, observed=False).size()                                                                                                                                                                       ## all cluster sizes 
        clust_dict = clust_comp.to_dict('index')

        self.clust = max((int(d['size']), d['cluster']) for d in clust_dict.values() if d['target'] == self.toi) 
        clust_size = clust_comp[['cluster', 'size']].groupby('cluster', as_index=False, observed=False).sum()

        wrt_str = [f"Cluster of interest (coi): {self.clust[1]}\n", 
               f"Cluster size (coi_size): {clust_size[clust_size['cluster'] == self.clust[1]]['size'].iat[0]}\n", 
               f"Number of targets in cluster (num_toi): {self.clust[0]}\n",
               ] 
        write_log(self.write, self.log_file, wrt_str)
        if plot: 
            wrt_str = f'\n[{timefmt()}] Plotting random forest clustering feature importance...\n'
            write_log(self.write, self.log_file, wrt_str)
            self.fgc.plot_feature_importance(thr_pvalue=0.01, top_n=n, num_cols=5, 
                                             save=Path(f'{self.outdir}/{self.fileID}cluster'))
        return 
    

    def get_rules(self, pvalue=0.001, min_support=0.6, min_lift=1.5, non_lift=1.2, min_conf=0.8, 
                  min_zhang=0.5, min_lev=0.05, min_conv=1.5, fpg_file=None, subsample=False):
        """
        Generate frequent patterns from cluster and target of interest, 
        then find association rules for all combinations up to given group size.
            pvalue: float, feature importance from random forest clustering (default=0.001)
            max_biomarkers: integer >1, maximum group size for biomarker identification (default=5) 
            min_support: float [0:1], frequency of all orthologues in a group appearing together in the dataset,
                         required for FPG and association rules (default=0.6) 
            min_lift: float >1, how much more often orthologues occur together would be expected if they 
                      were statistically independent; independence == 1 (default=1.5) 
            non_lift: float >1, lift threshold to use for non-target filtering (default=1.2)
            min_conf: float [0:1], the probability of the presence of a group given that another group is 
                      also present; completely dependant == 1 (default=0.8) 
            min_zhang: float [-1:1], positive value (>0) indicates association and negative value indicates 
                       dissociation; independence == 0 (default=0.5) 
            min_lev: float, difference between the observed frequency orthologues appearing together and 
                     the expected frequency if they were independent; independence == 0 (default=0.5) 
            min_conv: float >1, high conviction value means that one group is highly dependent on the 
                      other; independence == 1 (default=1.5)
            fpg_file: tab-separated file with precomputed FPGrowth results (default=None)
        """
        ## !! need to add parameters.tsv file
        wrt_str = [f'\nMinimum thresholds for frequent patterns and association rules:\n',
                   f'Max cluster importance p-value: {pvalue}\n',
                   f'Minimum support: {min_support}\n',
                   f'Minimum lift: {min_lift} (nonTarget lift: {non_lift})\n',
                   f'Minimum confidence: {min_conf}\n',
                   f'Minimum leverage: {min_lev}\n',
                   f"Minimum threshold for Zhang's metric: {min_zhang}\n"]
        write_log(self.write, self.log_file, wrt_str)
        
        def _get_features_df(target, min_lift, fpg_file=fpg_file, pvalue=pvalue, min_support=min_support, 
                             min_conf=min_conf, min_zhang=min_zhang, min_lev=min_lev, min_conv=min_conv, subsample=subsample):
            if target:
                all_target_df = self.fgc.data_clustering_ranked[(self.fgc.data_clustering_ranked['cluster'] == self.clust[1]) & 
                                                                (self.fgc.data_clustering_ranked['target'] == self.toi)] ## look at isolates from cluster AND target of interest
                tgt = f'Target ({self.toi})'
                if fpg_file:
                    if glob.glob(f'{self.outdir}/{self.fileID}Target_FPG.tsv*'):
                        fpg_file = glob.glob(f'{self.outdir}/{self.fileID}Target_FPG.tsv*')[0]
                clust_list = self.fgc.p_value_of_features_per_cluster[self.fgc.p_value_of_features_per_cluster[self.clust[1]] <= pvalue].index.to_list()
            else:
                all_target_df = self.fgc.data_clustering_ranked[(self.fgc.data_clustering_ranked['cluster'] != self.clust[1]) & ## look at isolates from other cluster(s)
                                                                (self.fgc.data_clustering_ranked['target'] != self.toi)] ## AND NOT target of interest
                non_df = self.fgc.p_value_of_features_per_cluster.drop(columns=self.clust[1])
                tgt = f'nonTarget ({" ".join([c for c in list(set(self.y)) if c != self.toi])})'
                if fpg_file:
                    if glob.glob(f'{self.outdir}/{self.fileID}nonTarget_FPG.tsv*'):
                        fpg_file = glob.glob(f'{self.outdir}/{self.fileID}nonTarget_FPG.tsv*')[0]
                clust_list = non_df[non_df.astype(float).le(pvalue).any(axis=1)].index.to_list()
            # Ensure clust_list is treated as a list so we always get a 2D DataFrame
            if isinstance(clust_list, str):
                clust_cols = [clust_list]
            else:
                clust_cols = list(clust_list)

            # -----------------------------
            # Safety guards (multiclass + small-n friendly)
            # -----------------------------
            # With 3-class targets it is possible that the selected cluster/target
            # slice has 0 samples, or that no features pass the p-value filter.
            # In those cases, TransactionEncoder/fpgrowth will crash on empty input.
            if all_target_df is None or all_target_df.shape[0] == 0:
                wrt_str = (
                    f"\n[{timefmt()}] WARNING: No samples found for {tgt}. "
                    f"Skipping frequent pattern mining and association rules for this subset.\n"
                )
                write_log(self.write, self.log_file, wrt_str)
                empty_rules = pd.DataFrame(
                    columns=[
                        'orthogroups', 'num_orthogroups', 'support', 'confidence',
                        'lift', 'leverage', 'conviction', 'zhangs_metric'
                    ]
                )
                if target:
                    self.target_rules = empty_rules
                else:
                    self.non_target_rules = empty_rules
                return

            if clust_cols is None or len(clust_cols) == 0:
                wrt_str = (
                    f"\n[{timefmt()}] WARNING: No important features passed the p-value filter "
                    f"for {tgt}. Skipping frequent pattern mining and association rules for this subset.\n"
                )
                write_log(self.write, self.log_file, wrt_str)
                empty_rules = pd.DataFrame(
                    columns=[
                        'orthogroups', 'num_orthogroups', 'support', 'confidence',
                        'lift', 'leverage', 'conviction', 'zhangs_metric'
                    ]
                )
                if target:
                    self.target_rules = empty_rules
                else:
                    self.non_target_rules = empty_rules
                return

            clust_df = all_target_df[clust_cols].astype(int).replace({1: True, 0: False})
            all_target_df = None

            # create orthogroup list for each isolate
            names_df = clust_df.apply(lambda x: np.where(x, x.name, None))
            wrt_str = [f'\n[{timefmt()}] Identifying frequent patterns for {tgt}...\n',
                       f'Number of samples used: {names_df.shape[0]}\n',
                       f'Number of important features: {names_df.shape[1]}\n',
                       f'\t{clust_cols}\n']
            write_log(self.write, self.log_file, wrt_str)

            if subsample:
                names_df = clust_df.apply(lambda x: np.where(x, x.name, None)).sample(frac=0.5, axis=1, ignore_index=True)
                wrt_str = f'\tNumber subsampled: {names_df.shape[1]}\n'
                write_log(self.write, self.log_file, wrt_str)

            lst = []
            for sublist in names_df.values.tolist():
                clean_sublist = [item for item in sublist if item is not None]
                lst.append(clean_sublist)
            names_df = None

            # If every transaction is empty (no features), abort gracefully.
            if (lst is None) or (len(lst) == 0) or (sum(len(x) for x in lst) == 0):
                wrt_str = (
                    f"\n[{timefmt()}] WARNING: No items available to encode for {tgt}. "
                    f"Skipping frequent pattern mining and association rules for this subset.\n"
                )
                write_log(self.write, self.log_file, wrt_str)
                empty_rules = pd.DataFrame(
                    columns=[
                        'orthogroups', 'num_orthogroups', 'support', 'confidence',
                        'lift', 'leverage', 'conviction', 'zhangs_metric'
                    ]
                )
                if target:
                    self.target_rules = empty_rules
                else:
                    self.non_target_rules = empty_rules
                return

            te = TransactionEncoder()
            te.fit(lst)
            te_ary = te.transform(lst, sparse=True)
            sprs_df = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)

            # Free-up mem
            clean_sublist = None
            lst = None
            te = None
            te_ary = None

            # Frequent pattern growth
            if not fpg_file:
                fpg_file = Path(f'{self.outdir}/{self.fileID}{tgt.split(" ")[0]}_FPG.tsv') 
                fpg_df = fpgrowth(sprs_df, min_support = min_support, use_colnames = True, max_len=self.max_biomk+1)
                fpg_df['itemsets'] = fpg_df['itemsets'].apply(lambda x: list(x) if pd.notna else list())
                fpg_df.sort_values(
                    'support', ascending=False).to_csv(
                        fpg_file, sep='\t', index=False)
            else:
                fpg_df = pd.read_csv(fpg_file, sep='\t')
                fpg_df['itemsets'] = fpg_df['itemsets'].apply(literal_eval)
                wrt_str = f'Using file {str(Path(fpg_file))} for frequent patterns dataset\n'
                write_log(self.write, self.log_file, wrt_str)
            sprs_df = None

            wrt_str = [f'\tNumber of frequent patterns for {tgt}: {fpg_df.shape[0]}\n',
                       f'\n[{timefmt()}] Generating association rules for {tgt}...\n']
            write_log(self.write, self.log_file, wrt_str)
            # If no frequent itemsets were found, skip association rule generation
            if fpg_df.empty:
               wrt_str = f'\n[{timefmt()}] No frequent itemsets found; skipping association rules.\n'
               write_log(self.write, self.log_file, wrt_str)
            # create an empty DataFrame with the expected columns so downstream code doesn't crash
               fpg_rules_df = pd.DataFrame(
                   columns=[
                     'antecedents',
                     'consequents',
                     'support',
                     'confidence',
                     'lift',
                     'leverage',
                     'conviction',
                     'zhangs_metric',
                   ]
               )
            else:
               fpg_rules_df = association_rules(fpg_df, metric="confidence", min_threshold=min_conf)


            fpgrules_file = Path(f'{self.outdir}/{self.fileID}{tgt.split(" ")[0]}_FPG_rules.tsv')

            fpg_columns = ['antecedents', 'consequents', 'support', 'confidence', 
                        'lift', 'leverage', 'conviction', 'zhangs_metric']
            fpg_rules_df = fpg_rules_df[[c for c in fpg_columns if c in fpg_rules_df.columns]]

            wrt_str = [f'\tNumber of association rules for {tgt}: {fpg_rules_df.shape[0]}\n',
                       f'\nMax calculated metrics values:\n{fpg_rules_df[fpg_columns].max(numeric_only=True)}\n',
                       f'\nMin calculated metrics values:\n{fpg_rules_df[fpg_columns].min(numeric_only=True)}\n']
            write_log(self.write, self.log_file, wrt_str)

            # Free-up mem
            fpg_df = None

            # filter by various threshold values 
            wrt_str = f'\n[{timefmt()}] Inititial filtering of association rules for {tgt} (min lift = {min_lift})...\n'
            write_log(self.write, self.log_file, wrt_str)
            fpg_rules_df = fpg_rules_df[(fpg_rules_df['lift'] >= min_lift) & 
                                        ((fpg_rules_df['conviction'] == 'inf') | (fpg_rules_df['conviction'] >= min_conv)) & 
                                        (fpg_rules_df['zhangs_metric'] >= min_zhang)].round(
                                            {c:4 for c in [
                                                'support', 'confidence', 'lift', 'leverage', 
                                                'conviction', 'zhangs_metric']})
            wrt_str = f'\tNumber of association rules for {tgt} after filtering: {fpg_rules_df.shape[0]}\n'
            write_log(self.write, self.log_file, wrt_str)
            
            # combine antecedant and consequent columns 
            fpg_rules_df['orthogroups'] = [sorted(frozenset.union(*X)) for X in fpg_rules_df[['antecedents', 'consequents']].values]
            fpg_rules_df['num_orthogroups'] = fpg_rules_df['orthogroups'].apply(lambda x: len(x))
            fpg_rules_df['orthogroups'] = fpg_rules_df['orthogroups'].apply(lambda x: str(x))
            fpg_columns = ['orthogroups', 'num_orthogroups', 'support', 'confidence', 
                        'lift', 'leverage', 'conviction', 'zhangs_metric']
            fpg_rules_df = fpg_rules_df[fpg_columns]

            # only keep one row with the same orthogroup list and symmetrical scoring metrics (support and lift)
            subset_cols = ['orthogroups', 'support', 'lift']
            fpg_rules_df.drop_duplicates(subset=[c for c in subset_cols if c in fpg_rules_df.columns], 
                                        ignore_index=True, keep='first', inplace=True)
            wrt_str = f'\tNumber of association rules for {tgt} after removing duplicates: {fpg_rules_df.shape[0]}\n'
            write_log(self.write, self.log_file, wrt_str)

            fpg_rules_df.to_csv(fpgrules_file, sep='\t', index=False)
            wrt_str = [f'All frequent patterns for {tgt} saved to: {fpg_file}\n',
                       f'Filtered association rules for {tgt} saved to: {fpgrules_file}\n']
            write_log(self.write, self.log_file, wrt_str)
            if target:
                self.target_rules = fpg_rules_df
            else:
                self.non_target_rules = fpg_rules_df
            return
        _get_features_df(target=True, min_lift=min_lift, fpg_file=fpg_file) 
        _get_features_df(target=False, min_lift=non_lift, fpg_file=fpg_file) 
        return 

    def get_biomarkers(self, rules_file=None): ## !! need to add to parameters file
        if rules_file:
            wrt_str = [f'\n[{timefmt()}] Reading previously generated association rules from files:\n']
            if glob.glob(f'{self.outdir}/{self.fileID}Target_FPG_rules.tsv*'):
                target_file = glob.glob(f'{self.outdir}/{self.fileID}Target_FPG_rules.tsv*')[0]
                wrt_str.append(f'\t{target_file}\n')
            else:
                try:
                    self.target_rules
                except NameError:
                    err_str = f'No target FPG dataframe or file "{self.outdir}/{self.fileID}Target_FPG_rules.tsv*" found'
                    raise FileNotFoundError(err_str)
            if glob.glob(f'{self.outdir}/{self.fileID}nonTarget_FPG_rules.tsv*'):
                nontarget_file = glob.glob(f'{self.outdir}/{self.fileID}nonTarget_FPG_rules.tsv*')[0]
                wrt_str.append(f'\t{nontarget_file}\n')
            else:
                try:
                    self.non_target_rules
                except NameError:
                    err_str = f'No non-target FPG dataframe or file "{self.outdir}/{self.fileID}nonTarget_FPG_rules.tsv*" found'
                    raise FileNotFoundError(err_str)
            write_log(self.write, self.log_file, wrt_str)
            self.target_rules = pd.read_csv(target_file, sep='\t')
            self.non_target_rules = pd.read_csv(nontarget_file, sep='\t')
        if not self.non_target_rules.empty:
            self.non_target_rules['orthogroups'] = self.non_target_rules['orthogroups'].apply(literal_eval)
            # get list of all single orthogroups in a rule from non_target_rules 
            non_lst = list(set([c for l in self.non_target_rules['orthogroups'].to_list() for c in l]))
        else:
            non_lst = []
        
        # filter target_rules to remove any rule containing a non-target orthogroup 
        def _filt_features(target_lst, non_lst):
            lst = [f for f in target_lst if f not in non_lst]
            if lst == target_lst: 
                return True
            else:
                return False
        if not self.target_rules.empty:
            self.target_rules['orthogroups'] = self.target_rules['orthogroups'].apply(literal_eval)
            wrt_str = f'\n[{timefmt()}] Final filtering of association rules for Target ({self.toi})...\n'
            write_log(self.write, self.log_file, wrt_str)
            self.filt_biomarkers = self.target_rules[self.target_rules['orthogroups'].apply(lambda x: _filt_features(x, non_lst))]
            self.final_feat = list(set([c for l in self.filt_biomarkers['orthogroups'].to_list() for c in l]))
        else:
            self.filt_biomarkers = pd.DataFrame()
            self.final_feat = []
        filt_file = Path(f'{self.outdir}/{self.fileID}filtered_rules.tsv') 
        self.filt_biomarkers.to_csv(filt_file, sep='\t', index=False)

        def _suggest_biomk(all_feat):
            feat_combos = list(combinations(set([o for l in all_feat for o in l]), self.max_biomk))
            sugg_feat = {}
            for s in feat_combos:
                n=0
                for i in range(len(all_feat)): 
                    if all(item in all_feat[i] for item in s):
                        n+=1
                sugg_feat[', '.join(s)] = n 
            sugg_feat_df = pd.Series(sugg_feat).to_frame(name='co_occurance').sort_values(by='co_occurance', ascending=False)
            return sugg_feat_df

        wrt_str = f'\n[{timefmt()}] Getting suggested biomarker groups for Target ({self.toi})...\n'
        write_log(self.write, self.log_file, wrt_str)

        if not self.filt_biomarkers.empty:
            self.suggested_df = _suggest_biomk(self.filt_biomarkers[self.filt_biomarkers['num_orthogroups'] == self.max_biomk+1]['orthogroups'].to_list()
                                        ).head(500)
            inf_str = "\n\t".join(self.suggested_df.head(10).index.to_list())
        else:
            self.suggested_df = pd.DataFrame()
            inf_str = 'NONE FOUND'
        
        biomk_file = Path(f'{self.outdir}/{self.fileID}biomarker_groups.tsv') 
        self.suggested_df.to_csv(biomk_file, sep='\t', index=True) 

        wrt_str = [f'\tNumber of association rules for target ({self.toi}) after filtering non-target features: {self.filt_biomarkers.shape[0]}\n',
                   f'Top suggested biomarker groups (co-occuring features):\n\t{inf_str}\n',
                   f'\nSuggested biomarker groups saved to: {biomk_file}\n',
                   f'\nFiltered association rules saved to: {filt_file}\n',
                   f'\tTotal number of features in association rules: {len(self.final_feat)}\n',
                   f'\t{self.final_feat}\n'] 
        write_log(self.write, self.log_file, wrt_str)
        return 
    
    def get_annotions(self, annot_file):
        wrt_str = f'\n[{timefmt()}] Getting gene names for suggested biomarker groups...\n'
        write_log(self.write, self.log_file, wrt_str)
        iso_list = self.isolate_clust[(self.isolate_clust['cluster'] == self.clust[1]) & 
                                      (self.isolate_clust[self.targets] == self.toi)][
                                          self.id_col].to_list() + ["Orthogroup"]
        orthoannot_df = pd.read_csv(annot_file, sep='\t')
        orthoannot_df = orthoannot_df[iso_list]
        annots_dict = {}
        for og in self.final_feat:
            x = orthoannot_df[orthoannot_df['Orthogroup'] == og][
                [c for c in orthoannot_df.columns if c != 'Orthogroup']] 
            x = [o.split(', ') for l in x.values.tolist() for o in l if str(o) != 'nan']
            annots_dict[og]= dict(Counter([re.sub(r'\w+_\d+ ', '', o) for l in x for o in l]))
        annots = []
        for k, v in annots_dict.items():
            prot_name = ['no annotations']
            if v.values():
                x = v
                prot_name = [max(x, key=x.get)]
                if max(x.values())/sum(x.values()) < 0.8 and sum(x.values())>10:
                    x.pop(max(x, key=x.get))
                    prot_name.append(max(x, key=x.get))
            annots.append([k, ', '.join(prot_name)])
        self.annots_df = pd.DataFrame(annots, columns=['Orthogroup', 'Annotation(s)'])
        return self.annots_df
