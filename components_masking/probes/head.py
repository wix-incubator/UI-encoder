"""The probe head: LinearProbe, a linear classifier fitted on frozen component features, with C
chosen by page-grouped cross-validation. It is the probe behind every number in the paper.
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
CV_FOLDS = 3

# no torch, no argparse, no paths: a head is (fit, predict) over numpy, so the training loop can
# import this module and score a live encoder with the SAME code the offline probe reports.


def _linear_pipeline(C):
    """the linear probe: standardize each feature dimension, then multinomial logistic regression
    with inverse-frequency class weights. One pipeline, so the scaler is fitted on training rows
    only.

    Args:
        C: inverse regularization strength of the logistic regression.

    Returns:
        an unfitted sklearn Pipeline.
    """
    return Pipeline([('sc', StandardScaler()),
                     ('lr', LogisticRegression(C=C, solver='lbfgs', max_iter=2000, tol=1e-4,
                                               class_weight='balanced'))])


def _cv_score(X, y, folds, C):
    """cross-validation for one C: fit on each fold's training rows, score balanced accuracy on its
    held-out rows. Runs in a worker process.

    Args:
        X: (n, d) training features.
        y: (n,) training labels.
        folds: [(training row positions, held-out row positions)], one pair per fold.
        C: inverse regularization strength.

    Returns:
        (C, mean balanced accuracy, per-fold balanced accuracies).
    """
    fold_scores = [float(balanced_accuracy_score(y[val_rows], _linear_pipeline(C).fit(X[train_rows], y[train_rows])
                                                 .predict(X[val_rows])))
                   for train_rows, val_rows in folds]
    return C, float(np.mean(fold_scores)), fold_scores


class LinearProbe:
    """the paper's linear probe. With C=None (the paper's setting) C is selected from c_grid by
    `folds`-fold cross-validation inside the training rows, grouped by page so no page is split
    between folds; the probe is then refitted on all training rows with that C. After fit, C_ is the
    chosen C and cv_rows the per-C scores.
    """

    name = 'linear'
    task = 'classification'

    def __init__(self, C=None, c_grid=None, folds=CV_FOLDS, seed=0, n_jobs=0, **_):
        """Args:
            C: a fixed C; None (the paper's setting) selects it by cross-validation.
            c_grid: the C values to try. Default: C_GRID.
            folds: number of cross-validation folds. Default: CV_FOLDS (3).
            seed: seeds the fold assignment.
            n_jobs: C values cross-validated in parallel; 0 = automatic.
            **_: other options, ignored.
        """
        self.C = C
        self.c_grid = list(c_grid or C_GRID)
        self.folds = folds
        self.seed = seed
        self.n_jobs = n_jobs
        self.C_ = C
        self.cv_rows = []

    def params(self):
        """the head's settings, recorded in metrics.json.
        Returns:
            {'C': chosen C, 'C_selection': 'cv' or 'fixed', 'cv_folds': folds or None}.
        """
        return {'C': self.C_, 'C_selection': 'fixed' if self.C is not None else 'cv',
                'cv_folds': None if self.C is not None else self.folds}

    def fit(self, X, y, groups=None):
        """select C (unless fixed), then fit the pipeline on all training rows.
        Args:
            X: (n, d) training features.
            y: (n,) training labels.
            groups: (n,) page id of each row; required when C is selected.
        Returns:
            self.
        """
        if self.C is None:
            self.C_ = self._select_C(X, y, groups)
        self.model_ = _linear_pipeline(self.C_).fit(X, y)
        return self

    def predict(self, X):
        """predict labels.
        Args:
            X: (n, d) features.
        Returns:
            (n,) predicted labels.
        """
        return self.model_.predict(X)

    def _select_C(self, X, y, groups):
        """the C with the best mean cross-validated balanced accuracy; ties go to the smaller C. Folds are
        stratified by class and grouped by page, and shared by every C. Also sets self.cv_rows.
        Args:
            X: (n, d) training features.
            y: (n,) training labels.
            groups: (n,) page id of each row.
        Returns:
            the chosen C.
        """
        if groups is None:
            raise ValueError('LinearProbe: page groups are required to select C by grouped CV; '
                             'pass groups=... or a fixed C')
        folds = list(StratifiedGroupKFold(n_splits=self.folds, shuffle=True,
                                          random_state=self.seed).split(X, y, groups))
        n_workers = self.n_jobs or min(len(self.c_grid), max(1, (os.cpu_count() or 4) // 4))
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_cv_score, X, y, folds, C): C for C in self.c_grid}
            progress = tqdm(as_completed(futures), total=len(futures), desc=f'cv[seed {self.seed}]',
                            unit='fit', dynamic_ncols=True)
            for future in progress:
                C, mean, per_fold = future.result()
                progress.set_postfix_str(f'C={C:g} cv_bal={mean:.4f}', refresh=True)
                results.append({'seed': self.seed, 'C': C, 'cv_balanced_acc': mean, 'folds': per_fold})
            progress.close()
        self.cv_rows = sorted(results, key=lambda row: row['C'])
        return min(results, key=lambda row: (-row['cv_balanced_acc'], row['C']))['C']
