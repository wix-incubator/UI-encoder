"""Metrics for the probe: macro F1 (the paper's headline number), balanced accuracy, top-1, the
per-class report and confusion matrix, and the human-readable report.txt."""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)

SIZE_TIERS = [(1, 3, 'sub/tiny 1-3'), (4, 15, 'small 4-15'), (16, 63, 'medium 16-63'), (64, 10 ** 9, 'large 64+')]
PRIMARY_METRIC = {'classification': 'balanced_acc'}


def scores(y_true, y_pred):
    """headline metrics.
    Args:
        y_true: true labels.
        y_pred: predicted labels.

    Returns:
        {'balanced_acc', 'macro_f1' (unweighted mean of the per-class F1), 'top1'}.
    """
    return {'balanced_acc': float(balanced_accuracy_score(y_true, y_pred)),
            'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'top1': float(accuracy_score(y_true, y_pred))}


def detail(y_true, y_pred):
    """the heavy per-class blobs that go into metrics.json.
    Args:
        y_true: true labels.
        y_pred: predicted labels.
    Returns:
        {'classification_report': sklearn's report as a dict,
         'confusion': {'labels': classes, 'matrix': counts, rows = true class}}.
    """
    labels = sorted(set(y_true) | set(y_pred))
    return {'classification_report': classification_report(y_true, y_pred, digits=4, zero_division=0,
                                                           labels=labels, output_dict=True),
            'confusion': {'labels': labels,
                          'matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist()}}


def report(title, y_true, y_pred, n_patches):
    """the text block written to report.txt: headline metrics, the per-class precision/recall/F1
    table, the confusion matrix, and the metrics again per component size tier (SIZE_TIERS).
    Args:
        title: heading of the block.
        y_true: true labels.
        y_pred: predicted labels.
        n_patches: how many patches each component's box covers, which sets its size tier.
    Returns:
        the block as text.
    """
    headline = scores(y_true, y_pred)
    lines = [f'=== {title} ===']
    lines.append(f'balanced acc {headline["balanced_acc"]:.4f} | macro F1 {headline["macro_f1"]:.4f} | '
                 f'top-1 {headline["top1"]:.4f}')
    labels = sorted(set(y_true) | set(y_pred))
    lines.append(classification_report(y_true, y_pred, digits=3, zero_division=0, labels=labels))
    confusion = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
    lines += ['confusion (rows=true):', confusion.to_string()]
    lines += ['', 'size tiers:']
    for min_patches, max_patches, label in SIZE_TIERS:
        in_tier = (n_patches >= min_patches) & (n_patches <= max_patches)
        if not in_tier.sum():
            continue
        with warnings.catch_warnings():
            # a size tier can legitimately hold no instance of a class the model predicts there;
            # balanced accuracy still averages recall over the classes that ARE present
            warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')
            tier_scores = scores(y_true[in_tier], y_pred[in_tier])
        lines.append(f'  {label}: n={int(in_tier.sum())} balanced {tier_scores["balanced_acc"]:.4f} '
                     f'top-1 {tier_scores["top1"]:.4f}')
    return '\n'.join(lines)


def mean_std(scores_by_seed):
    """mean and standard deviation of every metric across probe seeds.
    Args:
        scores_by_seed: {seed: {metric: value}}.
    Returns:
        ({metric: mean}, {metric: std}).
    """
    metrics = sorted({metric for seed_scores in scores_by_seed.values() for metric in seed_scores})
    return ({metric: float(np.mean([seed_scores[metric] for seed_scores in scores_by_seed.values()]))
             for metric in metrics},
            {metric: float(np.std([seed_scores[metric] for seed_scores in scores_by_seed.values()]))
             for metric in metrics})
