"""Component linear probe on the released Hugging Face dataset: N encoders -> N reports.

Adapted from the probe.py that produced the reported numbers. The two changes: the labels come
from the dataset's own `classes` column instead of a taxonomy file, and the probe is scored on the
test split only, because the validation split is not released. Feature extraction (embed.py), the
linear probe (head.py), scoring (scoring.py), the seed subsample and the C selection are the original code.

    cd components_masking
    python -m probes.probe --models ours=Wix/ui-encoder-ours --gpus 0,0,0,0 --root ../work
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_info

from .head import CV_FOLDS, LinearProbe
from . import scoring
from .embed import MAX_PATCHES, ENCODERS_PER_PASS, BASE_MODEL, IMAGE_PROCESSOR, embed_many
from .probe_dataset import (DATASET, N_PER_CLASS, PER_PAGE_CAP, ProbeDataset, balanced_sampling,
                            class_distribution_table)

HERE = Path(__file__).parent


def log(msg):
    """print immediately (unbuffered), so progress shows up in piped and teed logs.
    Args:
        msg: the text to print.
    """
    print(msg, flush=True)


def parse_encoders(args):
    """the encoders to score: from --models NAME=PATH ..., or the single --name/--model pair.
    Args:
        args: parsed command line.
    Returns:
        [(name, Hugging Face id or checkpoint dir)].
    """
    if args.models:
        encoders = []
        for arg in args.models:
            if '=' not in arg:
                raise SystemExit(f'--models expects NAME=PATH, got "{arg}"')
            name, path = arg.split('=', 1)
            encoders.append((name, path))
        return encoders
    if not args.name:
        raise SystemExit('need --name (with --model) or --models NAME=PATH ...')
    return [(args.name, args.model)]


def score_encoder(name, checkpoint, features, probe_data, args, out_root, train_rows_by_seed,
                  train_pages_by_seed, dist_text, dists):
    """fit and score the linear probe for one encoder, and write its outputs to <out_root>/<name>/:
    metrics.json, report.txt and preds_test.parquet. For each probe seed, the probe is fitted on the
    seed split's training rows and predicts every test component; the reported test metrics are
    those of the first seed's fit.
    Args:
        name: encoder name, e.g. 'ours'.
        checkpoint: its Hugging Face id or checkpoint dir (recorded in metrics.json).
        features: {split: (X, meta)} for this encoder, from embed_many.
        probe_data: the ProbeDataset (gives each test component's page).
        args: parsed command line (uses seeds, C, fit_jobs, n, dataset, max_patches and gpu_list).
        out_root: folder that gets one subfolder per encoder.
        train_rows_by_seed: {probe seed: training row positions in the seed split}, shared by every
            encoder.
        train_pages_by_seed: {probe seed: page id of each training row}, the groups for selecting C.
        dist_text: class-distribution table for report.txt.
        dists: the same distributions as {title: {class: count}}, for metrics.json.
    Returns:
        the report text, as a list of blocks.
    """
    X_train, meta_train = features['seed']
    X_test, meta_test = features['test']
    y_train, y_test = meta_train['cls'].to_numpy(), meta_test['cls'].to_numpy()
    first_seed = args.seeds[0]
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    preds, scores_by_seed, C_by_seed, cv_rows = {}, {}, {}, []
    for seed in args.seeds:
        train_rows = train_rows_by_seed[seed]
        probe = LinearProbe(C=args.C, seed=seed, n_jobs=args.fit_jobs)
        probe.fit(X_train[train_rows], y_train[train_rows], groups=train_pages_by_seed[seed])
        preds[seed] = probe.predict(X_test)
        scores_by_seed[seed] = scoring.scores(y_test, preds[seed])
        if seed == first_seed:                     # the reported test numbers are the first seed's fit
            fitted = probe
        C_by_seed[seed] = probe.C_
        cv_rows += probe.cv_rows
        if args.C is None:
            cv_summary = ' '.join(f'{row["C"]:g}:{row["cv_balanced_acc"]:.4f}' for row in probe.cv_rows)
            edge_note = ' [EDGE OF GRID]' if probe.C_ in (probe.c_grid[0], probe.c_grid[-1]) else ''
            log(f'  [{name}] seed {seed}: cv {cv_summary} -> C={probe.C_:g}{edge_note}')
        metric = scoring.PRIMARY_METRIC[probe.task]
        log(f'  [{name}] linear seed {seed} (C={probe.C_:g}): test {metric} '
            f'{scores_by_seed[seed][metric]:.4f} | ' +
            ' '.join(f'{key} {value:.4f}' for key, value in scores_by_seed[seed].items() if key != metric))

    pred = preds[first_seed]
    mean, std = scoring.mean_std(scores_by_seed)
    detail = scoring.detail(y_test, pred)
    report_parts = [f"run {name} | model {checkpoint} | dataset {args.dataset} | cap n={args.n} | "
                    f"seeds={args.seeds} | heads=['linear']", dist_text,
                    scoring.report('TEST — linear', y_test, pred, meta_test['n_patches'].to_numpy())]
    metrics = {'name': name, 'model': checkpoint, 'dataset': args.dataset,
               'dataset_source': probe_data.source, 'image_processor': IMAGE_PROCESSOR,
               'classes': sorted(set(y_test)), 'n': args.n,
               'seeds': args.seeds, 'heads': ['linear'], 'max_patches': args.max_patches,
               'gpus': args.gpu_list, 'feat_l2_norm_train': float(np.linalg.norm(X_train, axis=1).mean()),
               # CPU BLAS threads during the probe fit: the fit's sums, and so the last digits of
               # the scores, depend on this count (reproduce.sh sets 8)
               'blas_threads': sorted({pool['num_threads'] for pool in threadpool_info()
                                       if pool['user_api'] == 'blas'}),
               'distributions': dists,
               'head_detail': {'linear': {'task': fitted.task, 'params': fitted.params(),
                                          'test': scores_by_seed[first_seed],
                                          'test_per_seed': scores_by_seed,
                                          'test_seed_mean': mean, 'test_seed_std': std}}}
    metrics |= {f'test_{metric}': value for metric, value in scores_by_seed[first_seed].items()}
    metrics |= {'test_classification_report': detail['classification_report'],
                'test_confusion': detail['confusion']}
    metrics |= {'C': C_by_seed[first_seed], 'C_per_seed': C_by_seed,
                'C_selection': 'fixed' if args.C is not None else 'cv',
                'cv_grid': cv_rows, 'test_scores_per_seed': scores_by_seed}

    preds_table = pd.DataFrame({'comp_id': meta_test['comp_id'],
                                'page': meta_test[['comp_id']].merge(probe_data.pages('test'), on='comp_id',
                                                                 how='left')['page'].to_numpy(),
                                'true': y_test, 'pred': pred, 'pred_linear': pred,
                                'n_patches': meta_test['n_patches']})

    preds_table.to_parquet(out_dir / 'preds_test.parquet', index=False)
    (out_dir / 'report.txt').write_text('\n\n'.join(report_parts))
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=1, default=float))
    log(f'-> {out_dir}/report.txt | {out_dir}/metrics.json')
    return report_parts


def main():
    """load seed + test from the dataset, embed every page once with every encoder, draw the shared
    training subsample from seed, then fit and score each encoder (score_encoder)."""
    parser = argparse.ArgumentParser(description='component probe on the released dataset: '
                                                 'N encoders -> N reports')
    parser.add_argument('--name', default=None, help='run name for single-encoder mode, e.g. ours')
    parser.add_argument('--model', default=BASE_MODEL, help='HF id or checkpoint dir (single-encoder mode)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='NAME=PATH ... — score many encoders in ONE pass: each page is decoded '
                             'once and broadcast to every resident encoder (decoding is the '
                             'bottleneck, not the forward)')
    parser.add_argument('--models-per-pass', type=int, default=ENCODERS_PER_PASS,
                        help='how many encoders stay resident per GPU (~0.85GB each)')
    parser.add_argument('--dataset', default=DATASET,
                        help='Hugging Face dataset id, or a local directory of <split>-*.parquet files')
    parser.add_argument('--n', type=int, default=N_PER_CLASS, help='per-class cap on probe train components')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--C', type=float, default=None,
                        help='fixed C: skips the CV grid. Default: C selected per encoder by '
                             'page-grouped CV inside the seed split, as in the paper.')
    parser.add_argument('--max-patches', type=int, default=MAX_PATCHES)
    parser.add_argument('--batch-size', type=int, default=16, help='pages per forward')
    parser.add_argument('--workers', type=int, default=8, help='total dataloader workers (split across gpus)')
    parser.add_argument('--gpus', default='all', help="'all' | 'none' | comma list e.g. 0,1,2,3")
    parser.add_argument('--root', default=None, help='work dir for screenshots, cache and out '
                                                     '(default: next to this file)')
    parser.add_argument('--fit-jobs', type=int, default=0, help='parallel probe fits (0 = auto)')
    args = parser.parse_args()

    encoders = parse_encoders(args)
    if args.gpus == 'all':
        gpus = list(range(torch.cuda.device_count()))
    elif args.gpus in ('none', ''):
        gpus = []
    else:
        gpus = [int(gpu) for gpu in args.gpus.split(',') if gpu != '']
    args.gpu_list = gpus
    if len(gpus) != 4:
        log(f'WARNING: --gpus gives {len(gpus)} entries. The reported runs sharded pages over 4, and '
            'the training subsample follows that order, so scores will differ from the released ones. '
            'Pass four entries, repeating ids on smaller machines (e.g. --gpus 0,0,0,0).')

    root = Path(args.root) if args.root else HERE
    out_root = root / 'out'
    out_root.mkdir(parents=True, exist_ok=True)

    splits = ['seed', 'test']
    probe_data = ProbeDataset(args.dataset, root, splits=splits)
    log(f'encoders: {len(encoders)}, linear probe -> {out_root}')

    features = embed_many({split: probe_data[split] for split in splits}, str(probe_data.data_dir),
                          dict(encoders), args.max_patches, args.batch_size, args.workers, gpus,
                          encoders_per_pass=args.models_per_pass)

    # meta is produced by the shared _component_patches plan, so it is identical for every encoder:
    # the subsample (and therefore the paired comparison across encoders) is computed ONCE.
    first_features = features[encoders[0][0]]
    for name, _ in encoders[1:]:
        for split in splits:
            if not first_features[split][1]['comp_id'].equals(features[name][split][1]['comp_id']):
                raise SystemExit(f'component rows for "{name}" differ from "{encoders[0][0]}" on {split} — '
                                 f'refusing to score encoders on non-identical rows')
    meta_seed = first_features['seed'][1]
    y_seed = meta_seed['cls'].to_numpy()
    train_rows_by_seed, train_pages_by_seed = {}, {}
    for seed in args.seeds:
        train_rows_by_seed[seed], train_pages_by_seed[seed] = balanced_sampling(
            meta_seed, probe_data.pages('seed'), args.n, seed)

    first_seed = args.seeds[0]
    dists = {'seed_pool (natural)': probe_data.class_counts('seed'),
             f'probe_train (cap {args.n})':
                 pd.Series(y_seed[train_rows_by_seed[first_seed]]).value_counts().to_dict(),
             'test_eval (as-is)': probe_data.class_counts('test')}
    class_order = sorted(dists['seed_pool (natural)'], key=lambda cls: -dists['seed_pool (natural)'][cls])
    dist_text = ('class distributions\n' + class_distribution_table(dists, class_order) +
                 '\nprobe_train: no resampling, class_weight=balanced (each class capped at '
                 f'{args.n} unique components, <={PER_PAGE_CAP}/page)')
    log('')
    log(dist_text)
    log('')
    if args.C is not None:
        log(f'C FIXED at {args.C:g} for every seed — CV grid skipped. cv_grid empty in metrics.json.')
    else:
        log(f'selecting C by {CV_FOLDS}-fold page-grouped CV inside the seed split; '
            f'test is NOT used for selection')

    last_report = None
    for encoder_no, (name, checkpoint) in enumerate(encoders, 1):
        log(f'\n=== [{encoder_no}/{len(encoders)}] {name} ===')
        last_report = score_encoder(name, checkpoint, features[name], probe_data, args, out_root,
                                    train_rows_by_seed, train_pages_by_seed, dist_text, dists)
    if len(encoders) == 1 and last_report:
        log('\n' + '\n\n'.join(last_report))


if __name__ == '__main__':
    main()
