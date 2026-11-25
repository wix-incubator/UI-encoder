#!/usr/bin/env python3
"""Print the test results of a reproduction run as the README's results tables.

reproduce.sh calls this as its last step. It reads <root>/out/<encoder>/metrics.json, the file
components_masking/probes/probe.py writes for each encoder, and prints two Markdown tables:

1. Per-class test F1 in the layout of the README's highlight table. Frozen SigLIP2 is shown as
   an absolute F1; random, random+ and ours are shown as the change from SigLIP2, and the best
   mask per class is in bold. Classes where ours is best come first, then the rest, each group
   ordered by the gain of ours. The last row is macro F1.
2. Test macro F1 of all six encoders, with the Hugging Face seed-0 checkpoint scores and the
   gaps in F1 points. Gaps are calculated from the scores shown in the table. The reference and
   gap are shown only for a run of the released checkpoint with the release settings
   (is_release_run); any other run in a folder of the same name shows its own score alone.

If an encoder's metrics.json is missing, or is not a cross-validated run with test scores, that
encoder is listed as missing, left out of the tables, and the exit status is 1.
"""

import argparse
import json
import sys
from pathlib import Path

# (output directory under <root>/out, label in the tables), in the README's order
ENCODERS = [
    ('siglip2_base', 'SigLIP2, frozen'),
    ('random', 'random'),
    ('random-plus', 'random⁺'),
    ('random-plus-bal', 'random⁺, class-balanced loss'),
    ('ours', 'ours'),
    ('ours-bal', 'ours, class-balanced loss'),
]
BASE = 'siglip2_base'
MASKS = ['random', 'random-plus', 'ours']   # the highlight table's columns after SigLIP2

# Checkpoint-specific test macro F1 (%) from the Wix model cards linked in the README.
HF_SEED_0_MACRO_F1 = {
    'random': 62.66,
    'random-plus': 64.07,
    'random-plus-bal': 66.63,
    'ours': 66.00,
    'ours-bal': 68.13,
}
# Settings of the runs behind those scores: a run is compared with them only if it used these.
RELEASE_SETTINGS = {'seeds': [0], 'n': 1000, 'max_patches': 1024, 'C_selection': 'cv'}
TEST_COMPONENTS = 72507


def is_release_run(name, m):
    """Whether a metrics.json comes from scoring the released checkpoint `name` as the README does.

    The checkpoint must be the Hugging Face repo Wix/ui-encoder-<name> or a local folder named
    <name> (reproduce.sh downloads each checkpoint to <workdir>/models/<name>), the probe settings
    must be those of RELEASE_SETTINGS, and every test component must have been scored.
    """
    model = str(m.get('model', '')).rstrip('/')
    same_checkpoint = model == f'Wix/ui-encoder-{name}' or Path(model).name == name
    same_settings = all(m.get(key) == value for key, value in RELEASE_SETTINGS.items())
    scored = m.get('test_classification_report', {}).get('macro avg', {}).get('support')
    return same_checkpoint and same_settings and scored == TEST_COMPONENTS


def load_run(out_dir, name):
    """Return one encoder's test F1 in percent, or None if its run is missing.

    Args:
        out_dir: <root>/out of the run.
        name: encoder directory name, e.g. 'ours'.

    Returns:
        {'macro': test macro F1, 'per_class': {class: test F1}, 'release': is_release_run(...)},
        or None when metrics.json is absent or lacks CV-selected test scores.
    """
    path = out_dir / name / 'metrics.json'
    if not path.exists():
        return None
    m = json.loads(path.read_text())
    if m.get('C_selection') != 'cv' or 'test_macro_f1' not in m:
        return None
    report = m['test_classification_report']
    return {'macro': m['test_macro_f1'] * 100,
            'per_class': {c: report[c]['f1-score'] * 100 for c in m['classes']},
            'release': is_release_run(name, m)}


def fmt_delta(d):
    """Format a change in F1 points to one decimal: '+1.7', '−2.4', '0.0'."""
    s = f'{d:+.1f}'
    return '0.0' if s in ('+0.0', '-0.0') else s.replace('-', '−')


def highlight_table(runs):
    """Per-class F1 rows in the layout of the README's highlight table (see the module docstring).

    Args:
        runs: {encoder name: load_run(...) result} for SigLIP2 and the three masks.

    Returns:
        The table as a list of Markdown lines.
    """
    classes = list(runs[BASE]['per_class'])
    rows = [(c, runs[BASE]['per_class'][c], {k: runs[k]['per_class'][c] - runs[BASE]['per_class'][c]
                                            for k in MASKS}) for c in classes]

    def cells(deltas):
        shown = {k: fmt_delta(d) for k, d in deltas.items()}
        best = max(round(d, 1) for d in deltas.values())
        return [f'**{shown[k]}**' if round(deltas[k], 1) == best else shown[k] for k in MASKS]

    def ours_best(deltas):
        return round(deltas['ours'], 1) == max(round(d, 1) for d in deltas.values())

    rows.sort(key=lambda r: (not ours_best(r[2]), -r[2]['ours']))
    lines = ['| class | SigLIP2 | random | random⁺ | **ours** |', '|---|---:|---:|---:|---:|']
    for c, base, deltas in rows:
        lines.append(f'| {c} | {base:.1f} | ' + ' | '.join(cells(deltas)) + ' |')
    macro = {k: runs[k]['macro'] - runs[BASE]['macro'] for k in MASKS}
    lines.append(f'| **macro F1** | **{runs[BASE]["macro"]:.1f}** | ' + ' | '.join(cells(macro)) + ' |')
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--root', required=True, type=Path, help='the --root the probe was run with')
    args = ap.parse_args()

    out_dir = args.root / 'out'
    runs = {name: load_run(out_dir, name) for name, _ in ENCODERS}
    missing = [name for name, r in runs.items() if r is None]

    if all(runs[k] for k in [BASE] + MASKS):
        print('Per-class test F1 (change from frozen SigLIP2; bold = best mask per class)\n')
        print('\n'.join(highlight_table(runs)))
    print('\nTest macro F1\n')
    print('| encoder | test macro F1 (%) | HF seed-0 macro F1 (%) | gap (F1 points) |')
    print('|---|---:|---:|---:|')
    custom = []
    for name, label in ENCODERS:
        run = runs[name]
        if run is None:
            continue
        macro = round(run['macro'], 2)
        reference = HF_SEED_0_MACRO_F1.get(name) if run['release'] else None
        if reference is None:
            print(f'| {label} | {macro:.2f} | — | — |')
            if name in HF_SEED_0_MACRO_F1:
                custom.append(name)
        else:
            print(f'| {label} | {macro:.2f} | {reference:.2f} | {macro - reference:+.2f} |')
    if custom:
        print('\nno Hugging Face comparison for ' + ', '.join(custom) + ': the run is not the released '
              'checkpoint scored with the release settings (see is_release_run)')
    if missing:
        print('\nmissing or incomplete runs (no CV-selected test scores in <root>/out/<encoder>/metrics.json): '
              + ', '.join(missing))
    sys.exit(1 if missing else 0)


if __name__ == '__main__':
    main()
