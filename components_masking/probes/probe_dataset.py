"""probe_dataset.py turns the released dataset into a table with one row per
component (which page it's on, its box, its class), and saves each page's screenshot to disk so embed.py
can read it.

The released dataset already carries the class of every component, so this version only reads it.
balanced_sampling() and class_distribution_table() are unchanged.

Each released row becomes a page '<split>-<row index, zero-padded>'. The rows are in the order of
the original page ids and the zero-padding keeps that order when sorted, which the probe depends
on: embed.py sorts pages before sharding them over GPUs, and the seed subsample is drawn from the
resulting row order.
"""

import hashlib
import io
import json
from pathlib import Path

import pandas as pd
from filelock import FileLock
from huggingface_hub import HfApi
from PIL import Image

PER_PAGE_CAP = 5
N_PER_CLASS = 1000
PAGE_WIDTH = 1280               # CSS width every page was rendered at; boxes are in screenshot pixels
DATASET = 'Wix/ui-components'


def log(msg):
    """print immediately (unbuffered), so progress shows up in piped and teed logs.
    Args:
        msg: the text to print.
    """
    print(msg, flush=True)


def local_split_files(path, split):
    files = sorted(path.glob(f'{split}-*.parquet'))
    if not files:
        raise SystemExit(f'no {split}-*.parquet files in {path}')
    return files


def dataset_source(dataset, splits):
    """Identify the dataset contents used by a work directory."""
    path = Path(dataset)
    if path.is_dir():
        files = {}
        for split in splits:
            for shard in local_split_files(path, split):
                with shard.open('rb') as stream:
                    files[shard.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        return {'kind': 'parquet', 'files': files}

    info = HfApi().dataset_info(dataset)
    if not info.sha:
        raise RuntimeError(f'could not resolve a revision for dataset {dataset}')
    return {'kind': 'hub', 'id': info.id, 'revision': info.sha}


def load_split(dataset, split, revision=None):
    """the released split, with images left as their original bytes.
    Args:
        dataset: Hugging Face dataset id, or a local directory of <split>-*.parquet files.
        split: 'seed' or 'test'.
        revision: immutable Hub revision resolved for this run.
    Returns:
        a datasets.Dataset with one row per page (image, classes, rectangles, ...).
    """
    from datasets import Image as ImageFeature
    from datasets import load_dataset
    path = Path(dataset)
    if path.is_dir():
        files = [str(file) for file in local_split_files(path, split)]
        rows = load_dataset('parquet', data_files={split: files}, split=split)
    else:
        rows = load_dataset(dataset, split=split, revision=revision)
    return rows.cast_column('image', ImageFeature(decode=False))


def pixel_box_to_page_units(pixel_box, px_per_unit):
    """convert one released box into the format the rest of the code uses.
    The dataset gives boxes as corners in screenshot pixels, (x0, y0, x1, y1).
    embed.py, through rect_to_patch_box in masking.py, expects (x, y, w, h) in page units: the page's CSS pixels at its 1280-wide layout.
    load_components calls it once per component.
    Args:
        pixel_box: (x0, y0, x1, y1) in screenshot pixels, as released.
        px_per_unit: screenshot pixels per page unit (screenshot width / 1280).
    Returns:
        (x, y, w, h) in page units; ints when every value is whole, which holds for every released
        box.
    """
    x0, y0, x1, y1 = (coord / px_per_unit for coord in pixel_box)
    page_box = (x0, y0, x1 - x0, y1 - y0)
    if all(float(coord).is_integer() for coord in page_box):
        return tuple(int(coord) for coord in page_box)
    return page_box


def load_components(dataset, split, screenshots_dir, cache_dir, revision=None):
    """builds the component table of one split.

    When each case is used:
    - first run in a work folder: the split is read with load_split and the pages are walked in
      release order. Each page is named '<split>-<row index>' and each component
      '<page>-<component index>'; its box goes through pixel_box_to_page_units. The table is then saved
      to <cache_dir>/components_<split>.parquet.
    - every later run in the same folder (e.g. rerunning reproduce.sh): the saved table is read and
      returned at once; the screenshots are already on disk.
    - a page with no components is skipped.
    - a row whose classes and rectangles differ in length stops the run; this guards against a
      broken dataset copy and never happens on the released data.
    Args:
        dataset: Hugging Face dataset id or local directory, as in load_split.
        split: 'seed' or 'test'.
        screenshots_dir: where the screenshots are written.
        cache_dir: where the table is cached.
        revision: immutable Hub revision resolved for this run.
    Returns:
        DataFrame with one row per component: comp_id ('<page>-<n>'), page ('<split>-<row>'),
        x, y, w, h (page units) and cls.
    """
    cache_path = Path(cache_dir) / f'components_{split}.parquet'
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if all((screenshots_dir / f'{page}.webp').is_file() for page in cached['page'].unique()):
            return cached
    release = load_split(dataset, split, revision=revision)
    comp_rows = []
    for row_no, row in enumerate(release):
        if not row['classes']:
            continue
        if len(row['classes']) != len(row['rectangles']):
            raise SystemExit(f'{split} row {row_no}: classes/rectangles length mismatch')
        page, image_bytes = f'{split}-{row_no:05d}', row['image']['bytes']
        screenshot = screenshots_dir / f'{page}.webp'
        pending_screenshot = screenshot.with_suffix('.tmp')
        pending_screenshot.write_bytes(image_bytes)
        pending_screenshot.replace(screenshot)
        with Image.open(io.BytesIO(image_bytes)) as image:
            px_per_unit = image.size[0] / PAGE_WIDTH
        for comp_no, (cls, pixel_box) in enumerate(zip(row['classes'], row['rectangles'])):
            x, y, w, h = pixel_box_to_page_units(pixel_box, px_per_unit)
            comp_rows.append({'comp_id': f'{page}-{comp_no:03d}', 'page': page, 'x': x, 'y': y, 'w': w,
                              'h': h, 'cls': cls})
    components = pd.DataFrame(comp_rows, columns=['comp_id', 'page', 'x', 'y', 'w', 'h', 'cls'])
    pending = cache_path.with_suffix('.tmp')
    components.to_parquet(pending, index=False)
    pending.replace(cache_path)
    log(f'components[{split}]: {len(components)} components on {components.page.nunique()} of '
        f'{len(release)} pages -> {cache_path.name}')
    return components


def balanced_sampling(meta, pages, cap, seed):
    """picks the seed components the linear probe trains on: at most `cap` per class (1,000 in the
    paper). Page stratification limits frequent classes before sampling. Classes under the
    cap are kept whole, and the probe applies class_weight='balanced'. The probe scores
    the full test split.

    probe.py calls it once per probe seed, AFTER embedding (each page is embedded once anyway, so
    sampling earlier would save no GPU time), and every encoder trains on the same rows.

    A class with at most `cap` components is taken whole. For a larger class, the sampler
    shuffles rows and limits each page to PER_PAGE_CAP (5). It then draws up to `cap`
    components from those rows and tops up from the remaining components when needed.
    Finally, it shuffles all picks together.
    Every draw uses `seed` and pandas draws by position, so the result also depends on meta's row
    order (the page order embed.py produces).

    The sampler returns the selected rows and their page IDs for grouped cross-validation.
    Args:
        meta: the embedded components (comp_id, cls), in embedding order.
        pages: comp_id -> page table (ProbeDataset.pages('seed')).
        cap: most components per class (N_PER_CLASS = 1000 in the paper).
        seed: probe seed, used for every random draw.
    Returns:
        (row positions into meta, their page ids). The result also depends on meta's row order.
    """
    pool = meta.merge(pages, on='comp_id', how='left')
    picks = []
    for cls, class_rows in pool.groupby('cls'):
        if len(class_rows) > cap:
            class_rows = class_rows.sample(frac=1.0, random_state=seed).groupby('page').head(PER_PAGE_CAP)
            if len(class_rows) > cap:
                class_rows = class_rows.sample(n=cap, random_state=seed)
            else:
                leftover = pool[(pool['cls'] == cls) & ~pool['comp_id'].isin(set(class_rows['comp_id']))]
                class_rows = pd.concat([class_rows, leftover.sample(n=min(cap - len(class_rows), len(leftover)),
                                                                    random_state=seed)])
        picks.append(class_rows)
    picked = pd.concat(picks).sample(frac=1.0, random_state=seed)
    return picked.index.to_numpy(), picked['page'].to_numpy()


def class_distribution_table(dists, order=None):
    """creates a text table of class counts and shares, one column per distribution, plus a TOTAL row.
    Args:
        dists: {column title: {class: count}}.
        order: class order of the rows. Default: the first distribution's counts, largest first.
    Returns:
        the table as text.
    """
    columns = {}
    for title, counts in dists.items():
        total = max(int(sum(counts.values())), 1)
        columns[title] = {cls: f'{int(count)} ({100 * count / total:.1f}%)' for cls, count in counts.items()}
    classes = order or sorted(next(iter(dists.values())), key=lambda cls: -next(iter(dists.values()))[cls])
    table = pd.DataFrame({dist_title: [columns[dist_title].get(cls, '0 (0.0%)') for cls in classes]
                          for dist_title in dists}, index=classes)
    table.loc['TOTAL'] = [f'{int(sum(counts.values()))}' for counts in dists.values()]
    return table.to_string()


class ProbeDataset:
    """one object holds everything the probe needs to know about the data: the component table of each split,
      and the folder the screenshots were written to."""

    def __init__(self, dataset, root, splits=('seed', 'test')):
        """read the component table of each split (from the cache when present).
        Args:
            dataset: Hugging Face dataset id or local directory, as in load_split.
            root: work folder. Screenshots go to <root>/data/screenshots (data_dir, which embed.py
                reads), the cached tables to <root>/cache.
            splits: the splits to load.
        """
        self.data_dir = Path(root) / 'data'
        screenshots_dir = self.data_dir / 'screenshots'
        self.cache_dir = Path(root) / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.cache_dir / 'dataset.json'
        with FileLock(str(self.cache_dir / 'dataset.lock')):
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            self.source = dataset_source(dataset, splits)
            if manifest.exists():
                if json.loads(manifest.read_text()) != self.source:
                    raise SystemExit(f'dataset differs from the cache in {root}; use a fresh work directory')
            elif any(self.cache_dir.glob('components_*.parquet')) or any(screenshots_dir.iterdir()):
                raise SystemExit(f'cache in {root} has no dataset identity; use a fresh work directory')
            else:
                pending = manifest.with_suffix('.tmp')
                pending.write_text(json.dumps(self.source, indent=2) + '\n')
                pending.replace(manifest)
            self.tables = {
                split: load_components(dataset, split, screenshots_dir, self.cache_dir,
                                       revision=self.source.get('revision'))
                for split in splits
            }

    def __getitem__(self, split):
        """the component table of a split.
        Args:
            split: split name.
        Returns:
            DataFrame with comp_id, page, x, y, w, h and cls.
        """
        return self.tables[split]

    def pages(self, split):
        """which page each component is on.
        Args:
            split: split name.
        Returns:
            DataFrame with comp_id and page.
        """
        return self.tables[split][['comp_id', 'page']]

    def class_counts(self, split):
        """how many components each class has.
        Args:
            split: split name.
        Returns:
            {class: number of components}.
        """
        return self.tables[split]['cls'].value_counts().to_dict()
