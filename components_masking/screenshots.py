"""Screenshot lookup for probes/embed.py.

Trimmed from the original module, which also read the internal crawl output. The release reads the
Hugging Face dataset instead (probes/probe_dataset.py), so only screenshot_index is kept (renamed
for clarity, code unchanged): embed.py imports it to find each page's screenshot under <data_dir>/screenshots/.
"""

from pathlib import Path


def screenshot_index(data_dir):
    """finds the screenshot file of every page, so embed.py can open a page's image by its page id.
    ProbeDataset writes the screenshots (load_components), embed.py calls this once and then reads
    the image of each page it embeds from the returned paths.

    When each case is used:
    - the release (always): every screenshot is in one folder, <data_dir>/screenshots, written by
      ProbeDataset for both splits. The index is built from that folder.
    - the original internal crawl (never in the release): it stored screenshots in one folder per
      crawl shard, <data_dir>/shard_*/screenshots. When such folders exist, they are read instead,
      in sorted order, and if a page appears in more than one shard the first copy is kept.

    Example: a screenshot at /work/data/screenshots/seed-00000.webp is indexed as 'seed-00000'.

    Args:
        data_dir: folder that holds screenshots/*.webp (ProbeDataset.data_dir, i.e. <root>/data).

    Returns:
        {page id: Path of its .webp}; the page id is the file name without .webp.
    """
    screenshot_paths = {}
    root = Path(data_dir)
    for screenshots_dir in sorted(root.glob('shard_*/screenshots')) or [root / 'screenshots']:
        for file in Path(screenshots_dir).glob('*.webp'):
            screenshot_paths.setdefault(file.stem, file)
    return screenshot_paths
