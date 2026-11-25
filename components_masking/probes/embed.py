"""Component features: run each encoder on full pages and average the patch features under each
component's box.

A page is decoded and preprocessed once, then passed through every loaded encoder, so scoring N
encoders costs N forward passes but a single decode. Pages are sharded round-robin across GPUs.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from screenshots import screenshot_index
from PIL import Image
from tqdm.auto import tqdm
from training.jepa.masking import box_to_indices, rect_to_patch_box
from training.jepa.model import JepaEncoder
from transformers import AutoProcessor

HERE = Path(__file__).parent

BASE_MODEL = 'google/siglip2-so400m-patch16-naflex'
MAX_PATCHES = 1024
IMAGE_PROCESSOR = 'Siglip2ImageProcessor'
ENCODERS_PER_PASS = 13      # ~0.85GB VRAM per resident fp16 encoder; 13 keeps a 46GB card at ~11GB


def log(msg):
    """print immediately (unbuffered), so progress shows up in piped and teed logs.
    """
    print(msg, flush=True)


def best_device():
    """best available device for single-device runs: cuda, then Apple mps, then cpu.
    Returns:
        'cuda', 'mps' or 'cpu'.
    """
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class PageDataset(torch.utils.data.Dataset):
    """one item per page: the screenshot run through the SigLIP2 processor. Images are opened lazily,
    one page at a time, inside the DataLoader workers.
    """

    def __init__(self, page_ids, screenshot_paths, processor):
        """Args:
            page_ids: pages to load, in the order the DataLoader yields them.
            screenshot_paths: {page id: screenshot path}, from screenshots.screenshot_index.
            processor: the SigLIP2 NaFlex processor (aspect-preserving, at most MAX_PATCHES patches).
        """
        self.page_ids, self.screenshot_paths, self.processor = page_ids, screenshot_paths, processor

    def __len__(self):
        return len(self.page_ids)

    def __getitem__(self, i):
        """one processed page.
        Returns:
            {'page_id': page id, 'size': (width, height) of the screenshot in pixels,
             'pixel_values': (MAX_PATCHES, 768) pixels of each 16x16 patch, padded,
             'pixel_attention_mask': (MAX_PATCHES,) 1 for a real patch, 0 for padding,
             'spatial_shapes': (2,) rows and columns of the page's patch grid}.
        """
        page_id = self.page_ids[i]
        image = Image.open(self.screenshot_paths[page_id]).convert('RGB')
        processed = self.processor(images=[image], return_tensors='pt')
        return {'page_id': page_id, 'size': image.size,
                'pixel_values': processed['pixel_values'][0],
                'pixel_attention_mask': processed['pixel_attention_mask'][0],
                'spatial_shapes': processed['spatial_shapes'][0]}


def collate_pages(batch):
    """stack PageDataset items into one batch.
    Args:
        batch: list of PageDataset items.
    Returns:
        the same keys: 'page_id' and 'size' as lists, the tensors stacked along a new first (page)
        dimension.
    """
    return {'page_id': [item['page_id'] for item in batch], 'size': [item['size'] for item in batch],
            'pixel_values': torch.stack([item['pixel_values'] for item in batch]),
            'pixel_attention_mask': torch.stack([item['pixel_attention_mask'] for item in batch]),
            'spatial_shapes': torch.stack([item['spatial_shapes'] for item in batch])}


def _load_encoders(encoders, device):
    """get every encoder ready to run.
    Args:
        encoders: {name: checkpoint or encoder}. A checkpoint (Hugging Face id or local dir) is
            loaded with JepaEncoder.from_pretrained, fp16 on CUDA and fp32 otherwise, moved to
            `device` and put in eval mode. An encoder already in memory (nn.Module, e.g. the model a
            training loop is training) is used as is: not copied and not moved, only switched to eval
            mode.
        device: where loaded checkpoints go, e.g. 'cuda:0' or 'cpu'.
    Returns:
        (loaded, train_mode_before). loaded: {name: encoder ready to run}. train_mode_before:
        {name: was it in training mode?} for the in-memory encoders only, so embed_components can
        switch them back afterwards. The release passes checkpoints only, so it stays empty.
    """
    loaded, train_mode_before = {}, {}
    for name, source in encoders.items():
        if isinstance(source, torch.nn.Module):
            train_mode_before[name] = source.training
            loaded[name] = source.eval()
        else:
            encoder = JepaEncoder.from_pretrained(
                str(source), torch_dtype=torch.float16 if str(device).startswith('cuda') else torch.float32)
            loaded[name] = encoder.to(device).eval()
    return loaded, train_mode_before


def _component_patches(batch, components_by_page):
    """for every component on all the batch's pages: which patches its box covers. It depends only
    on page geometry, so it is computed once per batch and shared by every loaded encoder.

    Args:
        batch: a collate_pages batch (uses 'page_id', 'size' and 'spatial_shapes').
        components_by_page: {page id: that page's rows of the component table}.

    Returns:
        one flat list for the whole batch, with one entry per component per all its pages. (pages
        in batch order, each page's components in table order). Each entry is (page_pos, n_page_patches, comp_patches, comp_id,
        cls):
            page_pos: position of the component's page in the batch (0 .. batch size - 1).
            n_page_patches: that page's number of real patches (grid_h * grid_w).
            comp_patches: indices of the patches the component's box overlaps, in that page's
                patch sequence.
            comp_id: the component id.
            cls: its class.
        Components whose box covers no patch are skipped.
    """
    batch_components = []
    for page_pos, page_id in enumerate(batch['page_id']):
        grid_h, grid_w = batch['spatial_shapes'][page_pos].tolist()
        for comp in components_by_page[page_id].itertuples():
            # (top row, left col, n rows, n cols) of the patches the box overlaps
            comp_patch_box = rect_to_patch_box((comp.x, comp.y, comp.w, comp.h), grid_h, grid_w,
                                               *batch['size'][page_pos])
            if comp_patch_box is None:
                continue
            # those patches' positions in the encoder output: row * grid_w + col
            comp_patches = box_to_indices(grid_h, grid_w, comp_patch_box)
            comp_patches = comp_patches[comp_patches < grid_h * grid_w]
            if len(comp_patches) == 0:
                continue
            batch_components.append((page_pos, grid_h * grid_w, comp_patches, comp.comp_id, comp.cls))
    return batch_components


@torch.inference_mode()
def embed_components(components, data_dir, encoders, max_patches, batch_size, workers, device,
                     show_bar=True, processor=None, screenshot_paths=None, amp_dtype=None):
    """embed the components of one component table with every encoder. Every encoder sees the same
    batches and the same _component_patches list, so each encoder's X is row-for-row aligned with
    the shared meta.

    Args:
        components: component table (comp_id, page, x, y, w, h, cls), e.g. ProbeDataset[split].
        data_dir: folder with screenshots/ (ProbeDataset.data_dir).
        encoders: {name: checkpoint or in-memory encoder}, see _load_encoders.
        max_patches: patch budget per page for the processor (MAX_PATCHES = 1024 in the paper).
        batch_size: pages per forward pass.
        workers: DataLoader worker processes (0 = load in this process).
        device: where to run, e.g. 'cuda:0', 'mps' or 'cpu'.
        show_bar: show a progress bar.
        processor: a ready SigLIP2 processor, to skip building one.
        screenshot_paths: a ready {page id: path} index, to skip rebuilding it.
        amp_dtype: if set (e.g. torch.bfloat16), run the forward pass under autocast in this dtype.

    Returns:
        {name: (X, meta)}. X: (n_components, hidden_size) float32, one row per component: the
        mean of the encoder's features over every patch the component's box overlaps, even
        partly, so the averaged area can reach slightly past the box. meta: DataFrame with comp_id,
        cls and n_patches (how many patches were averaged), one row per row of X. Pages without a
        screenshot and components covering no patch are left out.
    """
    if processor is None:
        # Pin the Pillow-backed processor, so preprocessing does not depend on whether torchvision is installed.
        processor = AutoProcessor.from_pretrained(BASE_MODEL, max_num_patches=max_patches, use_fast=False)
        actual_processor = type(processor.image_processor).__name__
        if actual_processor != IMAGE_PROCESSOR:
            raise RuntimeError(f'Reproduction requires {IMAGE_PROCESSOR}; got {actual_processor}.')
    loaded, train_mode_before = _load_encoders(encoders, device)
    screenshot_paths = screenshot_index(data_dir) if screenshot_paths is None else screenshot_paths
    components_by_page = {page_id: page_rows for page_id, page_rows in components.groupby('page')}
    page_ids = [page for page in components_by_page if page in screenshot_paths]

    loader = torch.utils.data.DataLoader(
        PageDataset(page_ids, screenshot_paths, processor), batch_size=batch_size, num_workers=workers,
        collate_fn=collate_pages, pin_memory=str(device).startswith('cuda'),
        prefetch_factor=4 if workers else None)

    meta_rows, features = [], {name: [] for name in loaded}
    progress = tqdm(total=len(page_ids), desc=f'embed[{device}] x{len(loaded)}', unit='page',
                    dynamic_ncols=True, mininterval=2.0, disable=not show_bar)
    try:
        for batch in loader:
            batch_components = _component_patches(batch, components_by_page)
            pixel_values = batch['pixel_values'].to(device, non_blocking=True)
            pixel_mask = batch['pixel_attention_mask'].to(device, non_blocking=True)
            grid_shapes = batch['spatial_shapes'].to(device, non_blocking=True)
            for name, encoder in loaded.items():
                if amp_dtype is not None:
                    with torch.autocast(str(device).split(':')[0], dtype=amp_dtype):
                        patch_features = encoder.forward_full(pixel_values, pixel_mask, grid_shapes)
                else:
                    patch_features = encoder.forward_full(pixel_values, pixel_mask, grid_shapes)
                patch_features = patch_features.float().cpu().numpy()
                # [:n_page_patches] drops the padding slots after the page's real patches
                features[name].extend(patch_features[page_pos][:n_page_patches][comp_patches].mean(0)
                                      for page_pos, n_page_patches, comp_patches, _, _ in batch_components)
            meta_rows.extend({'comp_id': comp_id, 'cls': cls, 'n_patches': len(comp_patches)}
                             for _, _, comp_patches, comp_id, cls in batch_components)
            progress.update(len(batch['page_id']))
            progress.set_postfix(components=len(meta_rows), refresh=False)
    finally:
        progress.close()
        for name, was_training in train_mode_before.items():
            if was_training:
                encoders[name].train()

    hidden_size = next(iter(loaded.values())).hidden_size
    meta = pd.DataFrame(meta_rows) if meta_rows else pd.DataFrame(columns=['comp_id', 'cls', 'n_patches'])
    return {name: (np.stack(vectors) if vectors else np.zeros((0, hidden_size), dtype=np.float32), meta)
            for name, vectors in features.items()}


def _shard_worker(rank, gpus, shard_dir, table_paths, encoder_items, data_dir, max_patches,
                  batch_size, workers):
    """one GPU process of the multi-GPU path. It takes every len(gpus)-th page of the sorted page
    list, starting at `rank`, embeds them with all encoders, and saves each (encoder, split) result
    to <shard_dir>/<name>__<split>__<rank>.npy/.parquet for _merge_shards.

    Args:
        rank: this process's index, 0 .. len(gpus) - 1 (passed by torch.multiprocessing.spawn).
        gpus: GPU ids, one entry per process; this process runs on cuda:gpus[rank]. An id may
            repeat: 0,0,0,0 runs four processes on GPU 0.
        shard_dir: temporary folder for the shard files.
        table_paths: {split: path of that split's component table, as parquet}.
        encoder_items: tuple of (name, checkpoint) pairs.
        data_dir: folder with screenshots/.
        max_patches: patch budget per page.
        batch_size: pages per forward pass.
        workers: DataLoader workers of this process.
    """
    encoders = dict(encoder_items)
    for split, table_path in table_paths.items():
        components = pd.read_parquet(table_path)
        page_ids = sorted(components['page'].unique())
        shard = components[components['page'].isin(set(page_ids[rank::len(gpus)]))]
        embedded = embed_components(shard, data_dir, encoders, max_patches, batch_size, workers,
                                    f'cuda:{gpus[rank]}', show_bar=(rank == 0))
        for name, (X, meta) in embedded.items():
            stem = f'{shard_dir}/{name}__{split}__{rank}'
            # Match the reported evaluation's FP16 shard storage before merging in FP32.
            np.save(stem + '.npy', X.astype(np.float16))
            meta.to_parquet(stem + '.parquet', index=False)


def _merge_shards(shard_dir, names, splits, n_shards):
    """the shard files back into one result, concatenated in rank order. The resulting row order
    depends on the number of shards, which is why runs must use the same GPU count.

    Args:
        shard_dir: the folder _shard_worker wrote to.
        names: encoder names.
        splits: split names.
        n_shards: number of processes (len(gpus)).

    Returns:
        {name: {split: (X, meta)}}, X as float32.
    """
    merged = {}
    for name in names:
        merged[name] = {}
        for split in splits:
            stems = [f'{shard_dir}/{name}__{split}__{rank}' for rank in range(n_shards)]
            shard_X = [np.load(stem + '.npy') for stem in stems]
            shard_meta = [pd.read_parquet(stem + '.parquet') for stem in stems]
            X = np.concatenate([part for part in shard_X if len(part)]).astype(np.float32)
            merged[name][split] = (X, pd.concat([part_meta for part_meta in shard_meta if len(part_meta)],
                                                ignore_index=True))
    return merged


def embed_many(components_by_split, data_dir, encoders, max_patches=MAX_PATCHES, batch_size=16,
               workers=8, gpus=None, encoders_per_pass=ENCODERS_PER_PASS, device=None, processor=None,
               screenshot_paths=None, amp_dtype=None, quiet=False):
    """embed several splits with several encoders.

    Page-sharded across GPUs with EVERY encoder of a pass resident on each GPU: a page is decoded
    once and broadcast to all of them. Sharding by encoder instead (fewer encoders per GPU, all
    pages each) would multiply the screenshot decoding, which is the actual bottleneck.

    Args:
        components_by_split: {split: component table}.
        data_dir: folder with screenshots/.
        encoders: {name: checkpoint or in-memory encoder}.
        max_patches: patch budget per page.
        batch_size: pages per forward pass.
        workers: total DataLoader workers, divided among the GPU processes.
        gpus: GPU ids. Two or more entries: one process per entry, pages sharded round-robin
            (_shard_worker); the reported runs used four. None, empty or one entry: one process.
        encoders_per_pass: how many encoders are loaded at once; the rest run in further passes over
            the pages. 0 or None loads all of them at once.
        device: device of the single-process path. Default: the first in-memory encoder's device,
            else cuda:gpus[0], else best_device().
        processor: a ready SigLIP2 processor (single-process path only).
        screenshot_paths: a ready {page id: path} index (single-process path only).
        amp_dtype: autocast dtype for the forward pass (single-process path only).
        quiet: no log lines and no progress bars.

    Returns:
        {name: {split: (X, meta)}}, see embed_components.
    """
    names = list(encoders)
    splits = list(components_by_split)
    in_memory = any(isinstance(source, torch.nn.Module) for source in encoders.values())
    per_pass = len(names) if (in_memory or not encoders_per_pass) else encoders_per_pass
    passes = [names[start:start + per_pass] for start in range(0, len(names), per_pass)]
    n_pages = {split_name: components_by_split[split_name].page.nunique() for split_name in splits}
    if not quiet:
        log(f'embed_many: {len(names)} encoders in {len(passes)} pass(es) of <={per_pass} | '
            f'splits {n_pages} | patches<={max_patches} batch={batch_size} workers={workers} '
            f'gpus={gpus if gpus else best_device()}')

    result = {}
    for pass_no, pass_names in enumerate(passes, 1):
        pass_encoders = {encoder_name: encoders[encoder_name] for encoder_name in pass_names}
        if not quiet:
            log(f'  pass {pass_no}/{len(passes)}: {len(pass_names)} encoder(s) resident')
        if in_memory or not gpus or len(gpus) < 2:
            run_device = device
            if run_device is None:
                first_module = next((source for source in pass_encoders.values()
                                     if isinstance(source, torch.nn.Module)), None)
                run_device = (next(first_module.parameters()).device if first_module is not None
                              else (f'cuda:{gpus[0]}' if gpus else best_device()))
            for split in splits:
                embedded = embed_components(components_by_split[split], data_dir, pass_encoders,
                                            max_patches, batch_size, workers, run_device,
                                            show_bar=not quiet, processor=processor,
                                            screenshot_paths=screenshot_paths, amp_dtype=amp_dtype)
                for name, (X, meta) in embedded.items():
                    result.setdefault(name, {})[split] = (X.astype(np.float32), meta)
        else:
            with tempfile.TemporaryDirectory() as shard_dir:
                table_paths = {}
                for split in splits:
                    table_path = str(Path(shard_dir) / f'components_{split}.parquet')
                    components_by_split[split].to_parquet(table_path, index=False)
                    table_paths[split] = table_path
                torch.multiprocessing.spawn(
                    _shard_worker, nprocs=len(gpus),
                    args=(gpus, shard_dir, table_paths, tuple(pass_encoders.items()), data_dir,
                          max_patches, batch_size, max(1, workers // len(gpus))), join=True)
                merged = _merge_shards(shard_dir, pass_names, splits, len(gpus))
            for name, per_split in merged.items():
                result.setdefault(name, {}).update(per_split)
        if not quiet:
            for name in pass_names:
                log(f'    {name}: {({split_name: result[name][split_name][0].shape for split_name in splits})}')
    return result
