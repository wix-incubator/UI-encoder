"""Patch-grid geometry and the target-mask samplers used in the pretraining.
grid: (h, w): the page's patches: h rows, w columns
patch index: one number per patch: row × w + column. With w = 4, the patch at row 1, column 2 is 1×4+2 = 6
rect (x, y, w, h): a component's box in page units (pages are 1280 units wide)
box (top, left, rows, cols): the same kind of box, measured in patches

Geometry: rect_to_patch_box (with axis_span) maps a component's bounding box onto the page's patch
grid, and box_to_indices lists the patches that box covers. The linear probe uses the same two
functions to pool each component's patch features.

Mask samplers:
- sample_random_masks: target blocks placed without looking at the page. The paper's random arm
  draws 4 blocks of 15-20% of the page, random+ draws 16 blocks of 2-8%.
- sample_pay_masks (ours): components are visited smallest first, and each draws a share of its
  patches to hide (pay_rate); a contiguous slab inside the component brings it to that share,
  and the slab leaves visible context where it can. The paper's ours arm draws the shares from
  (0.5, 0.7); nothing outside components is masked.

Coordinates: pages are rendered 1280 units wide, and component boxes are (x, y, w, h) in those
units. Patch boxes are (top row, left column, rows, columns) on the page's (h, w) patch grid.
"""

import math

import numpy as np

PAGE_WIDTH = 1280.0


def sample_block(h, w, scale_range, aspect_range, rng):
    """sample random rectangle of patches inside the grid.
    Args:
        h: grid rows.
        w: grid columns.
        scale_range: (lo, hi) range of the rectangle's area, as a share of the grid.
        aspect_range: (lo, hi) range of its width / height.
        rng: numpy random Generator.
    Returns:
        (top, left, rows, columns) box, at least one patch on each side and inside the grid.
    """
    scale = rng.uniform(*scale_range)
    aspect = rng.uniform(*aspect_range)
    area = scale * h * w
    bh = min(h, max(1, round(math.sqrt(area / aspect))))
    bw = min(w, max(1, round(area / bh)))
    top = rng.integers(0, h - bh + 1)
    left = rng.integers(0, w - bw + 1)
    return top, left, bh, bw


def box_to_indices(h, w, box):
    """the patches a box covers, row by row (patch index = row * w + column).
    Args:
        h: grid rows.
        w: grid columns.
        box: (top, left, rows, columns).
    Returns:
        (rows * columns,) patch indices.
    """
    top, left, bh, bw = box
    rows = np.arange(top, top + bh)
    cols = np.arange(left, left + bw)
    return (rows[:, None] * w + cols[None, :]).ravel()


def ctx_candidates(h, w):
    """the patches the context may come from: every patch of the page. Both samplers remove the
    target patches from this list, and what is left is the context the encoder sees.
    Args:
        h: grid rows.
        w: grid columns.
    Returns:
        (h * w,) patch indices, 0 to h * w - 1.
    """
    return np.arange(h * w)


def axis_span(lo, size, n, page_size):
    """a helper of rect_to_patch_box, which finds the patches a component covers. The component's box is given in page units, and the
    patches form a grid, so rect_to_patch_box asks this function twice: once along x, for the
    patch columns the component covers, and once along y, for the patch rows.
    Along one axis, the component runs from lo to lo + size. This returns the patch cells (columns
    or rows) it touches. A cell counts even if the component covers only part of it, so no part of
    the component is left out.
    Args:
        lo: where the component starts on this axis, in page units.
        size: the component's width (or height), in page units.
        n: number of patch columns (or rows).
        page_size: the page's width (or height), in page units.
    Returns:
        (first cell, number of cells), or None if the component is outside the page.
    """
    cell = page_size / n
    first = int(np.floor(lo / cell))
    last = int(np.ceil((lo + size) / cell)) - 1
    first, last = max(0, first), min(n - 1, last)
    if last >= first:
        return first, last - first + 1
    c = lo + size / 2                       # degenerate: zero length or outside the page
    if not 0 <= c < page_size:
        return None
    return min(int(c / cell), n - 1), 1


def rect_to_patch_box(rect, h, w, img_w, img_h):
    """a component's bounding box on the page to its box on the patch grid.
    Args:
        rect: (x, y, width, height) in page units (the page is PAGE_WIDTH units wide).
        h: grid rows.
        w: grid columns.
        img_w: screenshot width in pixels.
        img_h: screenshot height in pixels; with img_w it gives the page's height in page units.
    Returns:
        (top, left, rows, columns), or None if the component is off the page.
    """
    x, y, rw, rh = rect
    page_h = img_h * PAGE_WIDTH / img_w
    row = axis_span(y, rh, h, page_h)
    col = axis_span(x, rw, w, PAGE_WIDTH)
    if row is None or col is None:
        return None
    return row[0], col[0], row[1], col[1]


def build_patch_class_map(h, w, insts, img_w, img_h, cls_to_idx):
    """the component class of every patch, for the class-balanced loss of the -bal encoders (it
    does not affect which patches are masked). Where components overlap, the smallest one owns the
    patch: boxes are painted largest first, by patch count and then by area on the page, so smaller
    ones paint over them.
    Args:
        h: grid rows.
        w: grid columns.
        insts: [(class name, (x, y, width, height))], the page's components in page units.
        img_w: screenshot width in pixels.
        img_h: screenshot height in pixels.
        cls_to_idx: class name to class index. Classes not in it (e.g. section, treated as
            background) are not painted.
    Returns:
        (h * w,) int16 class index of each patch, -1 for background.
    """
    cmap = np.full(h * w, -1, dtype=np.int16)
    boxes = []
    for cls, rect in insts:
        if cls not in cls_to_idx:
            continue
        box = rect_to_patch_box(rect, h, w, img_w, img_h)
        if box is not None:
            boxes.append((box[2] * box[3], rect[2] * rect[3], cls_to_idx[cls], box))
    boxes.sort(key=lambda b: (-b[0], -b[1]))
    for _, _, ci, box in boxes:
        cmap[box_to_indices(h, w, box)] = ci
    return cmap


def topup_slab(u, unm, w, k, rng):
    """the solid piece of a component that gets masked: k of its allowed patches, in whole lines
    across its longer side (columns of a wide component, rows of a tall one). The slab starts at a
    random line and grows one line at a time, to a random side, until it holds at least k allowed
    patches; the last line is taken only partly, so exactly k are returned. Patches that are not
    allowed are skipped, so the slab grows around them.
    Args:
        u: (n,) patch indices of the component's box.
        unm: (n,) bool, True where the patch may be masked.
        w: grid columns.
        k: number of patches to mask.
        rng: numpy random Generator.
    Returns:
        patch indices of the slab (k of them, or every allowed patch if there are fewer).
    """
    rows, cols = u // w, u % w
    r0, c0 = rows.min(), cols.min()
    bh, bw = rows.max() - r0 + 1, cols.max() - c0 + 1
    axis, cross = (rows - r0, cols - c0) if bh >= bw else (cols - c0, rows - r0)
    n_axis = max(bh, bw)
    line_new = np.bincount(axis[unm], minlength=n_axis)
    lo = hi = int(rng.integers(0, n_axis))
    new = int(line_new[lo])
    while new < k and (lo > 0 or hi < n_axis - 1):
        grow_hi = hi < n_axis - 1 and (lo == 0 or rng.uniform() < 0.5)
        if grow_hi:
            hi += 1
            new += int(line_new[hi])
        else:
            lo -= 1
            new += int(line_new[lo])
    sel = unm & (axis >= lo) & (axis <= hi)
    order = np.lexsort((cross[sel], axis[sel]))
    return u[sel][order][:k]


def sample_random_masks(h, w, rng, n_targets, rand_block_scale, target_aspect=(0.75, 1.5)):
    """the masks of the random and random+ encoders: n_targets random blocks, placed without
    looking at the page; the context is the rest of the page. The paper's random arm uses 4 blocks
    of 15-20% of the grid, random+ 16 blocks of 2-8%. sample_pay_masks also falls back to one
    small random block on a page where no component can be masked.
    Args:
        h: grid rows.
        w: grid columns.
        rng: numpy random Generator.
        n_targets: number of target blocks.
        rand_block_scale: (lo, hi) area of each block, as a share of the grid.
        target_aspect: (lo, hi) width / height of each block.
    Returns:
        (ctx, blocks, info): context patch indices (never empty); one patch-index array per target
        block (blocks may overlap); {'n_comp': 0, 'rects': []}.
    """
    tgt_idx = [box_to_indices(h, w, sample_block(h, w, rand_block_scale, target_aspect, rng))
               for _ in range(n_targets)]
    tgt_union = np.unique(np.concatenate(tgt_idx))
    ctx = np.setdiff1d(ctx_candidates(h, w), tgt_union)
    if len(ctx) == 0:
        ctx = np.array([0], dtype=np.int64)
    return ctx, tgt_idx, {'n_comp': 0, 'rects': []}


def pay_rate(rng, pband):
    """the share of a component that sample_pay_masks masks, drawn uniformly from pband for every
    component on every page. The paper's ours uses pband=(0.5, 0.7).
    Args:
        rng: numpy random Generator.
        pband: (lo, hi) range of the share.
    Returns:
        the share, between lo and hi.
    """
    return rng.uniform(*pband)


def sample_pay_masks(h, w, rng, insts, pband, n_targets=16, rand_block_scale=(0.02, 0.08),
                     target_aspect=(0.75, 1.5)):
    """the masks of the ours encoders. Components are visited smallest first. Each masks a share of
    itself (pay_rate, from pband) as one solid slab (topup_slab), counting the patches smaller
    components inside it have already masked. Its slab never takes the last patch it could still
    mask, so it leaves at least one of those visible; smaller overlapping components can still
    have masked all of a component's patches before its turn. The patches a component leaves
    visible are then protected from later component slabs. A component may contribute no slab,
    for example when it occupies only one patch.
    Each slab is one target block inside a component. If no slabs are placed, the sampler falls
    back to one random target block on the page.
    The first lines draw n_targets random blocks that are never used: in the research code they set
    a masking budget for variants explored during development. The draw is kept because it moves
    the random generator, so removing it would change every mask the encoders were trained with.
    Args:
        h: grid rows.
        w: grid columns.
        rng: numpy random Generator.
        insts: [(class name, patch indices of its box)], the page's components.
        pband: (lo, hi) range of the share each component masks; (0.5, 0.7) in the paper.
        n_targets: number of blocks in the unused random draw (16 in the paper).
        rand_block_scale: (lo, hi) area of each of those blocks, as a share of the grid.
        target_aspect: (lo, hi) width / height of each of those blocks.
    Returns:
        (ctx, blocks, info): context patch indices (never empty); one patch-index array per target
        block; {'n_comp': number of component slabs, 'rects': []}.
    """
    grid = h * w
    budget_boxes = [box_to_indices(h, w, sample_block(h, w, rand_block_scale, target_aspect, rng))
                    for _ in range(n_targets)]
    masked = np.zeros(grid, dtype=bool)
    blocks = []
    n_comp = 0
    order = sorted(insts, key=lambda g: len(g[1]))
    # the patches each component already placed left visible: no later slab may mask them
    protected = np.zeros(grid, dtype=bool)
    for j, (cls, u) in enumerate(order):
        frac = pay_rate(rng, pband)
        k = frac * len(u) - int(masked[u].sum())
        if k <= 0:
            protected[u[~masked[u]]] = True
            continue
        k = int(np.floor(k) + (rng.uniform() < k - np.floor(k)))
        unm = ~masked[u]
        unm = unm & ~protected[u]
        k = min(k, int(unm.sum()) - 1)
        if k <= 0:
            protected[u[~masked[u]]] = True
            continue
        take = topup_slab(u, unm, w, k, rng)
        masked[take] = True
        blocks.append(take)
        n_comp += 1
        protected[u[~masked[u]]] = True
    if not blocks:
        return sample_random_masks(h, w, rng, 1, (0.02, 0.08))
    tgt_union = np.flatnonzero(masked)
    ctx = np.setdiff1d(ctx_candidates(h, w), tgt_union)
    if len(ctx) == 0:
        ctx = np.array([0], dtype=np.int64)
    return ctx, blocks, {'n_comp': n_comp, 'rects': []}
