"""The I-JEPA predictor, packed: every target block of a page is predicted in a single pass.

The sequence is [context ; queries of block 1 ; ... ; queries of block k], and an attention mask
keeps the blocks independent: the context attends only to itself, and each block's queries attend
to the context and to their own block. Since the context never attends to a query, it is encoded
once for all blocks rather than once per block, and the blocks need no padding to a common length.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sincos_1d(pos, dim):
    """sine-cosine embedding of one coordinate.
    Args:
        pos: (...) positions.
        dim: embedding size (even).
    Returns:
        (..., dim) embeddings.
    """
    omega = torch.arange(dim // 2, dtype=torch.float32, device=pos.device) / (dim // 2)
    omega = 1.0 / (10000 ** omega)
    ang = pos[..., None].float() * omega
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


def sincos_2d(coords, dim):
    """sine-cosine embedding of a (row, column) grid position: half the size for each axis.
    Args:
        coords: (..., 2) row and column of each patch.
        dim: embedding size (divisible by 4).
    Returns:
        (..., dim) embeddings.
    """
    half = dim // 2
    return torch.cat([sincos_1d(coords[..., 0], half), sincos_1d(coords[..., 1], half)], dim=-1)


class Block(nn.Module):
    """a pre-norm transformer block whose attention takes a full (query, key) mask."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        """Args:
            dim: token width.
            heads: attention heads.
            mlp_ratio: MLP hidden width as a multiple of dim.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))
        self.heads = heads

    def forward(self, x, attn_mask):
        """Args:
            x: (batch, tokens, dim) tokens.
            attn_mask: (batch, 1, tokens, tokens) bool, True where a query row may attend to a key.
        Returns:
            (batch, tokens, dim) tokens.
        """
        # attn_mask [b, 1, s, s]: which keys each query row may attend to (not just key padding)
        b, s, d = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).reshape(b, s, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        h = h.transpose(1, 2).reshape(b, s, d)
        x = x + self.proj(h)
        x = x + self.mlp(self.norm2(x))
        return x


class JepaPredictor(nn.Module):
    """a narrow ViT that predicts the encoder features of the target patches from the context. The
    paper's runs use dim=384, depth=6, heads=6."""

    def __init__(self, encoder_dim, dim=384, depth=6, heads=6):
        """Args:
            encoder_dim: encoder feature size (1152 for SigLIP2 so400m).
            dim: predictor width.
            depth: number of transformer blocks.
            heads: attention heads per block.
        """
        super().__init__()
        self.proj_in = nn.Linear(encoder_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(dim))
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, encoder_dim)
        self.dim = dim
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, ctx, ctx_mask, ctx_coords, tgt_coords, tgt_valid, tgt_seg, ctx_sees_q=False):
        """predict every target block of each page in one pass.
        Each target patch enters as the mask token plus its position. The attention mask, for a
        context C and two blocks q_1, q_2 (V = may attend):
                    C     q_1   q_2
            C    [  V      .     .  ]
            q_1  [  V      V     .  ]
            q_2  [  V      .     V  ]

        Args:
            ctx: (batch, n_ctx, encoder_dim) context features from the encoder (padded).
            ctx_mask: (batch, n_ctx) bool, True for real context tokens.
            ctx_coords: (batch, n_ctx, 2) grid (row, column) of each context token.
            tgt_coords: (batch, n_tgt, 2) grid (row, column) of each target patch; all blocks of a
                page are concatenated in one row.
            tgt_valid: (batch, n_tgt) bool, True for real target patches.
            tgt_seg: (batch, n_tgt) block number of each target patch, starting at 1.
            ctx_sees_q: if True the context also attends to every query. With a single block this
                is exactly the standard I-JEPA predictor; with several blocks it couples them
                through the context.
        Returns:
            (batch, n_tgt, encoder_dim) predicted features of the target patches.
        """
        nc = ctx.shape[1]
        x = torch.cat([self.proj_in(ctx) + sincos_2d(ctx_coords, self.dim).to(ctx.dtype),
                       self.mask_token + sincos_2d(tgt_coords, self.dim).to(ctx.dtype)], dim=1)
        seg = torch.cat([torch.zeros_like(ctx_mask, dtype=tgt_seg.dtype),
                         tgt_seg * tgt_valid], dim=1)                 # 0 = ctx and padded queries
        valid = torch.cat([ctx_mask, tgt_valid], dim=1)
        allow = (seg[:, None, :] == 0) | (seg[:, :, None] == seg[:, None, :])
        if ctx_sees_q:
            allow = allow | (seg[:, :, None] == 0)
        allow = allow & valid[:, None, :]                            # padded tokens are never keys
        attn_mask = allow[:, None]
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return self.proj_out(self.norm(x[:, nc:]))
