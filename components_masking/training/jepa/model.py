"""The encoder: a SigLIP2 vision tower. Each released checkpoint is a plain Siglip2VisionModel, loaded
with JepaEncoder.from_pretrained.

forward_full encodes every patch of a page; the linear probe reads its features. forward_context and
compact are used in I-JEPA pretraining, where the encoder sees only the unmasked (context) patches."""

import torch
import torch.nn as nn
from transformers import Siglip2VisionModel
from transformers.masking_utils import create_bidirectional_mask


def compact(x, keep):
    """pack each sample's kept tokens to the front of a new, shorter tensor.
    Args:
        x: (batch, tokens, dim) tokens.
        keep: (batch, tokens) bool, the tokens to keep.
    Returns:
        (packed, mask): packed is (batch, most kept, dim), zero-padded after each sample's kept
        tokens; mask is (batch, most kept) bool, True for real tokens.
    """
    lengths = keep.sum(1)
    vmax = int(lengths.max())
    out = x.new_zeros(x.shape[0], vmax, x.shape[2])
    mask = torch.arange(vmax, device=x.device)[None] < lengths[:, None]
    out[mask] = x[keep]
    return out, mask


def enable_flash_attn():
    """let transformers use flash-attention-2 for SigLIP2, which it does not enable by default."""
    from transformers.models.siglip2.modeling_siglip2 import Siglip2PreTrainedModel
    Siglip2PreTrainedModel._supports_flash_attn = True


class JepaEncoder(nn.Module):
    """wraps a Siglip2VisionModel. The pooling head on top of the vision tower is frozen: the
    encoder is trained and read out through its patch features, never the pooled embedding."""

    def __init__(self, model):
        """Args:
            model: a transformers Siglip2VisionModel.
        """
        super().__init__()
        self.model = model
        for p in self.model.vision_model.head.parameters():
            p.requires_grad_(False)

    @property
    def vision(self):
        """the underlying vision transformer (embeddings, encoder, post-layernorm)."""
        return self.model.vision_model

    @classmethod
    def from_pretrained(cls, name_or_path, **kw):
        """load an encoder.
        Args:
            name_or_path: Hugging Face id or local checkpoint dir.
            **kw: passed to Siglip2VisionModel.from_pretrained (e.g. torch_dtype); with
                attn_implementation='flash_attention_2', flash attention is enabled first.
        Returns:
            a JepaEncoder.
        """
        if kw.get('attn_implementation') == 'flash_attention_2':
            enable_flash_attn()
        return cls(Siglip2VisionModel.from_pretrained(name_or_path, **kw))

    @property
    def hidden_size(self):
        """feature dimension per patch (1152 for so400m)."""
        return self.model.config.hidden_size

    def forward_full(self, pixel_values, pixel_attention_mask, spatial_shapes):
        """features for every patch of the full, unmasked page. This is what the probe reads.
        Args:
            pixel_values: (batch, patches, 768) patch pixels, from the SigLIP2 processor.
            pixel_attention_mask: (batch, patches), 1 for real patches, 0 for padding.
            spatial_shapes: (batch, 2) rows and columns of each page's patch grid.
        Returns:
            (batch, patches, hidden_size) features.
        """
        out = self.model(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask,
                         spatial_shapes=spatial_shapes)
        return out.last_hidden_state

    def forward_context(self, pixel_values, spatial_shapes, keep_mask):
        """pretraining only: encode just the context patches. The kept patches are packed together
        (compact) so masked ones cost no compute.
        Args:
            pixel_values: (batch, patches, 768) patch pixels.
            spatial_shapes: (batch, 2) rows and columns of each page's patch grid.
            keep_mask: (batch, patches) bool, True for the context patches.
        Returns:
            (features, mask): (batch, most kept, hidden_size) context features and the
            (batch, most kept) mask of real tokens.
        """
        embeds = self.vision.embeddings(pixel_values, spatial_shapes)
        ctx, ctx_mask = compact(embeds, keep_mask)
        attn_mask = create_bidirectional_mask(config=self.model.config, inputs_embeds=ctx, attention_mask=ctx_mask)
        hidden = self.vision.encoder(inputs_embeds=ctx, attention_mask=attn_mask).last_hidden_state
        return self.vision.post_layernorm(hidden), ctx_mask

    def save_pretrained(self, path):
        """save as a plain Siglip2VisionModel checkpoint (the format of the released weights).
        Args:
            path: output folder.
        """
        self.model.save_pretrained(path)
