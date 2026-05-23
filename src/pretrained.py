"""Adapter that loads pretrained DeiT-tiny weights into our IterativeTransformerBlock.

Theorem 3.1 is stated for a Post-Norm + RMSNorm block. DeiT uses Pre-Norm + LayerNorm,
so we cannot apply the theorem to a raw DeiT layer. Instead we extract only the trained
*linear* maps (W_Q, W_K, W_V, W_O for MSA; W_1, W_2 for FFN) from one DeiT layer and
load them into our IterativeTransformerBlock, whose normalization is the RMSNorm
analyzed in the paper. This keeps the architecture exactly the one the theorem
covers, while the MSA/FFN computations are driven by real trained weights.
"""

import torch
import torch.nn as nn

from src.model import IterativeTransformerBlock


def _copy_msa_weights(target_msa: nn.MultiheadAttention, deit_attention) -> None:
    """Copy DeiT's separate Q/K/V/O linear maps into nn.MultiheadAttention's packed buffers."""
    Wq = deit_attention.attention.query.weight.detach()
    Wk = deit_attention.attention.key.weight.detach()
    Wv = deit_attention.attention.value.weight.detach()
    bq = deit_attention.attention.query.bias.detach()
    bk = deit_attention.attention.key.bias.detach()
    bv = deit_attention.attention.value.bias.detach()
    Wo = deit_attention.output.dense.weight.detach()
    bo = deit_attention.output.dense.bias.detach()

    with torch.no_grad():
        target_msa.in_proj_weight.copy_(torch.cat([Wq, Wk, Wv], dim=0))
        target_msa.in_proj_bias.copy_(torch.cat([bq, bk, bv], dim=0))
        target_msa.out_proj.weight.copy_(Wo)
        target_msa.out_proj.bias.copy_(bo)


def _copy_ffn_weights(target_ffn: nn.Sequential, deit_layer) -> None:
    with torch.no_grad():
        target_ffn[0].weight.copy_(deit_layer.intermediate.dense.weight.detach())
        target_ffn[0].bias.copy_(deit_layer.intermediate.dense.bias.detach())
        target_ffn[2].weight.copy_(deit_layer.output.dense.weight.detach())
        target_ffn[2].bias.copy_(deit_layer.output.dense.bias.detach())


def build_block_from_deit(layer_index: int = 6, model_name: str = "facebook/deit-tiny-patch16-224"):
    """Return (IterativeTransformerBlock, ViTModel) with weights copied from `layer_index`.

    The returned block has:
      * MSA initialized from the chosen DeiT layer (Q/K/V/O packed for nn.MultiheadAttention),
      * FFN initialized from the same DeiT layer (intermediate / output dense),
      * RMSNorm γ = 1 (default), so γ_max = 1 in the bound.
    """
    from transformers import ViTModel

    deit = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
    deit.eval()
    cfg = deit.config

    block = IterativeTransformerBlock(
        dim=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        ffn_hidden_dim=cfg.intermediate_size,
    )
    block.eval()

    _copy_msa_weights(block.msa, deit.encoder.layer[layer_index].attention)
    _copy_ffn_weights(block.ffn, deit.encoder.layer[layer_index])
    return block, deit


@torch.no_grad()
def patch_embeddings_from_image(deit, image_tensor):
    """Run the DeiT patch+pos embeddings on a (B, 3, 224, 224) image, return (B, 197, 192)."""
    emb = deit.embeddings(image_tensor)
    return emb


def cifar10_to_deit_input(cifar_image_chw):
    """CIFAR-10 image (3, 32, 32) in [-1, 1] -> DeiT input (1, 3, 224, 224) properly normalized.

    DeiT-tiny was trained with ImageNet mean/std on 224x224 inputs.
    """
    import torch.nn.functional as F

    img = cifar_image_chw.unsqueeze(0)
    img = (img + 1.0) * 0.5
    img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (img - mean) / std
