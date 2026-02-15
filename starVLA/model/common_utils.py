from typing import Optional

import torch.nn as nn


def get_input_embedding_vocab_size(model: nn.Module) -> int:
    """Return input embedding vocab size (ZeRO-3 safe, no all-gather)."""
    embed = model.get_input_embeddings()
    weight = embed.weight
    ds_shape = getattr(weight, "ds_shape", None)
    if ds_shape is not None and len(ds_shape) > 0:
        return int(ds_shape[0])
    return int(weight.shape[0])


def resolve_llm_hidden_size(model: nn.Module) -> int:
    """Resolve text hidden size across Qwen-like config variants."""
    vl_config = getattr(model, "config", None)
    hidden_size: Optional[int] = getattr(vl_config, "hidden_size", None) if vl_config is not None else None
    if hidden_size is None and vl_config is not None:
        text_config = getattr(vl_config, "text_config", None)
        hidden_size = getattr(text_config, "hidden_size", None) if text_config is not None else None
    if hidden_size is None:
        model_core = getattr(model, "model", None)
        embed_tokens = getattr(model_core, "embed_tokens", None)
        if embed_tokens is not None and hasattr(embed_tokens, "weight"):
            hidden_size = int(embed_tokens.weight.shape[1])
    if hidden_size is None:
        raise AttributeError("Cannot resolve llm hidden_size from model config/model")
    return int(hidden_size)
