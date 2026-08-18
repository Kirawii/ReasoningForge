import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _attention_implementation(device: str) -> str:
    """Allow an explicit local attention override.

    AutoDL/local deployment adaptation: the local launcher explicitly sets
    CS336_ATTN_IMPLEMENTATION=sdpa. With no override, preserve the official
    checkpoint behavior exactly: eager on CPU and flash_attention_2 on GPU.
    """
    override = os.environ.get("CS336_ATTN_IMPLEMENTATION")
    if override:
        return override
    return "eager" if device == "cpu" else "flash_attention_2"


def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation=_attention_implementation(device),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer
