import random
from transformers.trainer_pt_utils import get_parameter_names
import numpy as np
from torch import nn
import torch
from pathlib import Path
import yaml


def load_config(config_path: str) -> dict:
    """
    Loads YAML configuration from the given file path
    and returns it as a Python dictionary.
    """
    config_file = Path(config_path).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed: int = 42):
    # Python's built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU)
    torch.manual_seed(seed)

    # PyTorch (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def expand_inputs_for_generation(
        expand_size = 1,
        input_ids = None,
        **model_kwargs,
):
    """Adapted from HF - Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
    # Do not call torch.repeat_interleave if expand_size is 1 because it clones
    # the input tensor and thus requires more memory although no change is applied
    if expand_size == 1:
        return input_ids, model_kwargs

    def _expand_dict_for_generation(dict_to_expand):
        for key in dict_to_expand:
            if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
            ):
                dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
        return dict_to_expand

    if input_ids is not None:
        input_ids = input_ids.repeat_interleave(expand_size, dim=0)

    model_kwargs = _expand_dict_for_generation(model_kwargs)
    return input_ids, model_kwargs


def get_decay_parameter_names(model):
    """
    *** MODIFIED ORIGINAL FROM HuggingFace - https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
    Get all parameter names that weight decay will be applied to.

    This function filters out parameters in two ways:
    1. By layer type (instances of layers specified in ALL_LAYERNORM_LAYERS)
    2. By parameter name patterns (containing 'bias', or variation of 'norm')
    """
    forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
    decay_parameters = get_parameter_names(model, [nn.LayerNorm], forbidden_name_patterns)
    return decay_parameters

