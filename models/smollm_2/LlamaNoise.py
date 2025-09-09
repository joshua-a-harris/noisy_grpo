import os
import logging
from transformers.models.llama import modeling_llama as _orig_mod
from models.smollm_2.llama_noise_mod import NoiseLlamaDecoderLayer

from models.NoiseModelMixin import NoiseModelMixin

_orig_mod.LlamaDecoderLayer = NoiseLlamaDecoderLayer

from transformers import (
    LlamaForCausalLM, )

logger = logging.getLogger(__name__)


class LlamaNoise(LlamaForCausalLM, NoiseModelMixin):
    pass

