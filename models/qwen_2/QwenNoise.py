import os
import logging
from transformers.models.qwen2 import modeling_qwen2 as _orig_mod
from models.qwen_2.qwen_noise_mod import NoiseQwen2DecoderLayer

from models.NoiseModelMixin import NoiseModelMixin

_orig_mod.Qwen2DecoderLayer = NoiseQwen2DecoderLayer

from transformers import (
    Qwen2ForCausalLM, )

logger = logging.getLogger(__name__)


class QwenNoise(Qwen2ForCausalLM, NoiseModelMixin):
    pass

