import torch
import torch.nn as nn
from typing import Optional, Tuple

# --- Assumed imports from Hugging Face transformers library ---
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2MLP, Qwen2RMSNorm, Cache

class NoiseQwen2DecoderLayer(nn.Module):
    """
    A modified Qwen2DecoderLayer with noise injection.
    """

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        # --- Original Qwen2 Attributes ---
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

        # --- Noise Attributes ---
        self.layer_idx = layer_idx
        self.add_noise = False
        self.noise_scaling_factor = 0.1

        self.register_buffer('avg_hidden_state_magnitude', torch.tensor(0.0))
        self.current_noise = None

    def set_noise_mode(self, add_noise: bool, noise_scaling_factor: Optional[float] = None):
        self.add_noise = add_noise
        if noise_scaling_factor is not None:
            self.noise_scaling_factor = noise_scaling_factor

    def _update_rolling_avg_magnitude(self, hidden_states: torch.Tensor):
        current_magnitude = hidden_states.detach().abs().mean()
        if self.avg_hidden_state_magnitude == 0:
            self.avg_hidden_state_magnitude = current_magnitude
        else:
            momentum = 0.99
            self.avg_hidden_state_magnitude = (momentum * self.avg_hidden_state_magnitude) + (
                        (1 - momentum) * current_magnitude)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Update hidden state magnitude
        self._update_rolling_avg_magnitude(hidden_states)
        # Check if autoregressive generation
        is_generating = use_cache and cache_position is not None and hidden_states.shape[1] == 1
        if is_generating:
            # Scale noise external noise relative to hidden state magnitude and add
            if self.add_noise:
                std = self.noise_scaling_factor * self.avg_hidden_state_magnitude
                noise = self.current_noise * std
                hidden_states = hidden_states + noise.type(hidden_states.dtype)  # Apply noise for the forward pass
                self.current_noise = noise
        else:
            # Just add noise set externally for parallel forward
            if self.add_noise:
                hidden_states = hidden_states + self.current_noise.type(hidden_states.dtype)
        # BELOW FROM ORIGINAL TRANSFORMERS
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

