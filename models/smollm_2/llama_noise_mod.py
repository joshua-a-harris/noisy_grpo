from typing import Optional

import torch
from transformers.models.llama.modeling_llama import GradientCheckpointingLayer, LlamaAttention, LlamaRMSNorm, LlamaMLP, Cache
from transformers.models.llama.configuration_llama import LlamaConfig

class NoiseLlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> tuple[torch.Tensor]:
        self._update_rolling_avg_magnitude(hidden_states)
        # NOTE - seems to be a bug in hf modeling_llama LlamaModel where use_cache is not passed down
        is_generating = hidden_states.shape[1] == 1
        if is_generating:
            if self.add_noise:
                std = self.noise_scaling_factor * self.avg_hidden_state_magnitude
                noise = self.current_noise * std
                hidden_states = hidden_states + noise.type(hidden_states.dtype)  # Apply noise for the forward pass
                self.current_noise = noise
        else:
            if self.add_noise:
                hidden_states = hidden_states + self.current_noise.type(hidden_states.dtype)
        # BELOW ORIGINAL FROM TRANSFORMERS
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