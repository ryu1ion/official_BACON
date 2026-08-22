"""Qwen3-VL-Moe flash-attn forward adapters for SnapKV / PyramidKV / AdaKV /
SparseMM / MixSparseMM / Mask. Mirrors `mixkv/qwen_model.py` (Qwen2-VL flavor)
but binds to `transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe`.

The cluster classes in `mixkv/mixkv_utils.py` are reused unchanged — they
operate on attention-layer KV tensors and are agnostic to whether the FFN
is dense (Qwen2-VL) or MoE (Qwen3-VL-A3B).

Imports are defensive: Qwen3-VL inherits MROPE from Qwen2-VL, but if HF
relocates the helpers under qwen3_vl_moe, we prefer the local copy.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache, DynamicCache, StaticCache
try:
    from transformers.masking_utils import create_causal_mask
except Exception:  # noqa: BLE001
    create_causal_mask = None
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

# ---- Helper imports (defensive: prefer qwen3_vl_moe, fall back to qwen2_vl). ----
try:
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        apply_rotary_pos_emb,
        repeat_kv,
    )
except Exception:  # noqa: BLE001
    apply_rotary_pos_emb = None
    from transformers.models.qwen2_vl.modeling_qwen2_vl import repeat_kv

try:
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
except Exception:  # noqa: BLE001
    from transformers.models.qwen2_vl.modeling_qwen2_vl import _flash_attention_forward

from bacon.bacon_utils import (
    init_snapkv,
    init_pyramidkv,
    init_adakv,
    init_sparsemm,
    init_mixsparsemm,
    init_mask,
    DynamicCacheSplitHeadFlatten,
)

try:
    from flash_attn import flash_attn_varlen_func
except Exception:  # noqa: BLE001
    flash_attn_varlen_func = None

logger = logging.get_logger(__name__)


def _mrope_section(self) -> List[int]:
    """Resolve the MROPE section list. Both Qwen2-VL and Qwen3-VL-Moe store
    it on the attention layer's config under `rope_scaling['mrope_section']`."""
    rope_scaling = getattr(self.config, "rope_scaling", None) or {}
    section = rope_scaling.get("mrope_section")
    if section is None:
        raise RuntimeError(
            "rope_scaling['mrope_section'] missing from config — Qwen3-VL "
            "compression hooks require multimodal RoPE."
        )
    return section


def _num_attention_heads(self) -> int:
    return getattr(self, "num_heads", self.config.num_attention_heads)


def _num_key_value_heads(self) -> int:
    return getattr(self, "num_key_value_heads", self.config.num_key_value_heads)


def _hidden_size(self) -> int:
    return getattr(self, "hidden_size", self.config.hidden_size)


def _attention_output_size(self) -> int:
    return getattr(self.o_proj, "in_features", _num_attention_heads(self) * self.head_dim)


def _past_key_values(past_key_value: Optional[Cache], kwargs) -> Optional[Cache]:
    return past_key_value if past_key_value is not None else kwargs.get("past_key_values")


def _project_qkv(self, hidden_states: torch.Tensor):
    bsz, q_len, _ = hidden_states.size()
    num_heads = _num_attention_heads(self)
    num_key_value_heads = _num_key_value_heads(self)

    query_states = self.q_proj(hidden_states).view(bsz, q_len, num_heads, self.head_dim)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, num_key_value_heads, self.head_dim)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, num_key_value_heads, self.head_dim)

    if hasattr(self, "q_norm"):
        query_states = self.q_norm(query_states)
    if hasattr(self, "k_norm"):
        key_states = self.k_norm(key_states)

    return (
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
    )


def _apply_qwen3_rope(self, query_states, key_states, value_states, position_ids, position_embeddings):
    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    if apply_rotary_pos_emb is not None:
        return (*apply_rotary_pos_emb(query_states, key_states, cos, sin), cos, sin)

    from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_multimodal_rotary_pos_emb

    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, _mrope_section(self)
    )
    return query_states, key_states, cos, sin


def qwen3vl_flash_attn2_forward_SnapKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    init_snapkv(self)
    past_key_value = _past_key_values(past_key_value, kwargs)
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states, key_states, value_states = _project_qkv(self, hidden_states)

    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"{self.__class__.__name__} requires a layer_idx for KV caching."
            )

    query_states, key_states, cos, sin = _apply_qwen3_rope(
        self, query_states, key_states, value_states, position_ids, position_embeddings
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if q_len == 1:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        else:
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states
            )
            past_key_value.update(
                key_states_compress, value_states_compress, self.layer_idx, cache_kwargs
            )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    sliding_window = None
    if (
        getattr(self.config, "use_sliding_window", False)
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= getattr(self.config, "max_window_layers", 10**9)
    ):
        sliding_window = self.config.sliding_window

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=getattr(self, "_flash_attn_uses_top_left_mask", False),
    )

    attn_output = attn_output.reshape(bsz, q_len, _attention_output_size(self)).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def qwen3vl_flash_attn2_forward_PyramidKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    init_pyramidkv(self)
    past_key_value = _past_key_values(past_key_value, kwargs)
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states, key_states, value_states = _project_qkv(self, hidden_states)

    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(f"{self.__class__.__name__} requires a layer_idx for KV caching.")

    query_states, key_states, cos, sin = _apply_qwen3_rope(
        self, query_states, key_states, value_states, position_ids, position_embeddings
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        if q_len == 1:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        else:
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states
            )
            past_key_value.update(
                key_states_compress, value_states_compress, self.layer_idx, cache_kwargs
            )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    sliding_window = None
    if (
        getattr(self.config, "use_sliding_window", False)
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= getattr(self.config, "max_window_layers", 10**9)
    ):
        sliding_window = self.config.sliding_window

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=getattr(self, "_flash_attn_uses_top_left_mask", False),
    )

    attn_output = attn_output.reshape(bsz, q_len, _attention_output_size(self)).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def _adakv_like_forward(self, hidden_states, attention_mask, position_ids, past_key_value,
                        cache_position, position_embeddings, init_fn, kwargs=None):
    """Shared body for AdaKV / SparseMM / MixSparseMM forwards (head-flat KV cache)."""
    init_fn(self)
    past_key_value = _past_key_values(past_key_value, kwargs or {})
    bsz, q_len, _ = hidden_states.size()

    query_states, key_states, value_states = _project_qkv(self, hidden_states)

    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(f"{self.__class__.__name__} requires a layer_idx for KV caching.")

    query_states, key_states, cos, sin = _apply_qwen3_rope(
        self, query_states, key_states, value_states, position_ids, position_embeddings
    )

    is_prefill = q_len != 1
    cache_kwargs = None
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

    if is_prefill:
        key_states_compress, value_states_compress = self.kv_cluster.update_kv(
            key_states, query_states, value_states
        )
        past_key_value.update(
            key_states_compress, value_states_compress, self.layer_idx, cache_kwargs
        )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        sliding_window = None
        if (
            getattr(self.config, "use_sliding_window", False)
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= getattr(self.config, "max_window_layers", 10**9)
        ):
            sliding_window = self.config.sliding_window

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=getattr(self, "_flash_attn_uses_top_left_mask", False),
        )

        attn_output = attn_output.reshape(bsz, q_len, _attention_output_size(self)).contiguous()
    else:
        if flash_attn_varlen_func is None:
            raise RuntimeError(
                "flash_attn_varlen_func is not available; AdaKV/SparseMM decoding requires flash-attn."
            )
        cache_kwargs["head_lens"] = self.kv_cluster.head_lens
        cache_kwargs["cu_klen"] = self.kv_cluster.cu_klen
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

        self.kv_cluster.klen_sum += _num_key_value_heads(self)
        self.kv_cluster.max_seqlen_k += 1
        self.kv_cluster.cu_klen += self.kv_cluster.cu_offset
        self.kv_cluster.head_lens += 1

        query_states = query_states.view(-1, self.num_key_value_groups, self.head_dim)
        key_states = key_states.view(-1, 1, self.head_dim)
        value_states = value_states.view(-1, 1, self.head_dim)

        cu_seqlens_q = self.kv_cluster.cu_qlen
        cu_seqlens_k = self.kv_cluster.cu_klen
        max_seqlen_q = 1
        max_seqlen_k = self.kv_cluster.max_seqlen_k
        target_dtype = query_states.dtype
        if target_dtype == torch.float32:
            target_dtype = self.q_proj.weight.dtype
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)
        attn_output = flash_attn_varlen_func(
            query_states, key_states, value_states,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True,
        )
        assert bsz == 1
        attn_output = attn_output.reshape(bsz, _num_attention_heads(self), q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(
            bsz, q_len, _attention_output_size(self)
        ).contiguous()

    attn_output = self.o_proj(attn_output)
    return attn_output, None


def qwen3vl_flash_attn2_forward_AdaKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    return _adakv_like_forward(
        self, hidden_states, attention_mask, position_ids, past_key_value,
        cache_position, position_embeddings, init_adakv, kwargs,
    )


def qwen3vl_flash_attn2_forward_SparseMM(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    return _adakv_like_forward(
        self, hidden_states, attention_mask, position_ids, past_key_value,
        cache_position, position_embeddings, init_sparsemm, kwargs,
    )


def qwen3vl_flash_attn2_forward_MixSparseMM(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    return _adakv_like_forward(
        self, hidden_states, attention_mask, position_ids, past_key_value,
        cache_position, position_embeddings, init_mixsparsemm, kwargs,
    )


def qwen3vl_flash_attn2_forward_Mask(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    init_mask(self)
    past_key_value = _past_key_values(past_key_value, kwargs)
    bsz, q_len, _ = hidden_states.size()

    query_states, key_states, value_states = _project_qkv(self, hidden_states)

    if past_key_value is not None and self.layer_idx is None:
        raise ValueError(f"{self.__class__.__name__} requires a layer_idx for KV caching.")

    query_states, key_states, cos, sin = _apply_qwen3_rope(
        self, query_states, key_states, value_states, position_ids, position_embeddings
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    for h in getattr(self, "head_list", []):
        if self.layer_idx == h[0]:
            query_states[:, h[1], :, :] = 0

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    sliding_window = None
    if (
        getattr(self.config, "use_sliding_window", False)
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= getattr(self.config, "max_window_layers", 10**9)
    ):
        sliding_window = self.config.sliding_window

    attn_output = _flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        q_len,
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=getattr(self, "_flash_attn_uses_top_left_mask", False),
    )

    attn_output = attn_output.reshape(bsz, q_len, _attention_output_size(self)).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def prepare_inputs_for_generation_qwen3vl(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    **kwargs,
):
    """Qwen3-VL-Moe generation input prep with compression state reset.

    Keep Hugging Face's Qwen3-VL behavior: position_ids are intentionally left
    for the model forward to prepare with its own rope_deltas state.
    """
    if not isinstance(past_key_values, tuple):
        if hasattr(past_key_values, "key_cache") and len(past_key_values.key_cache) == 0:
            text_model = getattr(self, "model", None)
            language_model = getattr(text_model, "language_model", None) or text_model
            layers = getattr(language_model, "layers", None) or getattr(self.model, "layers", [])
            for layer in layers:
                if hasattr(layer, "self_attn"):
                    layer.self_attn.kv_seq_len = 0

    model_inputs = super(self.__class__, self).prepare_inputs_for_generation(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        position_ids=position_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        use_cache=use_cache,
        **kwargs,
    )

    model_inputs["position_ids"] = None

    if cache_position is not None and cache_position[0] != 0:
        model_inputs["pixel_values"] = None
        model_inputs["pixel_values_videos"] = None

    return model_inputs


def adakv_qwen3vl_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    visual_pos_masks: Optional[torch.Tensor] = None,
    deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    """Model-level forward used by AdaKV / SparseMM / MixSparseMM on Qwen3-VL-Moe.
    Replaces the default `Qwen3VLMoeModel.forward` so the KV cache becomes
    `DynamicCacheSplitHeadFlatten` (per-head flat layout).

    This mirrors the Hugging Face Qwen3VLMoeTextModel forward rather than the
    Qwen2-VL forward: Qwen3-VL injects DeepStack visual features into the first
    decoder layers and its decoder layers return tensors, not layer-output
    tuples.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if getattr(self, "gradient_checkpointing", False) and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    return_legacy_cache = False
    if (
        use_cache
        and not isinstance(past_key_values, DynamicCacheSplitHeadFlatten)
        and not self.training
    ):
        past_key_values = DynamicCacheSplitHeadFlatten.from_legacy_cache(past_key_values)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    if create_causal_mask is None:
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
    else:
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if getattr(self, "gradient_checkpointing", False) and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                position_embeddings,
                causal_mask,
                text_position_ids,
                past_key_values,
                cache_position,
                **kwargs,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = layer_outputs

        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

        if output_attentions:
            all_self_attns += (None,)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = past_key_values if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
