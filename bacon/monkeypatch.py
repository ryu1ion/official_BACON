from importlib.metadata import version
import transformers

from bacon.mistral_model import mistral_flash_attn2_forward_AdaKV, mistral_flash_attn2_forward_MixSparseMM,  mistral_flash_attn2_forward_PyramidKV, mistral_flash_attn2_forward_SnapKV, \
                                   mistral_flash_attn2_forward_SparseMM, mistral_flash_attn2_forward_Mask
from bacon.mistral_model import prepare_inputs_for_generation_mistral_new, adaptive_MistralModel_forward

from bacon.qwen2_self import flash_attn_forward_adakv, flash_attn_forward_snapkv, qwen2_forward_adakv,flash_attn_forward_pyramidkv
from bacon.qwen_model import qwen_flash_attn2_forward_AdaKV, qwen_flash_attn2_forward_MixSparseMM, qwen_flash_attn2_forward_PyramidKV, qwen_flash_attn2_forward_SnapKV, \
                                qwen_flash_attn2_forward_SparseMM, qwen_flash_attn2_forward_Mask
from bacon.qwen_model import prepare_inputs_for_generation_qwen, adakv_qwen_forward

# Qwen3-VL-Moe forwards (lazy: only the function refs; the transformers module
# is touched inside replace_qwen3vl so importing this file on older transformers
# does not error).
from bacon.qwen3_vl_model import (
    qwen3vl_flash_attn2_forward_SnapKV,
    qwen3vl_flash_attn2_forward_PyramidKV,
    qwen3vl_flash_attn2_forward_AdaKV,
    qwen3vl_flash_attn2_forward_SparseMM,
    qwen3vl_flash_attn2_forward_MixSparseMM,
    qwen3vl_flash_attn2_forward_Mask,
    prepare_inputs_for_generation_qwen3vl,
    adakv_qwen3vl_forward,
)





def replace_mistral(method):

    if method == "pyramidkv":
        print("Using PyramidKV!")
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_PyramidKV

    elif method == "snapkv":
        print("Using SnapKV!")
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_SnapKV

    elif method == "adakv":
        print("Using AdaKV!")
        transformers.models.mistral.modeling_mistral.MistralModel.forward  = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_AdaKV

    elif method == "sparsemm":
        print("Using SparseMM!")
        transformers.models.mistral.modeling_mistral.MistralModel.forward  = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_SparseMM
    elif method == "mixsparsemm":
        print("Using MixSparseMM!")
        transformers.models.mistral.modeling_mistral.MistralModel.forward  = adaptive_MistralModel_forward
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_MixSparseMM

    elif method == 'mask':
        print("Mask Head")
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_Mask

    if method not in ["fullkv"]:
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral_new



def replace_qwen(method):
    if method == 'snapkv':
        print("Using SnapKV!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_SnapKV

    elif method == 'pyramidkv':
        print("Using PyramidKV!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_PyramidKV
    
    if method == "adakv":
        print("Using AdaKV!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.forward = adakv_qwen_forward
        
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_AdaKV

    elif method == "sparsemm":
        print("Using SparseMM!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.forward = adakv_qwen_forward
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_SparseMM

    elif method == 'mask':
        print("Mask Head")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_Mask
    
    elif method == "mixsparsemm":
        print("Using MixSparseMM!")
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel.forward = adakv_qwen_forward
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLFlashAttention2.forward = qwen_flash_attn2_forward_MixSparseMM
    if method not in ["fullkv"]:
        transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen

def _resolve_qwen3vl_module():
    """Return the qwen3_vl_moe modeling module, or raise with a clear message."""
    try:
        import transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe as mod
        return mod
    except ImportError as exc:
        raise ImportError(
            "transformers.models.qwen3_vl_moe is not available. "
            "Qwen3-VL-30B-A3B-Instruct requires transformers>=4.57. "
            "Activate the qwen3vl30b_kv_eval conda env created by "
            "scripts/setup_qwen3vl30b_eval_env.sh."
        ) from exc


def _resolve_qwen3vl_attn_class(mod):
    """Find the FlashAttention2 attention class. HuggingFace has renamed Qwen3-VL
    attention modules across releases; try the most likely names in order."""
    candidates = (
        "Qwen3VLMoeTextFlashAttention2",
        "Qwen3VLMoeFlashAttention2",
        "Qwen3VLMoeTextAttention",
        "Qwen3VLMoeAttention",
    )
    for name in candidates:
        cls = getattr(mod, name, None)
        if cls is not None:
            return name, cls
    raise AttributeError(
        f"None of {candidates!r} were found on {mod.__name__}. "
        "Check the installed transformers version's qwen3_vl_moe submodule."
    )


def _resolve_qwen3vl_model_class(mod):
    """Find the text-level Model class (the one whose forward we override for AdaKV)."""
    candidates = (
        "Qwen3VLMoeTextModel",
        "Qwen3VLMoeModel",
    )
    for name in candidates:
        cls = getattr(mod, name, None)
        if cls is not None:
            return name, cls
    raise AttributeError(
        f"None of {candidates!r} were found on {mod.__name__}."
    )


def _resolve_qwen3vl_clm_class(mod):
    """Find the ConditionalGeneration class (top-level model)."""
    candidates = (
        "Qwen3VLMoeForConditionalGeneration",
    )
    for name in candidates:
        cls = getattr(mod, name, None)
        if cls is not None:
            return name, cls
    raise AttributeError(
        f"None of {candidates!r} were found on {mod.__name__}."
    )


def replace_qwen3vl(method):
    """Monkey-patch Qwen3-VL-Moe attention / model / prepare_inputs.

    Mirrors `replace_qwen(method)` for Qwen2-VL but binds to the qwen3_vl_moe
    module. Called from `lmms_eval.models.qwen3_vl.Qwen3_VL.__init__` based on
    the METHOD env var.
    """
    mod = _resolve_qwen3vl_module()
    attn_name, attn_cls = _resolve_qwen3vl_attn_class(mod)
    model_name, model_cls = _resolve_qwen3vl_model_class(mod)
    clm_name, clm_cls = _resolve_qwen3vl_clm_class(mod)

    if method == "snapkv":
        print(f"Using SnapKV on {attn_name}!")
        attn_cls.forward = qwen3vl_flash_attn2_forward_SnapKV

    elif method == "pyramidkv":
        print(f"Using PyramidKV on {attn_name}!")
        attn_cls.forward = qwen3vl_flash_attn2_forward_PyramidKV

    elif method == "adakv":
        print(f"Using AdaKV on {attn_name}, {model_name}!")
        model_cls.forward = adakv_qwen3vl_forward
        attn_cls.forward = qwen3vl_flash_attn2_forward_AdaKV

    elif method == "sparsemm":
        print(f"Using SparseMM on {attn_name}, {model_name}!")
        model_cls.forward = adakv_qwen3vl_forward
        attn_cls.forward = qwen3vl_flash_attn2_forward_SparseMM

    elif method == "mixsparsemm":
        print(f"Using MixSparseMM on {attn_name}, {model_name}!")
        model_cls.forward = adakv_qwen3vl_forward
        attn_cls.forward = qwen3vl_flash_attn2_forward_MixSparseMM

    elif method == "mask":
        print(f"Mask Head on {attn_name}")
        attn_cls.forward = qwen3vl_flash_attn2_forward_Mask

    if method not in ["fullkv"]:
        clm_cls.prepare_inputs_for_generation = prepare_inputs_for_generation_qwen3vl


def replace_internvl(method):
    if method=="adakv":
        print("Using Adakv")
        transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward=qwen2_forward_adakv
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward=flash_attn_forward_adakv
    elif method=="snapkv":
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward=flash_attn_forward_snapkv
        print("Using Snapkv")
    elif method=="pyramidkv":
        transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward=flash_attn_forward_pyramidkv
        print("Using pyramidkv")


