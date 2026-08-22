"""lmms-eval wrapper for Qwen3-VL-Moe (e.g. Qwen3-VL-30B-A3B-Instruct).

Adapted from `lmms_eval.models.qwen2_vl`. The KV-cache compression hook
selection mirrors the Qwen2-VL wrapper: at __init__ time we read the METHOD
env var and call `replace_qwen3vl(...)` from `bacon.monkeypatch`.
"""

import gc
import os
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from transformers import Qwen3VLMoeForConditionalGeneration  # type: ignore
except ImportError:  # transformers<4.57
    Qwen3VLMoeForConditionalGeneration = None  # filled in lazily below

from bacon.monkeypatch import replace_qwen3vl
from bacon.bacon_utils import clear_bacon_trace


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


@register_model("qwen3_vl")
class Qwen3_VL(lmms):
    """Qwen3-VL-Moe model wrapper.

    model_args (passed via --model_args key=value,...):
      pretrained          : HF model id (default Qwen/Qwen3-VL-30B-A3B-Instruct)
      device, device_map  : device placement (default auto / cuda)
      batch_size          : eval batch size (default 1)
      use_cache           : KV cache (default True)
      attn_implementation : flash_attention_2 (default), sdpa, or eager
      dtype               : bfloat16 (default), float16, float32
      max_pixels          : processor option (default 16384*28*28)
      min_pixels          : processor option (default 32*28*28)
      max_num_frames      : processor option for videos (unused for image tasks)
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-30B-A3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: str = "flash_attention_2",
        dtype: str = "bfloat16",
        max_pixels: int = 16384 * 28 * 28,
        min_pixels: int = 32 * 28 * 28,
        max_num_frames: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # ---- MixKV / BACON method dispatch via env vars. ----
        method = os.getenv("METHOD", None)
        if method == "adakv":
            replace_qwen3vl("adakv")
        elif method == "pyramidkv":
            replace_qwen3vl("pyramidkv")
        elif method == "snapkv":
            replace_qwen3vl("snapkv")
        elif method == "sparsemm":
            replace_qwen3vl("sparsemm")
        elif method == "mixsparsemm":
            replace_qwen3vl("mixsparsemm")
        elif method in ("mask", "mask_random"):
            replace_qwen3vl("mask")
        else:
            eval_logger.info("Qwen3-VL: METHOD env var unset/unknown — using Full KV")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        torch_dtype = _DTYPES.get(str(dtype).lower(), torch.bfloat16)

        # Lazy-resolve the model class if the top-level import failed (older transformers).
        global Qwen3VLMoeForConditionalGeneration
        if Qwen3VLMoeForConditionalGeneration is None:
            try:
                from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (  # type: ignore
                    Qwen3VLMoeForConditionalGeneration as _Cls,
                )
                Qwen3VLMoeForConditionalGeneration = _Cls
            except ImportError as exc:
                raise ImportError(
                    "Qwen3VLMoeForConditionalGeneration not available. "
                    "Install transformers>=4.57 in the qwen3vl30b_kv_eval env."
                ) from exc

        eval_logger.info(
            f"Loading {pretrained} (dtype={torch_dtype}, attn_implementation={attn_implementation})"
        )
        self._model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype=torch_dtype,
            device_map=self.device_map,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
        ).eval()

        # Processor: Qwen3-VL uses the same min_pixels/max_pixels API as Qwen2-VL.
        try:
            self.processor = AutoProcessor.from_pretrained(
                pretrained, max_pixels=max_pixels, min_pixels=min_pixels
            )
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(pretrained)

        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.empty_cache_every = int(os.getenv("MIXKV_EMPTY_CACHE_EVERY", "1"))

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    # ----- Properties -----
    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen3_VL")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def _release_after_generate(self, force_empty_cache: bool = True) -> None:
        clear_bacon_trace(self.model)
        gc.collect()
        if force_empty_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_message(self, context: str, visual):
        content = []
        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):
            content.append({"type": "video", "video": visual})
        elif isinstance(visual, Image.Image):
            content.append({"type": "image", "image": visual.convert("RGB")})
        elif isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
            content.extend({"type": "image", "image": v.convert("RGB")} for v in visual)
        content.append({"type": "text", "text": context})
        return [{"role": "user", "content": content}]

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk_idx, chunk in enumerate(chunks):
            clear_bacon_trace(self.model)
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)

            gen_kwargs = all_gen_kwargs[0]
            until = [self.tokenizer.decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(
                        f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}"
                    )

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            messages = [
                self._build_message(
                    context,
                    visuals[i] if i < len(visuals) else None,
                )
                for i, context in enumerate(contexts)
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                images_kwargs={"max_pixels": self.max_pixels, "min_pixels": self.min_pixels},
                num_frames=self.max_num_frames,
                padding=True, return_tensors="pt",
                return_dict=True,
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 128
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            pad_token_id = self.tokenizer.pad_token_id
            input_lengths = [int(in_ids.shape[0]) for in_ids in inputs.input_ids]
            with torch.inference_mode():
                cont = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )

            generated_ids_trimmed = [
                out_ids[input_len:].detach().cpu()
                for input_len, out_ids in zip(input_lengths, cont)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans.strip()

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

            del (
                answers,
                cont,
                generated_ids_trimmed,
                input_lengths,
                inputs,
                messages,
                visuals,
            )
            self._release_after_generate(
                self.empty_cache_every > 0 and (chunk_idx + 1) % self.empty_cache_every == 0
            )

        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
