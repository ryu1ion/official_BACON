#!/usr/bin/env bash
# Evaluate LLaVA-NeXT-Mistral-7B on a multimodal benchmark with a chosen
# KV-compression backbone (snapkv / pyramidkv / adakv / sparsemm) and a
# chosen within-head score (`base`, `mixkv`, or `bacon`).
#
# Override any knob via env vars, e.g.
#   BUDGET=128 TASK=docvqa SELECT=bacon bash scripts/eval/mistral.sh
set -euo pipefail

METHOD="${METHOD:-snapkv}"
SELECT="${SELECT:-bacon}"
BUDGET="${BUDGET:-64}"
TASK="${TASK:-mmmu_val}"
RATIO="${RATIO:-0.1}"
MASK_RATIO="${MASK_RATIO:-0.1}"
MISTRAL_MODEL="${MISTRAL_MODEL:-liuhaotian/llava-v1.6-mistral-7b}"
OUTPUT_DIR="${OUTPUT_DIR:-./logs}"
RESULT_DIR="${RESULT_DIR:-./eval_results/mistral_results}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC="${NPROC:-2}"
LIMIT="${LIMIT:-}"

case "$SELECT" in
    base)  select_method="attn" ;;
    mixkv) select_method="headwisemixkv" ;;
    bacon) select_method="bacon" ;;
    *)
        echo "ERROR: unknown SELECT='$SELECT' (expected base | mixkv | bacon)" >&2
        exit 2 ;;
esac

if [[ "$SELECT" == "bacon" ]]; then
    export BACON_MODE="${BACON_MODE:-elt}"
    export BACON_LOCAL_K="${BACON_LOCAL_K:-5}"
    export BACON_TRACE_R="${BACON_TRACE_R:-4}"
    export BACON_LAMBDA_C="${BACON_LAMBDA_C:-0.5}"
    export BACON_TOPK_FRAC="${BACON_TOPK_FRAC:-2.0}"
fi

export METHOD BUDGET RATIO MASK_RATIO
export SELECT_METHOD="$select_method"
export CUDA_VISIBLE_DEVICES

mkdir -p "$RESULT_DIR" "$OUTPUT_DIR"

extra_args=()
if [[ -n "$LIMIT" ]]; then extra_args+=(--limit "$LIMIT"); fi

log_file="${RESULT_DIR}/${TASK}_${METHOD}_${BUDGET}_${SELECT}.log"
echo "==> Mistral  task=${TASK}  backbone=${METHOD}  select=${SELECT}  budget=${BUDGET}"

python3 -m accelerate.commands.launch \
    --num_processes="$NPROC" \
    --main_process_port "$((54300 + RANDOM % 200))" \
    -m lmms_eval \
        --model llava \
        --model_args "pretrained=${MISTRAL_MODEL},conv_template=mistral_instruct" \
        --tasks "$TASK" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix llava_v1.6_mistral \
        --output_path "$OUTPUT_DIR" \
        --gen_kwargs temperature=0 \
        --verbosity=INFO \
        "${extra_args[@]}" \
    2>&1 | tee "$log_file"
