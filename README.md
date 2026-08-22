<div align=center>
<h1> 🥓 Last But Not Least: Boundary Attention CalibratiON for Multimodal KV Cache Compression </h1>

<p>
Tianhao Chen<sup>1</sup>, Yuheng Wu<sup>1</sup>, Kelu Yao<sup>3</sup>, Xiaogang Xu<sup>4</sup>, <b>Xiaobin Hu</b><sup>2,&Dagger;</sup>, <b>Dongman Lee</b><sup>1,&Dagger;</sup>
</p>

<p>
<sup>1</sup>KAIST&nbsp;&nbsp;<sup>2</sup>National University of Singapore&nbsp;&nbsp;<sup>3</sup>Zhejiang Laboratory&nbsp;&nbsp;<sup>4</sup>The Chinese University of Hong Kong<br>
<sup>&Dagger;</sup>Corresponding authors
</p>

<a href="https://arxiv.org/abs/2606.14782"><img src="https://img.shields.io/badge/arXiv-2606.14782-b31b1b.svg" alt="arXiv"></a>
<a href="https://ryu1ion.github.io/official_BACON/"><img src="https://img.shields.io/badge/Project-Page-4a3a26.svg" alt="Project Page"></a>
<img src="https://img.shields.io/badge/EMNLP%202026-Main-blueviolet.svg" alt="EMNLP 2026 Main">
</div>

<p align="center">💡<i>  <strong>BACON</strong> — a training-free, plug-and-play attention-score calibration that recovers <strong>boundary-emergent evidence</strong> lost to observation-window aggregation, with <strong>no extra hyperparameter on the user side</strong>. </i></p>

## 🔥 News

* **`2026.08.22`** Code released.
* **`2026.08.21`** BACON is accepted to **EMNLP 2026 Main Conference**.

## 🧭 TL;DR

Multimodal LLMs need long visual contexts, which inflate the KV cache and
decoding latency. Existing compressors score tokens by **observation-window
attention**, but that aggregation can dilute sparse, answer-critical visual
evidence. **BACON** keeps the window score as a stable backbone and calibrates
it with **last-query attention**, filtered through **intra-layer coherence**
and **inter-layer persistence**. It is plug-and-play across compression methods
(SnapKV / PyramidKV / AdaKV / SparseMM) and adds **+7.5% average / +30.9% peak**
under the most aggressive cache budget, with no extra inference cost.

## 🏗 Repository layout

```
.
├── README.md
├── LICENSE
├── pyproject.toml
├── bacon/                  # core: backbones, MixKV, BACON calibration
│   ├── __init__.py
│   ├── bacon_utils.py      # `compute_bacon_score`, `_bacon_q_pooled`, KV clusters
│   ├── monkeypatch.py      # `replace_qwen` / `replace_qwen3vl` / `replace_mistral` / `replace_internvl`
│   ├── qwen_model.py
│   ├── qwen2_self.py
│   ├── qwen3_vl_model.py
│   └── mistral_model.py
├── csrc/                   # optional CUDA kernel for flattened head-wise cache
├── lmms-eval/              # evaluation harness (vendored, MixKV-compatible)
├── visual_head/            # head-score priors (qwen.json, llava-*.json, …)
├── scripts/
│   ├── eval/
│   │   ├── qwen.sh         # Qwen2-VL-7B-Instruct
│   │   ├── qwen3_vl.sh     # Qwen3-VL-30B-A3B-Instruct
│   │   ├── mistral.sh      # LLaVA-NeXT-Mistral-7B
│   │   └── internvl2.sh    # InternVL3-8B
│   └── others/
│       └── viz.sh
├── distribution_qwen.py
└── mistral_distribution.py
```

## 🛠 Installation

```bash
conda create -n bacon python=3.10 -y
conda activate bacon

# Required PyTorch / CUDA stack (matches MixKV).
pip install packaging torch==2.5.1
pip uninstall -y ninja && pip cache purge && pip install ninja --no-cache-dir

# Optional flattened-cache CUDA kernel. Speeds up AdaKV / SparseMM decoding.
# If your GPU's virtual architecture differs, edit the `-arch` flag inside
# csrc/build.py before running `make`.
cd csrc && make
cd ..

# Project + flash-attn.
pip install -e .
pip install flash-attn==2.4.1 --no-build-isolation
pip install qwen-vl-utils

# Evaluation harness.
cd lmms-eval && pip install -e . && cd ..
```

If you skip the `csrc` step, BACON / MixKV / the baselines still run; you only
lose the optimised flattened KV-cache update path used by AdaKV and SparseMM.

## 📦 Model and dataset preparation

Use Hugging Face caches; avoid hard-coded local paths.

```bash
export HF_HOME=/path/to/hf_cache
export TRANSFORMERS_CACHE=/path/to/hf_cache
export HF_DATASETS_CACHE=/path/to/hf_cache
```

Pre-trained backbones used in the paper:

| Backbone                  | HF model id                                          |
| ------------------------- | ---------------------------------------------------- |
| Qwen2-VL-7B-Instruct      | `Qwen/Qwen2-VL-7B-Instruct`                          |
| Qwen3-VL-30B-A3B-Instruct | `Qwen/Qwen3-VL-30B-A3B-Instruct`                     |
| LLaVA-NeXT-Mistral-7B     | `liuhaotian/llava-v1.6-mistral-7b`                   |
| InternVL3-8B              | `OpenGVLab/InternVL3-8B`                             |

Qwen3-VL-30B-A3B requires `transformers >= 4.57` with the `qwen3_vl_moe`
modeling module; the other backbones use the standard MixKV / EMNLP26 stack
(`transformers==4.46.2`).

Datasets are downloaded automatically by `lmms-eval` on first use (MMMU,
DocVQA, ChartQA, TextVQA, TextCaps).

## 🚀 Running evaluation

The evaluation entry points are four short shell scripts under
`scripts/eval/`. Each script exposes the same knobs through environment
variables:

| Variable    | Meaning                                                                     | Default                       |
| ----------- | --------------------------------------------------------------------------- | ----------------------------- |
| `METHOD`    | Backbone compressor: `snapkv` / `pyramidkv` / `adakv` / `sparsemm` / `mixsparsemm` | `snapkv`                  |
| `SELECT`    | Within-head score: `base` / `mixkv` / `bacon`                                | `bacon`                       |
| `BUDGET`    | Per-head retention budget (e.g. `64`, `128`, `256`)                          | `64`                          |
| `TASK`      | `lmms-eval` task name: `mmmu_val`, `docvqa`, `chartqa`, `textvqa`, `textcaps` | `mmmu_val`                   |
| `LIMIT`     | Optional `--limit N` for fast sanity checks                                  | unset                         |
| `OUTPUT_DIR`| Path for sample logs                                                         | `./logs`                      |
| `QWEN_MODEL`/`QWEN3_MODEL`/`MISTRAL_MODEL`/`INTERNVL_MODEL` | HF model id                                  | see scripts          |

Examples:

```bash
# Qwen2-VL × SnapKV × budget 64 × DocVQA, with BACON calibration
SELECT=bacon BUDGET=64 TASK=docvqa bash scripts/eval/qwen.sh

# Same backbone, baseline (no calibration), for direct comparison
SELECT=base  BUDGET=64 TASK=docvqa bash scripts/eval/qwen.sh

# MixKV head-wise blending on top of the same backbone
SELECT=mixkv BUDGET=64 TASK=docvqa bash scripts/eval/qwen.sh

# Qwen3-VL-30B-A3B-Instruct on MMMU at budget 128 with BACON
SELECT=bacon BUDGET=128 TASK=mmmu_val bash scripts/eval/qwen3_vl.sh

# LLaVA-NeXT-Mistral on ChartQA at budget 128 with BACON
SELECT=bacon BUDGET=128 TASK=chartqa bash scripts/eval/mistral.sh

# InternVL3 on TextVQA, AdaKV backbone, BACON
METHOD=adakv SELECT=bacon BUDGET=128 TASK=textvqa bash scripts/eval/internvl2.sh

# Quick smoke test (2 samples) of the BACON path
LIMIT=2 bash scripts/eval/qwen.sh
```

## 🧪 Reproducing paper results

The default values in `scripts/eval/*.sh` correspond to the BACON paper
configuration. To reproduce the main table, sweep `METHOD ∈ {snapkv, adakv,
sparsemm, mixsparsemm}`, `BUDGET ∈ {64, 128, 256}`, `TASK ∈ {mmmu_val,
docvqa, chartqa, textvqa, textcaps}`, and `SELECT ∈ {base, mixkv, bacon}`
through the same scripts. Numerical results are written under
`./eval_results/<model>_results/` and per-sample logs under `./logs/`.

## 📌 Citation

If our work is useful to your research, please consider citing:

```bibtex
@article{chen2026last,
  title={Last But Not Least: Boundary Attention CalibratiON for Multimodal KV Cache Compression},
  author={Chen, Tianhao and Wu, Yuheng and Yao, Kelu and Xu, Xiaogang and Hu, Xiaobin and Lee, Dongman},
  journal={arXiv preprint arXiv:2606.14782},
  year={2026}
}
```

## 👍 Acknowledgements

BACON is implemented on top of the open-source release of
[MixKV](https://github.com/xuyang-liu16/MixKV) (ICLR 2026), which itself
builds on [SparseMM](https://github.com/CR400AF-A/SparseMM) and the
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) harness. We thank
the authors of those projects for their excellent open-source contributions.
