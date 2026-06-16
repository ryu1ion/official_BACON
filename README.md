# BACON

**Last But Not Least: Boundary Attention Calibration for Multimodal KV Cache Compression**

> Tianhao Chen, Yuheng Wu, Kelu Yao, Xiaogang Xu, Xiaobin Hu, Dongman Lee

[![arXiv](https://img.shields.io/badge/arXiv-2606.14782-b31b1b.svg)](https://arxiv.org/abs/2606.14782)
[![Project Page](https://img.shields.io/badge/Project-Page-4a3a26.svg)](https://ryu1ion.github.io/official_BACON/)

## TL;DR

Multimodal LLMs need long visual contexts, which inflate the KV cache and
decoding latency. Existing compressors score tokens by **observation-window
attention**, but that aggregation can dilute sparse, answer-critical visual
evidence. **BACON** keeps the window score as a stable backbone and calibrates
it with **last-query attention**, filtered through **intra-layer coherence**
and **inter-layer persistence**. It is plug-and-play across compression methods
(SnapKV / PyramidKV / AdaKV / SparseMM) and adds **+7.5% average / +30.9% peak**
under the most aggressive cache budget, with no extra inference cost.

## Code Release

The code is **not yet public**. We will release the full implementation,
training-free integration scripts for the supported MLLMs (LLaVA-NeXT,
InternVL3, Qwen2-VL, Qwen3-VL-30B-A3B), and the evaluation pipelines **after the review process is complete**.

Watch / star the repo to be notified when the code lands.

## Citation

```bibtex
@article{chen2026bacon,
  title  = {Last But Not Least: Boundary Attention Calibration for Multimodal KV Cache Compression},
  author = {Chen, Tianhao and Wu, Yuheng and Yao, Kelu and Xu, Xiaogang and Hu, Xiaobin and Lee, Dongman},
  year   = {2026},
  eprint = {2606.14782},
  archivePrefix = {arXiv}
}
```

## Contact

For questions about the paper, please reach out to the corresponding authors:
**Xiaobin Hu** and **Dongman Lee**.
