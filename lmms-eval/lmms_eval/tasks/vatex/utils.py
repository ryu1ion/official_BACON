"""Utility hooks for the VATEX video-captioning task in lmms-eval.

The HF dataset ``lmms-lab/vatex_from_url`` only ships YouTube URLs, so videos
must be pre-downloaded into ``VATEX_VIDEO_ROOT`` as ``<videoID>.mp4`` (other
common suffixes are also tried). The ``videoID`` follows the public VATEX
naming convention ``<youtube_id>_<start_ms>_<end_ms>``.

Metric: the primary path computes CIDEr / BLEU-4 / METEOR / ROUGE_L via
``pycocoevalcap`` (same path as ``coco_cap``). When pycocoevalcap is not
installed, a normalized token-F1 + exact-match fallback is used and tagged as
``fallback_token_f1`` in the metric value so it cannot be confused with real
CIDEr.
"""

import collections
import hashlib
import json
import os
import re
import string
from pathlib import Path

import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.video_loader import get_cache_dir, get_video

VATEX_METRICS = ("CIDEr", "Bleu_4", "METEOR", "ROUGE_L")

with open(Path(__file__).parent / "_default_template_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = [line for line in raw_data if "!function" not in line]
    _config = yaml.safe_load("".join(safe_data))


def _resolve_video_root() -> str:
    """Return the directory where VATEX videos are expected to live.

    ``VATEX_VIDEO_ROOT`` (env) takes priority; otherwise we fall back to
    ``${HF_HOME}/vatex/Videos`` (mirroring NextQA's ``cache_dir`` layout).
    If neither is set we return a sentinel path so that import-time work
    succeeds; the actual lookup will fail loudly with FileNotFoundError
    at first call.
    """
    env_root = os.environ.get("VATEX_VIDEO_ROOT", "").strip()
    if env_root:
        return env_root
    try:
        return get_cache_dir(_config, "Videos")
    except KeyError:  # HF_HOME unset
        return os.path.expanduser("~/.cache/huggingface/vatex/Videos")


def _video_root() -> str:
    # Re-resolve on every call so env tweaks between import and use take effect.
    return _resolve_video_root()


def _video_lookup_attempts(video_id: str):
    root = _video_root()
    out = []
    for suffix in ("mp4", "MP4", "mkv", "webm", "avi", "mov"):
        out.append(os.path.abspath(os.path.join(root, f"{video_id}.{suffix}")))
    return out


def vatex_filter_to_subset(dataset):
    """Filter the loaded HF dataset to a frozen subset of videoIDs.

    When ``VATEX_SUBSET_MANIFEST`` points at a JSON file containing a list of
    videoID strings, keep only rows whose videoID is in that list. Order
    inside ``dataset`` is preserved (HF ``filter`` is stable), so passing
    ``--limit N`` then yields the same first N rows across every cell.

    When the env var is unset or the path missing, the dataset is returned
    unchanged so the task remains usable without the manifest.
    """
    path = os.environ.get("VATEX_SUBSET_MANIFEST", "").strip()
    if not path or not os.path.exists(path):
        return dataset
    with open(path) as f:
        keep = set(json.load(f))

    def _row_id(row):
        return row.get("videoID") or row.get("video_name") or row.get("video")

    return dataset.filter(lambda r: _row_id(r) in keep)


def vatex_doc_to_visual(doc):
    video_id = doc.get("videoID") or doc.get("video_name") or doc.get("video")
    if not video_id:
        raise KeyError(f"VATEX doc is missing a videoID-like field. Keys: {list(doc.keys())}")
    try:
        return [get_video(_video_root(), video_id, suffix="mp4")]
    except FileNotFoundError:
        for path in _video_lookup_attempts(video_id):
            if os.path.exists(path):
                return [path]
        raise FileNotFoundError(
            f"VATEX video for id={video_id!r} not found. Set VATEX_VIDEO_ROOT to "
            f"the directory containing <videoID>.mp4 files. Looked at: "
            f"{_video_lookup_attempts(video_id)}"
        )


def vatex_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    lmms_eval_specific_kwargs = lmms_eval_specific_kwargs or {}
    pre = lmms_eval_specific_kwargs.get("pre_prompt", "") or ""
    post = lmms_eval_specific_kwargs.get("post_prompt", "") or ""
    return f"{pre}{post}".strip() or "Describe the video in one sentence."


def vatex_doc_to_target(doc):
    refs = doc.get("enCap") or doc.get("captions") or []
    if isinstance(refs, str):
        refs = [refs]
    return refs


def _stable_int_id(video_id: str) -> int:
    """Map a string videoID to a stable non-negative int (COCO eval needs ints)."""
    h = hashlib.md5(str(video_id).encode("utf-8")).hexdigest()
    return int(h[:15], 16)  # ~60 bits, fits comfortably in int64


def vatex_process_results(doc, results):
    pred = results[0] if results else ""
    video_id = doc.get("videoID") or doc.get("video_name") or doc.get("video") or ""
    refs = vatex_doc_to_target(doc)
    payload = {
        "video_id": str(video_id),
        "image_id": _stable_int_id(str(video_id)),
        "pred": pred,
        "answer": list(refs) if refs else [""],
    }
    return {metric: payload for metric in VATEX_METRICS}


# ---------- pycocoevalcap-backed aggregation (primary path) ---------------- #

_SCORER_KEYS = {
    "Bleu_1": ("Bleu", 4),
    "Bleu_2": ("Bleu", 4),
    "Bleu_3": ("Bleu", 4),
    "Bleu_4": ("Bleu", 4),
    "METEOR": ("Meteor", None),
    "ROUGE_L": ("Rouge", None),
    "CIDEr": ("Cider", None),
}


def _coco_aggregate(results, metric, args=None):
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    except Exception as exc:  # noqa: BLE001 - we want a broad catch + clear log
        eval_logger.warning(
            f"pycocoevalcap not importable ({exc.__class__.__name__}: {exc}); "
            f"falling back to token-F1 surrogate for metric={metric}."
        )
        return _fallback_aggregate(results, metric)

    gts, res = {}, {}
    for r in results:
        iid = r["image_id"]
        refs = r.get("answer") or [""]
        gts[iid] = [{"caption": str(a)} for a in refs]
        res[iid] = [{"caption": str(r.get("pred", ""))}]

    try:
        tokenizer = PTBTokenizer()
        gts_t = tokenizer.tokenize(gts)
        res_t = tokenizer.tokenize(res)
    except Exception as exc:  # noqa: BLE001
        eval_logger.warning(
            f"PTBTokenizer failed ({exc!r}); falling back to token-F1 surrogate for metric={metric}."
        )
        return _fallback_aggregate(results, metric)

    if metric.startswith("Bleu"):
        scorer = Bleu(4)
        score, _ = scorer.compute_score(gts_t, res_t)
        n = int(metric.split("_")[-1])
        value = float(score[n - 1])
    elif metric == "METEOR":
        scorer = Meteor()
        score, _ = scorer.compute_score(gts_t, res_t)
        value = float(score)
    elif metric == "ROUGE_L":
        scorer = Rouge()
        score, _ = scorer.compute_score(gts_t, res_t)
        value = float(score)
    elif metric == "CIDEr":
        scorer = Cider()
        score, _ = scorer.compute_score(gts_t, res_t)
        value = float(score)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Optionally persist a submission file (best-effort; never crashes the run).
    try:
        path = generate_submission_file(f"vatex_{metric}_predictions.json", args)
        stored = [{"video_id": r["video_id"], "image_id": r["image_id"], "caption": r["pred"], "answer": r["answer"]} for r in results]
        with open(path, "w") as f:
            json.dump(stored, f, indent=2)
    except Exception:  # noqa: BLE001
        pass

    return value


# ---------- token-F1 + exact-match fallback ------------------------------- #

_PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}]")


def _normalize(s: str) -> str:
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_f1(pred: str, ref: str) -> float:
    pt = _normalize(pred).split()
    rt = _normalize(ref).split()
    if not pt or not rt:
        return 0.0
    common = collections.Counter(pt) & collections.Counter(rt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(pt)
    r = n / len(rt)
    return 2 * p * r / (p + r)


def _fallback_aggregate(results, metric):
    """Best-of-references token-F1 (logged as ``metric=fallback_token_f1``)."""
    if not results:
        return 0.0
    total = 0.0
    for r in results:
        pred = r.get("pred", "")
        refs = r.get("answer") or [""]
        best = max((_token_f1(pred, ref) for ref in refs), default=0.0)
        total += best
    score = total / max(1, len(results))
    eval_logger.warning(
        f"vatex metric={metric!r} reported as fallback_token_f1 (mean={score:.4f}) "
        f"because pycocoevalcap is unavailable."
    )
    return score


def vatex_aggregate_cider(results, args=None):
    return _coco_aggregate(results, "CIDEr", args)


def vatex_aggregate_bleu4(results, args=None):
    return _coco_aggregate(results, "Bleu_4", args)


def vatex_aggregate_meteor(results, args=None):
    return _coco_aggregate(results, "METEOR", args)


def vatex_aggregate_rougel(results, args=None):
    return _coco_aggregate(results, "ROUGE_L", args)
