"""MixKV + BACON KV-cache compression utilities.

`BACON` (Boundary Attention CalibratiON) is implemented inside
:mod:`bacon.bacon_utils` as a plug-in score calibration that augments any
backbone retention score `B` with boundary-emergent evidence. It is exposed
via the public select_method name ``'bacon'`` (see scripts/eval/*).
"""

from .bacon_utils import (  # noqa: F401
    compute_bacon_score,
    clear_bacon_trace,
)
