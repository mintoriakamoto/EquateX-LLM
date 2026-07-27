# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the bounded in-memory metric histories.

``_decimate_aligned_histories`` keeps the backend's live-display buffers from
growing without limit over a long run while preserving the parallel-array
alignment the SSE replay and /metrics endpoints rely on. The heavy module-level
imports of ``core/training/training.py`` are stubbed so this imports under a
CPU-only, no-network runner (same preamble as test_training_stop_watchdog.py).
"""

from __future__ import annotations

import contextlib
import logging
import sys
import types as _types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_SAVED: dict = {}


def _stub(name, mod):
    _SAVED[name] = sys.modules.get(name)
    sys.modules[name] = mod


_lg = _types.ModuleType("loggers")
_lg.get_logger = lambda name: logging.getLogger(name)
_stub("loggers", _lg)
_stub("structlog", _types.ModuleType("structlog"))
_mpl = _types.ModuleType("matplotlib")
_plt = _types.ModuleType("matplotlib.pyplot")
_plt.Figure = type("Figure", (), {})
_mpl.pyplot = _plt
_stub("matplotlib", _mpl)
_stub("matplotlib.pyplot", _plt)
_hw = _types.ModuleType("utils.hardware")
_hw.prepare_gpu_selection = lambda *a, **k: (None, None)
_stub("utils.hardware", _hw)
_npl = _types.ModuleType("utils.native_path_leases")
_npl.native_path_secret_removed_for_child_start = lambda: contextlib.nullcontext()
_npl.run_without_native_path_secret = lambda fn: fn
_stub("utils.native_path_leases", _npl)
_pth = _types.ModuleType("utils.paths")
_pth.outputs_root = lambda *a, **k: "/tmp/outputs"
_stub("utils.paths", _pth)

_TRAINING_PRE_IMPORTED = "core.training.training" in sys.modules

from core.training.training import (  # noqa: E402
    _MAX_METRIC_HISTORY_POINTS,
    _decimate_aligned_histories,
)

for _name in (
    "loggers",
    "structlog",
    "matplotlib",
    "matplotlib.pyplot",
    "utils.hardware",
    "utils.native_path_leases",
    "utils.paths",
):
    _prev = _SAVED.get(_name)
    if _prev is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _prev

if not _TRAINING_PRE_IMPORTED:
    sys.modules.pop("core.training.training", None)
    sys.modules.pop("core.training", None)


def _simulate_run(total_steps: int, max_points: int):
    """Append `total_steps` monotonic points to three aligned lists, decimating
    after each append exactly as the pump loop does."""
    steps: list = []
    loss: list = []
    lr: list = []
    for step in range(1, total_steps + 1):
        steps.append(step)
        loss.append(round(1.0 / step, 6))
        lr.append(step * 2)
        _decimate_aligned_histories([steps, loss, lr], max_points)
    return steps, loss, lr


def test_history_is_bounded_over_a_long_run():
    steps, loss, lr = _simulate_run(total_steps = 60_000, max_points = 1_000)
    assert len(steps) <= 1_000
    assert len(steps) == len(loss) == len(lr)


def test_alignment_is_preserved_after_decimation():
    steps, loss, lr = _simulate_run(total_steps = 60_000, max_points = 1_000)
    # loss[i] and lr[i] must still correspond to steps[i].
    for i, step in enumerate(steps):
        assert loss[i] == round(1.0 / step, 6)
        assert lr[i] == step * 2


def test_endpoints_and_full_range_retained():
    steps, _, _ = _simulate_run(total_steps = 60_000, max_points = 1_000)
    assert steps[0] == 1
    assert steps[-1] == 60_000
    # The early part of the run is thinned but never dropped entirely.
    assert any(step < 1_000 for step in steps)


def test_steps_stay_monotonic():
    steps, _, _ = _simulate_run(total_steps = 60_000, max_points = 1_000)
    assert all(steps[i] < steps[i + 1] for i in range(len(steps) - 1))


def test_independent_groups_stay_self_aligned():
    # A group of a different length (e.g. eval, logged less often) decimates on
    # its own without disturbing alignment.
    gs: list = []
    gv: list = []
    for step in range(1, 5_001):
        gs.append(step)
        gv.append(step * step)
        _decimate_aligned_histories([gs, gv], 1_000)
    assert len(gs) == len(gv) <= 1_000
    assert all(gv[i] == gs[i] ** 2 for i in range(len(gs)))


def test_noop_below_cap_and_on_empty():
    under = [1, 2, 3]
    _decimate_aligned_histories([under], 10)
    assert under == [1, 2, 3]

    empty: list = []
    _decimate_aligned_histories([empty], 10)
    assert empty == []

    # No lists at all must not raise.
    _decimate_aligned_histories([], 10)


def test_default_cap_is_sane():
    assert isinstance(_MAX_METRIC_HISTORY_POINTS, int)
    assert _MAX_METRIC_HISTORY_POINTS >= 1_000
