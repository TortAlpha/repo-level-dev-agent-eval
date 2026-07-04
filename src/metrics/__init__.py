"""Evaluation metrics computed from recorded runs (see docs/spec.md).

- ``records``  — load ``runs.jsonl`` rows into typed ``RunRecord`` objects.
- ``compute``  — turn records into a ``MetricSet`` of spec metrics.
- ``report``   — CLI that prints the metrics, grouped (e.g. by agent_mode).
"""

from .compute import EXTERNAL_METRICS, MetricSet, compute_metrics, group_by
from .pricing import PRICES, ModelPrice, estimate_cost
from .records import (
    RunRecord,
    attach_quality,
    attach_sizes,
    load_quality,
    load_runs,
    load_sizes,
)

__all__ = [
    "EXTERNAL_METRICS",
    "MetricSet",
    "ModelPrice",
    "PRICES",
    "RunRecord",
    "attach_quality",
    "attach_sizes",
    "compute_metrics",
    "estimate_cost",
    "group_by",
    "load_quality",
    "load_runs",
    "load_sizes",
]
