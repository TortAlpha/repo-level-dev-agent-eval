"""Run one PR-derived benchmark task and score the agent's solution.

Split by concern:
- ``collection`` — task specs parsed from ``collection.csv``.
- ``workspace``  — per-run repo copy and the agent's captured patch.
- ``evaluation`` — sandboxed regression + hidden-test scoring.
- ``review``     — reviewer-model quality scoring.
- ``runner``     — the CLI wiring it all together.
"""

from .collection import TaskSpec, load_collection
from .runner import main

__all__ = ["TaskSpec", "load_collection", "main"]
