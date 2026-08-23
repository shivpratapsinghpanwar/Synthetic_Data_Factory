"""Stage contract: every stage produces one machine-readable StageResult.

The kaggle_runner collects /kaggle/working after a run, so a stage that writes
``stage_<name>.json`` there is automatically retrievable and appears in the
run summary's artifact list. An agent loop reads that JSON - never raw logs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import output_dir


@dataclass
class StageResult:
    stage: str
    success: bool
    metrics: dict = field(default_factory=dict)
    error: str = ""
    outputs: list = field(default_factory=list)  # artifact file names
    duration_s: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def write_result(
    result: StageResult, dest: Path | None = None, suffix: str = ""
) -> Path:
    """Write stage_<name>[_<suffix>].json. The suffix (usually the class)
    keeps repeated invocations of one stage in a single session from
    overwriting each other's results."""
    folder = dest or output_dir()
    folder.mkdir(parents=True, exist_ok=True)
    name = f"stage_{result.stage}_{suffix}.json" if suffix else f"stage_{result.stage}.json"
    path = folder / name
    path.write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path
