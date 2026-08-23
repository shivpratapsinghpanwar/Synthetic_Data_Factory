"""Provenance manifest for every synthetic image the factory produces.

Non-negotiable design rule: no synthetic image exists without a manifest row
naming it as synthetic and recording exactly how it was made (backend, base
model, checkpoint, seed, prompt). This is what makes runs reproducible, lets
quality gates trace bad samples back to their source, and guarantees the data
can never silently masquerade as real patient imagery.

Format: JSON Lines, one record per image, append-only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

REQUIRED_FIELDS = (
    "image_id",
    "file",
    "cls",
    "backend",
    "base_model",
    "checkpoint",
    "seed",
    "prompt",
)


class ManifestError(RuntimeError):
    pass


def new_record(**fields) -> dict:
    """Build a validated manifest record. Timestamps and the synthetic flag
    are added here so callers cannot forget them."""
    missing = [f for f in REQUIRED_FIELDS if f not in fields or fields[f] in (None, "")]
    # seed 0 is valid; repair the false positive from the emptiness check
    if "seed" in missing and fields.get("seed") == 0:
        missing.remove("seed")
    if missing:
        raise ManifestError(f"manifest record missing field(s): {', '.join(missing)}")

    record = dict(fields)
    record["synthetic"] = True
    record.setdefault(
        "created_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    return record


def append(path: Path, record: dict) -> None:
    if not record.get("synthetic"):
        raise ManifestError("refusing to write a record not marked synthetic=True")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records
