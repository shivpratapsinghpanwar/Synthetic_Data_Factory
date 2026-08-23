"""Grouped, stratified, deterministic train/val/test splitting.

Two invariants matter more than exact fractions:

1. **No group leakage.** All images of one physical lesion (group_id) land in
   the same split. HAM10000 re-images the same lesion up to 6 times; letting
   near-duplicates straddle train/test would inflate every downstream metric.
2. **Determinism.** The same (records, seed) always yields the same split, on
   any machine, in any process. Synthetic data is only ever added to *train*;
   val and test stay real-only forever, so they must never shift under us.
"""

from __future__ import annotations

import random
from collections import defaultdict

from .data.base import ImageRecord

# A class needs at least this many groups before carving out val/test;
# below it, everything goes to train and the class is flagged instead.
MIN_GROUPS_TO_SPLIT = 5


def grouped_stratified_split(
    records: list[ImageRecord],
    *,
    seed: int,
    val_frac: float,
    test_frac: float,
) -> tuple[dict[str, list[ImageRecord]], dict]:
    """Return ({"train": [...], "val": [...], "test": [...]}, stats)."""
    by_class: dict[str, dict[str, list[ImageRecord]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        by_class[rec.cls][rec.group_id].append(rec)

    splits: dict[str, list[ImageRecord]] = {"train": [], "val": [], "test": []}
    per_class: dict[str, dict[str, int]] = {}
    warnings: list[str] = []

    for cls in sorted(by_class):
        groups = sorted(by_class[cls].items())  # stable order before shuffling
        total = sum(len(recs) for _, recs in groups)

        if len(groups) < MIN_GROUPS_TO_SPLIT:
            warnings.append(
                f"class {cls!r} has only {len(groups)} groups; all {total} images -> train"
            )
            for _, recs in groups:
                splits["train"].extend(recs)
            per_class[cls] = {"train": total, "val": 0, "test": 0}
            continue

        # str seeds hash stably (SHA-512 path of random.seed), so this is
        # reproducible across processes and platforms.
        rng = random.Random(f"{seed}:{cls}")
        rng.shuffle(groups)

        want_test = round(total * test_frac)
        want_val = round(total * val_frac)
        counts = {"train": 0, "val": 0, "test": 0}

        for _, recs in groups:
            if counts["test"] < want_test:
                target = "test"
            elif counts["val"] < want_val:
                target = "val"
            else:
                target = "train"
            splits[target].extend(recs)
            counts[target] += len(recs)

        if counts["train"] == 0:
            warnings.append(f"class {cls!r}: fractions left no training groups")
        per_class[cls] = counts

    stats = {
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "per_class": per_class,
        "totals": {name: len(recs) for name, recs in splits.items()},
        "warnings": warnings,
    }
    _assert_no_leakage(splits)
    return splits, stats


def _assert_no_leakage(splits: dict[str, list[ImageRecord]]) -> None:
    seen: dict[str, str] = {}
    for split_name, recs in splits.items():
        for rec in recs:
            prior = seen.setdefault(rec.group_id, split_name)
            if prior != split_name:
                raise AssertionError(
                    f"group {rec.group_id!r} appears in both {prior!r} and {split_name!r}"
                )
