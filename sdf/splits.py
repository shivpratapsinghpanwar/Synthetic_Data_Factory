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
    splits: dict[str, list[ImageRecord]] = {"train": [], "val": [], "test": []}

    # Curated splits are honored verbatim (a Roboflow-style export may hold
    # augmented near-duplicates of one base image inside train; re-splitting
    # them randomly would leak variants across the boundary). If any record
    # is pre-assigned to test/val, the splitter stops carving that split from
    # the free records.
    free: list[ImageRecord] = []
    fixed_counts = {"train": 0, "val": 0, "test": 0}
    for rec in records:
        if rec.fixed_split:
            if rec.fixed_split not in splits:
                raise ValueError(f"invalid fixed_split {rec.fixed_split!r} on {rec.image_id}")
            splits[rec.fixed_split].append(rec)
            fixed_counts[rec.fixed_split] += 1
        else:
            free.append(rec)
    warnings: list[str] = []
    if fixed_counts["test"]:
        test_frac = 0.0
    if fixed_counts["val"]:
        val_frac = 0.0
    if (fixed_counts["val"] or fixed_counts["test"]) and test_frac > 0.0:
        warnings.append(
            "curated split present but no curated TEST - test is being carved "
            "from free records; prefer a curated test for evaluation"
        )

    by_class: dict[str, dict[str, list[ImageRecord]]] = defaultdict(lambda: defaultdict(list))
    for rec in free:
        by_class[rec.cls][rec.group_id].append(rec)
    per_class: dict[str, dict[str, int]] = {}

    for cls in sorted(by_class):
        groups = sorted(by_class[cls].items())  # stable order before shuffling
        total = sum(len(recs) for _, recs in groups)

        if val_frac == 0.0 and test_frac == 0.0:
            for _, recs in groups:
                splits["train"].extend(recs)
            per_class[cls] = {"train": total, "val": 0, "test": 0}
            continue

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

        # A nonzero fraction must never silently round to zero for a class
        # that is large enough to split (banker's rounding: round(0.5) == 0).
        want_test = max(1, round(total * test_frac)) if test_frac > 0 else 0
        want_val = max(1, round(total * val_frac)) if val_frac > 0 else 0
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
        "fixed": fixed_counts,
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
