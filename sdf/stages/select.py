"""select stage: compose the committee's best synthetic set for one class.

Several generator arms (mdx, sd15_lora, sdxl_lora) sample into the same
synthetic/<cls>/ pool, each row tagged with its backend in the manifest.
This stage picks the ``count`` images that go into training:

1. drop anything a quality gate flagged as memorized, plus flat images and
   exact duplicates (PIL-only screen - always runs);
2. score every candidate with the real-only detector when one is available
   (softmax probability of the correct class = realism/label-fidelity proxy);
3. blend arms round-robin, each arm's queue sorted by score, so the
   committee contributes its best rather than one arm dominating.

Output: selection_<cls>.json next to the manifest. The augment stage honors
it automatically - classes without a selection file keep today's take-all
behavior.

Usage: python -m sdf run-stage select --opt cls=<class> --opt count=100 \
    [--opt syn_root=...] [--opt detector_dir=.../detector/real_only] \
    [--opt min_score=0.2]
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .. import manifest as manifest_mod
from ..config import PipelineConfig, output_dir
from ..quality import pixel_screen
from .base import StageResult
from .augment import MANIFEST_NAME, _flagged_images


def _detector_scores(detector_dir: Path, paths: list[Path], cls: str) -> list[float] | None:
    """Probability of ``cls`` per image under the saved detector, or None
    when no usable detector/torch is present (selection then falls back to
    manifest order within each arm)."""
    model_file = detector_dir / "model.pt"
    labels_file = detector_dir / "labels.json"
    report_file = detector_dir / "report.json"
    if not (model_file.exists() and labels_file.exists()):
        return None
    try:
        import torch
        from torch.utils.data import DataLoader

        from .. import detect
    except ImportError:
        return None

    labels = json.loads(labels_file.read_text(encoding="utf-8"))
    if cls not in labels:
        return None
    model_name = "efficientnet_b0"
    if report_file.exists():
        model_name = json.loads(report_file.read_text(encoding="utf-8")).get(
            "model", model_name
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = detect.build_model(model_name, len(labels))
    model.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))
    model = model.to(device).eval()

    rows = [{"path": str(p), "cls": cls, "source": "synthetic"} for p in paths]
    loader = DataLoader(
        detect.make_dataset(rows, labels, train=False), batch_size=32, num_workers=0
    )
    scores: list[float] = []
    target = labels[cls]
    with torch.no_grad():
        for pixels, _y in loader:
            probs = torch.softmax(model(pixels.to(device)), dim=1)
            scores.extend(probs[:, target].cpu().tolist())
    return scores


def run(cfg: PipelineConfig, opts: dict | None = None) -> StageResult:
    started = time.time()
    opts = opts or {}
    cls = str(opts.get("cls", ""))
    if not cls:
        return StageResult(
            stage="select", success=False, error="pass --opt cls=<class>",
            duration_s=round(time.time() - started, 2),
        )
    count = int(opts.get("count", cfg.generator.sample_count))
    min_score = float(opts.get("min_score", 0.0))
    syn_root = Path(str(opts.get("syn_root", "")) or output_dir())

    rows = [
        r for r in manifest_mod.read(syn_root / MANIFEST_NAME)
        if r.get("cls") == cls and r.get("synthetic")
    ]
    if not rows:
        return StageResult(
            stage="select", success=False,
            error=f"no synthetic manifest rows for class {cls!r} under {syn_root}",
            duration_s=round(time.time() - started, 2),
        )

    flagged = _flagged_images(syn_root)
    candidates = []
    dropped = {"flagged": 0, "missing_file": 0, "flat": 0, "duplicate": 0}
    for row in rows:
        name = Path(row["file"]).name
        path = syn_root / row["file"]
        if name in flagged:
            dropped["flagged"] += 1
            continue
        if not path.exists():
            dropped["missing_file"] += 1
            continue
        candidates.append((path, row.get("backend", "unknown")))

    screen = pixel_screen([p for p, _b in candidates])
    bad = set(screen["flat"])
    dropped["flat"] = len(bad)
    for _kept, dupe in screen["exact_duplicates"]:
        bad.add(dupe)
        dropped["duplicate"] += 1
    candidates = [(p, b) for p, b in candidates if p.name not in bad]

    detector_dir = Path(
        str(opts.get("detector_dir", "")) or output_dir() / "detector" / "real_only"
    )
    scores = _detector_scores(detector_dir, [p for p, _b in candidates], cls)
    scored = scores is not None
    if not scored:
        scores = [0.0] * len(candidates)

    # Round-robin across arms, best-first within each arm.
    queues: dict[str, list[tuple[float, str]]] = {}
    for (path, backend), score in zip(candidates, scores):
        if scored and score < min_score:
            continue
        queues.setdefault(backend, []).append((score, path.name))
    for q in queues.values():
        q.sort(key=lambda item: -item[0])

    selected: list[str] = []
    per_backend: dict[str, int] = {}
    while len(selected) < count and any(queues.values()):
        for backend in sorted(queues):
            if queues[backend] and len(selected) < count:
                _score, name = queues[backend].pop(0)
                selected.append(name)
                per_backend[backend] = per_backend.get(backend, 0) + 1

    selection = {
        "cls": cls,
        "selected": selected,
        "per_backend": per_backend,
        "scored_by_detector": scored,
        "detector_dir": str(detector_dir) if scored else "",
        "candidates": len(candidates),
        "dropped": dropped,
    }
    out_file = syn_root / f"selection_{cls}.json"
    out_file.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"[select] {cls}: {len(selected)}/{len(candidates)} kept "
          f"per_backend={per_backend} scored={scored}", flush=True)

    return StageResult(
        stage="select",
        success=len(selected) > 0,
        metrics=selection,
        outputs=[out_file.name],
        error="" if selected else "no candidates survived selection",
        duration_s=round(time.time() - started, 2),
    )
