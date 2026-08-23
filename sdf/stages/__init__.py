"""Pipeline stages. Each stage: (PipelineConfig, opts: dict|None) -> StageResult."""

from . import audit, augment, evaluate, probe_ml, quality_gate, sample, train_detector, train_lora
from .base import StageResult, write_result  # noqa: F401

REGISTRY = {
    "audit": audit.run,
    "probe_ml": probe_ml.run,
    "train_lora": train_lora.run,
    "sample": sample.run,
    "quality_gate": quality_gate.run,
    "augment": augment.run,
    "train_detector": train_detector.run,
    "evaluate": evaluate.run,
}
