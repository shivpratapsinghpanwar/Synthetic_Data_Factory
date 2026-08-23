"""Pipeline stages. Each stage is a function (PipelineConfig) -> StageResult."""

from . import audit
from .base import StageResult, write_result  # noqa: F401

REGISTRY = {
    "audit": audit.run,
}
