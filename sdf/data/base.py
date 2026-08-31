"""Adapter interface: one uniform shape for every medical imaging dataset.

Adding a new modality (chest X-ray, MRI, ...) means writing one adapter that
returns ImageRecords; every downstream stage - splits, audit, training,
evaluation - stays unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


class DataError(RuntimeError):
    """The dataset is absent, unreadable, or structurally wrong."""


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: Path
    cls: str        # condition/diagnosis label
    group_id: str   # physical entity (lesion, patient); split boundaries respect it
    exists: bool    # file actually present on disk
    # Pre-assigned split ("train"/"val"/"test") for datasets that ship with
    # curated splits we must honor; empty = let the splitter decide.
    fixed_split: str = ""


class DatasetAdapter(ABC):
    """One concrete dataset. Constructed with the pipeline config."""

    name: str = ""

    @abstractmethod
    def index(self) -> tuple[list[ImageRecord], dict]:
        """Return (records, report).

        ``report`` is a JSON-safe dict describing what was discovered: which
        root was used, where the metadata lives, and what is missing. It goes
        verbatim into the audit stage output.
        """


def get_adapter(cfg) -> DatasetAdapter:
    """Look up the adapter for ``cfg.dataset.name``."""
    # Imported here to avoid a cycle (adapters import base).
    from .folder_class import FolderClassAdapter
    from .ham10000 import HAM10000Adapter
    from .imagechd import ImageCHDAdapter

    adapters = {
        HAM10000Adapter.name: HAM10000Adapter,
        ImageCHDAdapter.name: ImageCHDAdapter,
        FolderClassAdapter.name: FolderClassAdapter,
    }
    try:
        return adapters[cfg.dataset.name](cfg)
    except KeyError:
        raise DataError(
            f"no adapter named {cfg.dataset.name!r}; available: {sorted(adapters)}"
        ) from None
