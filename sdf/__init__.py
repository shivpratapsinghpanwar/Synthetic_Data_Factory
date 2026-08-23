"""Synthetic Data Factory pipeline.

Generates realistic synthetic medical-condition images and measures whether
they actually improve anomaly/condition detectors. Stages run locally for
tests and on Kaggle (via ``kaggle_runner``) for real data and GPUs.

Entry point: ``python -m sdf run-stage <name>``
Design: docs/pipeline_design.md
"""

__version__ = "0.1.0"
