"""Shared constants that are independent of pipeline execution."""

from __future__ import annotations

from pathlib import Path


ARTICLE_ID = "1ceee528ed7ee8"
ARTICLE_ROOT = Path("articles") / ARTICLE_ID

DEFAULT_DATASET_YAML = Path("shared/py/myproj/conf/dataset/completejourney/10_interim.yaml")
DEFAULT_CAUSAL_DISCOVERY_DIR = ARTICLE_ROOT / "artifacts" / "causal_discovery"
DEFAULT_CAUSAL_INFERENCE_DIR = ARTICLE_ROOT / "artifacts" / "causal_inference"

SUPPORTED_DISCOVERY_ALGORITHMS = ("pc", "ges", "lingam", "notears")

__all__ = [
    "ARTICLE_ID",
    "ARTICLE_ROOT",
    "DEFAULT_CAUSAL_DISCOVERY_DIR",
    "DEFAULT_CAUSAL_INFERENCE_DIR",
    "DEFAULT_DATASET_YAML",
    "SUPPORTED_DISCOVERY_ALGORITHMS",
]
