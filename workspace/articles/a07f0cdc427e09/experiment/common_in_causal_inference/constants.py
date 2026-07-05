"""因果探索・因果推論で共有する定数。

ここには両 pipeline から同じ意味で参照される定数だけを置く。推論 mode、
estimand、標準誤差 type など、推論パイプラインに閉じた選択肢は
``causal_inference_pipeline.constants`` に残す。
"""

from __future__ import annotations

from pathlib import Path


ARTICLE_ID = "a07f0cdc427e09"
ARTICLE_ROOT = Path("articles") / ARTICLE_ID

DEFAULT_DATASET_YAML = Path("shared/py/myproj/conf/dataset/completejourney/10_interim.yaml")
DEFAULT_CAUSAL_DISCOVERY_DIR = ARTICLE_ROOT / "artifacts" / "causal_discovery"
DEFAULT_CAUSAL_INFERENCE_DIR = ARTICLE_ROOT / "artifacts" / "causal_inference"

SUPPORTED_DISCOVERY_ALGORITHMS = ("pc", "ges", "lingam", "notears")
