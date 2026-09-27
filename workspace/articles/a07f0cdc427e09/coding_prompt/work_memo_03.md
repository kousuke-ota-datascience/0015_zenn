# 疑問点
以下の疑問に対し、現状の実装理由を明らかにした上で必要に応じて修正せよ

## 1. 構成の違い

なぜ、以下の3つでファイル命名ポリシーが違うのか。人間が読むときに混乱する可能性が高いと見ている

- a07f0cdc427e09/experiment/causal_discovery_pipeline/
    - config_loader.py
    - config_schema.py
- a07f0cdc427e09/experiment/causal_inference_pipeline/
    - config.py
- a07f0cdc427e09/experiment/common_in_causal_inference/
    - config.py

## 2. constants の不在

- a07f0cdc427e09/experiment/causal_discovery_pipeline

配下 には constants.py は不要なのか

## 3. 薄い wrapper の意義

### 3.1. loader

以下の２つは薄いラッパーとして残しているとのことだが、その意義はなにか？
ほぼ同じことをしているならば、LogicalTableDataLoader を呼び出せば良くないか？

- a07f0cdc427e09/experiment/causal_discovery_pipeline/data/loader.py
- a07f0cdc427e09/experiment/causal_inference_pipeline/data/loader.py

### 3.2. feature_engineering / preprocessing

- a07f0cdc427e09/experiment/causal_discovery_pipeline/
    - feature_engineering.py
    - preprocessing.py

の中身は以下の通り。これは [ポルターガイスト] ではないか？

**feature_engineering.py**
```python
"""後方互換用の特徴量 engineering import。"""

from .features.engineering import *  # noqa: F403
```

**preprocessing.py**
```python
"""後方互換用の preprocessor import。"""

from .features.builder import AbstractPreprocessor, CompleteJourneyPreprocessor

__all__ = ["AbstractPreprocessor", "CompleteJourneyPreprocessor"]
```

## 4. feature 配下のファイル構成の整合性

discovery, inference 配下に共通して存在するクラスファイル feature について。

各々以下のような構成になっているが、ファイル命名/構成に共通性が低いと感じられる。
causal_inference_pipeline の方は、encoder.py selectors.py と行った形で、
特徴量生成（=feature engineering）のさらに細分化されたレイヤで、「何をやりたいか」で切っているように見受けられるが、
causal_discovery_pipeline は "engineering.py" とひとまとめにされている

- a07f0cdc427e09/experiment/causal_discovery_pipeline/features
    - builder.py
    - engineering.py

- a07f0cdc427e09/experiment/causal_inference_pipeline/features
    - aggregations.py
    - builder.py
    - config.py
    - encoders.py
    - selectors.py
