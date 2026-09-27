# 疑問点

## 1. 共通化残ファイル群

以下のファイル群は、common_in_causal_inference に移動するのが適切なのではないか？
移動しなかった理由を明らかにし、必要に応じて移動させよ

- a07f0cdc427e09/experiment/
    - causal_inference_pipeline
        - constants.py
        - config.py
    - causal_discovery_pipeline/
        - config_loader.py
        - config_schema.py
        - schemas.py
        - graph_utils.py

## 2. 類似機能を持つと推測されるファイル

以下２つは共通化できなかったのか？
その理由はなにか？必要に応じて統合せよ。

- a07f0cdc427e09/experiment/
    - causal_discovery_pipeline/data/loader.py
    - causal_inference_pipeline/data/loader.py

以下２つの違いはなにか？
- a07f0cdc427e09/experiment/causal_discovery_pipeline/data_loader.py
- a07f0cdc427e09/experiment/causal_discovery_pipeline/data/loader.py

