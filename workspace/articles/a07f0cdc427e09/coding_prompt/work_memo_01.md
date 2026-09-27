# 1. やりたいこと

## 1.0. INTRODUCTION

以下ディレクトリ配下に格納されているプログラム軍に関し、リファクタリングを行いたい

- workspace/articles/a07f0cdc427e09

## 1.1. クラスファイル構成の統合

以下2つのクラスファイルの構成は、その構成が大きく異なっている。
どちらの構成がより合理的か判別の上、統合せよ

- workspace/articles/a07f0cdc427e09/experiment
    - causal_discovery_pipeline/
    - causal_inference_pipeline/

## 1.2. 重複機能の切り出し

以下2つのクラスファイルの中には、重複する機能が実装されている

- workspace/articles/a07f0cdc427e09/experiment
    - causal_discovery_pipeline/
    - causal_inference_pipeline/

### 例
- データロード機能
    - workspace/articles/a07f0cdc427e09/experiment/causal_inference_pipeline/
        - data/loader.py
    - workspace/articles/a07f0cdc427e09/experiment/causal_discovery_pipeline/
        - config_loader.py

上記例以外にも、機能が重複しているものがあればリストアップした上で切り出し、以下に格納せよ

- workspace/articles/a07f0cdc427e09/experiment
    - common_in_causal_inference

## 1.3. "Thin entrypoint"の統合

因果探索→因果推論のパイプラインは現状以下２つに別れている。
これらのファイルは thin entrypoint である

- workspace/articles/a07f0cdc427e09/experiment/
    - 因果探索: 03_causal_discovery_completejourney.py
    - 因果推論: 04_causal_inference_completejourney.py

これらのファイルを統合し、以下としたい

- 05_causal_discovery_inference_completejourney.py

# 2. 実装時の方針

## 2.1. articacts 出力対象

生成物（output）は、以下ディレクトリに作成すること。

- workspace/articles/a07f0cdc427e09/artifacts
    - features: 変換後の特徴量
    - causal_discovery: 因果探索実施時の生成物
    - causal_inference: 因果推論実施時の生成物
    - docs/_build/html: docstring によるAPI定義書

artifacts配下に登録するものは、git管理対象外としたい。
逆に言うと、experiment 配下のファイルは、まとめてgit管理対象としても問題のない構成にしたい

## 2.2. docstring (sphinx)

- 現状は英語だが、日本語で書くこと
- また、各機能の概要についても厚めに記載すること
