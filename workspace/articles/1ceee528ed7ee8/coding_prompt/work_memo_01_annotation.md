# プロンプト修正方針

こちらの修正
> causal_discovery          # 因果探索分析のクラス群
> causal_inference          # 因果推論分析のクラス群
> causal_pipeline_runtime   # 上位制御層

と合わせて、以下の指摘事項に対しその妥当性を評価した上で
プロンプトの markdown source code を再出力せよ

## 1. section識別のためのprefix number 追加

以下のようにあるが、

> # 目的
> # 重要な設計方針
> ## A. `--validate-only` / `--dry-run` は causal discovery に入れない

このようにしてほしい

> # 1. 目的
> # 2. 重要な設計方針
> ## 2.A. `--validate-only` / `--dry-run` は causal discovery に入れない


## 2. section構成

> # コーディングエージェント向け実装プロンプト

は全体を表現するものだよね？なのでstage（section）を表すものじゃないよね？
こうなるのが正しいのではないか？

```markdown
コーディングエージェント向け実装プロンプト
> # 0. INTRODUCTION
> # 1. 目的
> # 2. 重要な設計方針
```


## 3. 修正方針の文脈明示

> ## A. `--validate-only` / `--dry-run` は causal discovery に入れない
> 
> `--validate-only` / `--dry-run` は因果探索アルゴリズムの一部ではない。  
> これらは **統合パイプラインの実行制御モード** である。
> 
> したがって、以下は禁止する。

これらの解説がいきなり入るのは唐突。本来は以下の文脈があったはずである（不足/間違いあれば補正のこと）

> もともとは、 causal discovery を行う上で、alpha sensitivity などを行っていたが、
> それらはcausal discovery そのものではないのでモードで分けようとしたが、discovery は stage であり、 pipeline 全体ではない

## 4. レビュー評価の採点基準

以下のようにあるが、採点基準もなく "8/10" と言われても実装が難しい

> 以下の各観点について、レビュー評点を最低でも **8 / 10 以上** に引き上げる。

