# 0118 Review_002 adjudication

- Entry_ID: `0118`
- Review: `Review_0118_00_002.md` / `Review_0118_10_002.md`
- Review判定: 00=`要修正（Moderate）`, 10=`要修正（Moderate）`
- Coder判定: FindingsをすべてACCEPT

## 00 Findings

### F00-001 — 事故具体化型・道路回避型のContent Evidence traceability不足

**ACCEPT。**

Reviewの指摘どおり、旧00は事故・怪我等の具体化と「道路が木を避けた」型を主要異伝として保持していたが、列挙済み典拠からその内容を直接戻せなかった。追加Evidenceを推測で補わず、LIFULL HOME'Sで直接確認できる共有核「伐採を試みる→関係者に怖いことが起こる→伐採できず木が残る」まで縮退した。

事故・怪我・死亡の具体像、道路回避・道路線形変更は未固定事項へ戻し、2.3の主要異伝から除外。3.1では中心命題と各流通Evidenceの役割を分離した。

- 00 correction commit: `3ad16ee65d5b2e442aad287fbc7e79654dbc982e`

## 10 Findings

### F10-001 — D05/D09で独立Evidenceのない禁止規範を再生成

**ACCEPT。**

00が「切るな」という独立した禁止規範を固定していない以上、`D05.RUL.TABOO` / `D09.NOR.TABOOIZATION` は過剰だった。共有核へ直接対応するよう、D05を `D05.RUL.JINX_RULE / I`、D09を `D09.CAU.DIRECT_CAUSE / I` へ変更した。

同時に、00の縮退に合わせて道路配置をD08根拠から除外し、D04は変容Evidence不足のため `U` を維持。D10 taxonomy gap、D17=`U`、D18=`1/0/0 / I` は維持した。

- 10 correction commit: `3ab7084ea196b607b254859436b32bc9d27424a1`

## QA

- 事故具体化・道路回避型を直接Content Evidenceなしに使用していない。
- D05/D09は条件・帰結から禁止規範を自動生成していない。
- D11→D13→D15は「伐採試行→不運付与→広い災い」で一貫。
- D17はtriggerの反転ではなく `U` を維持。
- 次状態: `再レビュー待 / 002`。
