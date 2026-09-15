# 0158 Review_003 adjudication

- Entry_ID: `0158`
- Review_Seq: `003`
- Review対象: `0158_10_analysis.md`
- Review対象blob: `c2c250ff1790b37314b54c327c2cbbaf7d5854be`
- 判定: Review findingsを全件ACCEPT

## Findings

### F10-001 — D06 `T6` が研究上の不確定性をtruth stanceへ移している

- 裁定: **ACCEPT**
- 対応: D06を `D06.T4 / I` へ再裁定。
- 理由: 『弘賢随筆』系記録は日付・場所・舟・女性・箱・記号・対応を起きた出来事として具体的に叙述する。女性等の正体が不明であることは出来事自体の提示態度とは分離する。江戸期随筆・奇談の事実報告的語りを現行taxonomyへ写像する部分には解釈を含むためStatusは `I`。

### F10-002 — D19 `NATIONAL_PUBLIC` の実流通Evidence不足

- 裁定: **ACCEPT**
- 対応: D19を `U` へ保守化。
- 理由: 複数文書への収録、国立公文書館Web公開、研究・展示は媒体・所蔵・公開経路を示すが、社会的・地理的な実受容範囲を直接固定しない。

## 変更

- 00変更: なし。Review_003で00はPass済み。
- 10 correction commit: `0c447d505ce78cb59336fdcf04207cfd154f18ff`
- 10 post blob: `7b2e7eaeed4081cc21d1fe3febc7004b645b0976`

## QA

- tutorial構造: 維持
- Version Scope: 江戸期漂着譚中心、UFO解釈分離を維持
- D06: `T4 / I`
- D19: `U`
- D17/D20: `U`を維持
- 状態: `再レビュー待 / 003`
