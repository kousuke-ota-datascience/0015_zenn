# 0003 Review_003 Coder裁定

- Entry_ID: `0003`
- 対象Review: `Review_0003_00_003.md`, `Review_0003_10_003.md`
- 00判定: Pass。正本変更なし。
- 10判定: 要修正（Moderate）を受理。

## Finding裁定

### F10-001 — D01

- Reviewer指摘: 1972年の最古確認を1970年代の生成時期へ変換している。
- Coder裁定: `ACCEPT`
- 理由: 1972年採録は存在確認点であり、生成年代を1970年代へ限定する独立History Evidenceではない。1956年類似句も直接祖型と確定できない。
- 修正: `D01.G3 / D` を撤回し、コードなし・`Status=U` とする。

### F10-002 — D04

- Reviewer指摘: 同時点Variationの並存から `ORAL_VARIATION` / `ACCRETION` という変容機構を強く推定している。
- Coder裁定: `ACCEPT`
- 理由: 1972年と1986年、また1986年内の複数Variationの差異は確認できるが、伝播系列・変容方向・追加過程は固定できない。Variationの存在と生成機構を分離する。
- 修正: `D04.VAR.ORAL_VARIATION` および `D04.VAR.ACCRETION` を撤回し、コードなし・`Status=U` とする。

## 結果

- `0003_00_contents.md`: 変更なし。
- `0003_10_analysis.md`: D01/D04を保守化。
- 修正後10 blob: `5267babadc521d4c5e33c395ea8471cbd396cf35`
- 修正commit: `227d35da9fcc6cce75633543f43e93f16a8c8b6b`
- 次状態: `再レビュー待`（Review_004待ち）。