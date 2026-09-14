# 0001 Review_002 Coder裁定

- Entry_ID: `0001`
- 対象Review: `Review_0001_00_002.md`, `Review_0001_10_002.md`
- 00判定: Pass（修正なし）
- 10判定: 要修正（Major）

## 裁定

### D10 因果源存在論

Reviewer指摘を受理する。旧値 `D10.HUM.INDIVIDUAL_HUMAN` は、「妖怪・幽霊を固定できない」ことを「通常の人間である」ことへ読み替えており、Scope Evidenceより強かった。

修正後はコードを置かず `Status=U` とする。初期Scopeでは名前付き女性主体までは固定できるが、通常人間・妖怪・死者霊等の存在論を一意に確定しない。

### D18 作用レイヤー

Reviewer指摘を受理する。1979年6月15日の登校拒否Evidenceは、1979年初頭〜春というVersion Scopeより後続であるため、Scoped D18のL3根拠には使用しない。

修正後は `D18.L1=1; D18.L2=0; D18.L3=0`。6月の登校拒否は00側の後続流通史・X Evidenceとして保持する。

## 修正結果

- 00: 変更なし（Review_002 Pass）
- 10: commit `6285afdaaeafac0d79ddbc429bd1fc771a3ad2f5`
- 10 blob: `29fa0f28bd3f3a6b51680c990d5572f2b7edfd51`
- 次状態: `再レビュー待`
- 次Review: `Review_003`
