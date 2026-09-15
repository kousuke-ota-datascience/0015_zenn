# 0025 Review_001 Coder adjudication

- Entry_ID: `0025`
- Entry: 不幸の手紙
- Review source:
  - `docs/99_work/review_10_each_lore/0025/Review_0025_00_001.md`
  - `docs/99_work/review_10_each_lore/0025/Review_0025_10_001.md`

# 1. 00_contents adjudication

## F00-001 / F00-002 — tutorial正規構造

**受理。**

典拠章を `0.2.1–0.2.3`、Content章を `1.1–1.10` へ再配置した。

## F00-003 — 0024とのEntry境界先取り

**受理。**

00から「報酬型／制裁型を別Entryとする」という分析判断を除去し、報酬を含む型・制裁強調型をEvidence上のVariationとして保持した。Entry境界は10のVersion Scopeで定義する。

## F00-004 — sense-making機能評価

**受理。**

「恐怖回避策」「不確実性を制御可能化」等の分析表現をEvidence正本から除き、不転送なら不幸、期限内に転送するという観察可能な文面規則へ戻した。

## F00-005 — 1977年X Evidence

**維持。**

1977年自治体広報の実流通、受信者不安、破棄・派出所相談勧告を直接X Evidenceとして保持し、自殺報道は原記事未確認のため事実認定に用いない。

# 2. 10_analysis adjudication

## F10-001 — 0024とのEntry境界

**受理。**

Version Scopeで、単なる名称差ではなく以下の安定差により0024と分離すると明記した。

- Primary incentive: 幸運獲得ではなく不幸回避
- Primary consequence: `CURSE_MISFORTUNE`
- Control rule: `TRANSFER_SUBSTITUTE`

## F10-002 — D04

**再判定。**

`D04.VAR.RETELLING`を撤回した。1970年「50時間・29人」と1977年「数日・20人」の差は、再叙述一般より**自己複製文面の数値パラメータ変異**とみなす方が適切であり、現行Childに直接対応がないためtaxonomy gapとしてコード空欄・Status `I` で保持する。

## F10-003 — D07〜D18

**維持。**

ReviewerがPASSとした意味形成・因果骨格を維持する。

## F10-004 — D17

**コード維持・根拠限定。**

`TRANSFER_SUBSTITUTE`を維持するが、災厄そのものが他者へ物理的に転嫁されるとは断定しない。受信者が同じ制裁規則を次の受信者へ再付与し、自分は規則遵守によって制裁条件から外れる構造として記述した。

## F10-005 — D18

**維持。**

1977年はVersion Scope内であり、自治体広報が実流通・不安・行政対応を直接記録するため `L3=1` を維持する。

## F10-006 — 書式

**受理。**

Entry_IDを`0025`へ修正し、0180基準のD01〜D21正規書式へ統一した。R4作業ログ・旧10比較はCoding正本から除去した。

# 3. 修正後QA

- 00/10責務分離: PASS
- Entry境界: 10 Version Scopeで明文化
- D04: taxonomy gap（数値パラメータ変異）
- D07〜D18: 基本モデル維持
- D17: `TRANSFER_SUBSTITUTE` 維持、意味範囲限定
- D18: `L1=1; L2=1; L3=1` 維持
- canonical format: PASS

# 4. 修正版checkpoint

- 00 commit: `2ffb49cfa53ec12c2699edd65f6eaa08ff40eff8`
- 00 blob: `c6f9cae7ea5b462faa2ad015bf9d83414c26eca2`
- 10 commit: `3c38c7611873f6179239855f478feac86e7a51ef`
- 10 blob: `3ff11cb289f8f441a8e80c2dc1e62e6da83682c9`

Coder再作業完了。次状態は `再レビュー待`。