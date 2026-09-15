# 0118 Review_003 Adjudication

- Entry_ID: `0118`
- 伝承: 函館山の切れない木
- Review cycle: `003`
- Review対象00 commit: `3ad16ee65d5b2e442aad287fbc7e79654dbc982e`
- Review対象10 commit: `3ab7084ea196b607b254859436b32bc9d27424a1`
- SHA gate: exact match
- Coder correction 00 commit: `f5d9e6f3f4abe8c7fda23a058a318d9fc366f07d`
- Coder correction 10 commit: `ce51b3ab96cb91a2f9ba031cede4d25895bdbd8b`
- 最終状態: `再レビュー待 / 003`

## 1. Review_0118_00_003

### F00-001 — tutorial必須フォーマットの複数不一致

- 裁定: **ACCEPT**
- 理由: Review指摘どおり、内容上のEvidence修正は前cycleで達成していたが、0.2の資料フィールド、1.9の固定3区分、2.3の5列表をtutorialと異なる形へ簡略化していた。
- 対応:
  - `0.2.1` を `資料名 / URL・書誌情報・所蔵情報 / 対象箇所 / 参照確認日 / 証拠上の位置付け` の固定フィールドへ復旧。
  - `0.2.2` を `資料名 / URL・書誌情報 / 対象箇所 / 参照確認日 / 一次資料との関係 / 証拠上の注意` へ復旧。
  - `0.2.3` を `資料名 / URL・書誌情報 / 参照確認日 / 用途` へ復旧。
  - `1.9` を `明示された帰結 / 暗示された帰結 / 確認されない帰結` の3区分へ復旧。
  - `2.3` にtutorial所定の5列表を配置し、直接Content Evidenceを固定できない事故具体化型・道路回避型は主要異伝として採用しないことを明記。
- 内容判断: 前cycleのEvidence縮退方針を維持。新たな事故細部・道路回避型は追加していない。

## 2. Review_0118_10_003

### F10-001 — D01〜D21の記載形式がtutorial所定形式に一致しない

- 裁定: **ACCEPT**
- 理由: Review指摘どおり、コード値自体の主要問題は解消済みだったが、中心質問の短縮、`判定根拠` と `この伝承における現れ方` のインライン化によりformat gateを満たしていなかった。
- 対応:
  - D01〜D21をtutorialの所定見出し・中心質問へ復旧。
  - 各Dimensionを `コード値 → 判定根拠 → この伝承における現れ方` の独立ブロックへ再配置。
  - D03見出しの `（Version Scope）` を含む固定文言を復旧。
  - D10 taxonomy gapはD10内部の留保として保持し、近似Childへ押し込まない。
- コード判断: Review_003で維持可能とされた内容を変更しない。主要値は D05=`JINX_RULE/I`、D09=`DIRECT_CAUSE/I`、D10=taxonomy gap/I、D17=`U`、D18=`1/0/0 / I`、D19=`LOCAL_TRADITION/I`、D20=`U`。

## 3. QA

- 00 tutorial structure: PASS
- 00 fixed source fields: PASS
- 00 section 1.9 fixed three-way outcome structure: PASS
- 00 section 2.3 five-column table: PASS
- 10 tutorial headings/questions: PASS
- 10 independent rationale / manifestation blocks: PASS
- Review_002 substantive corrections preserved: PASS
- 00→10 traceability: PASS

## 4. Coder conclusion

Review_003の2 FindingをともにACCEPTし、format正規化を完了した。内容上の再裁定は行っていない。Reviewerによる次Reviewまで `再レビュー待 / 003` とする。
