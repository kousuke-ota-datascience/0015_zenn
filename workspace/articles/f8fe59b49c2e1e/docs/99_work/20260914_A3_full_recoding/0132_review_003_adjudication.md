# 0132 Review_003 Adjudication

- Entry_ID: `0132`
- 伝承: 八幡の藪知らず
- Review cycle: `003`
- Review対象00 commit: `b6f67d9abbfcde5d5a1410314662d3c2f6526fb1`
- Review対象10 commit: `40fc6f8a9e43f8049d0757db832189e17e5558a0`
- SHA gate: exact match
- Coder correction 00 commit: `085d12b6af6c02d59992877b1b18a63b46488504`
- Coder correction 10 commit: `065552cbd19c7600ed91d014740545b405df0913`
- 最終状態: `再レビュー待 / 003`

## 1. Review_0132_00_003

### F00-001 — tutorialの固定資料フィールド・帰結区分を独自化

- 裁定: **ACCEPT**
- 理由: Review指摘どおり、1749年『葛飾記』のEvidence追加自体は妥当だったが、0.2の資料フィールドを独自名称へ変更し、1.9でtutorial固定3区分以外の分類を前面化していた。
- 対応:
  - `0.2.1` を `資料名 / URL・書誌情報・所蔵情報 / 対象箇所 / 参照確認日 / 証拠上の位置付け` に統一。
  - `0.2.2` を `資料名 / URL・書誌情報 / 対象箇所 / 参照確認日 / 一次資料との関係 / 証拠上の注意` に統一。
  - `0.2.3` を `資料名 / URL・書誌情報 / 参照確認日 / 用途` に統一。
  - `1.9` を `明示された帰結 / 暗示された帰結 / 確認されない帰結` の固定3区分へ復旧。
  - 1749年『葛飾記』、1830年代『江戸名所図会』、1857年石碑、1881年錦絵の時点差は所定フィールド本文へ格納。
- 内容判断: 1749年『葛飾記』をEarliest AttestationとするReview_002対応を維持し、生成時期とは分離した。

## 2. Review_0132_10_003

### F10-001 — D01〜D21のtutorial記載形式が簡略化されている

- 裁定: **ACCEPT**
- 理由: Review指摘どおり、コード内容に新規Majorはなかったが、見出し・中心質問・独立ブロックを簡略化していた。
- 対応:
  - D01〜D21をtutorial所定の見出し・中心質問へ復旧。
  - 各Dimensionを `コード値 → 判定根拠 → この伝承における現れ方` の独立ブロックへ再配置。
  - D03見出しの `（Version Scope）` を含む固定文言を復旧。
  - D01/D02の1749年『葛飾記』根拠を維持し、GenerationとEarliest Attestationを明確に分離。
- コード判断: Review_003で維持可能とされた値を変更しない。主要値は D01=`G0/I`、D02=`BOOK/I`、D04=`CONTESTED_VERSION/I`、D17=`OBEY_TABOO/D`、D18=`1/0/1 / I`、D20=`U`、D21=`A2/D`。

## 3. QA

- 00 tutorial structure: PASS
- 00 fixed source fields: PASS
- 00 section 1.9 fixed three-way outcome structure: PASS
- 00 Earliest Attestation = 1749 『葛飾記』: PASS
- 10 tutorial headings/questions: PASS
- 10 independent rationale / manifestation blocks: PASS
- D01 Generation / D02 Earliest Attestation separation: PASS
- Review_002 substantive corrections preserved: PASS
- 00→10 traceability: PASS

## 4. Coder conclusion

Review_003の2 FindingをともにACCEPTし、format正規化を完了した。1749年Evidenceを含む前cycleの実質判断は維持する。Reviewerによる次Reviewまで `再レビュー待 / 003` とする。
