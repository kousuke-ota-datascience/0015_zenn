# 0137 Review_003 Coder Adjudication

- Entry_ID: `0137`
- 伝承: 犬鳴村
- Review_Seq: `003`
- 対象Review: `Review_0137_00_003.md`, `Review_0137_10_003.md`
- adjudication日: `2026-09-16`
- Coder結論: Review_003の全Findingを採用し、00/10正本を修正。再レビュー待へ移行する。

## 1. SHA gate

Review_003対象blobと修正前正本blobの一致を確認した。

- 00 target/current before fix: `b2df0647a8d8480b5357b06a6020a171f4fc26ef`
- 10 target/current before fix: `42c5d41e1ae8104f71db48d56f2971b537ccd86e`

SHA gate: PASS。

## 2. Review_0137_00_003

### F00-001 — 原ページ未固定資料を一次・同時代資料として分類

- 判定: **ACCEPT**
- 理由: 1999年元Webページを直接固定できていない以上、研究論文・後代保存転載による確認を一次・同時代Evidenceとして扱うのはsource role不整合である。
- 修正: 1999年Web記録を0.2.1から外し、0.2.2に後代保存転載として位置付け、原ページ・保存URL未固定と明記した。鳥飼研究は0.2.3の二次研究資料として保持し、1999年を「研究・保存記録から確認する早期定点」として扱う。
- correction commit: `0d811ede94f373d3058e121aa45cd91795eb1421`
- corrected blob: `82bcd114d2bf681bf1554a487bb4a26ffe6d20f4`

### F00-002 — 1.6がtutorial固定フィールド形式でない

- 判定: **ACCEPT**
- 理由: 修正前は条件・規則・禁忌等を一段落へまとめていた。
- 修正: 1.6を `条件 / 規則 / 禁忌 / 対処・回避 / 利用法` へ分解した。併せて1.9をtutorial所定3区分、2.3を所定5列、2.4を `変容上の重要点` へ統一した。
- correction commit: `0d811ede94f373d3058e121aa45cd91795eb1421`
- corrected blob: `82bcd114d2bf681bf1554a487bb4a26ffe6d20f4`

## 3. Review_0137_10_003

### F10-001 — D19 `NATIONAL_PUBLIC / D` が流通範囲を過大評価

- 判定: **ACCEPT**
- 理由: 全国公開映画・一般書籍・公開Webの存在は媒体到達可能性を示すが、Version Scopeの伝承が全国的大衆へ実際に受容されたことを直接固定しない。
- 修正: D19を Primary `D19.NET.OPEN_FORUM_WEB`、Secondary `D19.LOC.LOCAL_TRADITION`、Status `I` へ変更し、`NATIONAL_PUBLIC` を撤回した。00のsource role修正に合わせ、原Web未固定に依存するD12/D14等のdirectnessも保守化した。
- correction commit: `b31402dd45a0dddde6f2de1fb41eef4045990327`
- corrected blob: `77170de70e82b35c2c63c3a85df438245bbad586`

## 4. QA

- 1999 source role / directness consistency: PASS
- tutorial 1.6 / 1.9 / 2.3 / 2.4: PASS
- 00→10 traceability: PASS
- D17 trigger-inversion avoidance: PASS
- D18 zero-bit handling: PASS (`1/0/0`, Status I)
- D19 actual circulation scope: PASS
- real old settlement vs legendary village separation: PASS

## 5. Handoff

Coder correction complete. Reviewer approval前のため `完了` にはせず、`再レビュー待 / 003` とする。