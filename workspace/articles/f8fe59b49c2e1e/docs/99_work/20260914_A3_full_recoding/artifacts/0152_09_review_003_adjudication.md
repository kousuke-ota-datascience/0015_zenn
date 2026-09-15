# 0152 Review_003 Coder Adjudication

- Entry_ID: `0152`
- 伝承: 青木ヶ原樹海で方位磁針が狂う
- Review_Seq: `003`
- 対象Review: `Review_0152_00_003.md`, `Review_0152_10_003.md`
- adjudication日: `2026-09-16`
- Coder結論: Review_003の全Findingを採用し、00/10正本を修正。再レビュー待へ移行する。

## 1. SHA gate

Review_003対象blobと修正前正本blobの一致を確認した。

- 00 target/current before fix: `38f4ddb565b127c352f16bd9977b71b19d5068bc`
- 10 target/current before fix: `31298df0d1c68003d4776618c1bcc5de093f392d`

SHA gate: PASS。

## 2. Review_0152_00_003

### F00-001 — 1.6 / 1.9 の固定フォーマット不一致

- 判定: **ACCEPT**
- 理由: 修正前00は1.6の所定5フィールドと1.9の所定3区分を厳密には保持していなかった。
- 修正: 1.6を `条件 / 規則 / 禁忌 / 対処・回避 / 利用法`、1.9を `明示された帰結 / 暗示された帰結 / 確認されない帰結` へ整形した。2.3の列名、2.4見出しもtutorialへ統一し、科学的反証と伝承Contentの責務分離は維持した。
- correction commit: `8a5b1eef3941344534b2924888cbf2eb0c8b59cc`
- corrected blob: `76f4274e231776c8032b144c218e0a241753ae3f`

## 3. Review_0152_10_003

### F10-001 — D19 `NATIONAL_PUBLIC / D` のDirect Evidence不足

- 判定: **ACCEPT**
- 理由: 全国放送番組・一般書籍・公開Webの存在は到達可能性や媒体種別を示すが、全国的大衆への実流通範囲をDirectには固定しない。
- 修正: D19を Primary `D19.NET.OPEN_FORUM_WEB`、Primary Parent `D19.NET`、Secondaryなし、Status `D` へ変更した。2007年Web記録と2009年Yahoo!知恵袋は公開Web上での実流通を直接確認できるためこの範囲ではDとする。`NATIONAL_PUBLIC` は撤回した。
- correction commit: `ff46a9995446dcd08ddcb008c152178234befc5b`
- corrected blob: `02f680bbec38678cf6b884388ed2a75b41faf7b1`

## 4. QA

- tutorial 1.6 / 1.9 / 2.3 / 2.4: PASS
- Evidence / Analysis責務分離: PASS
- scientific debunking vs lore Content separation: PASS
- D17 external safety advice not imported: PASS
- D18 zero-bit handling: PASS (`1/0/1`, Status I)
- D19 actual circulation scope: PASS
- D20 external researchers not treated as lore-internal privileged holders: PASS

## 5. Handoff

Coder correction complete. Reviewer approval前のため `完了` にはせず、`再レビュー待 / 003` とする。