# 0132 Review_004 adjudication

- Entry_ID: `0132`
- Review cycle: `004`
- Review対象00 blob: `1029bd541c2a5b38638dbf86fdca7ec4de08f3d6`
- Review対象10 blob: `953b9377aab847ec8b8842470dde62b89815a327`
- SHA gate: exact match
- Corrected 00 commit: `69065c135d4cc22331abf59d5e4d85e6a4073fbd`
- Corrected 00 blob: `02eac89640e67b6593522eb243f7b77c02bc9892`
- Corrected 10 commit: `d913f0ff43fec59f1098a4edbcead1bbbe752e69`
- Corrected 10 blob: `cc7914ea9e891e7a8359aa3ccea3938a6b7e6f77`
- 最終状態: `再レビュー待 / 004`

## Findings adjudication

### Review_0132_00_004
- F00-001: **ACCEPT**。『葛飾記』『江戸名所図会』、1857年石碑、1881年作品を「歴史上の原資料」として示しつつ、今回の実アクセスEvidenceは公立図書館の抄録・復刻案内、自治体・図書館解説であることを0.2で分離。0.2.1では原本直接固定なしを明示した。
- F00-002: **ACCEPT**。3.2=`短い原文引用`、3.3=`証拠上の保留事項`へ修正。

### Review_0132_10_004
- F10-001: **ACCEPT**。D13〜D21の中心質問と2.4節見出しをtutorial所定表記へ復元。
- F10-002: **ACCEPT**。D19 Secondary `NATIONAL_PUBLIC`を撤回し、`LOCAL_TRADITION / I`に限定。
- 00のEvidence階層整理に連動し、歴史Contentの原本未実見をD05/D07/D08/D11〜D17等のStatus `I`へ保守的に反映。D21は現代自治体資料で具体地点を直接固定できるため`D`を維持。

## QA

- earliest attestation 1749 ≠ origin を維持。
- 原資料名と実アクセス資料のdirectnessを分離。
- 由来諸説を単一史実へ統合しない。
- D19は媒体アクセス可能性から全国的大衆流通を推定しない。
