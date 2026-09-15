# 0158 Review_001 Adjudication

- Entry_ID: `0158`
- Entry: 虚舟
- Review sequence: `001`
- Review 00 target blob: `abea52522f465b4252eb772de989428e584ebbd0`
- Review 10 target blob: `8c7ca4ba077d0d9867b7a02a50aa7bcafd066c5b`
- Corrected 00 commit: `e6a656cbc30e03c8e4620a77aea49b2fdd989753`
- Corrected 00 blob: `969be78ca449264dba70641e8e05769a2f8d1261`
- Corrected 10 commit: `b922d5297e56811cfcd3af4e7fc1fb6aeccd11d9`
- Corrected 10 blob: `c2c250ff1790b37314b54c327c2cbbaf7d5854be`

## 1. SHA gate

Review対象blobと修正開始時のmain正本blobは00/10ともexact match。Review_001は現行正本へ適用可能と判定した。

## 2. 00 findings adjudication

### F00-001 書誌情報を詳細Content Evidenceとして使用

**採用。** NDL書誌中心の構造を撤回し、国立公文書館所蔵『弘賢随筆』「うつろ舟の蛮女」を主要Evidenceへ置換した。1803年常陸国漂着、舟の形状、女性、箱、記号、沖へ戻す終端を資料位置付きで再構築した。

### F00-002 3.1 traceability不足

**採用。** Content要素ごとに典拠・位置・Evidence roleを分解し、女性・箱・記号・漂着・返送等を個別に追跡可能にした。

### F00-003 1.3必須列不足

**採用。** tutorial指定の「典拠上確認できる情報」「典拠位置」を追加した。

### F00-004 2.4変容機構の分析先取り

**採用。** 江戸期資料と近現代UFO解釈の差をHistory Evidenceとして記録し、「再符号化」等の分析裁定を00から除去した。

## 3. 10 findings adjudication

### F10-001 00 Evidence不足

**採用。** 00を国立公文書館所蔵資料・研究資料で再構築した後、そのEvidenceだけでD01〜D21を再判定した。

### F10-002 D07/D08の複合Status

**採用。** D07=`I`、D08=`I` の単一Statusへ正規化した。

### F10-003 D10 Conflict不適切

**採用。** `ANOMALOUS_EXPERIENCE` vs `ARTIFACT_DEVICE` のConflictを撤回し、D10を `D10.OBJ.ARTIFACT_DEVICE` Primary、`D10.HUM.INDIVIDUAL_HUMAN` Secondary、Status `I` とした。

### F10-004 D17 OBSERVATIONAL_ONLY過剰推論

**採用。** 海へ戻す行為は単発事件対応であり一般化可能な制御規則ではないため、D17=`U`。

### F10-005 D20 NO_ONE_KNOWS過剰推論

**採用。** 「保持者を確認できない」を積極的な「誰も知らない」へ変換せず、D20=`U`。

### F10-006 tutorial形式

**採用。** 階層番号・中心質問をtutorial準拠へ復元した。

## 4. 横断QA

- 00の主要Contentは国立公文書館所蔵資料へtrace可能。
- D07/D08を含め複合Statusなし。
- D10は具体的な舟・女性の存在論へ統一され、Conflict乱用なし。
- D13=`MANIFEST_ONLY` は、具体的な攻撃・呪詛等を固定できず、出現・観測そのものが作用であるというEvidenceに対応。
- D16=`SEQUENTIAL_EPISODE` は漂着→観察→推測→対応→沖へ戻す段階進行に対応。
- D17/D20はEvidence欠如を強い否定コードへ変換していない。
- D18=`1/0/0 / I` は研究・展示を社会現実効果へ自動写像していない。

## 5. 結論

Review_001指摘対応は完了。00/10とも `再レビュー待 / 001` へ移行可能。Reviewer Pass前のため `完了` にはしない。