# 0157 Review_001 Adjudication

- Entry_ID: `0157`
- Entry: 新郷村キリストの墓
- Review sequence: `001`
- Review 00 target blob: `5ddf78132729d5592ee4d43b2296b57ff3f52be5`
- Review 10 target blob: `98511d298cdbd560c3569aea07f5f2c0a1c986ae`
- Corrected 00 commit: `b81a1b4cbb38c1b13d0c67caec028ebc6a1495c9`
- Corrected 00 blob: `7ebed15dff9ae0712515346bd470b22c4a4c2a58`
- Corrected 10 commit: `5fec5c6feb9972cd522e59aea8518895e39369d4`
- Corrected 10 blob: `8d7c22cea63d899e57a4be3bfa558f73fd8e7f91`

## 1. SHA gate

Review対象blobと修正開始時のmain正本blobは00/10ともexact match。Review_001は現行正本へ適用可能と判定した。

## 2. 00 findings adjudication

### F00-001 tutorialの典拠区分・必須フィールド

**採用。** `0.2.1` をtutorial正規の「一次資料・同時代資料」へ戻し、各資料に資料名、URL/書誌、対象箇所、確認日、証拠上の位置付けを付与。1.3〜1.5と3.1もtutorial指定列へ整形した。

### F00-002 1.7の意味付与操作先取り

**採用。** 「通説的なキリスト伝と日本地域史を別の因果系列へ再構成」という分析文を00から除去し、身代わり、再渡来、戸来居住、死亡、墓比定というContentだけを記録した。

### F00-003 2.4の変容・社会効果分析

**採用。** 墓比定、伝承館、祭礼、自治体提示をHistory Evidenceとして時点別に記録し、「制度化」「相互補強」等の分析裁定を00から除去した。

## 3. 10 findings adjudication

### F10-001 複合Status D/I

**採用。** 全Dimensionを単一の `D/I/U/NA/C` へ正規化した。

### F10-002 D11〜D15が受容者モデル

**採用。** 受容者の「知る→信念変更」を撤回し、伝承内部を、身代わり後の再渡来→戸来での居住・生涯→死亡→墓という系列へ再構成した。

- D11: `D11.MOV.ENTER_PASS_CROSS / I`
- D12: `D12.FOC.PROTAGONIST_EXPERIENCER / I`
- D13: code blank / `I`; taxonomy gap = 通常の移住・定住・生涯経路形成
- D14: `D14.NEU / I`
- D15: `D15.BOD.DEATH / I`
- D16: `D16.EVT.SEQUENTIAL_EPISODE / I`

D13はEvidence不足ではなく、現行taxonomyに通常の移住・定住作用を表すChildがないためgapとして記録した。`REALITY_REPLACEMENT`への近似押込めは行わない。

### F10-003 D10 Conflict

**採用。** D10を `D10.HUM.INDIVIDUAL_HUMAN / I` に一本化。キリスト本人を伝承内部主体とし、竹内文書はEvidence/cueとしてD08側へ分離した。

### F10-004 D06 Conflict

**採用。** 現在の新郷村公式提示に基づき `D06.T6 / I`。媒体間の提示差をDimension-level Conflictへ拡大しない。

### F10-005 D20 特権情報保持者

**採用。** 資料保管者・自治体を真相保持者へ変換せず、D20=`U`。

### F10-006 tutorial形式

**採用。** `2.1.1`〜`2.4.3` の階層番号と中心質問を復元した。

## 4. 横断QA

- 00はEvidence/Content責務に限定され、Dコード分析を先取りしていない。
- 10のD11→D16は同一の伝承内部world modelで連続する。
- D18=`1/0/1 / I` は、伝承内系列、受容自体の非作用、墓・祭礼・伝承館という現実実践を分離している。
- D20は資料アクセス優位と伝承内特権知識を混同していない。
- 複合Statusなし。

## 5. 結論

Review_001指摘対応は完了。00/10とも `再レビュー待 / 001` へ移行可能。Reviewer Pass前のため `完了` にはしない。