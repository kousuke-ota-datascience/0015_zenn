# 0250 Review_001 Adjudication

- Entry: `0250 七人ミサキ`
- Review seq: `001`
- Review targets: `Review_0250_00_001.md`, `Review_0250_10_001.md`
- Result: **全Finding ACCEPT / 修正済み**

## 1. SHA gate

Review対象blobと修正着手時の正本は一致していた。

- 00 target blob: `d35a65204e134c5acf181b3fbf80ea9317d3fa30`
- 10 target blob: `742fee4a8e43332f27181a5db8bbb7a18596450e`

作業中に00へ並行更新が入り、GitHub更新時に409を検知したため旧blobへ上書きしなかった。現行00を再取得するとReview_001の00 Findingを解消する内容へ更新済みだったため、その新版を維持した。

## 2. 00 Findings

### F00-001 — tutorial正規構造を大幅に欠く
**ACCEPT.**

現行00を `#0〜#3`、`0.2.1〜0.2.3`、`1.1〜1.10`、`2.1〜2.4`、`3.1〜3.3` のtutorial構造へ全面再配置済み。

### F00-002 — tutorial必須内容・記載粒度不足
**ACCEPT.**

主体、場所、物、条件・規則、明示／暗示／未確認帰結、主要異伝5列表、Content→source→position対応表を追加済み。

### F00-003 — Version Scopeを00で裁定
**ACCEPT.**

「共有核として採用」等の分析裁定を除去し、1927、1928、1935、1938、1943年の各採録が何を記録するかを資料単位で保持した。

### F00-004 — 中心機構を00で確定
**ACCEPT.**

交代・成仏・被害を各採録のContentとして記録し、変容機構・中心因果の裁定は10へ分離した。

## 3. 10 Findings

### F10-001 — tutorial形式不足
**ACCEPT.**

D01〜D21を所定の中心質問、コード値、判定根拠、「この伝承における現れ方」の独立ブロックへ全面展開した。

### F10-002 — 複合Status
**ACCEPT.**

D10/D14/D18/D19を含め、各Dimensionを単一の `D/I/U/NA/C` に統一した。

### F10-003 — D01が最古確認を生成時期へ変換
**ACCEPT.**

D01を `U` に変更。1935〜1938年は循環型の確認時点であり生成時点ではないと明記した。

### F10-004 — D04が異伝並存から変容を推定
**ACCEPT.**

D04を `U` に変更。異伝差は確認するが、版間の時系列的再話・増補経路を固定しない。

### F10-005 — D20 LOCAL_RESIDENT
**ACCEPT.**

D20を `U` に変更。地域伝承の採録と、伝承内部の特権情報保持者を区別した。

### F10-006 — D06 T3 directness
**ACCEPT.**

D06を `U` に変更。採録者による「地域伝承」ラベルだけから共同体既知事実T3へ引き上げない。

## 4. 維持した主要判断

- D02/D03: `MAGAZINE / I`
- D09: `RECURRENCE_RULE` + `DIRECT_CAUSE / I`
- D10: `GHOST_SPIRIT / I`
- D13: `PERSON_TRANSFER` + `TARGETING / I`
- D14: `NEG / I`
- D15: `DEATH` + `IDENTITY_TRANSFORMATION / I`
- D16: `CHAIN_SPREAD` + `SUCCESSIVE_VICTIMS / I`
- D17: `U`
- D18: `1/0/0 / I`
- D19: `LOCAL_TRADITION` + `REGIONAL / I`
- D21: `A2 / I`

## 5. QA

- earliest attestation ≠ origin: PASS
- 1927年型と1935年以後循環型の分離: PASS
- 00 Evidence / 10 Analysis責務分離: PASS
- tutorial固定構造: PASS
- D01〜D21存在: PASS
- 複合Statusなし: PASS
- D20特権情報構造の過剰推定なし: PASS

## 6. correction checkpoints

- 00 correction commit: `6952caa9e115c2c771a47a48400006d45068369b`
- 00 current blob: `52812650b26c0958a429ec6e2a80b28f37934509`
- 10 correction commit: `b6a3e996ff8cc08594ba2ecf1ef443e8b4f5340f`
- 10 current blob: `baba62aa783a682155a93c2fea004db9db5acb1d`

Reviewer Pass前のため、Entry状態は `再レビュー待 / 001` とする。
