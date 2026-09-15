# 0152 Review_001 Coder裁定

- Entry_ID: `0152`
- 対象Review: `Review_0152_00_001.md`, `Review_0152_10_001.md`
- 修正前00 blob: `e8de25b6d8dd8c89267fece918c3ea15b6f0aa3a`
- 修正前10 blob: `1b324b655298fe5cea0fab84f7bcc52e9cf69808`
- 00判定: 要修正（Major）を受理。
- 10判定: 要修正（Major）を受理。

## 1. 00 Finding裁定

### F00-001 — INTRODUCTIONで因果鎖を統合

- Coder裁定: `ACCEPT`
- 理由: 実在地質、磁気説明、方位磁針異常、道迷いを一つの因果鎖として統合することはD09〜D15のAnalysis責務。
- 修正: INTRODUCTIONでは伝承命題、自然科学上の地質事実、検証・反証結果を資料層として分離した。

### F00-002 — summaryで「伝承の核」を意味形成モデルとして裁定

- Coder裁定: `ACCEPT`
- 理由: 「磁気異常→機能不全→経路判断不能」を伝承の意味形成核として確定するのはCoding責務。
- 修正: summaryは、方位磁針が使えないという流通命題、溶岩・地質Anchor、現地測定・大学解説、派生機器、明示帰結をEvidenceとして列挙する形へ戻した。

### F00-003 — 伝承史2.4で変容機構を裁定

- Coder裁定: `ACCEPT`
- 理由: 「疑似科学的説明の増補」「反証を伴う再流通」はD04の分析概念であり00では時系列事実と差分だけを記録すべき。
- 修正: 2.4を「時系列上確認できる差分」とし、2007年時点の派生要素・反証資料の併存だけを記録した。

## 2. 10 Finding裁定

### F10-001 — Entry_IDゼロ埋め違反

- Coder裁定: `ACCEPT`
- 修正: `152` を `0152` へ修正した。

### F10-002 — D15 `AVOIDANCE_ROUTE_CHANGE` の定義不一致

- Coder裁定: `ACCEPT`
- 理由: `D15.BEH.AVOIDANCE_ROUTE_CHANGE` は「場所や行動を避ける」帰結であり、伝承上の終端である方向・帰還経路の判断不能を表さない。現行taxonomyでは `D15.MND.MEMORY_COGNITION` が「記憶喪失、混乱等」を含み、方向認知の混乱に最も近い。
- 修正: D15を `D15.MND.MEMORY_COGNITION / I` へ変更し、死亡・永久失踪は付加しなかった。

### F10-003 — D20が外部科学者を特権情報保持者としている

- Coder裁定: `ACCEPT`
- 理由: 科学者が俗説の真偽を外部から測定できることと、伝承世界内で秘密・追加情報を特権的に保持することは別。
- 修正: D20をコードなし・`Status=U` とした。

### F10-004 — D18 vector Status `D` が過強

- Coder裁定: `ACCEPT`
- 理由: L1とL3には正Evidenceがあるが、L2=0は不在判断である。
- 修正: `L1=1; L2=0; L3=1` を維持し、Statusを `I` へ変更した。

### F10-005 — D08 `UNSUPPORTED_ASSERTION` がEvidence不足をcueへ変換

- Coder裁定: `ACCEPT`
- 理由: 研究上の裏付け不足は受容者が受け取る手掛かりではない。実際に追跡できる受容入口はテレビ・Web等で提示された既成命題と青木ヶ原の実在地・地質情報である。
- 修正: Primaryを `D08.CLM.IMPORTED_MEDIA_CLAIM`、Secondaryを `D08.HIS.PLACE_NAME_RUIN` とした。

### F10-006 — D04 Secondary `ACCRETION` の時系列directness不足

- Coder裁定: `ACCEPT`
- 理由: 2007年資料にGPS等の派生が存在することは確認できるが、それらが中心命題より後に増補された過程を連続したHistory Evidenceで固定できない。
- 修正: D04 Primary `D04.CON.DEBUNK_RECIRCULATION / I` を維持し、Secondary `ACCRETION` を撤回した。

## 3. 結果

- `0152_00_contents.md`: Content Evidenceと科学的検証を責務分離。
  - 修正後blob: `38f4ddb565b127c352f16bd9977b71b19d5068bc`
  - 修正commit: `a8231c662de9ead82c1cdbc5ef464da61d9daa06`
- `0152_10_analysis.md`: D08/D15/D18/D20等をReviewに従い再裁定。
  - 修正後blob: `31298df0d1c68003d4776618c1bcc5de093f392d`
  - 修正commit: `d8cd093e4b026f23e644320766da620d07f385cd`
- 次状態: `再レビュー待 / 001`（Review_002待ち）。