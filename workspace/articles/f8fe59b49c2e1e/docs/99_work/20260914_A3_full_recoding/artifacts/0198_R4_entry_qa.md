# 0198 R4 Entry QA

- R3 freeze: `ebc2fc8400e97feaa85c5bfecb5be7b39ae821e3`
- 旧 `0198_10_analysis.md`: 存在せず。

## QA結果

### 起源・Scope
- inventoryの「2006〜」は直接Evidence不足のため不採用。
- 2007-04-18の名称確認、4/21の具体手順・実況・補足を初期形成Scopeとする。
- 2007年以前の古来伝承説、大学研究説等を前史として推定しない。

### Coding
- D04 `LIVE_COCREATION` + `ACCRETION`: 実況中の記載漏れ・終了法追加・別投稿者による補足を直接確認できる。
- D05 `COLLAB_THREAD` + `LIVE_1P`: 単独怪談ではなく複数実践者・閲覧者の相互作用が構造的。
- D09: 異常報告を霊的主体へ帰属するため `AGENCY_ATTRIBUTION` Primary。
- D10: 「霊」「もう一人」という解釈を踏まえ `GHOST_SPIRIT` Primary、ぬいぐるみは媒介物としてSecondary。
- D11: 読むことではなく儀式実行が発動条件。`PERFORM_RITUAL` Primary。
- D13: 初期ログで確定するのは音・声・人影等の顕現であり、身体攻撃・死亡を必須化しない。`MANIFEST_ONLY` Primary。
- D15: 実現帰結は恐怖。死亡・重傷は採用しない。
- D17: 明示的な終了法があり `RITUAL_CLOSURE` Primary。順序・操作遵守をSecondary。
- D20: R3 Secondary `SUBCULTURE_VETERAN` は推論が強いため削除し、`PER.EXPERIENCER` 単独へ保守化する。初期手順知識は「聞いた／実践した」人物に置かれるが、専門制度的保持者は確認できない。

### D18
- L1=1: 儀式後に霊的・異常現象が起こるという伝承内因果。
- L2=0: 手順を読む／知るだけでは怪異の作用対象にならない。
- L3=1: 同一初期スレッド内で、提示された手順を読んだ別参加者が実際に自宅で儀式を開始し実況する投稿を確認できる。これは伝承受容が現実行動を生じさせた独立X Evidenceであり、単なる流通件数ではない。

### H3 QA
Primary exactly 1、Secondary 0〜2、Parent derived、StatusとChildを分離。

## 結論
R3を基本維持し、D20 Secondaryのみ削除して新規Coding正本へ採用する。外部00/10 Review前のため `レビュー待`。
