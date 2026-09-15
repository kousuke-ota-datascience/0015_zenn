記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

Evidence正本は同一ディレクトリの `0118_00_contents.md`、R1/R2/R3/R4監査成果物は `docs/99_work/20260914_A3_full_recoding/` 配下を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0118`
- `伝承エントリ名称`: 函館山の切れない木
- `Macro_Category`: 場所・祟り・禁忌
- `Entry_Type`: 地域怪談／場所禁忌
- `Version_Scope`: 函館山の登山道路周辺にある特定の木について、伐採・撤去を試みると関係者に事故・不幸・災いが起きるため、木が残されているとする地域怪談。道路が木を避けたという説明は伝承内異伝として扱い、史実認定しない。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示

### D01 生成年代
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`
- 判定根拠: 1980年代成立を直接裏付ける資料はなく、安全な成立年代を固定できない。
- この伝承における現れ方: 2000年代には既知対象として流通していた可能性があるが、成立時期は未確定。

### D02 最古確認流通媒体
- Primary Child / Value: `D02.WEB.WEBSITE` — Webサイト
- Primary Parent: `D02.WEB`
- Secondary: なし
- Status: `D`
- 判定根拠: 2008年掲示板由来記録は現在参照できるものが転載であり原ログを直接固定できないため、R1ではH補助に限定した。直接確認できる最古の安全な流通定点として2016年LIFULL HOME'S Web記事を採る。
- この伝承における現れ方: 一般向け都市伝説調査のWeb記事で中心命題が直接再提示される。

### D03 確認流通媒体ポートフォリオ
- Primary Child / Value: `D03.WEB.WEBSITE` — Webサイト
- Primary Parent: `D03.WEB`
- Secondary: `D03.AVD.VIDEO_SITE` — 動画サイト; `D03.PRT.BOOK` — 書籍
- Status: `D`
- 判定根拠: 2016年Web、2019年映像作品、2023年書籍収録を直接確認できる。
- この伝承における現れ方: 地域怪談がWeb・映像・出版へ媒体横断的に再提示される。

### D04 生成・変容パターン
- Primary Child / Value: `D04.STB.STABILIZED_CANON` — 定型化・カノン化
- Primary Parent: `D04.STB`
- Secondary: `D04.MED.CROSS_MEDIA` — クロスメディア化
- Status: `I`
- 判定根拠: 「伐採しようとすると災いが起きるため木が残る」という核は複数資料で安定する。複数媒体での再提示は確認できるが、時系列的な媒体移行過程まで十分に再構成できないためMEDIUM_SHIFTとはしない。
- この伝承における現れ方: 事故の種類や道路回避説明は揺れても、識別核を保ったまま複数媒体へ展開する。

### D05 提示形式
- Primary Child / Value: `D05.RUL.TABOO` — 禁忌
- Primary Parent: `D05.RUL`
- Secondary: `D05.HRS.LOCAL_HEARSAY` — 地元伝聞
- Status: `I`
- 判定根拠: 実践核は「その木を切ってはいけない／切ろうとすると災いが起きる」という禁止規則で、由来は地域伝聞型として語られる。
- この伝承における現れ方: 過去の事故談が現在の「木に手を出すな」という禁止境界を成立させる。

### D06 真実性提示
- Primary Child / Value: `D06.T5` — 条件付き信念
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 事故史は未確認だが、木が残る景観と怪談が「切ると災いがあるかもしれない」という信念を支える。
- この伝承における現れ方: 事実断定まで要求せず、念のため切らないという行動を合理化する程度の本当らしさで機能する。

## 2.2. 意味形成

### D07 意味形成対象
- Primary Child / Value: `D07.PSE.ENVIRONMENTAL_ANOMALY` — 環境異常
- Primary Parent: `D07.PSE`
- Secondary: `D07.PSE.DANGEROUS_PLACE` — 危険・怪異場所
- Status: `I`
- 判定根拠: 00が意味形成上明示する問いは「通常なら工事で除去されそうな木がなぜ道路周辺に残っているのか」という景観上の不可解さ。その説明結果として木・場所が危険属性を帯びる。
- この伝承における現れ方: 現存する木や道路配置が、単なる施工事情ではなく怪異の痕跡として読み直される。

### D08 意味形成契機
- Primary Child / Value: `D08.TRC.PHYSICAL_TRACE` — 物的痕跡
- Primary Parent: `D08.TRC`
- Secondary: `D08.CLM.UNSUPPORTED_ASSERTION` — 根拠未提示の主張
- Status: `I`
- 判定根拠: 木が現在も残り、道路が避けるように見える景観が物的手掛かりとなり、「伐採すると事故が起きた」という未確認主張が接続される。
- この伝承における現れ方: 目の前の景観を過去の祟りの痕跡として読むことで怪談が補強される。

### D09 意味付与操作
- Primary Child / Value: `D09.NOR.TABOOIZATION` — 禁忌化
- Primary Parent: `D09.NOR`
- Secondary: `D09.CAU.DIRECT_CAUSE` — 直接原因化
- Status: `I`
- 判定根拠: 「木を切る→災いが起きる」という因果を与え、通常の伐採・撤去を「してはいけない行為」へ変換する。
- この伝承における現れ方: 景観の不可解さが「過去に切ろうとして災いが起きたため、以後切ってはいけない」という禁止規則で閉じられる。

### D10 因果源存在論
- Primary Child / Value: `D10.SPC.SPECIFIC_PLACE` — 特定場所
- Primary Parent: `D10.SPC`
- Secondary: `D10.SUP.IMPERSONAL_CURSE` — 非人格的呪力
- Status: `I`
- 判定根拠: 因果は函館山の特定の木／地点へ固定され、安定した人格霊主体は確認できない。場所・対象に付着した非人格的な祟りとして最も安全に表現できる。
- この伝承における現れ方: 同じ伐採行為でも、この木に手を加えた場合だけ災いが起きるとされる。

## 2.3. 因果・行動モデル

### D11 発動・接触条件
- Primary Child / Value: `D11.MAN.CREATE_ALTER_MANIPULATE` — 作る・加工・操作する
- Primary Parent: `D11.MAN`
- Secondary: なし
- Status: `D`
- 判定根拠: 木を切る・撤去する・工事で改変しようとする操作が発動条件。
- この伝承における現れ方: 見る・近づくだけでなく、人間が自然物へ加工を加えようとした時点で禁忌破りになる。

### D12 作用対象
- Primary Child / Value: `D12.OTH.VICTIM_TARGET` — 特定被害者
- Primary Parent: `D12.OTH`
- Secondary: なし
- Status: `D`
- 判定根拠: 災いを受けるとされるのは伐採・工事を試みた作業員や関係者。
- この伝承における現れ方: 通行人一般ではなく、木へ手を加えた者が対象化される。

### D13 作用機構
- Primary Child / Value: `D13.FAT.CURSE_MISFORTUNE` — 呪詛・不運付与
- Primary Parent: `D13.FAT`
- Secondary: なし
- Status: `I`
- 判定根拠: 事故・怪我・死亡等の具体例は異伝差があり、共有核では非物理的な「災い・不幸」の付与として安定する。
- この伝承における現れ方: 通常の作業を祟りが事故・不幸へ結び付ける。

### D14 帰結極性
- Primary Child / Value: `D14.NEG` — 負
- Primary Parent: なし
- Secondary: なし
- Status: `D`
- 判定根拠: 作業関係者への事故・不幸と工事断念が中心帰結。
- この伝承における現れ方: 木へ手を加えることは一貫して負の結果へ接続される。

### D15 帰結領域
- Primary Child / Value: `D15.OPP.LUCK_MISFORTUNE` — 幸運・不運
- Primary Parent: `D15.OPP`
- Secondary: なし
- Status: `D`
- 判定根拠: 具体的重傷・死亡を固定できず、共有核は「事故・不幸・災い」という一般的不運。
- この伝承における現れ方: 結果細部が変わっても「切ると良くないことが起きる」が禁忌を維持する。

### D16 因果時間構造
- Primary Child / Value: `D16.EVT.SEQUENTIAL_EPISODE` — エピソード内段階進行
- Primary Parent: `D16.EVT`
- Secondary: なし
- Status: `I`
- 判定根拠: 伐採試行→災い→中止→木が残る、という複数段階が一つの因果系列として意味形成に必要。一定の潜伏時間や期限は示されない。
- この伝承における現れ方: 作業から結果までを段階的な由来譚として再構成する。

### D17 回避・制御方式
- Primary Child / Value: `D17.AVO.DO_NOT_ENGAGE` — 接触回避
- Primary Parent: `D17.AVO`
- Secondary: なし
- Status: `D`
- 判定根拠: 安定した解除儀礼は確認できず、木を切らない・手を加えないことが最も確実な回避。
- この伝承における現れ方: 木をそのまま残す不作為が安全策になる。

### D18 作用レイヤー
- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `0`
- Status: `I`
- 判定根拠: L1では伐採→災い→断念の伝承内因果が成立。噂を聞くこと自体は危害条件ではない。道路設計・伐採判断が実際に噂で変化したことを示す独立X Evidenceはない。
- この伝承における現れ方: 伝承内部の工事断念を現実の行政判断へ遡及しない。

## 2.4. 社会的分布・現実接続

### D19 流通範囲
- Primary Child / Value: `D19.LOC.LOCAL_TRADITION` — 地域伝承圏
- Primary Parent: `D19.LOC`
- Secondary: `D19.MAS.NATIONAL_PUBLIC` — 全国的大衆
- Status: `C`
- 判定根拠: 内容は函館山の具体地点へ固定された地域怪談で、後代には一般Web調査・映像・書籍を通じ地域外にも再流通する。ただし広域浸透度を一意に確定できない。
- この伝承における現れ方: 地域文脈が伝承の核でありながら、全国の受容者もメディアを通じ知り得る。

### D20 特権情報保持者
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`
- 判定根拠: 地元住民・作業関係者が追加情報を持つ可能性はあるが、誰が特権的真相保持者かを固定できるEvidenceがない。
- この伝承における現れ方: 公開流通している怪談の核と、未確認の事故史・工事史を誰が知るかは別問題であり、推測で埋めない。

### D21 現実アンカー
- Primary Child / Value: `D21.A2` — 具体的実在対象
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 函館山という実在地点と登山道路周辺の特定木へ固定される一方、伝承対象木を自治体・管理記録で正式同定できていない。
- この伝承における現れ方: 実在景観への固定が本当らしさを与えるが、祟りや事故史の実在性とは分離される。

# 3. QA・保留事項

- D01=U。Pilotの1980年代推定は維持しない。
- D02は2008転載ではなく、直接確認できる2016 Webを採用。
- D10は特定場所＋非人格的呪力で暫定処理。自然物そのもの／自然物に宿る固有の霊的効力を直接表すChild不足をGlobal Reconciliationのtaxonomy gap候補とする。
- D15は未確認事故を重傷・死亡へ具体化しない。
- D18 L3=0。現実の道路設計変更・工事中止を示す独立X Evidenceなし。
- D20=U。地元住民を特権保持者と推定しない。