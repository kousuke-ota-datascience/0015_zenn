記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

Evidence正本は同一ディレクトリの `0113_00_contents.md`、R1/R2/R3/R4監査成果物は `docs/99_work/20260914_A3_full_recoding/` 配下を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0113`
- `伝承エントリ名称`: 井の頭公園のボートに乗ると別れる
- `Macro_Category`: 恋愛・縁・吉凶
- `Entry_Type`: ジンクス／場所俗信
- `Version_Scope`: 井の頭恩賜公園の井の頭池で、恋人同士が二人でボートに乗ると、その後二人は別れるという場所固定型ジンクス。主要な因果説明として「井の頭弁財天が嫉妬して仲を裂く」を含むが、弁財天説明を共有核の必須条件とはしない。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示

### D01 生成年代
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`
- 判定根拠: 2016年以前の成立時期を固定できるEvidenceがない。2016年の確認年を生成時期へ置換しない。
- この伝承における現れ方: 現代に広く知られた場所ジンクスだが、いつ共有核が成立したかは不明。

### D02 最古確認流通媒体
- Primary Child / Value: `D02.WEB.WEBSITE` — Webサイト
- Primary Parent: `D02.WEB`
- Secondary: なし
- Status: `D`
- 判定根拠: 現在日時を固定できる最古確認点として2016-08-05公開のLIFULL HOME'S記事／調査で中心命題を直接確認できる。これは起源媒体の推定ではなく最古確認媒体のコード。
- この伝承における現れ方: 一般向けWeb調査記事上で「カップルでボートに乗ると別れる」が流通する。

### D03 確認流通媒体ポートフォリオ
- Primary Child / Value: `D03.WEB.WEBSITE` — Webサイト
- Primary Parent: `D03.WEB`
- Secondary: なし
- Status: `D`
- 判定根拠: LIFULL HOME'Sおよび三鷹市サイトで伝承の実流通を確認。
- この伝承における現れ方: 地域の噂が一般向け・自治体Webコンテンツとして再提示される。

### D04 生成・変容パターン
- Primary Child / Value: `D04.STB.STABILIZED_CANON` — 定型化・カノン化
- Primary Parent: `D04.STB`
- Secondary: `D04.VAR.RETELLING` — 再話
- Status: `I`
- 判定根拠: 2016年調査と自治体による後代紹介で「カップルでボート→別れる」という短い核が安定している一方、弁財天嫉妬等の説明を伴って再話される。
- この伝承における現れ方: 説明細部は変わっても、場所・乗船・破局の三項関係は維持される。

### D05 提示形式
- Primary Child / Value: `D05.RUL.JINX_RULE` — ジンクス規則
- Primary Parent: `D05.RUL`
- Secondary: なし
- Status: `D`
- 判定根拠: 「恋人同士でボートに乗ると別れる」という条件と恋愛帰結を結ぶジンクス規則として直接提示される。
- この伝承における現れ方: 個別物語を必要とせず一文のif-then規則として流通する。

### D06 真実性提示
- Primary Child / Value: `D06.T5` — 条件付き信念
- Primary Parent: なし
- Secondary: なし
- Status: `D`
- 判定根拠: 三鷹市サイトでも「噂」として提示され、確定事実ではなく「XならYらしい」という信念形式を取る。
- この伝承における現れ方: 完全に信じなくても、破局を避けたい者の意思決定へ入り得る。

## 2.2. 意味形成

### D07 意味形成対象
- Primary Child / Value: `D07.DCF.FATE_OMEN` — 運命・予兆
- Primary Parent: `D07.DCF`
- Secondary: なし
- Status: `I`
- 判定根拠: 恋愛関係が将来続くか別れるかという不確実な未来を、特定行為と結び付けて予測可能にする。
- この伝承における現れ方: 「この恋愛は続くか」という不確実性が、ボート乗船の有無へ接続される。

### D08 意味形成契機
- Primary Child / Value: `D08.CLM.UNSUPPORTED_ASSERTION` — 根拠未提示の主張
- Primary Parent: `D08.CLM`
- Secondary: なし
- Status: `D`
- 判定根拠: 資料が直接示す出発点は「昔からの言い伝え」という履歴証拠ではなく、「ボートに乗ると別れる」という噂・主張そのもの。
- この伝承における現れ方: 因果証拠を伴わない短い噂が、将来の破局を考える枠組みになる。

### D09 意味付与操作
- Primary Child / Value: `D09.PPR.CORRELATION_RULE` — 相関規則化
- Primary Parent: `D09.PPR`
- Secondary: `D09.AGN.AGENCY_ATTRIBUTION` — 主体化; `D09.CAU.DIRECT_CAUSE` — 直接原因化
- Status: `D/I`
- 判定根拠: 乗船と破局を規則的対応として結び、弁財天嫉妬型では結果を神格の意図へ主体化する。また再話上は「乗ったため別れた」という直接因果へ変換される。
- この伝承における現れ方: 通常は独立なレジャー行為と関係破綻が、一つの予測・因果規則へ圧縮される。

### D10 因果源存在論
- Primary Child / Value: `D10.SPC.SPECIFIC_PLACE` — 特定場所
- Primary Parent: `D10.SPC`
- Secondary: `D10.SUP.DEITY_DIVINE` — 神格・超越主体
- Status: `C`
- 判定根拠: 最小核では井の頭池という場所に効力が固定される裸ジンクスとして成立する一方、主要説明では井の頭弁財天が嫉妬する人格神格として因果源になる。
- この伝承における現れ方: 場所そのものが効く型と、神格の意思が効く型を同一化せず併存させる。

## 2.3. 因果・行動モデル

### D11 発動・接触条件
- Primary Child / Value: `D11.MOV.RIDE_BOARD` — 乗る
- Primary Parent: `D11.MOV`
- Secondary: `D11.SOC.RELATION_FAMILY` — 家族・恋愛関係になる
- Status: `D`
- 判定根拠: 恋人関係にある二人が井の頭池のボートへ乗ることが発動条件。
- この伝承における現れ方: 公園訪問一般ではなく、カップルという関係属性と乗船行為の組合せが条件になる。

### D12 作用対象
- Primary Child / Value: `D12.KIN.PARTNER_FRIEND` — 恋人・友人
- Primary Parent: `D12.KIN`
- Secondary: なし
- Status: `D`
- 判定根拠: 作用対象はボートに乗った恋人同士。
- この伝承における現れ方: 個人の身体ではなく親密関係の当事者が対象化される。

### D13 作用機構
- Primary Child / Value: `D13.FAT.CURSE_MISFORTUNE` — 呪詛・不運付与
- Primary Parent: `D13.FAT`
- Secondary: なし
- Status: `I`
- 判定根拠: 物理攻撃ではなく、恋愛上の不運として破局をもたらす非物理作用が中心。
- この伝承における現れ方: 場所または弁財天の効力が、後の関係悪化・別離へ変換される。

### D14 帰結極性
- Primary Child / Value: `D14.NEG` — 負
- Primary Parent: なし
- Secondary: なし
- Status: `D`
- 判定根拠: 中心帰結は恋愛関係の破綻。
- この伝承における現れ方: 乗船後の利益や保護は共有核に含まれない。

### D15 帰結領域
- Primary Child / Value: `D15.SOC.RELATION_BREAKDOWN` — 関係破綻
- Primary Parent: `D15.SOC`
- Secondary: なし
- Status: `D`
- 判定根拠: 「別れる」という明示終端を最も直接表す。
- この伝承における現れ方: 二者の恋愛関係が継続しなくなることが作用の終端。

### D16 因果時間構造
- Primary Child / Value: `D16.STA.STATIC_RULE` — 静的規則
- Primary Parent: `D16.STA`
- Secondary: なし
- Status: `D`
- 判定根拠: 期限・潜伏期間を規定せず、「乗れば別れる」という一般条件対応が主題。後日性はあるが待ち時間自体は意味形成の中心ではない。
- この伝承における現れ方: 時間進行の筋書きより、条件と帰結の対応表として記憶される。

### D17 回避・制御方式
- Primary Child / Value: `D17.AVO.DO_NOT_ENGAGE` — 接触回避
- Primary Parent: `D17.AVO`
- Secondary: なし
- Status: `D`
- 判定根拠: 最も安定した回避は恋人同士でボートへ乗らないこと。参拝による中和は共有核へ含めない。
- この伝承における現れ方: デートコースからボートを外せば発動条件を避けられる。

### D18 作用レイヤー
- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `0`
- Status: `I`
- 判定根拠: L1では乗船した恋人に破局因果が設定される。噂を聞くこと自体は破局条件ではない。認知調査は流通Evidenceであり、噂によるボート利用・観光等の現実変化を示す独立X Evidenceではない。
- この伝承における現れ方: 因果効力は伝承内の条件成立者へ限定される。

## 2.4. 社会的分布・現実接続

### D19 流通範囲
- Primary Child / Value: `D19.MAS.NATIONAL_PUBLIC` — 全国的大衆
- Primary Parent: `D19.MAS`
- Secondary: `D19.LOC.REGIONAL` — 地域・地方
- Status: `C`
- 判定根拠: 三鷹市の地域情報として定着する一方、2016年の一般向けインターネット調査でも57人が認知している。ただし調査本文から地理的代表性を確定できず、全国的大衆と地域流通の境界を一意に固定しない。
- この伝承における現れ方: 場所は東京に固定されるが、噂の受容者は地域外にも広がる。

### D20 特権情報保持者
- Primary Child / Value: `D20.NON.COMMON_KNOWLEDGE` — 一般共有
- Primary Parent: `D20.NON`
- Secondary: なし
- Status: `I`
- 判定根拠: 内部者・専門家だけが知る構造ではなく、自治体サイトや一般向け調査で公開共有される。一般共有という分布判断自体は推論を含む。
- この伝承における現れ方: 特定の伝承保持者を介さず公園利用者が規則を知り得る。

### D21 現実アンカー
- Primary Child / Value: `D21.A2` — 具体的実在対象
- Primary Parent: なし
- Secondary: なし
- Status: `D`
- 判定根拠: 井の頭恩賜公園、井の頭池、ボート場、井の頭弁財天堂という具体的実在対象へ固定される。
- この伝承における現れ方: 場所名を任意の池へ置換すると別の場所ジンクスになる。

# 3. QA・保留事項

- D01は成立年代Evidence不足のためU。
- D02は「起源媒体」ではなく、現在最古に確認できる実流通媒体として2016年Webを採用。
- D10は裸ジンクス型と弁財天嫉妬型の因果源差をCで保持。
- D16は明示的な期限・潜伏期間がないためSTATIC_RULE。
- D18 L3は独立X Evidence不足のため0。
- D19は調査の地理代表性を確定できずC。
- 新規taxonomy gapなし。