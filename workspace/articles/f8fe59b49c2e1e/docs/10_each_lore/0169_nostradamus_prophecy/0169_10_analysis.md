記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

Evidence正本は `0169_00_contents.md`、R1〜R4監査成果物は `docs/99_work/20260914_A3_full_recoding/` 配下を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0169`
- `伝承エントリ名称`: ノストラダムスの大予言
- `Macro_Category`: 予言・終末俗説
- `Entry_Type`: 予測命題・解釈伝承
- `Version_Scope`: 五島勉が1973年に刊行した『ノストラダムスの大予言』を主要アンカーとし、ノストラダムスの予言詩を1999年7月の人類規模破局へ対応付けた日本の大衆形。危機対象の解釈差、続編・テレビ等の増幅、1999年後の再解釈は変容層として含める。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示

### D01 生成年代
- Primary Child / Value: `D01.G3` — 1970年代
- Primary Parent: なし
- Secondary: なし
- Status: `D`
- 判定根拠: 日本の代表的大衆形は1973年刊行書に直接固定できる。

### D02 最古確認流通媒体
- Primary Child / Value: `D02.PRT.BOOK` — 書籍
- Primary Parent: `D02.PRT`
- Secondary: なし
- Status: `D`
- 判定根拠: 1973年版をNDL書誌で直接確認。

### D03 確認流通媒体ポートフォリオ
- Primary Child / Value: `D03.PRT.BOOK` — 書籍
- Primary Parent: `D03.PRT`
- Secondary: `D03.BRD.TELEVISION` — テレビ
- Status: `D/I`
- 判定根拠: 書籍は直接固定。テレビ増幅は00の流通史記述から保持するが個別番組は未固定。

### D04 生成・変容パターン
- Primary Child / Value: `D04.REC.CONTEXT_UPDATE` — 時代適応
- Primary Parent: `D04.REC`
- Secondary: `D04.MED.MASS_ADAPTATION` — マスメディア化
- Status: `I`
- 判定根拠: 恐怖の大王の正体を時代ごとの危機へ置換し、出版・テレビ等で大規模に再提示する。1998年にも「最終解答編」が刊行される。

### D05 提示形式
- Primary Child / Value: `D05.PRP.PREDICTIVE_CLAIM` — 予測命題
- Primary Parent: `D05.PRP`
- Secondary: `D05.PRP.EXPLANATORY_CLAIM` — 説明命題
- Status: `D`
- 判定根拠: 1999年7月の未来破局を予測し、古い詩をその根拠として説明する。

### D06 真実性提示
- Primary Child / Value: `D06.T5` — 条件付き信念
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 受容者は予言を完全な既知事実ではなく、「本当に起こるかもしれない」未来命題として信じうる。

## 2.2. 意味形成

### D07 意味形成対象
- Primary Child / Value: `D07.ISU.FUTURE_SOCIAL_CHANGE` — 将来・社会変化
- Primary Parent: `D07.ISU`
- Secondary: `D07.DCF.FATE_OMEN` — 運命・予兆
- Status: `D/I`
- 判定根拠: 将来の人類規模変化を、古い予言が示す予兆として理解する。

### D08 意味形成契機
- Primary Child / Value: `D08.TRC.TEXT_DOCUMENT` — 文書・記載
- Primary Parent: `D08.TRC`
- Secondary: `D08.CLM.UNSUPPORTED_ASSERTION` — 根拠未提示の主張
- Status: `D/I`
- 判定根拠: 予言詩という文書と、それを1999年へ対応させる解釈主張が契機になる。

### D09 意味付与操作
- Primary Child / Value: `D09.PPR.OMEN_FORECAST` — 予兆・予測
- Primary Parent: `D09.PPR`
- Secondary: `D09.HST.HISTORICAL_ANCHOR` — 歴史接続
- Status: `D/I`
- 判定根拠: 16世紀の詩を未来の具体的年月・危機へ接続し、予兆として読む。

### D10 因果源存在論
- Primary Child / Value: `D10.OBJ.INFORMATION_CONTENT` — 情報内容
- Primary Parent: `D10.OBJ`
- Secondary: `D10.HUM.INDIVIDUAL_HUMAN` — 個人
- Status: `I`
- 判定根拠: 受容者に作用する直接の権威源は予言内容であり、ノストラダムスという歴史人物がその権威を補強する。

## 2.3. 因果・行動モデル

### D11 発動・接触条件
- Primary Child / Value: `D11.INF.READ_VIEW_MEDIA` — 読む・媒体を見る
- Primary Parent: `D11.INF`
- Secondary: なし
- Status: `I`
- 判定根拠: 予言を読む・見ることが受容者の因果モデルへの入口。

### D12 作用対象
- Primary Child / Value: `D12.AUD.GENERAL_PUBLIC` — 一般公衆
- Primary Parent: `D12.AUD`
- Secondary: なし
- Status: `I`
- 判定根拠: 予言の受容対象は日本の読者・視聴者を含む一般公衆。

### D13 作用機構
- Primary Child / Value: `D13.COG.INFORMATION_INDUCED_ACTION` — 情報誘導・行動誘発
- Primary Parent: `D13.COG`
- Secondary: `D13.COG.MENTAL_INFLUENCE` — 精神干渉
- Status: `I`
- 判定根拠: 予言情報が受容者の判断・行動を誘導し、恐怖・不安を生じさせる。予言自体が破局を発生させるとはコードしない。

### D14 帰結極性
- Primary Child / Value: `D14.NEG` — 負
- Primary Parent: なし
- Secondary: なし
- Status: `D`
- 判定根拠: 中心予測は人類規模の破局・滅亡。

### D15 帰結領域
- Primary Child / Value: `D15.MND.FEAR_TRAUMA` — 恐怖・トラウマ
- Primary Parent: `D15.MND`
- Secondary: なし
- Status: `I`
- 判定根拠: Evidence正本で確認できる受容者側の中心帰結は終末不安。集団パニックを独立Evidenceなしに固定しない。

### D16 因果時間構造
- Primary Child / Value: `D16.DLY.DEADLINE` — 期限付き
- Primary Parent: `D16.DLY`
- Secondary: なし
- Status: `D`
- 判定根拠: 1999年7月という明示期限が予言構造の中心。

### D17 回避・制御方式
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`
- 判定根拠: 共有核として安定した回避・解除法を固定できない。

### D18 作用レイヤー
- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `0`
- Status: `I`
- 判定根拠: 伝承内部には未来破局因果がある。話を知ること自体が超常危害条件ではなく、流通規模と独立した社会現実効果を区別する。

## 2.4. 社会的分布・現実接続

### D19 流通範囲
- Primary Child / Value: `D19.MAS.NATIONAL_PUBLIC` — 全国的大衆
- Primary Parent: `D19.MAS`
- Secondary: なし
- Status: `D`
- 判定根拠: 1973年出版と後代マスメディアを通じ全国規模に流通。

### D20 特権情報保持者
- Primary Child / Value: `D20.NON.COMMON_KNOWLEDGE` — 一般共有
- Primary Parent: `D20.NON`
- Secondary: なし
- Status: `I`
- 判定根拠: 予言内容は大衆出版され、秘密の内部情報ではない。解釈者の知識差を真相独占とはみなさない。

### D21 現実アンカー
- Primary Child / Value: `D21.A4` — 史実・記録・既存伝承を因果統合
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 歴史人物、予言詩、1973年出版、1999年という現実時点を一つの予言モデルへ統合する。
