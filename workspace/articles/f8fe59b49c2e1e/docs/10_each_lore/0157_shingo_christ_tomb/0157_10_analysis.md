記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

Evidence正本は同一ディレクトリの `0157_00_contents.md`、R1/R2/R3/R4監査成果物は `docs/99_work/20260914_A3_full_recoding/` 配下を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0157`
- `伝承エントリ名称`: 新郷村キリストの墓
- `Macro_Category`: 異説史・地域伝承
- `Entry_Type`: 命題・伝説複合型
- `Version_Scope`: イエス・キリストはゴルゴタで死亡せず弟イスキリが身代わりとなり、本人は日本へ渡来して現在の青森県新郷村付近で生涯を終え、現地の塚がその墓であるとする代替歴史説。竹内文書を根拠とする昭和期の顕在化と、後代の墓・伝承館・祭礼・観光への定着を含む。渡来経路、年齢、家族、祭唄・家紋等の追加傍証は異伝として分離する。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示

### D01 生成年代
- Primary Child / Value: `D01.G1` — 1868–1944
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 新郷村公式の後代説明が、竹内古文書から現在型仮説が出たのを1935年とする。これは古代出来事の確認年ではなく、現在型伝承の顕在化年代への写像である。

### D02 最古確認流通媒体
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`
- 判定根拠: 1935年に仮説が現れたという公式後代説明はあるが、当時の最古実流通媒体を直接固定できていない。

### D03 確認流通媒体ポートフォリオ
- Primary Child / Value: `D03.WEB.WEBSITE` — Webサイト
- Primary Parent: `D03.WEB`
- Secondary: `D03.PRT.BOOK` — 書籍
- Status: `D`
- 判定根拠: 現在は新郷村公式Webで一般公開され、後代出版でも流通を確認できる。地域口承を最古・主要媒体として推測しない。

### D04 生成・変容パターン
- Primary Child / Value: `D04.MED.COMMERCIALIZATION` — 商品・観光化
- Primary Parent: `D04.MED`
- Secondary: `D04.VAR.LOCALIZATION` — 地域化
- Status: `D/I`
- 判定根拠: 代替歴史説が実在の墓、伝承館、祭礼、観光PRへ制度化され、新郷村固有の地域伝承として固定されている。

### D05 提示形式
- Primary Child / Value: `D05.PRP.FACT_CLAIM` — 事実主張
- Primary Parent: `D05.PRP`
- Secondary: `D05.PRP.EXPLANATORY_CLAIM` — 説明命題
- Status: `D/I`
- 判定根拠: 「キリストは日本で没した／この塚が墓」という歴史事実型主張を、竹内文書等によって説明する形式を取る。

### D06 真実性提示
- Primary Child / Value: `D06.T6` — 真偽未確定
- Primary Parent: なし
- Secondary: なし
- Status: `C`
- 判定根拠: 現在の新郷村公式は仮説を紹介しつつ墓の真偽判断を受容者へ委ねる。一方、伝承の再話には歴史事実型 `D06.T4` として提示する形もある。現行公式提示をPrimaryにする。

## 2.2. 意味形成

### D07 意味形成対象
- Primary Child / Value: `D07.ISU.INFORMATION_VOID` — 情報空白
- Primary Parent: `D07.ISU`
- Secondary: なし
- Status: `I`
- 判定根拠: キリストの死後・墓所・生涯について、通説と異なる「知られていない真相」を代替史で埋める。

### D08 意味形成契機
- Primary Child / Value: `D08.TRC.TEXT_DOCUMENT` — 文書・記載
- Primary Parent: `D08.TRC`
- Secondary: `D08.HIS.PLACE_NAME_RUIN` — 地名・遺構・記念物
- Status: `D`
- 判定根拠: 竹内文書と現地の塚が、代替歴史説を問題化・現実化する主要手掛かりとして提示される。

### D09 意味付与操作
- Primary Child / Value: `D09.HST.HISTORICAL_ANCHOR` — 歴史接続
- Primary Parent: `D09.HST`
- Secondary: `D09.CAU.DIRECT_CAUSE` — 直接原因化
- Status: `I`
- 判定根拠: キリスト史と日本地域史を直接接続し、文書と墓を同一の歴史系列へ統合する。

### D10 因果源存在論
- Primary Child / Value: `D10.HUM.INDIVIDUAL_HUMAN` — 個人
- Primary Parent: `D10.HUM`
- Secondary: `D10.OBJ.INFORMATION_CONTENT` — 情報内容
- Status: `C`
- 判定根拠: 伝承内部ではキリストという歴史人物の行動が代替史を成立させる。一方、受容者にとって根拠・説明源は竹内文書の情報内容であり、両層を競合的に保持する。

## 2.3. 因果・行動モデル

### D11 発動・接触条件
- Primary Child / Value: `D11.INF.LEARN_KNOW` — 知る
- Primary Parent: `D11.INF`
- Secondary: なし
- Status: `I`
- 判定根拠: 現代受容者が代替歴史モデルへ入る契機は、文書・墓に基づく主張を知ることにある。

### D12 作用対象
- Primary Child / Value: `D12.AUD.GENERAL_PUBLIC` — 一般公衆
- Primary Parent: `D12.AUD`
- Secondary: なし
- Status: `I`
- 判定根拠: 代替歴史主張が直接働きかける対象は、伝承を受容し世界史解釈を更新する公衆である。

### D13 作用機構
- Primary Child / Value: `D13.RST.REALITY_REPLACEMENT` — 現実置換
- Primary Parent: `D13.RST`
- Secondary: なし
- Status: `I`
- 判定根拠: 通説的なキリスト史を「実は日本で没した」という別の歴史状態へ置換して提示する。

### D14 帰結極性
- Primary Child / Value: `D14.NEU` — 中立
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 共有核の主帰結は危害・利益ではなく歴史解釈の置換である。

### D15 帰結領域
- Primary Child / Value: `D15.KNW.BELIEF_REVISION` — 信念変更
- Primary Parent: `D15.KNW`
- Secondary: `D15.COL.COMMUNITY_CHANGE` — 共同体変化
- Status: `I`
- 判定根拠: 個人側では歴史認識が更新され、現実の地域側では墓・祭礼・観光という共同体実践へ展開している。

### D16 因果時間構造
- Primary Child / Value: `D16.STA.STATIC_ATTRIBUTE` — 静的属性・同一性
- Primary Parent: `D16.STA`
- Secondary: なし
- Status: `I`
- 判定根拠: 現在の識別核は「この塚はキリストの墓である」という静的同一性主張であり、古代の移動物語はその説明層である。

### D17 回避・制御方式
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `NA`
- 判定根拠: 危害因果を回避・解除・利用することがVersion Scopeの構成要素ではない。

### D18 作用レイヤー
- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `1`
- Status: `D/I`
- 判定根拠: L1では身代わり・渡来・墓という伝承内部の歴史因果がある。L2は聞くこと自体が超常作用条件ではない。L3は墓・伝承館・祭礼・観光PRという現実の地域実践が公式資料で確認できる。

## 2.4. 社会的分布・現実接続

### D19 流通範囲
- Primary Child / Value: `D19.LOC.LOCAL_TRADITION` — 地域伝承圏
- Primary Parent: `D19.LOC`
- Secondary: `D19.MAS.NATIONAL_PUBLIC` — 全国的大衆
- Status: `D/I`
- 判定根拠: 新郷村という地域に強く固定される一方、出版・自治体Web・観光PRを通じ地域外へ広く流通する。

### D20 特権情報保持者
- Primary Child / Value: `D20.ORG.INSTITUTION_AUTHORITY` — 組織・権限主体
- Primary Parent: `D20.ORG`
- Secondary: `D20.EXP.RELIGIOUS_FOLKLORE` — 宗教・伝承専門家
- Status: `I`
- 判定根拠: 伝承館・自治体等が資料・地域伝承史を集積し、公衆より追加情報へアクセスしやすい。単なる地元住民一般を真相保持者とはしない。

### D21 現実アンカー
- Primary Child / Value: `D21.A4` — 史実・記録・既存伝承を因果統合
- Primary Parent: なし
- Secondary: なし
- Status: `I`
- 判定根拠: 歴史人物キリスト、竹内文書、実在塚、1935年の発見伝承、現代祭礼・観光実践を一つの代替歴史モデルへ統合する。
