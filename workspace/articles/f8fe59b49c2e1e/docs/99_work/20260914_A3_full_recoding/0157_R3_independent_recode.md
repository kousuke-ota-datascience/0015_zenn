# 0157 R3 Independent Recode

- Evidence input: strengthened `0157_00_contents.md`
- Version Scope: `0157_R2_version_scope.md`
- Old 10: 未参照。R3 freeze後に比較する。

## D01–D21

- D01: `D01.G1` — 1868–1944; Status `I`。新郷村公式の後代説明が現在型仮説の顕在化を1935年とする。古代出来事の年代ではない。
- D02: codeなし; Status `U`。1935年に仮説が現れたことは確認できるが、その最古実流通媒体を直接固定できない。
- D03: Primary `D03.WEB.WEBSITE`, Parent `D03.WEB`; Secondary `D03.PRT.BOOK`; Status `D`。自治体公式Webと後代出版で流通を確認。
- D04: Primary `D04.VAR.ACCRETION`, Parent `D04.VAR`; Secondary `D04.VAR.LOCALIZATION`; Status `I`。代替キリスト史へ地域文化・墓・祭礼等の補強要素が付加され、新郷村へ場所固定される。
- D05: Primary `D05.PRP.FACT_CLAIM`, Parent `D05.PRP`; Secondary `D05.PRP.EXPLANATORY_CLAIM`; Status `D/I`。「キリストは日本で没した／この塚が墓」という歴史事実型主張を、竹内文書で説明する。
- D06: `D06.T6` — 真偽未確定; Status `D`。現在の自治体公式は仮説を紹介しつつ真偽判断を読者へ委ねる。
- D07: Primary `D07.ISU.INFORMATION_VOID`, Parent `D07.ISU`; Status `I`。キリストの死後・墓所・生涯について通説と異なる情報空白を代替史で埋める。
- D08: Primary `D08.TRC.TEXT_DOCUMENT`, Parent `D08.TRC`; Secondary `D08.HIS.PLACE_NAME_RUIN`; Status `D`。竹内文書と現地の塚が主要な証拠手掛かりとして提示される。
- D09: Primary `D09.HST.HISTORICAL_ANCHOR`, Parent `D09.HST`; Secondary `D09.CAU.DIRECT_CAUSE`; Status `I`。キリスト史と日本地域史を接続し、文書・墓を一つの代替歴史へ統合する。
- D10: Primary `D10.OBJ.INFORMATION_CONTENT`, Parent `D10.OBJ`; Secondary `D10.HUM.INDIVIDUAL_HUMAN`; Status `I`。受容者にとって代替歴史の因果源・根拠は竹内文書の情報内容であり、伝承内ではキリストという人物の行動が中心。
- D11: Primary `D11.INF.READ_VIEW_MEDIA`, Parent `D11.INF`; Status `I`。代替史は文書内容への接触・受容によって認識可能になる。伝承内の古代出来事そのものには単一の発動条件を置きにくい。
- D12: Primary `D12.AUD.GENERAL_PUBLIC`, Parent `D12.AUD`; Status `I`。文書・墓の代替歴史主張が作用するのは受容者・公衆の世界認識。
- D13: Primary `D13.RST.REALITY_REPLACEMENT`, Parent `D13.RST`; Status `I`。通説的キリスト史を「実は日本で没した」という別の歴史状態へ置換して提示する。
- D14: `D14.NEU`; Status `I`。共有核は危害・利益ではなく代替歴史の提示自体。
- D15: Primary `D15.KNW.BELIEF_REVISION`, Parent `D15.KNW`; Secondary `D15.COL.COMMUNITY_CHANGE`; Status `I`。受容者の歴史認識を変え、現実側では地域文化・観光実践へ接続する。
- D16: Primary `D16.STA.STATIC_ATTRIBUTE`, Parent `D16.STA`; Secondary なし; Status `I`。「この塚はキリストの墓である」という同一性主張が現在の核。古代移動譚はその説明物語。
- D17: codeなし; Status `NA`。危害因果の回避・解除を中心とする伝承ではない。
- D18: `L1=1; L2=0; L3=1`; Status `D/I`。L1では身代わり・渡来・墓という伝承内歴史因果がある。L3は墓・伝承館・祭礼・観光という現実実践を公式資料で確認。聞くこと自体が超常作用条件ではないためL2=0。
- D19: Primary `D19.LOC.LOCAL_TRADITION`, Parent `D19.LOC`; Secondary `D19.MAS.NATIONAL_PUBLIC`; Status `D/I`。地域に固定された伝承だが出版・観光PR等で広域流通。
- D20: Primary `D20.EXP.RELIGIOUS_FOLKLORE`, Parent `D20.EXP`; Secondary `D20.ORG.INSTITUTION_AUTHORITY`; Status `I`。竹内文書・伝承史の追加情報は伝承研究・資料保持主体や地域公式側へ偏る。
- D21: `D21.A4`; Status `I`。歴史人物キリスト、竹内文書、実在塚、昭和期発見伝承、現代祭礼を代替歴史因果へ統合する。

## QA before freeze

- 1935年を古代史の実証へ転用しない。
- 竹内文書原資料未確認をD02等で過剰推定しない。
- L3=1は公式に確認できる祭礼・伝承館・観光実践へ限定。
- D17は回避構造がないためNA。
- H3 Primary exactly 1 / Secondary <=2。
