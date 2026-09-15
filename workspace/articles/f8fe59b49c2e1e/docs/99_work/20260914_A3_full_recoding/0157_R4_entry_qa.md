# 0157 R4 Entry QA

- R3 freeze: `68f901f86daa9a2f0e53d00a80ac791bc9b3fa3e`
- Old 10 compared after freeze: `0157_10_analysis.md` (pilot coding transcription)

## 差分裁定

- D01: 旧Uを更新。新郷村公式が現在型仮説の顕在化を1935年と説明するため `D01.G1 / I`。
- D02: Uを維持。1935年の具体的流通媒体は未固定。
- D03: 旧 `LOCAL_ORAL/I` は直接根拠不足。現在直接確認できる自治体WebをPrimary、出版をSecondary。
- D04: R3 `ACCRETION` より、現実の伝承館・祭礼・観光化を直接表す旧 `D04.MED.COMMERCIALIZATION` をPrimaryに採用し、地域化をSecondary。
- D05: 旧地元伝聞より、共有核の論理形式である `FACT_CLAIM` + `EXPLANATORY_CLAIM` を採用。
- D06: 現在の自治体公式は真偽判断を開く一方、伝承は歴史事実型でも流通する。Primary `D06.T6`、Status `C` とし、 competing form として `D06.T4` を注記する。
- D07: 新Evidenceで代替史が埋める「キリストの生涯・墓所の情報空白」を明確化できるため `D07.ISU.INFORMATION_VOID/I`。
- D08: 旧の墓だけでなく、竹内文書が根拠として中心的。`TEXT_DOCUMENT` Primary + `PLACE_NAME_RUIN` Secondary。
- D09: `HISTORICAL_ANCHOR` を維持し、通説と地域史を直接接続する操作として記録。
- D10: 旧 `INFORMAL_GROUP` は因果主体に合わない。伝承内部ではキリストという歴史人物、認識論上は竹内文書の情報内容が競合するため `D10.HUM.INDIVIDUAL_HUMAN` Primary + `D10.OBJ.INFORMATION_CONTENT` Secondary、Status `C`。
- D11: `D11.INF.LEARN_KNOW/I` を採用。代替史の受容への入口は当該主張を知ること。
- D12: 旧地域共同体より、主張が作用する一般受容者を `D12.AUD.GENERAL_PUBLIC` Primary。地域共同体はL3側の実践主体として分離。
- D13: 旧 `MANIFEST_ONLY` は不適。受容者の歴史世界モデルを置換する `D13.RST.REALITY_REPLACEMENT/I` を採用。
- D14: NEU維持。
- D15: 旧 `UNCERTAINTY_PRESERVED` のみでは弱い。Primary `BELIEF_REVISION`、Secondary `COMMUNITY_CHANGE`。自治体公式が真偽を開くことはD06で保持。
- D16: `STATIC_ATTRIBUTE` 維持。
- D17: 回避・解除概念がVersion Scopeに非該当なので旧Uではなく `NA`。
- D18: 旧L3=0を更新。墓・伝承館・継続祭礼・観光PRが公式資料で直接確認できるため `L1=1,L2=0,L3=1`。
- D19: `LOCAL_TRADITION` Primaryを維持し、全国出版・観光流通をSecondary `NATIONAL_PUBLIC`。
- D20: 旧地元住民のみの情報優位は直接性不足。伝承館・自治体という資料保持・提示主体を `D20.ORG.INSTITUTION_AUTHORITY` Primary、伝承・宗教知識を扱う主体を `D20.EXP.RELIGIOUS_FOLKLORE` Secondary。
- D21: A4維持。

## QA

- 1935年を古代史の事実証拠へ転用していない: Pass。
- 竹内文書原資料未確認を明示: Pass。
- L3は実在する地域実践に直接対応: Pass。
- U/NA/C分離: D02=U, D17=NA, D06/D10=C。
- H3 Primary exactly 1 / Secondary <=2: Pass。
