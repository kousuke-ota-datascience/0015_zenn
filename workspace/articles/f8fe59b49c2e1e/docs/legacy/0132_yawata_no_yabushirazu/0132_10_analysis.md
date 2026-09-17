記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

伝承内容・典拠・異伝・証拠上の不確実性は、同一ディレクトリの `0132_00_contents.md` を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0132`
- `伝承エントリ名称`: 八幡の藪知らず
- `Macro_Category`: 場所・異界・祟り・禁忌
- `Entry_Type`: 禁足地伝承／場所伝説
- `Version_Scope`: 千葉県市川市八幡の八幡不知森について、中へ立ち入ってはならず、禁を破って入ると出られなくなる、または祟り・災いを受けるとする禁足地伝承。複数由来説は別説として保持し、徳川光圀侵入譚は主要派生として含む。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示の次元

### 2.1.1. D01: 生成年代
**いつ成立したか**
- Primary Child / Value: `D01.G0`
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**
1749年『葛飾記』の公的本文抜粋に共有核が記録されるため、少なくとも前近代には存在したと推定できる。1749年を生成時点とはしない。

**この伝承における現れ方**
少なくとも1749年には禁足・非帰還・祟りが既知の里諺として記録されている。

### 2.1.2. D02: 最古確認流通媒体
**現在の証拠で、伝承が実際に人から人へ流通したことを最も古く確認できる媒体は何か**
- Primary Child / Value: `D02.PRT.BOOK`
- Primary Parent: `D02.PRT`
- Secondary: なし
- Status: `I`

**判定根拠**
1749年『葛飾記』という書物への記録を公立図書館の本文抜粋から確認する。原本未実見のため `I`。

**この伝承における現れ方**
地域の禁足知識が前近代の地誌系書物へ記録される。

### 2.1.3. D03: 確認流通媒体ポートフォリオ（Version Scope）
**今回コードするVersion Scopeで、伝承が受容者へ流通したことを確認できる媒体は何か**
- Primary Child / Value: `D03.PRT.BOOK`
- Primary Parent: `D03.PRT`
- Secondary: `D03.WEB.WEBSITE`
- Status: `I`

**判定根拠**
前近代書物と現代自治体・図書館Webで再提示を確認するが、前近代資料は公的抄録・案内経由である。

**この伝承における現れ方**
書物と現代Webの双方で禁足・祟り・由来説明が提示される。

### 2.1.4. D04: 生成・変容パターン
**時間とともにどう変形したか**
- Primary Child / Value: `D04.CON.CONTESTED_VERSION`
- Primary Parent: `D04.CON`
- Secondary: なし
- Status: `I`

**判定根拠**
神聖地、将門、八門遁甲、日本武尊、入会地等、同じ禁足の理由を説明する複数Versionが併存する。派生順序は固定しない。

**この伝承における現れ方**
「入るな」という規則を共有しつつ、由来説明が複数競合する。

### 2.1.5. D05: 提示形式
**どんなコミュニケーション形式で提示されるか**
- Primary Child / Value: `D05.RUL.TABOO`
- Primary Parent: `D05.RUL`
- Secondary: `D05.HYB.NARRATIVE_EXPLANATION`
- Status: `I`

**判定根拠**
中心核は「入ってはいけない」という禁忌であり、歴史資料Contentは公的抄録・案内経由で確認するため `I`。

**この伝承における現れ方**
禁足規則に、複数の由来説明や光圀譚が付随する。

### 2.1.6. D06: 真実性提示
**どんな「本当らしさ」を要求するか**
- Primary Child / Value: `D06.T3`
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**
1749年本文抜粋は「昔より里諺に云ひ伝へたり」とし、共同体既知の言い伝えとして提示する。

**この伝承における現れ方**
昔から共有される地域知識として禁足が提示される。

## 2.2. 意味形成の次元

### 2.2.1. D07: 意味形成対象
**何が不可解・不確実なのか**
- Primary Child / Value: `D07.PSE.DANGEROUS_PLACE`
- Primary Parent: `D07.PSE`
- Secondary: `D07.PSE.SPATIAL_ROUTE_ANOMALY`
- Status: `I`

**判定根拠**
実在する藪が危険な禁足地とされ、「入ると出られない」とする型が公的抄録・解説で確認できる。

**この伝承における現れ方**
市街地内の具体的空間が、侵入すると非帰還・祟りが起こる場所として区別される。

### 2.2.2. D08: 意味形成契機
**何を手掛かりに問題化されるか**
- Primary Child / Value: `D08.CLM.INHERITED_SAYING`
- Primary Parent: `D08.CLM`
- Secondary: `D08.HIS.PLACE_NAME_RUIN`
- Status: `I`

**判定根拠**
里諺としての既存伝承と、八幡不知森という名称・石碑・現存地点が手掛かりになる。

**この伝承における現れ方**
受容者は既存の禁足伝承と具体地点を対応させる。

### 2.2.3. D09: 意味付与操作
**不可解なものをどう理解可能にするか**
- Primary Child / Value: `D09.NOR.TABOOIZATION`
- Primary Parent: `D09.NOR`
- Secondary: `D09.HST.HISTORICAL_ANCHOR`
- Status: `I`

**判定根拠**
特定空間への侵入をしてはいけない行為として規範化し、複数の歴史・伝説的由来へ接続する。

**この伝承における現れ方**
立入制限が歴史・宗教・社会史的説明と結び付けられる。

### 2.2.4. D10: 因果源存在論
**原因を何として世界に置くか**
- Primary Child / Value: `D10.SPC.SPECIFIC_PLACE`
- Primary Parent: `D10.SPC`
- Secondary: `D10.SUP.IMPERSONAL_CURSE`
- Status: `I`

**判定根拠**
異常効力は八幡不知森という特定場所へ固定され、単一人格主体を全Versionで必須としない。

**この伝承における現れ方**
危険は任意の藪ではなく、この場所自体へ固定される。

## 2.3. 因果・行動モデルの次元

### 2.3.1. D11: 発動・接触条件
**何を契機に因果系へ入るか**
- Primary Child / Value: `D11.MOV.ENTER_PASS_CROSS`
- Primary Parent: `D11.MOV`
- Secondary: なし
- Status: `I`

**判定根拠**
藪内部へ立ち入ることが禁忌違反・非帰還・祟りの条件として、公的抄録・解説から確認できる。

**この伝承における現れ方**
境界を越えて内部へ入ることが因果系への入口となる。

### 2.3.2. D12: 作用対象
**誰／何に作用するか**
- Primary Child / Value: `D12.FOC.PROTAGONIST_EXPERIENCER`
- Primary Parent: `D12.FOC`
- Secondary: なし
- Status: `I`

**判定根拠**
禁を破って内部へ入った人物本人が非帰還・祟り・警告の対象になる。

**この伝承における現れ方**
侵入者本人が作用対象となる。

### 2.3.3. D13: 作用機構
**因果源が対象へ何をするか**
- Primary Child / Value: `D13.RST.SPATIAL_DISTORTION`
- Primary Parent: `D13.RST`
- Secondary: `D13.FAT.CURSE_MISFORTUNE`
- Status: `I`

**判定根拠**
「入ると出られない」という空間作用と、「禁を破ると祟り・災いがある」という作用が公的抄録・解説から確認できる。

**この伝承における現れ方**
侵入者は帰還不能になる、または災いを受けるとされる。

### 2.3.4. D14: 帰結極性
**結果は正・負・中立・混合か**
- Primary Child / Value: `D14.NEG`
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**
非帰還・祟り・災いはいずれも負の帰結である。

**この伝承における現れ方**
侵入は危害・非帰還へ結び付けられる。

### 2.3.5. D15: 帰結領域
**何の領域が最終的に変わるか**
- Primary Child / Value: `D15.OPP.LUCK_MISFORTUNE`
- Primary Parent: `D15.OPP`
- Secondary: `D15.LIF.DISAPPEARANCE`
- Status: `I`

**判定根拠**
祟り型では広い災い、非帰還型では共同体へ戻れない結果が語られる。死亡を全型へ一般化しない。

**この伝承における現れ方**
不特定の災い、または藪から戻れない状態が帰結となる。

### 2.3.6. D16: 因果時間構造
**発動条件・原因成立から主作用／主帰結までの関係は、時間上どのように編成されているか**
- Primary Child / Value: `D16.STA.STATIC_RULE`
- Primary Parent: `D16.STA`
- Secondary: なし
- Status: `I`

**判定根拠**
「この場所へ入れば出られない／祟りが生じる」という恒常的条件規則であり、期限・段階進行を必要としない。

**この伝承における現れ方**
時期に依存せず、境界越えと負の帰結が結び付く。

### 2.3.7. D17: 回避・制御方式
**結果をどう回避・制御・利用できるか**
- Primary Child / Value: `D17.RUL.OBEY_TABOO`
- Primary Parent: `D17.RUL`
- Secondary: なし
- Status: `I`

**判定根拠**
「入る者がいない」という里諺や立入禁止が独立した共同体規範として確認されるが、歴史Contentは間接Evidenceである。

**この伝承における現れ方**
内部へ入らないという禁足遵守が回避方式となる。

### 2.3.8. D18: 作用レイヤー
**因果効力は伝承内／受容者／社会現実のどこに及ぶか**
- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `1`
- Status: `I`

**判定根拠**
L1は伝承内の非帰還・祟り。L2は情報接触のみで発動するEvidenceなし。L3は歴史資料に記録された立入忌避・禁止を限定的な現実行動Evidenceとして扱う。不在判断を含むため `I`。

**この伝承における現れ方**
伝承内因果に加え、少なくとも歴史記録上は「入らない／入らせない」という共同体行動へ接続する。

## 2.4. 社会的埋め込みの次元

### 2.4.1. D19: 流通範囲
**誰の間に伝承が流通するか**
- Primary Child / Value: `D19.LOC.LOCAL_TRADITION`
- Primary Parent: `D19.LOC`
- Secondary: なし
- Status: `I`

**判定根拠**
市川市八幡の具体地点・地域史に不可分な伝承圏は確認できる。書物・図像・Webで地域外から閲覧可能であることだけでは全国的一般大衆の実流通を示さないため `NATIONAL_PUBLIC` は付さない。

**この伝承における現れ方**
作用地点と共同体規範は八幡不知森の地域伝承圏へ固定される。

### 2.4.2. D20: 特権情報保持者
**誰が真相・追加情報を持つか**
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**
複数由来説が併存し、誰か一者へ真相保持を固定できるEvidenceがない。

**この伝承における現れ方**
由来諸説の併存と、特権的真相保持者の存在は区別する。

### 2.4.3. D21: 現実アンカー
**実在世界へどの程度固定されるか**
- Primary Child / Value: `D21.A2`
- Primary Parent: なし
- Secondary: なし
- Status: `D`

**判定根拠**
八幡不知森は市川市八幡に実在し、自治体資料で現存地点・竹藪・鳥居・祠等を直接確認できる。

**この伝承における現れ方**
任意の禁足地ではなく、具体的実在地点であることがEntry識別条件となる。
