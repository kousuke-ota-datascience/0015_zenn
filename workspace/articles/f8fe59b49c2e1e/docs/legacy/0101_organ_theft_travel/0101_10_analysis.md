記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

Evidence正本は同一ディレクトリの `0101_00_contents.md` を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0101`
- `伝承エントリ名称`: 海外旅行で臓器を抜かれる
- `Macro_Category`: 犯罪・社会不安
- `Entry_Type`: FOAF
- `Version_Scope`: 1991年Deseret Newsで同時代に確認できる成人旅行者／出張者kidney-heist型。旅行・出張中の成人男性が酒場・ホテルバーで女性と接触し、薬物で無力化された後に腎臓を摘出されたと説明される型を対象とする。同記事内の第1異伝（本人がホテルへ助けを求め、闇市場への売却が説明される型）と追加異伝（背中の新しい傷、警察、医師による腎臓摘出確認を伴う型）をContent Scopeに含める。1990年代後半のkidney/organ theft関連newsgroup増加は近縁family一般のHistory Evidenceとして保持するが、現Version Scopeの成人旅行者型のBBS流通・ネット再増幅を直接示すものとは扱わない。1980年代の子ども臓器盗難rumor一般、浴槽・メモ等の未固定細部は必須条件にしない。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示の次元

### 2.1.1. D01: 生成年代

**いつ成立したか**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

1991年新聞は成人旅行者型の存在確認点であり、生成時期そのものを示さない。独立した成立History Evidenceは固定できていない。

**この伝承における現れ方**

1991年までには成人旅行者kidney-heist型が流通しているが、それ以前の成立時期は未確定である。

### 2.1.2. D02: 最古確認流通媒体

**現在の証拠で、伝承が実際に人から人へ流通したことを最も古く確認できる媒体は何か**

- Primary Child / Value: `D02.PRT.NEWSPAPER` — 新聞
- Primary Parent: `D02.PRT`
- Secondary: なし
- Status: `D`

**判定根拠**

1991-11-15のDeseret Newsで成人旅行者・薬物・腎臓摘出・闇市場という型と、身体痕跡・医療確認を伴う追加異伝を直接確認できる。

**この伝承における現れ方**

読者から寄せられた噂が新聞紙面上でurban legendとして可視化される。

### 2.1.3. D03: 確認流通媒体ポートフォリオ（Version Scope）

**今回コードするVersion Scopeで、伝承が受容者へ流通したことを確認できる媒体は何か**

- Primary Child / Value: `D03.PRT.NEWSPAPER` — 新聞
- Primary Parent: `D03.PRT`
- Secondary: なし
- Status: `D`

**判定根拠**

1991年Deseret News本文で、現Version Scopeの成人旅行者／出張者型を直接確認できる。一方、Donovan 2002が示す1996〜1998年のkidney/organ theft関連newsgroup投稿増加については個別投稿本文を固定しておらず、現Version Scopeと同一内容がBBSで流通したことは確認できないためD03には含めない。

**この伝承における現れ方**

現Version Scopeの具体内容を直接確認できる流通媒体は新聞である。

### 2.1.4. D04: 生成・変容パターン

**時間とともにどう変形したか**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

1991年記事内には複数異伝があるが、その先後・派生方向は固定できない。Donovan 2002の投稿量増加はkidney/organ theft legend family一般のHistory Evidenceであり、現Version Scopeの成人旅行者型そのものがネット上で再増幅したことを固定しないため `DIGITAL_REAMPLIFICATION` は採用しない。

**この伝承における現れ方**

複数異伝と近縁familyのネット流通増加は確認できるが、現Version Scopeの変容機構は未確定である。

### 2.1.5. D05: 提示形式

**どんなコミュニケーション形式で提示されるか**

- Primary Child / Value: `D05.HRS.FOAF` — FOAF
- Primary Parent: `D05.HRS`
- Secondary: なし
- Status: `I`

**判定根拠**

1991年記事は読者が最近聞いた具体的被害談として投稿し、Brunvandがurban legendとして応答する。後代の全媒体を一律に同じ提示形式とはしない。

**この伝承における現れ方**

具体的な成人旅行者の被害談として、近接伝聞的な事件話の形を取る。

### 2.1.6. D06: 真実性提示

**どんな「本当らしさ」を要求するか**

- Primary Child / Value: `D06.T2` — 近接伝聞事実
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**

1991年の投稿は具体的被害事例として提示される一方、投稿者自身またはBrunvandがurban legendである可能性・判断を明示する。受容者向け原話そのものを直接採録した資料ではないためIとする。

**この伝承における現れ方**

「旅行者に実際に起きた事件らしい」という事件伝聞として流通する。

## 2.2. 意味形成の次元

### 2.2.1. D07: 意味形成対象

**何が不可解・不確実なのか**

- Primary Child / Value: `D07.ICT.HIDDEN_CRIMINAL_PRACTICE` — 隠れた犯罪慣行
- Primary Parent: `D07.ICT`
- Secondary: なし
- Status: `I`

**判定根拠**

薬物で無力化された空白時間に腎臓摘出・売買が秘密裏に行われたと説明される。誘拐・暴行という一般的対人被害はこの中心問題に内包され、独立した意味形成対象として追加しない。

**この伝承における現れ方**

通常の旅行空間の背後に、表から見えない臓器摘出犯罪が存在すると理解される。

### 2.2.2. D08: 意味形成契機

**何を手掛かりに問題化されるか**

- Primary Child / Value: `D08.STY.FOAF_REPORT` — 近接伝聞
- Primary Parent: `D08.STY`
- Secondary: なし
- Status: `I`

**判定根拠**

現Version Scopeの両異伝は、読者が「最近聞いた具体的被害談」として新聞へ持ち込んだ近接伝聞によって受容者側で問題化される。追加異伝にのみ現れる背中の新しい傷は、当該異伝内部では有力なcueだが、Scope横断で共通する契機ではないためPrimaryへ置かず、Secondaryとしても独立安定構造へ引き上げない。

**この伝承における現れ方**

「旅行者が薬物で無力化され腎臓を抜かれたらしい」という具体的被害報告が、隠れた犯罪慣行を問題化する入口となる。

### 2.2.3. D09: 意味付与操作

**不可解なものをどう理解可能にするか**

- Primary Child / Value: `D09.AGN.AGENCY_ATTRIBUTION` — 主体化
- Primary Parent: `D09.AGN`
- Secondary: `D09.CAU.HIDDEN_CAUSE` — 隠れた原因化
- Status: `I`

**判定根拠**

薬物による無力化と臓器喪失を、人間犯罪者による秘密の摘出行為へ帰属する。明示Evidenceのない禁忌化は加えない。

**この伝承における現れ方**

不可解な身体損失が、見えない犯罪行為の結果として説明される。

### 2.2.4. D10: 因果源存在論

**原因を何として世界に置くか**

- Primary Child / Value: なし — **taxonomy gap: 個人／集団／組織未特定の人間加害主体**
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**

1991年Content Evidenceから、薬物投与・腎臓摘出の因果源が超自然ではなく人間加害主体であることは明確である。一方、酒場・ホテルバーで接触する女性と摘出実行者が同一人物か複数人か、非制度的集団・組織かは固定できない。これは因果源自体が不明なのではなく、「人間加害主体までは確定するが、個人／集団／組織未特定」という構造を現行D10 Childで表現できないtaxonomy gapである。近似Childへ押し込まない。

**この伝承における現れ方**

人間犯罪者が原因であることは明確だが、その人数・組織形態は未特定であり、現行taxonomyでは直接コードできない。

## 2.3. 因果・行動モデルの次元

### 2.3.1. D11: 発動・接触条件

**何を契機に因果系へ入るか**

- Primary Child / Value: `D11.SOC.MEET_INTERACT` — 会う・会話する
- Primary Parent: `D11.SOC`
- Secondary: なし
- Status: `D`

**判定根拠**

1991年記事の両異伝で直接確認できる入口は、旅行者・出張者が酒場・ホテルバーで女性と接触すること。加害者側の不可視な「選択」を補わない。

**この伝承における現れ方**

通常の社交的接触が被害エピソードの入口となる。

### 2.3.2. D12: 作用対象

**誰／何に作用するか**

- Primary Child / Value: `D12.OTH.VICTIM_TARGET` — 被害者・標的
- Primary Parent: `D12.OTH`
- Secondary: なし
- Status: `D`

**判定根拠**

薬物投与・臓器摘出を受ける成人旅行者・出張者が作用対象である。

**この伝承における現れ方**

一般の旅行者・出張者が犯罪被害の対象になる。

### 2.3.3. D13: 作用機構

**因果源が対象へ何をするか**

- Primary Child / Value: `D13.PHY.BODY_TRANSFORMATION` — 身体変容
- Primary Parent: `D13.PHY`
- Secondary: なし
- Status: `I`

**判定根拠**

中心作用は身体内部から腎臓を摘出する不可逆的改変であり、1991年記事で確認できる。一方、制度・集団による監禁・行動強制は明示されず、`COERCE_CONFINE` を過程から推定してSecondaryに置かない。

**この伝承における現れ方**

身体内部そのものが盗難対象となり、身体状態が不可逆に変化する。

### 2.3.4. D14: 帰結極性

**結果は正・負・中立・混合か**

- Primary Child / Value: `D14.NEG` — 負
- Primary Parent: なし
- Secondary: なし
- Status: `D`

**判定根拠**

腎臓喪失・重大身体被害が中心帰結である。

**この伝承における現れ方**

被害者に利益をもたらす型ではない。

### 2.3.5. D15: 帰結領域

**何の領域が最終的に変わるか**

- Primary Child / Value: `D15.BOD.SEVERE_INJURY` — 重傷・障害
- Primary Parent: `D15.BOD`
- Secondary: なし
- Status: `I`

**判定根拠**

腎臓喪失という重大な身体損失はScope Contentへ直接接続できる。一方、将来の選択肢・運命が狭まることは伝承内で明示されないため、一般医学的推論から `FUTURE_CONSTRAINT` を追加しない。

**この伝承における現れ方**

被害エピソードは腎臓を失った重い身体被害として終わる。

### 2.3.6. D16: 因果時間構造

**発動条件・原因成立から主作用／主帰結までの関係は、時間上どのように編成されているか**

- Primary Child / Value: `D16.EVT.SEQUENTIAL_EPISODE` — エピソード内段階進行
- Primary Parent: `D16.EVT`
- Secondary: なし
- Status: `I`

**判定根拠**

接触→薬物による無力化→腎臓摘出→事後発覚という一エピソード内の段階進行が確認できる。被害者が犯行中に意識・認識を失っていることは、主作用が時間的に遅延して発生することを意味しないため `DELAYED` は採用しない。

**この伝承における現れ方**

社交的接触から無力化・摘出・事後発覚へ、単一の被害エピソード内で順序立って進む。

### 2.3.7. D17: 回避・制御方式

**結果をどう回避・制御・利用できるか**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

1991年追加異伝では、警察が背中の傷を認め、医師が診察して既に腎臓が摘出されていることを確認する。しかし、診断・確認によって臓器喪失を解除する、被害を防ぐ、結果を制御するといった機能はScope Evidenceから固定できない。医師の登場だけで `MEDICAL_PROFESSIONAL` を制御方式として採用せず、事前回避規則も因果展開から逆算しない。

**この伝承における現れ方**

被害後の診断経路は確認できるが、結果を回避・解除・制御する方法は現Evidenceでは未確定である。

### 2.3.8. D18: 作用レイヤー

**因果効力は伝承内／受容者／社会現実のどこに及ぶか**

- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `0`
- Status: `I`

**判定根拠**

物語内部の犯罪者→旅行者作用は確認できる。受容行為自体が被害条件となるEvidence、流通が社会制度・市場等を変えた独立X Evidenceは固定していない。不在値を含むvector全体はIとする。

**この伝承における現れ方**

確認できる因果効力は物語内部に留まる。

## 2.4. 社会的埋め込みの次元

### 2.4.1. D19: 流通範囲

**誰の間に伝承が流通するか**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

1991年Deseret Newsで成人旅行者型が紙面上に存在することは確認できるが、この具体Versionがどの社会集団・地理範囲へどの程度流通したかを独立に固定できない。Antonijević 2007の国際性とDonovan 2002のnewsgroup増加はorgan-theft legend family一般のEvidenceであり、現Version Scopeへ `TRANSNATIONAL` や `OPEN_FORUM_WEB` を移植しない。

**この伝承における現れ方**

成人旅行者型の存在は確認できるが、その社会的流通範囲は現Evidenceでは未確定である。

### 2.4.2. D20: 特権情報保持者

**誰が真相・追加情報を持つか**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

第1異伝では被害者本人がホテルへ助けを求め、追加異伝では警察・医師が身体状態を確認する。Scope内で情報確定経路が異なるため、単一の特権情報保持者Childへ固定しない。

**この伝承における現れ方**

被害の真相が誰によって確定されるかは異伝により異なる。

### 2.4.3. D21: 現実アンカー

**実在世界へどの程度固定されるか**

- Primary Child / Value: `D21.A1` — 一般的現実背景
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**

旅行、酒場、病院、外科処置、臓器市場等の現実的要素を用いるが、特定制度・事件・組織が成立条件として固定されていない。現実の臓器取引一般をScopeの制度条件へ引き上げない。

**この伝承における現れ方**

一般的な旅行・医療・犯罪の現実背景に埋め込まれる。

# 3. 横断QA

## 3.1. Review_004対応

- D07: `HIDDEN_CRIMINAL_PRACTICE` を中心対象とし、重複する `ABDUCTION_ASSAULT` Secondaryを除外。
- D08: Scope横断で確認できる受容者側の近接伝聞をPrimaryへ置き、一異伝固有の身体痕跡をPrimary/Secondaryから除外。
- D16: `SEQUENTIAL_EPISODE` を維持し、意識喪失を時間的遅延へ読み替えず `DELAYED` Secondaryを除外。
- Review_003で確定したD03/D04/D10/D17/D19の保守化を維持。

## 3.2. Sense-making chain

```text
通常の旅行・出張中に、認識できない時間帯で腎臓を失う隠れた犯罪被害
→ 近接伝聞の具体的被害報告を手掛かりに問題化
→ 薬物・臓器喪失を人間犯罪者の行為へ主体・原因帰属
→ 人間加害主体であることは明確だが、主体形態を表すChildが不足（taxonomy gap）
→ 酒場・ホテルバーで見知らぬ女性と接触
→ 成人旅行者・出張者が被害対象となる
→ 腎臓摘出による身体変容
→ 重傷・障害として終わる
→ 回避・制御方式は未確定
```

Evidenceにない犯罪ネットワーク構造、監禁、長期将来制約、浴槽・メモ、Scope外familyの国際流通を分析値へ持ち込まない。