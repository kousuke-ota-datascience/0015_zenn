記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

Evidence正本は同一ディレクトリの `0081_00_contents.md`、R1/R2/R3監査成果物は `docs/99_work/20260914_A3_full_recoding/` 配下を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0081`
- `伝承エントリ名称`: ピアスの白い糸
- `Macro_Category`: 身体・医療・食品
- `Entry_Type`: 物語・伝説
- `Version_Scope`: 1994年までに現代伝説として採録された共有核。ピアス穴付近から白い糸状物が現れ、それを身体内部の重要な神経（典型的には視神経）と説明し、引くと失明する／視覚を失うと因果付ける型を対象とする。実際に失明する物語型と警告命題型は同一Entry内の異伝として扱う。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示

### 2.1.1. D01: 生成年代
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

1994年採録は直接確認できるが、それ以前の最初期口承・成立年代は固定できない。刊行年を生成年代へ置換せず、1980年代の若者文化という背景だけでも埋めない。

**この伝承における現れ方**

近現代のピアス文化を前提とする伝承だが、共有核がどの年代に成立したかは未確定である。

### 2.1.2. D02: 最古確認流通媒体
- Primary Child / Value: `D02.PRT.BOOK` — 書籍
- Primary Parent: `D02.PRT` — 印刷・書簡
- Secondary: なし
- Status: `D`

**判定根拠**

現在直接確認できる最古の流通媒体は1994年刊『ピアスの白い糸―日本の現代伝説』である。それ以前の口承媒体は現Evidenceで固定しない。

**この伝承における現れ方**

少なくとも1994年には、個別の噂が現代伝説集の一話として読者へ提示されていた。

### 2.1.3. D03: 確認流通媒体ポートフォリオ（Version Scope）
- Primary Child / Value: `D03.PRT.BOOK` — 書籍
- Primary Parent: `D03.PRT` — 印刷・書簡
- Secondary: なし
- Status: `D`

**判定根拠**

Scopeの共有核が1994年書籍へ採録・提示されたことは直接確認できる。学校口承・テレビ等は具体的な流通証拠を固定していないため追加しない。

**この伝承における現れ方**

白い糸と失明の因果核が活字化され、再話可能な現代伝説として保存される。

### 2.1.4. D04: 生成・変容パターン
- Primary Child / Value: `D04.VAR.RETELLING` — 再話
- Primary Parent: `D04.VAR` — 再話・変異
- Secondary: なし
- Status: `I`

**判定根拠**

実際に引いて失明する物語型と、引くと失明すると警告される命題型が共存し、白い糸と失明の因果核を保ちながら再話される。

**この伝承における現れ方**

登場人物や説明者、失明が実際に起きるかは変化する一方、疑似解剖学的な世界規則は維持される。

### 2.1.5. D05: 提示形式
- Primary Child / Value: `D05.PRP.PREDICTIVE_CLAIM` — 予測命題
- Primary Parent: `D05.PRP` — 命題・主張形式
- Secondary: `D05.RUL.WARNING` — 警告
- Status: `I`

**判定根拠**

Scope全体を貫く最小の伝達単位は「白い糸を引けば失明する」という条件付き予測であり、実践上は「引くな」という警告へ変換される。FOAF形式は異伝の一つに留める。

**この伝承における現れ方**

短い身体知識としても、体験談としても流通できるが、再話を横断する中心は行為と結果を結ぶ危険予測である。

### 2.1.6. D06: 真実性提示
- Primary Child / Value: `D06.T5` — 条件付き信念
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**

「白い糸を引けば失明する」という規則を事実らしい危険知識として提示するが、Version Scopeを特定一件の直接体験事実へ限定しない。

**この伝承における現れ方**

聞き手は、医学的に検証済みか否かとは別に、「その条件なら危険らしい」という実践的信念を与えられる。

## 2.2. 意味形成

### 2.2.1. D07: 意味形成対象
- Primary Child / Value: `D07.BHF.BODY_ANOMALY` — 身体異常
- Primary Parent: `D07.BHF` — 身体・健康・食品
- Secondary: `D07.BHF.MEDICAL_RISK` — 医療リスク
- Status: `D`

**判定根拠**

耳から見える白い糸状物の正体と、身体改変後にそれを操作した場合の重大な身体リスクが中心問題である。

**この伝承における現れ方**

見えない身体内部を「一本の白い糸」で可視化し、ピアス後の不可解な身体徴候へ簡単な説明を与える。

### 2.2.2. D08: 意味形成契機
- Primary Child / Value: `D08.TRC.PHYSICAL_TRACE` — 物的痕跡
- Primary Parent: `D08.TRC` — 痕跡・証拠
- Secondary: `D08.DEX.BODILY_SENSATION` — 身体感覚・身体徴候
- Status: `I`

**判定根拠**

目に見える白い糸状物という物的・身体的徴候が、「これは何か」という問題化を開始させる。

**この伝承における現れ方**

抽象的な医学的不安ではなく、耳から垂れる可視的なものが意味形成の手掛かりになる。

### 2.2.3. D09: 意味付与操作
- Primary Child / Value: `D09.CAT.TYPE_ASSIGNMENT` — 類型化
- Primary Parent: `D09.CAT` — カテゴリー化
- Secondary: `D09.CAU.DIRECT_CAUSE` — 直接原因化; `D09.NOR.TABOOIZATION` — 禁忌化
- Status: `I`

**判定根拠**

正体不明の白い糸を「視神経」という既知の身体構造へ割り当て、引く行為を失明の直接原因とし、その操作を禁止すべき行為へ変換する。

**この伝承における現れ方**

「何か分からない糸」が「視神経」という名前を得ることで、原因・結果・禁止規則が一度に成立する。

### 2.2.4. D10: 因果源存在論
- Primary Child / Value: `D10.NAT.BIOPHYSIO_PROCESS` — 生理・生物過程
- Primary Parent: `D10.NAT` — 自然・生物物理過程
- Secondary: なし
- Status: `I`

**判定根拠**

伝承内では、身体内部の神経構造とそれを引くことで起きる生理学的損傷が因果源として置かれる。医学的に正しいかどうかとは別問題である。

**この伝承における現れ方**

超自然主体を置かず、人体の「隠れた配線」によって危険を説明する。

## 2.3. 因果・行動モデル

### 2.3.1. D11: 発動・接触条件
- Primary Child / Value: `D11.MAN.CREATE_ALTER_MANIPULATE` — 作る・加工・操作する
- Primary Parent: `D11.MAN` — 操作・儀式
- Secondary: `D11.SEN.VISUAL_EXPOSURE` — 見る
- Status: `D`

**判定根拠**

危害を発動させる中心条件は白い糸を引く・操作すること。糸を見ることは発見条件だが、見るだけでは失明しないためSecondary。

**この伝承における現れ方**

異物を取り除こうとする日常的な操作が、伝承内では取り返しのつかない危険行為へ反転する。

### 2.3.2. D12: 作用対象
- Primary Child / Value: `D12.FOC.PRACTITIONER` — 実践者
- Primary Parent: `D12.FOC` — 焦点人物
- Secondary: なし
- Status: `I`

**判定根拠**

D13の生理変化を直接受けるのは、白い糸を操作した本人である。

**この伝承における現れ方**

自分の身体から出た糸を引いた本人が、その行為の結果として視覚を失う。

### 2.3.3. D13: 作用機構
- Primary Child / Value: `D13.PHY.PHYSIOLOGICAL_CHANGE` — 生理変化
- Primary Parent: `D13.PHY` — 身体・物質作用
- Secondary: なし
- Status: `D`

**判定根拠**

白い糸の操作によって視覚機能が失われるという身体機能変化が主作用である。

**この伝承における現れ方**

神経を物理的に引き抜くという単純な機械モデルが、即時の視覚喪失へ結び付けられる。

### 2.3.4. D14: 帰結極性
- Primary Child / Value: `D14.NEG` — 負
- Primary Parent: なし
- Secondary: なし
- Status: `D`

**判定根拠**

失明・視力喪失という重大な危害が中心帰結である。

**この伝承における現れ方**

軽い失敗ではなく、不可逆的な身体障害が身体改変への恐怖を最大化する。

### 2.3.5. D15: 帰結領域
- Primary Child / Value: `D15.BOD.SEVERE_INJURY` — 重傷・障害
- Primary Parent: `D15.BOD` — 身体・健康
- Secondary: `D15.BEH.AVOIDANCE_ROUTE_CHANGE` — 回避・経路変更
- Status: `I`

**判定根拠**

主帰結は重大な視覚障害。警告型では、聞き手が白い糸を触らないという行動回避も独立した実践的帰結になる。

**この伝承における現れ方**

身体障害の物語が、そのまま「同じものを見ても触らない」という意思決定へ翻訳される。

### 2.3.6. D16: 因果時間構造
- Primary Child / Value: `D16.EVT.IMMEDIATE` — 即時
- Primary Parent: `D16.EVT` — 単一エピソード・事象内
- Secondary: なし
- Status: `I`

**判定根拠**

代表形では白い糸を引く操作と視覚喪失が直結し、長い潜伏・遅延を必要としない。

**この伝承における現れ方**

「引いた瞬間に取り返しがつかなくなる」という時間圧縮が警告効果を強める。

### 2.3.7. D17: 回避・制御方式
- Primary Child / Value: `D17.RUL.OBEY_TABOO` — 禁忌遵守
- Primary Parent: `D17.RUL` — 規則遵守
- Secondary: `D17.AVO.DO_NOT_ENGAGE` — 接触回避
- Status: `I`

**判定根拠**

実践的な制御規則は「白い糸を引いてはいけない」。対象への操作を避けることで危害を回避する。

**この伝承における現れ方**

不可解な身体徴候へ手を出さないという単純な禁忌が、失明を避ける唯一の行動規則として機能する。

### 2.3.8. D18: 作用レイヤー
- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `0`
- Status: `D`

**判定根拠**

伝承内部では操作→失明という因果がある。話を読む・聞くこと自体が失明を発動する自己適用規則はなく、独立X Evidenceも未確認。

**この伝承における現れ方**

因果作用は基本的に、物語内の身体と操作行為に閉じている。

## 2.4. 社会的分布・現実接続

### 2.4.1. D19: 流通範囲
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

1994年の日本の現代伝説集への採録は確認できるが、この個別話が当時どの社会範囲まで流通していたかを直接固定するEvidenceが不足する。

**この伝承における現れ方**

広く知られた印象と、測定可能な流通範囲のEvidenceを分離する。

### 2.4.2. D20: 特権情報保持者
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

友人・医師・周囲の人物が「視神経」と説明する異伝はあるが、Scope全体で一貫した特権情報保持者を固定できない。

**この伝承における現れ方**

危険知識の出所は再話ごとに変わり、必ず医療専門家だけが知る構造ではない。

### 2.4.3. D21: 現実アンカー
- Primary Child / Value: `D21.A1` — 一般的現実背景
- Primary Parent: なし
- Secondary: なし
- Status: `D`

**判定根拠**

ピアス、耳、眼、神経という一般的現実対象を用いるが、特定の病院・人物・商品・事件へ依存しない。

**この伝承における現れ方**

実在する身体部位と医学用語が、誤った因果モデルへもっともらしさを与える。

# 3. R4 QA

## 3.1. causal chain

```text
D11: 白い糸状物を引く／操作する
→ D12: 操作した本人が直接対象になる
→ D13: 視覚機能を失う生理変化が起きる
→ D15: 重大な視覚障害として帰結する
```

D12/D13/D15は接続する。

## 3.2. L3

社会的なピアス回避効果を推測で付与せず、独立X Evidenceがないため `L3=0`。

## 3.3. U / NA / C

- D01=`U`: 採録年から成立年代を逆算しない。
- D19=`U`: 個別話の流通社会範囲が未固定。
- D20=`U`: 情報保持者が異伝で変動。
- NA: なし。
- C: なし。

## 3.4. taxonomy gap

現行Scopeは既存taxonomyで表現可能。新Child要求なし。

## 3.5. 旧10との差分

R3 freeze後に旧10を確認した。主要差分は、D02 `U→BOOK`（1994年書籍を「最古確認できる媒体」として扱う規則の再適用）、D04 `STABILIZED_CANON→RETELLING`、D05 `FOAF/C→PREDICTIVE_CLAIM + WARNING`、D06 `T4→T5`、D08を異伝競合Cではなく白い糸という共通の物的徴候へ統合、D09 `DIRECT_CAUSE→TYPE_ASSIGNMENT`をPrimaryへ、D19をEvidence不足としてUにした点である。差分は現行coding rulesとVersion Scopeを優先した結果である。