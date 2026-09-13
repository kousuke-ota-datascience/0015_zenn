記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

伝承内容・典拠・異伝・証拠上の不確実性は、同一ディレクトリの `0113_00_contents.md` を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `113`
- `伝承エントリ名称`: 井の頭公園のボートに乗ると別れる
- `Macro_Category`: ジンクス・俗信
- `Entry_Type`: 俗信・ジンクス
- `Version_Scope`: 井の頭池で恋人同士がボートに乗ると後に別れるという場所ジンクス。弁財天嫉妬説は主要因果異伝として参照する。

# 2. 分析概念次元

各次元は、単にコード値を記録するだけでなく、**当該伝承においてその次元が具体的にどのように現れているか**まで自然言語で記述する。

## 2.1. 来歴・流通・提示の次元

### 2.1.1. D01: 生成年代

**いつ成立したか**

- Primary Child / Value: `D01.G4` — 1980年代
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**

Pilotは昭和後期のデートスポット俗信として1980年代を推定する。

**この伝承における現れ方**

現代には広く定型化しているが、今回1980年代の最初期一次資料は固定できていない。年代はtask 3で追加資料と再照合するが、task 2では既存値を保持する。

### 2.1.2. D02: 最古確認流通媒体

**現在の証拠で、伝承が実際に人から人へ流通したことを最も古く確認できる媒体は何か**

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

噂の最初の実流通媒体を直接確定できない。

**この伝承における現れ方**

口コミ先行と考えられるが、口承を最古と断定できる直接証拠はない。最古媒体は推測で埋めず `U` を保持する。

### 2.1.3. D03: 確認流通媒体ポートフォリオ（Version Scope）

**今回コードするVersion Scopeで、伝承が受容者へ流通したことを確認できる媒体は何か**

- Primary Child / Value: `D03.ORL.PEER_ORAL` — 友人・仲間内伝承
- Primary Parent: `D03.ORL`
- Secondary: なし
- Status: `I`

**判定根拠**

デートスポットのジンクスとして仲間・恋人間で伝えられるというPilot判断を保持する。

**この伝承における現れ方**

「あそこのボートに乗ると別れるらしい」という短い警告が、場所利用前の意思決定に直接入り込む。旧末尾メモ `D03:U` と本文値が矛盾するため、task 3およびExcel正本照合で再確認する。

### 2.1.4. D04: 生成・変容パターン

**時間とともにどう変形したか**

- Primary Child / Value: `D04.STB.STABILIZED_CANON`
- Primary Parent: `D04.STB`
- Secondary: なし
- Status: `I`

**判定根拠**

「カップル＋井の頭池ボート→別れる」という短い核が長期に安定する。

**この伝承における現れ方**

原因説明が変化しても場所・行為・帰結の三項関係は維持される。弁財天嫉妬説は核へ付加される説明として扱う。

### 2.1.5. D05: 提示形式

**どんなコミュニケーション形式で提示されるか**

- Primary Child / Value: `D05.RUL.JINX_RULE`
- Primary Parent: `D05.RUL`
- Secondary: なし
- Status: `D`

**判定根拠**

伝承は条件と結果を一文で結ぶジンクス規則として直接提示される。

**この伝承における現れ方**

完結した事件物語を必要とせず、「乗ると別れる」という予測規則だけで流通可能である。

### 2.1.6. D06: 真実性提示

**どんな「本当らしさ」を要求するか**

- Primary Child / Value: `D06.T5` — 条件付き信念
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**

強い事実断定ではなく、「気にする人は避ける」程度の条件付き信念として機能する。

**この伝承における現れ方**

信じ切らなくても、恋愛上の損失が大きいため回避行動を合理化し得る。完全な事実認定を要求しないまま行動へ作用する。

## 2.2. 意味形成の次元

### 2.2.1. D07: 意味形成対象

**何が不可解・不確実なのか**

- Primary Child / Value: `D07.DCF.MISFORTUNE_STREAK` — 不運・成功失敗の偏り
- Primary Parent: `D07.DCF`
- Secondary: なし
- Status: `D`

**判定根拠**

「なぜ恋人同士が別れるのか」という恋愛上の不運を場所・行為へ結び付ける。

**この伝承における現れ方**

一般的に起こり得る破局を、井の頭池でのボート乗船という特定過去行為で説明可能にする。

### 2.2.2. D08: 意味形成契機

**何を手掛かりに問題化されるか**

- Primary Child / Value: `D08.CLM.INHERITED_SAYING`
- Primary Parent: `D08.CLM`
- Secondary: なし
- Status: `I`

**判定根拠**

個別破局の調査から生じるより、既にある「乗ると別れる」という言い伝えが先行する。

**この伝承における現れ方**

噂を知った後の破局が、ジンクスの確認例として再解釈される。伝承規則そのものが出来事を見る枠組みになる。

### 2.2.3. D09: 意味付与操作

**不可解なものをどう理解可能にするか**

- Primary Child / Value: `D09.NOR.GOOD_BAD_VALUATION`
- Primary Parent: `D09.NOR`
- Secondary: なし
- Status: `I`

**判定根拠**

ボート乗船を恋愛上の「悪い選択」として価値付ける。

**この伝承における現れ方**

本来中立的なレジャー行為が、関係維持の観点では避けるべき行為へ規範化される。

### 2.2.4. D10: 因果源存在論

**原因を何として世界に置くか**

- Primary Child / Value: `D10.SPC.SPECIFIC_PLACE`
- Primary Parent: `D10.SPC`
- Secondary: なし
- Status: `I`

**判定根拠**

Pilotは井の頭池という特定場所を因果源としている。

**この伝承における現れ方**

同じボートでも他の池ではなく、井の頭池で乗ることに意味がある。弁財天嫉妬型では因果源が人格神格へ移るため、Version Scope差はtask 3で再評価する。

## 2.3. 因果・行動モデルの次元

### 2.3.1. D11: 発動・接触条件

**何を契機に因果系へ入るか**

- Primary Child / Value: `D11.MOV.RIDE_BOARD`
- Primary Parent: `D11.MOV`
- Secondary: なし
- Status: `D`

**判定根拠**

恋人同士でボートに乗ることが明示的条件である。

**この伝承における現れ方**

公園訪問だけではなく乗船行為が因果系への入口になる。

### 2.3.2. D12: 作用対象

**誰／何に作用するか**

- Primary Child / Value: `D12.KIN.PARTNER_FRIEND`
- Primary Parent: `D12.KIN`
- Secondary: なし
- Status: `D`

**判定根拠**

ジンクスの対象は恋人関係の二人である。

**この伝承における現れ方**

個人の健康ではなく、二者間関係そのものが作用対象になる。

### 2.3.3. D13: 作用機構

**因果源が対象へ何をするか**

- Primary Child / Value: `D13.FAT.CURSE_MISFORTUNE`
- Primary Parent: `D13.FAT`
- Secondary: なし
- Status: `D`

**判定根拠**

乗船後に破局という不運が付与されるとする。

**この伝承における現れ方**

弁財天型では嫉妬が吉凶作用を人格化し、純ジンクス型では作用主体を明示しない。

### 2.3.4. D14: 帰結極性

**結果は正・負・中立・混合か**

- Primary Child / Value: `D14.NEG`
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**

帰結は恋愛関係の破綻で一貫して負である。

**この伝承における現れ方**

利得型・中立型は中心Versionにない。

### 2.3.5. D15: 帰結領域

**何の領域が最終的に変わるか**

- Primary Child / Value: `D15.SOC.RELATION_BREAKDOWN`
- Primary Parent: `D15.SOC`
- Secondary: なし
- Status: `D`

**判定根拠**

「別れる」が直接の予測結果である。

**この伝承における現れ方**

身体や財産ではなく、親密関係の継続可否が変化するとされる。

### 2.3.6. D16: 因果時間構造

**発動条件・原因成立から主作用／主帰結までの関係は、時間上どのように編成されているか**

- Primary Child / Value: `D16.DLY.DELAYED`
- Primary Parent: `D16.DLY`
- Secondary: なし
- Status: `I`

**判定根拠**

ボート乗船直後ではなく、その後の不特定時点で破局が起こる。

**この伝承における現れ方**

遅延があるため、後の別れを遡って乗船へ帰属できる。

### 2.3.7. D17: 回避・制御方式

**結果をどう回避・制御・利用できるか**

- Primary Child / Value: `D17.AVO.DO_NOT_ENGAGE`
- Primary Parent: `D17.AVO`
- Secondary: なし
- Status: `I`

**判定根拠**

最も単純な回避法は恋人同士でボートに乗らないことである。

**この伝承における現れ方**

ジンクスを知ることがデートコース選択を変える。

### 2.3.8. D18: 作用レイヤー

**因果効力は伝承内／受容者／社会現実のどこに及ぶか**

- Primary Child / Value: `D18.L1=1; D18.L2=0; D18.L3=0`
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**

Pilotは伝承内部の乗船→破局のみを採用している。

**この伝承における現れ方**

L1ではジンクス因果が作動する。一方、現実にボートを避けるという受容者行動が容易に想定され、Web上でもその行動選択が語られる。L2=0はtask 3で再検討余地があるが、task 2では既存値を保持する。

## 2.4. 社会的埋め込みの次元

### 2.4.1. D19: 流通範囲

**誰の間に伝承が流通するか**

- Primary Child / Value: `D19.MAS.NATIONAL_PUBLIC`
- Primary Parent: `D19.MAS`
- Secondary: なし
- Status: `I`

**判定根拠**

東京の一地点に固定された内容だが、Web調査・メディアを通じ全国的に知られる。

**この伝承における現れ方**

流通範囲は広いが、作用地点は極めて局所的という組合せを持つ。

### 2.4.2. D20: 特権情報保持者

**誰が真相・追加情報を持つか**

- Primary Child / Value: `D20.NON.COMMON_KNOWLEDGE`
- Primary Parent: `D20.NON`
- Secondary: なし
- Status: `I`

**判定根拠**

噂の内容は一般に共有され、秘密保持者を必要としない。

**この伝承における現れ方**

地域案内や自治体サイトでも紹介されるほど公開された共同知となっている。

### 2.4.3. D21: 現実アンカー

**実在世界へどの程度固定されるか**

- Primary Child / Value: `D21.A2` — 具体的実在対象
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**

井の頭恩賜公園、井の頭池、弁財天、ボート場という実在対象に強く固定される。

**この伝承における現れ方**

場所名を任意に置換するとこのEntry固有性が失われる。
