記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

伝承内容・典拠・異伝・証拠上の不確実性は、同一ディレクトリの `0178_00_contents.md` を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0178`
- `伝承エントリ名称`: 猿夢
- `Macro_Category`: ネット怪談
- `Entry_Type`: 物語・伝説
- `Version_Scope`: 2000年8月2日の保存転載で確認できる一人称反復夢型。2003年「猿夢＋」等の受容者側派生はHistory Evidenceとして分離し、D11〜D18のScopeへ混ぜない。

# 2. 分析概念次元

各次元は、コード値、判定根拠、この伝承における現れ方の順で記述する。

## 2.1. 来歴・流通・提示の次元

### 2.1.1. D01: 生成年代
**いつ成立したか**
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**
2000年8月2日は保存転載で確認できる最古級定点であり、生成年そのものではない。

**この伝承における現れ方**
少なくとも2000年には掲示板投稿として流通していたが、それ以前の形成有無は固定できない。

### 2.1.2. D02: 最古確認流通媒体
**現在の証拠で、伝承が実際に人から人へ流通したことを最も古く確認できる媒体は何か**
- Primary Child / Value: `D02.WEB.FORUM`
- Primary Parent: `D02.WEB`
- Secondary: なし
- Status: `I`

**判定根拠**
保存転載が匿名掲示板投稿を示すが、原ページ未固定である。

**この伝承における現れ方**
初期確認形は匿名掲示板の一人称怪談である。

### 2.1.3. D03: 確認流通媒体ポートフォリオ（Version Scope）
**今回コードするVersion Scopeで、伝承が受容者へ流通したことを確認できる媒体は何か**
- Primary Child / Value: `D03.WEB.FORUM`
- Primary Parent: `D03.WEB`
- Secondary: `D03.WEB.WEBSITE`
- Status: `I`

**判定根拠**
掲示板投稿として保存され、現在は保存Webページから確認できる。原投稿未固定なので `I` とする。

**この伝承における現れ方**
掲示板投稿が後代Web保存を介して再閲覧される。

### 2.1.4. D04: 生成・変容パターン
**時間とともにどう変形したか**
- Primary Child / Value: `D04.VAR.ACCRETION`
- Primary Parent: `D04.VAR`
- Secondary: なし
- Status: `I`

**判定根拠**
00で特定した後代起源検討資料により、少なくとも2003年5月には元話を読んだ別人物が類似夢を語る「猿夢＋」が確認される。絶対的初出は固定しない。

**この伝承における現れ方**
投稿者自身の反復夢から、読者側へ類似夢が及ぶ派生が追加される。

### 2.1.5. D05: 提示形式
**どんなコミュニケーション形式で提示されるか**
- Primary Child / Value: `D05.EXP.RETROSPECTIVE_1P`
- Primary Parent: `D05.EXP`
- Secondary: なし
- Status: `I`

**判定根拠**
保存転載では投稿者が自身の夢体験を一人称で回想する。原投稿未固定のためStatusは `I`。

**この伝承における現れ方**
自己体験として語られるネット怪談である。

### 2.1.6. D06: 真実性提示
**どんな「本当らしさ」を要求するか**
- Primary Child / Value: `D06.T1`
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**
保存転載上では投稿者自身が経験した夢として提示される。原ページ未固定のため `D` にはしない。

**この伝承における現れ方**
読者には一人称の直接体験報告として差し出される。

## 2.2. 意味形成の次元

### 2.2.1. D07: 意味形成対象
**何が不可解・不確実なのか**
- Primary Child / Value: `D07.ANO.DREAM_SLEEP_ANOMALY`
- Primary Parent: `D07.ANO`
- Secondary: `D07.ANO.UNEXPLAINED_EVENT`
- Status: `I`

**判定根拠**
同じ夢が4年後に前回の続きから再開し、危害系列と警告が継続する。

**この伝承における現れ方**
単発の悪夢ではなく連続性を持つ反復夢として問題化される。

### 2.2.2. D08: 意味形成契機
**何を手掛かりに問題化されるか**
- Primary Child / Value: `D08.DEX.DREAM_SLEEP_EVENT`
- Primary Parent: `D08.DEX`
- Secondary: なし
- Status: `I`

**判定根拠**
投稿者が経験すると語る夢そのものが契機である。Contentは保存転載からの確認なので `I`。

**この伝承における現れ方**
睡眠中の夢経験が異常認識の入口になる。

### 2.2.3. D09: 意味付与操作
**不可解なものをどう理解可能にするか**
- Primary Child / Value: `D09.UNK.DELIBERATE_NONRESOLUTION`
- Primary Parent: `D09.UNK`
- Secondary: なし
- Status: `I`

**判定根拠**
夢の原因、放送主体、反復理由は説明されない。

**この伝承における現れ方**
異常な連続夢という現象だけが残り、真相は閉じられない。

### 2.2.4. D10: 因果源存在論
**原因を何として世界に置くか**
- Primary Child / Value: `D10.PHN.DREAM_SLEEP_PHENOMENON`
- Primary Parent: `D10.PHN`
- Secondary: なし
- Status: `I`

**判定根拠**
独立した怪物・霊・人物主体は固定できず、反復する夢現象そのものが最も保守的である。

**この伝承における現れ方**
原因主体を擬人化せず、異常な夢現象として保持する。

## 2.3. 因果・行動モデルの次元

### 2.3.1. D11: 発動・接触条件
**何を契機に因果系へ入るか**
- Primary Child / Value: `D11.PAS.SLEEP_DREAM`
- Primary Parent: `D11.PAS`
- Secondary: なし
- Status: `I`

**判定根拠**
睡眠・夢状態になることで無人駅と電車の出来事へ入る。保存転載由来のため `I`。

**この伝承における現れ方**
夢を見ることが危害系列への入口になる。

### 2.3.2. D12: 作用対象
**誰／何に作用するか**
- Primary Child / Value: `D12.FOC.PROTAGONIST_EXPERIENCER`
- Primary Parent: `D12.FOC`
- Secondary: なし
- Status: `I`

**判定根拠**
焦点人物は投稿者本人であり、他乗客は投稿者が目撃する先行被害者である。

**この伝承における現れ方**
他乗客への危害を目撃した後、投稿者自身へ順番が近づく。

### 2.3.3. D13: 作用機構
**因果源が対象へ何をするか**
- Primary Child / Value: なし — **taxonomy gap: 投稿者へ危害が迫る／処理対象として順番が来る**
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**
他乗客には身体処理が実現するが、投稿者本人は実際に処理される前に覚醒する。D12対象の投稿者に対する作用として `PHYSICAL_ATTACK` を確定すると、実現した他乗客への危害と未実現の投稿者脅威を混同するため、taxonomy gapとして保守化する。

**この伝承における現れ方**
投稿者は処理順序へ組み込まれ、危害が直前まで迫るが、実害成立前に覚醒する。

### 2.3.4. D14: 帰結極性
**結果は正・負・中立・混合か**
- Primary Child / Value: `D14.NEG`
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**
残酷な危害と投稿者への脅威として提示される。保存転載由来なので `I`。

**この伝承における現れ方**
利益・中立ではなく明確な危険として描かれる。

### 2.3.5. D15: 帰結領域
**何の領域が最終的に変わるか**
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**
投稿者の終端は二度の覚醒と次回警告であり、持続的傷害・死亡・トラウマ等を固定できない。

**この伝承における現れ方**
危害は迫るが、投稿者の持続的終端状態は不明である。

### 2.3.6. D16: 因果時間構造
**発動条件・原因成立から主作用／主帰結までの関係は、時間上どのように編成されているか**
- Primary Child / Value: `D16.REC.RECURRENT`
- Primary Parent: `D16.REC`
- Secondary: なし
- Status: `I`

**判定根拠**
一度覚醒して収束した後、4年後に同じ夢が前回の続きから再び起こる。条件成立のたび必ず再発する規則は示されないため `TRIGGERED_RECURRENCE` ではなく `RECURRENT` とする。

**この伝承における現れ方**
一回の夢の後、長い間隔を置いて不定期に同じ系列が再開する。

### 2.3.7. D17: 回避・制御方式
**結果をどう回避・制御・利用できるか**
- Primary Child / Value: なし — **taxonomy gap: 夢から意図的に覚醒して離脱する**
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**
投稿者は夢だと認識し、覚醒によって危害場面から離脱する。物理的逃走ではないため既存FLEE系へ近似しない。

**この伝承における現れ方**
制御は夢世界内の移動ではなく、睡眠状態そのものから抜けることで行われる。

### 2.3.8. D18: 作用レイヤー
**因果効力は伝承内／受容者／社会現実のどこに及ぶか**
- `D18.L1`: `1`
- `D18.L2`: `0`
- `D18.L3`: `0`
- Status: `I`

**判定根拠**
2000年Scopeでは夢内因果のみ確認する。読者感染型は後代派生なのでL2へ混ぜず、独立した社会現実効果も固定しない。

**この伝承における現れ方**
怪異作用は2000年型では投稿者の夢内に留まる。

## 2.4. 社会的埋め込みの次元

### 2.4.1. D19: 流通範囲
**誰の間に伝承が流通するか**
- Primary Child / Value: `D19.NET.OPEN_FORUM_WEB`
- Primary Parent: `D19.NET`
- Secondary: なし
- Status: `I`

**判定根拠**
匿名掲示板投稿として保存されていることから公開Web流通を推定できるが、原ログ未固定のため `I`。

**この伝承における現れ方**
ネット掲示板上の不特定受容者へ提示される怪談である。

### 2.4.2. D20: 特権情報保持者
**誰が真相・追加情報を持つか**
- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**
夢の正体や反復理由を知る主体は固定できない。

**この伝承における現れ方**
警告は存在するが、真相を説明できる主体は示されない。

### 2.4.3. D21: 現実アンカー
**実在世界へどの程度固定されるか**
- Primary Child / Value: `D21.A1`
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**
具体的実在地点ではなく匿名掲示板投稿と一般的な駅・乗り物イメージに弱く接続する。

**この伝承における現れ方**
現実の固有場所より、投稿記録そのものが主な現実接点である。