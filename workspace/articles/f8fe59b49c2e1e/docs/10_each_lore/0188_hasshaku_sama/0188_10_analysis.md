記載内容は固定baselineの理論設計・Parent/Child code system・coding rulesに従う。Evidenceは `0188_00_contents.md` を参照する。

# 1. 伝承エントリ基本情報
- `Entry_ID`: `0188`
- `伝承エントリ名称`: 八尺様
- `Macro_Category`: ネット怪談
- `Entry_Type`: 物語・伝説
- `Version_Scope`: 2008年8月26日、2ちゃんねる「死ぬほど洒落にならない怖い話を集めてみない？196」レス908〜916。後代派生は除外。

# 2. 分析概念次元

### D01 生成年代
- Primary: `D01.G6` — 2000年代
- Status: `D`
2008年原型を直接確認。

### D02 最古確認流通媒体
- Primary: `D02.WEB.FORUM`
- Parent: `D02.WEB`
- Status: `D`

### D03 確認流通媒体
- Primary: `D03.WEB.FORUM`
- Parent: `D03.WEB`
- Status: `D`
原型Scopeは公開掲示板。

### D04 生成・変容
- Primary: `D04.STB.SINGLE_FIXED`
- Parent: `D04.STB`
- Status: `D`
9レスに分割された一続きの完成回想譚。

### D05 提示形式
- Primary: `D05.EXP.RETROSPECTIVE_1P`
- Parent: `D05.EXP`
- Status: `D`
十年以上前の自身の体験を一人称回想する。

### D06 真実性提示
- Primary: `D06.T1`
- Status: `D`
自身の直接体験として提示される。

### D07 意味形成対象
- Primary: `D07.ANO.UNKNOWN_EXISTENCE`
- Parent: `D07.ANO`
- Secondary: `D07.DCF.UNEXPLAINED_DEATH_LOSS`
- Status: `D/I`
未知存在の正体と、魅入られた若者の死が問題化される。

### D08 意味形成契機
- Primary: `D08.DEX.DIRECT_EVENT`
- Parent: `D08.DEX`
- Secondary: `D08.DEX.PERCEPTUAL_ANOMALY`
- Status: `D`
異常に背の高い女性姿と奇妙な声への直接遭遇。

### D09 意味付与操作
- Primary: `D09.CAT.TYPE_ASSIGNMENT`
- Parent: `D09.CAT`
- Secondary: `D09.AGN.AGENCY_ATTRIBUTION`
- Status: `D/I`
祖母が遭遇を「八尺様」と同定し、過去の死と現在の追跡を同存在へ帰属する。

### D10 因果源存在論
- Primary: `D10.SUP.YOKAI_ENTITY`
- Parent: `D10.SUP`
- Status: `I`
女性姿、意思、声模倣、追跡を行う人格怪異。幽霊・死者霊までは特定しない。

### D11 発動・接触条件
- Primary: `D11.SEN.VISUAL_EXPOSURE`
- Parent: `D11.SEN`
- Secondary: `D11.CON.LOCATION_STATE`
- Status: `D/I`
地域内で八尺様を視認した後に「魅入られた」と判定される。

### D12 作用対象
- Primary: `D12.FOC.PROTAGONIST_EXPERIENCER`
- Parent: `D12.FOC`
- Status: `D`

### D13 作用機構
- Primary: `D13.REL.TARGETING`
- Parent: `D13.REL`
- Status: `D/I`
八尺様が特定主人公を魅入り、声を模倣して誘引し、脱出まで追跡する。

### D14 帰結極性
- Primary: `D14.NEG`
- Status: `I`

### D15 帰結領域
- Primary: `D15.LIF.FUTURE_CONSTRAINT`
- Parent: `D15.LIF`
- Secondary: `D15.BEH.AVOIDANCE_ROUTE_CHANGE`
- Status: `D`
主人公は生存するが祖父母の地域へ戻れず、祖父の葬儀にも参加できない。

### D16 因果時間構造
- Primary: `D16.PRG.STAGED_PROGRESSION`
- Parent: `D16.PRG`
- Status: `D/I`
視認→標的化→夜間接近→護衛脱出→長期回避→地蔵破壊後日談と段階進行。

### D17 回避・制御方式
- Primary: `D17.RUL.OBEY_TABOO`
- Parent: `D17.RUL`
- Secondary: `D17.RIT.RELIGIOUS_SPECIALIST`
- Status: `D`
部屋を出ない、声に応じない、札を持つ、目を開けない等の手順とKの祈祷で脱出する。

### D18 作用レイヤー
- Value: `D18.L1=1; D18.L2=0; D18.L3=0`
- Status: `D/I/I`
物語内作用のみ。読者感染規則と独立社会現実X Evidenceはない。

### D19 流通範囲
- Primary: `D19.NET.OPEN_FORUM_WEB`
- Parent: `D19.NET`
- Status: `D`

### D20 特権情報保持者
- Primary: `D20.INS.LOCAL_RESIDENT`
- Parent: `D20.INS`
- Secondary: `D20.EXP.RELIGIOUS_FOLKLORE`
- Status: `D/I`
祖父母・地域血縁者が規則を知り、Kが専門的対処知を持つ。

### D21 現実アンカー
- Primary: `D21.A1`
- Status: `I`
一般的な農村・祖父母宅という実在背景はあるが、地域名は匿名化され具体的実在対象へ固定できない。

# 3. R4 QA
- pre-SHA: `4d762489c220d89837df9edf603d84780b2195f9`
- R3 freeze: `9d59272c2d141ef05a8e4f9ec80633ceb3fb7a21`
- 旧10なし。
- 主人公の実現帰結を死亡へ誤拡張せず、D15は長期生活制約。
- D18=`1/0/0`。
- H3 Primary exactly 1、Secondary 0–2、Parent derivedを確認。
