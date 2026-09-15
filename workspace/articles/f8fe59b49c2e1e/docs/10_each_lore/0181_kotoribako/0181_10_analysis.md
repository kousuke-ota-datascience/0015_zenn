記載内容は固定baselineの理論設計・Parent/Child code system・coding rulesに従う。Evidenceは `0181_00_contents.md` を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `0181`
- `伝承エントリ名称`: コトリバコ
- `Macro_Category`: ネット怪談
- `Entry_Type`: 物語・伝説
- `Version_Scope`: 2005年6月6日の初回遭遇譚から、2005年6月11日までに直接確認できる早期体系化まで。初回遭遇譚と後続増補の追加時点は区別して保持する。6月11日以後の長期考察・後代翻案は除外する。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示

### D01 生成年代
- Primary Child / Value: `D01.G6` — 2000年代
- Secondary: なし
- Status: `D`

2005年6月の初期投稿群を直接確認できる。2005年以前の同名伝承を示す独立資料は今回確認していない。

### D02 最古確認流通媒体
- Primary Child / Value: `D02.WEB.FORUM` — Web掲示板・フォーラム
- Primary Parent: `D02.WEB`
- Secondary: なし
- Status: `D`

2005年6月の2ちゃんねる投稿と同時代過去ログから、公開Web掲示板での流通を最古確認媒体として固定する。

### D03 確認流通媒体ポートフォリオ
- Primary Child / Value: `D03.WEB.FORUM` — Web掲示板・フォーラム
- Primary Parent: `D03.WEB`
- Secondary: なし
- Status: `D`

初回投稿、専用スレッド、民俗板への持込みのいずれも公開掲示板上で確認できる。

### D04 生成・変容パターン
- Primary Child / Value: `D04.VAR.ACCRETION` — 増補
- Primary Parent: `D04.VAR`
- Secondary: `D04.COL.SERIALIZATION` — シリーズ化
- Status: `D`

6月6日の「危険な箱との遭遇」から数日で、女性・子どもへの危害、等級、複数世代管理、歴史由来が追加される。さらに専用スレ・続編として連続展開する。

### D05 提示形式
- Primary Child / Value: `D05.EXP.RETROSPECTIVE_1P` — 一人称回想
- Primary Parent: `D05.EXP`
- Secondary: `D05.HYB.NARRATIVE_EXPLANATION` — 物語＋解説
- Status: `D/I`

初回投稿者は前月の出来事を自分が居合わせた経験として回想する。その後、Mらから聞いた箱の由来・管理知識が説明として重なる。

### D06 真実性提示
- Primary Child / Value: `D06.T1` — 直接体験事実
- Secondary: なし
- Status: `D`

初回投稿は投稿者自身の経験として提示される。ただし、箱の実在や呪力を外部的に実証するコードではない。

## 2.2. 意味形成

### D07 意味形成対象
- Primary Child / Value: `D07.DCF.UNEXPLAINED_DEATH_LOSS` — 不可解な死・喪失
- Primary Parent: `D07.DCF`
- Secondary: `D07.ANO.UNEXPLAINED_EVENT` — 説明不能事象
- Status: `I`

早期体系化では、女性・子どもの死が箱の呪いによって説明される。初回のMの異常反応や緊急処置は、その危険性を具体化する出来事として配置される。

### D08 意味形成契機
- Primary Child / Value: `D08.DEX.DIRECT_EVENT` — 出来事への直接遭遇
- Primary Parent: `D08.DEX`
- Secondary: `D08.STY.FOAF_REPORT` — 近接伝聞
- Status: `D/I`

投稿者はSによる箱持込みとMの反応・処置を直接経験する。箱の由来・過去の処理についてはM、父、祖父らの知識が伝聞として加わる。

### D09 意味付与操作
- Primary Child / Value: `D09.HST.HISTORICAL_ANCHOR` — 歴史接続
- Primary Parent: `D09.HST`
- Secondary: `D09.CAU.HIDDEN_CAUSE` — 隠れた原因化
- Status: `D/I`

実在する1868年の隠岐騒動を、箱の製法伝来と報復呪術の由来へ組み込む。同時に、説明しにくい死亡・身体異常を見えない呪力へ帰属する。隠岐騒動と箱の接続が史実であるとは認定しない。

### D10 因果源存在論
- Primary Child / Value: `D10.OBJ.CURSED_OBJECT` — 呪物
- Primary Parent: `D10.OBJ`
- Secondary: `D10.SUP.IMPERSONAL_CURSE` — 非人格的呪力
- Status: `D/I`

組木箱そのものが危険作用を保持する対象として置かれ、作用は人格的怪異より怨念・呪いとして説明される。

## 2.3. 因果・行動モデル

### D11 発動・接触条件
- Primary Child / Value: `D11.CON.LOCATION_STATE` — 位置・状態条件
- Primary Parent: `D11.CON`
- Secondary: `D11.MAN.TAKE_OWN_CARRY` — 持つ・所有・持ち帰る
- Status: `I/D`

早期体系では箱を対象家・人物の近くへ置くことで作用するとされる。初回ではSが納屋から箱を持ち出してA宅へ運び込むことが具体的な遭遇入口になる。開封は必須条件として確認できない。

### D12 作用対象
- Primary Child / Value: `D12.GRP.DEMOGRAPHIC_GROUP` — 属性集団
- Primary Parent: `D12.GRP`
- Secondary: `D12.OTH.VICTIM_TARGET` — 特定被害者
- Status: `D/I`

早期体系で主要な呪い対象は女性・子どもとされる。初回ではSが具体的な危害対象として扱われる。

### D13 作用機構
- Primary Child / Value: `D13.FAT.CURSE_MISFORTUNE` — 呪詛・不運付与
- Primary Parent: `D13.FAT`
- Secondary: `D13.PHY.PHYSIOLOGICAL_CHANGE` — 生理変化
- Status: `D/I`

箱の呪いが危害を付与し、早期体系では身体内部の異常や吐血等を経て死亡へ至ると説明される。呪力と身体結果を別レベルで保持する。

### D14 帰結極性
- Primary Child / Value: `D14.NEG` — 負
- Secondary: なし
- Status: `D/I`

中核帰結は重篤な危害・死・家系への攻撃であり、負に明確に偏る。

### D15 帰結領域
- Primary Child / Value: `D15.BOD.DEATH` — 死亡
- Primary Parent: `D15.BOD`
- Secondary: `D15.BOD.SEVERE_INJURY` — 重傷・障害
- Status: `D/I`

早期体系化で女性・子どもの死亡が主帰結として語られる。初回のSはMの介入により死亡を回避したとされるため、個別Sの死亡事実を主張するものではない。

### D16 因果時間構造
- Primary Child / Value: `D16.STA.ENDURING_CONDITION` — 持続状態
- Primary Parent: `D16.STA`
- Secondary: なし
- Status: `I`

箱は複数世代にわたり管理され、長期間危険性を保持する呪物として体系化される。個々の接触から発症までの時間は一律に固定されない。

### D17 回避・制御方式
- Primary Child / Value: `D17.RIT.RELIGIOUS_SPECIALIST` — 宗教専門家
- Primary Parent: `D17.RIT`
- Secondary: `D17.CST.CONTAINMENT` — 封印・隔離
- Status: `D/I`

Mが神職家系の父へ連絡して儀式的処置を行い、その後は箱を安置・管理する。専門知識を持つ家系への介入依存が制御モデルの中心にある。

### D18 作用レイヤー
- Primary Child / Value: `D18.L1=1; D18.L2=0; D18.L3=0`
- Status: `D/D/I`

L1では箱が人物へ作用する。初回本文には、話しただけで呪いが取り憑くわけではない旨があるためL2=0。Scope内の掲示板拡散・考察は流通そのものであり、独立した社会現実X EvidenceはないためL3=0。

## 2.4. 社会的埋め込み

### D19 流通範囲
- Primary Child / Value: `D19.NET.OPEN_FORUM_WEB` — 公開Web・掲示板
- Primary Parent: `D19.NET`
- Secondary: なし
- Status: `D`

初期形成クラスターは2ちゃんねるの公開板と専用スレッド上で形成・流通する。

### D20 特権情報保持者
- Primary Child / Value: `D20.EXP.RELIGIOUS_FOLKLORE` — 宗教・伝承専門家
- Primary Parent: `D20.EXP`
- Secondary: `D20.PER.FAMILY_BLOODLINE` — 家族・家系
- Status: `D/I`

Mの父・祖父ら神職家系が箱の識別・処理知識を持つと物語内で示される。地域一般ではなく、専門性と家系継承へ保守的に限定する。

### D21 現実アンカー
- Primary Child / Value: `D21.A4` — 史実・記録・既存伝承を因果統合
- Secondary: なし
- Status: `D`

実在する1868年の隠岐騒動を、箱の製法伝来という怪談上の因果由来へ明示的に統合する。A4は「歴史接続が作品構造に必要」という判定であり、その接続自体の史実性を認定するものではない。

# 3. R4 QA要約

- pre-SHA: `15e4277b3be93a189cc168fa20f62cfb5a63f048`
- R3 freeze: `2453520af5a0aa8d99e175f896e4df8f6a160db5`
- 旧10は存在せず、新規Coding正本。
- R3からD20 Secondaryのみ `LOCAL_RESIDENT` → `FAMILY_BLOODLINE` へ保守化。
- 1868年隠岐騒動の実在と、箱製法伝来説の史実性を分離。
- 被差別集落に関する怪談設定を現実地域・住民の属性へ一般化しない。
- H3 Primary exactly 1、Secondary 0–2、Parent derivedを確認。
- D18は `1/0/0`。読者感染なし、独立L3 Evidenceなし。
