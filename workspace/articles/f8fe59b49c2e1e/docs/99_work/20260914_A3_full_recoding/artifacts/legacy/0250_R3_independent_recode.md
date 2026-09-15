# 0250 R3 Independent Recode

R3 freeze前に旧10・旧Excel coding値は参照していない。Evidenceは `0250_00_contents.md` と固定baseline code systemのみを使用した。

| Dimension | Primary Child / Value | Parent | Secondary | Status | 根拠 |
|---|---|---|---|---|---|
| D01 | `D01.G1` | — | — | `D` | 循環型は1935〜1938年の近代採録で直接確認できる。前近代へ遡及しない。 |
| D02 | `D02.PRT.MAGAZINE` | `D02.PRT` | — | `D` | 近代の民俗・伝承雑誌掲載によって七人ミサキの流通を直接確認できる。 |
| D03 | `D03.PRT.MAGAZINE` | `D03.PRT` | — | `D` | 今回Scopeの循環型を直接確認できる流通媒体は印刷採録。 |
| D04 | `D04.VAR.RETELLING` | `D04.VAR` | `D04.VAR.ACCRETION` | `I` | 地域ごとに由来・場所・被害形を変えつつ、七人集団と循環規則が反復される。 |
| D05 | `D05.HRS.LOCAL_HEARSAY` | `D05.HRS` | — | `I` | 地域伝承として採録される提示形式。 |
| D06 | `D06.T3` | — | — | `I` | 地域で既知の怪異・伝承として採録される。 |
| D07 | `D07.DCF.UNEXPLAINED_DEATH_LOSS` | `D07.DCF` | `D07.ANO.UNKNOWN_EXISTENCE` | `I` | 七人の死者霊が新たな死・取り込みを生む構造が中心。 |
| D08 | `D08.CLM.INHERITED_SAYING` | `D08.CLM` | `D08.STY.COMMUNITY_REPETITION` | `I` | 既存の地域伝承を採録した資料群が出発点。 |
| D09 | `D09.PPR.RECURRENCE_RULE` | `D09.PPR` | `D09.CAU.DIRECT_CAUSE` | `I` | 新規被害者の加入と古参霊の離脱を反復規則として説明する。 |
| D10 | `D10.SUP.GHOST_SPIRIT` | `D10.SUP` | — | `D/I` | 因果源は成仏できない死者霊の七人集団。 |
| D11 | `D11.PAS.SPONTANEOUS_SELECTION` | `D11.PAS` | — | `I` | 被害者は意図的な儀式ではなく遭遇・対象化により循環へ取り込まれる。 |
| D12 | `D12.OTH.VICTIM_TARGET` | `D12.OTH` | — | `I` | 主作用対象は七人ミサキに新たに取られる人物。 |
| D13 | `D13.TRN.PERSON_TRANSFER` | `D13.TRN` | `D13.FAT.CURSE_MISFORTUNE` | `I` | 被害者が次の構成員となり、異常条件が人から人へ継承される。 |
| D14 | `D14.NEG` | — | — | `D/I` | 死・取り込み・発熱等の負帰結が中心。 |
| D15 | `D15.BOD.DEATH` | `D15.BOD` | `D15.LIF.IDENTITY_TRANSFORMATION` | `I` | 代表的循環型では被害者が死亡し、次の七人ミサキ側へ組み込まれる。 |
| D16 | `D16.TRN.CHAIN_SPREAD` | `D16.TRN` | `D16.TRN.SUCCESSIVE_VICTIMS` | `I` | 新規被害者→次の構成員→次の被害者という連鎖が構造の核心。 |
| D17 | — | — | — | `U` | 共有核から確立した回避・解除法を一意に決められない。 |
| D18 | `L1=1; L2=0; L3=0` | — | — | `D/I` | 伝承内因果は明瞭。受容者感染規則や独立した社会現実X Evidenceは今回Scopeで固定しない。 |
| D19 | `D19.LOC.LOCAL_TRADITION` | `D19.LOC` | `D19.LOC.REGIONAL` | `D/I` | 高知・愛媛等を中心とする地域伝承として複数地域で採録される。 |
| D20 | `D20.INS.LOCAL_RESIDENT` | `D20.INS` | — | `I` | 由来・遭遇規則等の追加情報は地域伝承保持者側に置かれる。特定の単一権威者は固定しない。 |
| D21 | `D21.A2` | — | — | `D` | 高知県宿毛市等の具体的実在地域に採録が固定されるが、単一史実起源は採らない。 |

## QA notes

- D01は1927年の名称採録と1935年以後の循環型を区別し、前近代へ遡及しない。
- D13/D16は「七人」の固定人数ではなく、新規被害者との交代で集団が持続する伝播構造を優先した。
- D18 L3は塚・祭祀の記述だけから独立社会効果を推定せず0とする。
- 地域異伝に固有の戦国人物・事故・遍路等をD21 A4へ引き上げない。

R3: frozen.