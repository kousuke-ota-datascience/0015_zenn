# 0198 R3 Independent Recode

## 前提
- Entry_ID: `0198`
- 伝承: 一人かくれんぼ
- pre-SHA: `ed97ce65ee6856da024c91a0a14bbbc202353306`
- Version Scope: 2007-04-18〜04-21の初期プロトコル形成クラスター
- 旧10はR3 freeze前に参照しない。

## 独立コーディング

| D | Primary | Parent | Secondary | Status | 根拠 |
|---|---|---|---|---|---|
| D01 | `D01.G6` | — | — | D | 2007年4月ログを直接確認。 |
| D02 | `D02.WEB.FORUM` | `D02.WEB` | — | D | 公開Web掲示板。 |
| D03 | `D03.WEB.FORUM` | `D03.WEB` | — | D | Scope全体が掲示板上で形成・流通。 |
| D04 | `D04.COL.LIVE_COCREATION` | `D04.COL` | `D04.VAR.ACCRETION` | D | 実況・質問・失敗・補足を受けて終了法や注意点が追加される。 |
| D05 | `D05.HYB.COLLAB_THREAD` | `D05.HYB` | `D05.EXP.LIVE_1P` | D | 複数参加者のリアルタイム実践報告と閲覧者応答。 |
| D06 | `D06.T1` | — | — | D | 実践者が自分に現在起きている経験として報告。客観的超常認定ではない。 |
| D07 | `D07.ANO.UNEXPLAINED_EVENT` | `D07.ANO` | `D07.ANO.UNKNOWN_EXISTENCE` | D/I | 儀式後の音・声・テレビ異常・気配等と「もう一人」の存在が不可解さの中心。 |
| D08 | `D08.DEX.DIRECT_EVENT` | `D08.DEX` | `D08.DEX.PERCEPTUAL_ANOMALY` | D | 実践者自身が音・声・気配等を直接知覚したと実況する。 |
| D09 | `D09.AGN.AGENCY_ATTRIBUTION` | `D09.AGN` | `D09.CAT.TYPE_ASSIGNMENT` | I | 室内異常を「霊」「もう一人」等の作用主体へ帰属し、降霊現象として類型化する。 |
| D10 | `D10.SUP.GHOST_SPIRIT` | `D10.SUP` | `D10.OBJ.CURSED_OBJECT` | I | 儀式は霊的存在の招来・遭遇として理解され、加工したぬいぐるみが媒介物になる。 |
| D11 | `D11.MAN.PERFORM_RITUAL` | `D11.MAN` | `D11.MAN.CREATE_ALTER_MANIPULATE` | D | 一連の儀式手順の実行が発動条件で、ぬいぐるみ加工がその一部。読むだけでは発動しない。 |
| D12 | `D12.FOC.PRACTITIONER` | `D12.FOC` | — | D | 儀式を実行した本人が異常体験の対象。 |
| D13 | `D13.MAN.MANIFEST_ONLY` | `D13.MAN` | `D13.REL.TARGETING` | I | 初期ログで必須の主作用は異常な存在・音・声等の顕現。実践者を探す／狙う解釈がSecondary。実害は必須ではない。 |
| D14 | `D14.NEG` | — | — | I | 恐怖・危険・終了失敗への不安が中心。 |
| D15 | `D15.MND.FEAR_TRAUMA` | `D15.MND` | — | D/I | 初期参加者は強い恐怖を実況するが、死亡・重傷は確認された実現帰結ではない。 |
| D16 | `D16.EVT.SEQUENTIAL_EPISODE` | `D16.EVT` | — | D | 準備→開始→隠れる→異常報告→終了という一回の儀式エピソード内進行。 |
| D17 | `D17.RIT.RITUAL_CLOSURE` | `D17.RIT` | `D17.RUL.PROCEDURAL_RULE` | D | 塩水等を用いる明示的終了法が提示され、規則に従って儀式を閉じることが重要。 |
| D18 | `L1=1 / L2=0 / L3=1` | — | — | D/I/D | 伝承内で異常作用が主張される。読むだけでは作用しない。一方、掲示板投稿を受けた別参加者が実際に儀式を実践・実況する現実行動を同一Scopeで直接確認できる。 |
| D19 | `D19.NET.OPEN_FORUM_WEB` | `D19.NET` | — | D | 公開掲示板。 |
| D20 | `D20.PER.EXPERIENCER` | `D20.PER` | `D20.INS.SUBCULTURE_VETERAN` | I | 手順・対処知識は実践経験者や掲示板内の知識保持者から提示されるが、制度的専門家ではない。 |
| D21 | `D21.A1` | — | — | I | 自宅、浴室、テレビ等の一般的現実背景に依存するが特定実在地点・制度へ固定されない。 |

## QA
- H3 Primary exactly 1、Secondary 0〜1、Parent derived。
- D11は情報受容ではなく実践行為を発動条件に固定。
- D13で死亡・身体攻撃を必須化せず、初期ログで直接共有される顕現・気配を中心にする。
- D15は恐怖。後代の重篤事故説を逆輸入しない。
- D18 L3=1は、同一初期スレッド内で別参加者がプロトコルを受容後に実践する独立投稿をX Evidenceとする。L2は読むだけで怪異対象化されないため0。
