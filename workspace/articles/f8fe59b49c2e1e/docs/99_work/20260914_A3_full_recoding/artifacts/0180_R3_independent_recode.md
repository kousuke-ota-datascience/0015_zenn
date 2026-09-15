# 0180 R3 Independent Recode

## 前提
- Entry_ID: `0180`
- 伝承: きさらぎ駅
- pre-SHA: `390a59182cffc7c6982c45af037e732c514c4fc0`
- Version Scope: 2004年1月の2ちゃんねる初期実況、レス98〜635
- 旧 `0180_10_analysis.md` は本freeze前に参照していない。

## 独立コーディング

| Dimension | Primary | Parent | Secondary | Status | 根拠 |
|---|---|---|---|---|---|
| D01 | `D01.G6` 2000年代 | — | — | D | 最古直接確認は2004年1月。 |
| D02 | `D02.WEB.FORUM` Web掲示板・フォーラム | `D02.WEB` | — | D | 2ちゃんねる公式過去ログを直接確認。 |
| D03 | `D03.WEB.FORUM` Web掲示板・フォーラム | `D03.WEB` | — | D | Version Scope自体が公開掲示板実況。 |
| D04 | `D04.COL.LIVE_COCREATION` 実況共同生成 | `D04.COL` | `D04.VAR.ACCRETION` 増補 | D/I | 初期形は参加者の検索・助言・解釈を取り込む実況共同生成。後代に帰還ルール・異界駅一般化等の増補。 |
| D05 | `D05.HYB.COLLAB_THREAD` 参加型実況 | `D05.HYB` | `D05.EXP.LIVE_1P` 一人称実況 | D | 投稿者の逐次報告と参加者応答で成立。 |
| D06 | `D06.T1` 直接体験事実 | — | — | D | 投稿者自身が進行中の経験として提示。真実性の客観判定ではない。 |
| D07 | `D07.PSE.SPATIAL_ROUTE_ANOMALY` 空間・経路異常 | `D07.PSE` | `D07.PSE.HIDDEN_VANISHED_PLACE` 隠れた・消えた場所 | D/I | 通常路線から説明不能な駅・経路へ接続し、通常地理へ回収できない。 |
| D08 | `D08.DEX.DIRECT_EVENT` 出来事への直接遭遇 | `D08.DEX` | `D08.TRC.MISSING_ALTERED_RECORD` 欠落・改変記録 | D/I | 投稿者が異常運行・未知駅に直接遭遇し、駅名・位置が通常検索で確認できないことが手掛かりになる。 |
| D09 | `D09.UNK.DELIBERATE_NONRESOLUTION` 非解決維持 | `D09.UNK` | `D09.CAT.TYPE_ASSIGNMENT` 類型化 | D/I | 参加者は異界・あの世等へ類型化するが、原因・世界構造・帰還は確定しない。 |
| D10 | `D10.PHN.ANOMALOUS_EXPERIENCE` 異常体験 | `D10.PHN` | `D10.SPC.ROUTE_CONNECTION` 経路・接続 | C | 因果主体を確定せず、異常な経路接続・体験そのものが中心。異界そのものを確定因果源にしない。 |
| D11 | `D11.PAS.ACCIDENT_INVOLVEMENT` 事故・出来事に巻き込まれる | `D11.PAS` | `D11.MOV.RIDE_BOARD` 乗る | I | 特別な儀式ではなく通常の帰宅乗車中に非意図的に異常へ巻き込まれる。 |
| D12 | `D12.FOC.PROTAGONIST_EXPERIENCER` 主人公・体験者 | `D12.FOC` | — | D | 作用の中心は「はすみ」。 |
| D13 | `D13.RST.SPATIAL_DISTORTION` 空間異常 | `D13.RST` | — | I | 路線・距離・駅・位置関係が通常地理と整合しない。別世界への置換までは確定しない。 |
| D14 | `D14.NEG` 負 | — | — | I | 恐怖、孤立、位置不明、不審者の車内という危険状態で終了。 |
| D15 | `D15.KNW.UNCERTAINTY_PRESERVED` 不確実性維持 | `D15.KNW` | `D15.MND.FEAR_TRAUMA` 恐怖・トラウマ | D/I | 真相・帰還は未確定のまま残り、投稿者は強い恐怖を示す。失踪・死亡は確定帰結にしない。 |
| D16 | `D16.EVT.SEQUENTIAL_EPISODE` エピソード内段階進行 | `D16.EVT` | — | D | 列車異常→未知駅→線路移動→老人→トンネル→車と一続きの段階的エピソード。 |
| D17 | `D17.UNA.NO_KNOWN_ESCAPE` 回避法なし | `D17.UNA` | `D17.RIT.LEGAL_OFFICIAL` 公的・法的介入 | I/D | 警察・家族・参加者助言を試すが確実な帰還法は成立しない。110番は直接記録される。 |
| D18 | `L1=1 / L2=0 / L3=1` | — | — | D/I/D | 伝承内因果は明確。受容者が読むこと自体は危害条件でない。後代に遠州鉄道が「きさらぎ駅」を公式コンテンツとして展開しており、流通に由来する現実側の組織行動が独立に確認できるためL3=1。 |
| D19 | `D19.NET.OPEN_FORUM_WEB` 公開Web・掲示板 | `D19.NET` | `D19.MAS.NATIONAL_PUBLIC` 全国的大衆 | D/I | 原型は公開掲示板。後代には広範なネット怪談・映像等へ展開。 |
| D20 | `D20.PER.EXPERIENCER` 体験者本人 | `D20.PER` | `D20.UNK.NO_ONE_KNOWS` 誰も知らない | D/I | 個別局面の情報は投稿者のみが持つ一方、現象全体の原因・真相保持者は示されない。 |
| D21 | `D21.A2` 具体的実在対象 | — | — | D | 新浜松、比奈、現実の私鉄利用が初期形の現実アンカーとして具体的に明示される。 |

## H3 QA
- 各H3次元はPrimary exactly 1。
- Secondaryは0〜1件。
- ParentはChildから一意に導出。
- `U/NA/C` はstatusとしてのみ使用。

## causal QA
- D11は「通常乗車そのものが再現手順」ではなく、非意図的に巻き込まれることをPrimaryとした。
- D13は後代の異界設定を逆輸入せず、初期実況で直接必要な `SPATIAL_DISTORTION` に限定した。
- D15で「投稿途絶＝物理失踪」を採用していない。

## D18 QA
- L1=1: 初期実況内で空間異常・危険状態が人物へ作用する。
- L2=0: 読者・スレ参加者が受容だけで怪異の因果対象になる構造はない。
- L3=1: 遠州鉄道公式による「きさらぎ駅」コンテンツ展開という現実側の組織行動を00が独立資料で確認している。

## taxonomy note
現時点で必須のtaxonomy gapは認めない。D13 `SPATIAL_DISTORTION`、D17 `NO_KNOWN_ESCAPE` で初期形の中心構造を表現可能。
