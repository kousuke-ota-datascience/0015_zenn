# 0060 タクシー幽霊 — R3 Independent Recode

- Entry_ID: `0060`
- Version Scope正本: `docs/99_work/20260914_A3_full_recoding/0060_R2_version_scope.md`
- Evidence正本: `docs/10_each_lore/0060_ghost_taxi/0060_00_contents.md`
- R1監査補助: `docs/99_work/20260914_A3_full_recoding/0060_R1_evidence_audit.md`
- baseline: control plane記載の固定blob
- 旧 `0060_10_analysis.md`: **未参照**
- 旧Excel coding値: **未参照**

# 1. 独立判定

## D01 生成年代
- Value: `D01.G2` — 1945–1969
- Status: `I`
- 根拠: 朝里2023が1950年代には新聞掲載され広く知られていたと整理するため、少なくとも1950年代までに成立済みと推定できる。単一初出を直接確認していないためI。

## D02 最古確認流通媒体
- Primary: `D02.PRT.NEWSPAPER`
- Parent: `D02.PRT`
- Status: `I`
- 根拠: 朝里2023が1950年代の新聞掲載を明示する。新聞原文・紙名・日付を本作業で直接確認していないためI。これ以前の媒体は推測しない。

## D03 確認流通媒体ポートフォリオ
- Primary: `D03.PRT.NEWSPAPER`
- Parent: `D03.PRT`
- Secondary: なし
- Status: `I`
- 根拠: Version Scopeの流通について明示的に確認できる歴史媒体は1950年代の新聞である。口承・書籍等は存在可能性を推測で追加しない。

## D04 生成・変容パターン
- Primary: `D04.VAR.RETELLING`
- Parent: `D04.VAR`
- Secondary: `D04.VAR.LOCALIZATION`, `D04.REC.EVENT_ATTACHMENT`
- Status: `D`
- 根拠: 全国型は乗客属性・乗車地点・目的地・不在化の様式を変えつつ再話され、青山墓地・深泥池等へ地域化される。さらに東日本大震災後には災害死者の文脈へ付着する。

## D05 提示形式
- Primary: `D05.DOC.NEWS_STYLE`
- Parent: `D05.DOC`
- Secondary: なし
- Status: `I`
- 根拠: 最古に確認できる実流通が1950年代新聞掲載であるため、現在固定できる早い提示形はニュース・記事形式。ただし個別記事本文未確認のためI。

## D06 真実性提示
- Value: `D06.T6` — 真偽未確定
- Status: `I`
- 根拠: 朝里2023はこれを全国に伝わる怪談として扱い、個々の再話は死者との接触を事実らしく語る一方、単一の実事件として確定しない。Scope全体では真偽を開いた伝承として扱うのが保守的。

## D07 意味形成対象
- Primary: `D07.ANO.UNEXPLAINED_EVENT` — 説明不能事象
- Parent: `D07.ANO`
- Secondary: `D07.ANO.UNKNOWN_EXISTENCE` — 未知存在
- Status: `I`
- 根拠: 日常の乗客が移動途中・終端で通常の生者としての連続性を失うことが中心問題であり、その人物が何者だったかという存在論的不確実性も独立して残る。

## D08 意味形成契機
- Primary: `D08.DEX.DIRECT_EVENT` — 出来事への直接遭遇
- Parent: `D08.DEX`
- Secondary: `D08.STY.WITNESS_REPORT` — 目撃証言
- Status: `I`
- 根拠: 物語内では運転手自身が異常な乗客事案へ遭遇する。伝承としては、その体験報告が受容者の手掛かりになる。

## D09 意味付与操作
- Primary: `D09.CAT.TYPE_ASSIGNMENT` — 類型化
- Parent: `D09.CAT`
- Secondary: `D09.AGN.AGENCY_ATTRIBUTION` — 主体化
- Status: `I`
- 根拠: 通常の乗客として説明できない事象を「幽霊・死者の乗客」と再分類し、異常の背後に死者主体を置くことで理解可能にする。

## D10 因果源存在論
- Primary: `D10.SUP.GHOST_SPIRIT` — 幽霊・死者霊
- Parent: `D10.SUP`
- Secondary: なし
- Status: `D`
- 根拠: Scoped coreは最終的に乗客を死者・幽霊として理解する型を対象とし、朝里2023でも「タクシー幽霊」として明示される。

## D11 発動・接触条件
- Primary: `D11.PAS.SPONTANEOUS_SELECTION` — 偶然選ばれる・遭遇する
- Parent: `D11.PAS`
- Secondary: `D11.SOC.MEET_INTERACT` — 会う・会話する
- Status: `I`
- 根拠: 通常のタクシー営業は再現的に怪異を発動する儀式ではなく、運転手が偶然その乗客に当たることが因果系への入口。実際の接触は乗車・会話という社会的相互行為で成立する。

## D12 作用対象
- Primary: `D12.FOC.PROTAGONIST_EXPERIENCER` — 主人公・体験者
- Parent: `D12.FOC`
- Secondary: なし
- Status: `I`
- 根拠: D13の主作用を「死者が通常の乗客として顕現し、運転手に経験されること」と置くと、その直接対象は体験者である運転手。

## D13 作用機構
- Primary: `D13.MAN.MANIFEST_ONLY` — 顕現のみ
- Parent: `D13.MAN`
- Secondary: なし
- Status: `I`
- 根拠: 全国型の共有核に共通するのは、死者・幽霊が通常の乗客として生者のタクシー内に現れること。車内での瞬間消失は一主要異伝だが全型共通でないためPrimaryにしない。

## D14 帰結極性
- Value: `D14.NEU` — 中立
- Status: `I`
- 根拠: 怪異は不気味だが、共有核では運転手への身体危害・呪い・利益を必須としない。中心帰結は認識の転換である。

## D15 帰結領域
- Primary: `D15.KNW.BELIEF_REVISION` — 信念変更
- Parent: `D15.KNW`
- Secondary: `D15.KNW.REVELATION_KNOWLEDGE` — 真相・知識獲得
- Status: `I`
- 根拠: 通常の生者と思っていた乗客が死者・幽霊だったと理解されることで、体験者の出来事解釈が更新される。故人確認型では遺影・家人等により追加知識も得られる。

## D16 因果時間構造
- Primary: `D16.EVT.SEQUENTIAL_EPISODE` — エピソード内段階進行
- Parent: `D16.EVT`
- Secondary: なし
- Status: `D`
- 根拠: 接触→乗車・移動→不在化／目的地到着→故人確認という複数段階が一つの短いエピソード内で順に進む。

## D17 回避・制御方式
- Primary: `D17.NON.OBSERVATIONAL_ONLY` — 観測のみ
- Parent: `D17.NON`
- Secondary: なし
- Status: `I`
- 根拠: 全国型の共有核には安定した回避手順・祓い・正答等がなく、体験後に異常を認識する構造が中心。対処法がないことを不可避とは読まない。

## D18 作用レイヤー
- `D18.L1=1`
- `D18.L2=0`
- `D18.L3=0`
- Status: `D`
- 根拠: L1は伝承内部で死者の顕現と体験者の認識変化が起きる。話を聞いた受容者が因果対象になる自己適用規則はなく、全国型の流通が現実制度・市場等を変えたX Evidenceも今回確認していない。

## D19 流通範囲
- Primary: `D19.MAS.NATIONAL_PUBLIC` — 全国的大衆
- Parent: `D19.MAS`
- Secondary: `D19.LOC.REGIONAL` — 地域・地方
- Status: `D`
- 根拠: 朝里2023は全国各地に話があり、青山墓地・深泥池等の地域版が知られると整理する。

## D20 特権情報保持者
- Primary / Parent / Secondary: なし
- Status: `U`
- 根拠: 故人確認型では遺族・目的地住人が追加情報を持つが、車内消失型では同じ保持者構造が必須ではない。全国型共有核として一意に固定できない。

## D21 現実アンカー
- Value: `D21.A1` — 一般的現実背景
- Status: `D`
- 根拠: 全国型の成立には実在する交通職業・タクシーという一般的都市背景が不可欠だが、共有核は特定一地点・企業・人物へ依存しない。

# 2. causal precheck

```text
D11: 通常営業中に偶然、異常な乗客へ遭遇する
→ D12: 運転手が直接の体験者となる
→ D13: 死者・幽霊が通常の乗客として顕現する
→ D15: 生者だと思っていた乗客を死者・幽霊として再解釈する
```

接続は成立する。

# 3. D18 L3 precheck

地域事件で警察対応等が確認される例があっても、それを全国型の流通による社会現実効果へ一般化しない。0060自身について独立X Evidenceを確認していないため `L3=0`。

# 4. U / NA / C precheck

- D20=`U`: 主要異伝間で情報保持者構造を一意に固定できないため。
- NA: なし。
- C: なし。

# 5. taxonomy gap precheck

全国型の共有核は既存taxonomyで表現可能。車内瞬間消失という異伝単独では0059同様にD13「対象の消失／不在化」gapが再出するが、0060のScope全体のPrimary作用は死者の顕現としたため、0060自体のPrimary codingには新Childを要求しない。gap候補はGlobal Reconciliationへ保持する。

# 6. R3 freeze

本ファイルは旧 `0060_10_analysis.md` および旧Excel coding値を参照せずに作成した独立判定である。以後のR4で初めて旧10を参照する。