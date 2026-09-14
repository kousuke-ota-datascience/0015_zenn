記載内容については、以下3文書のインストラクションに従うこと。

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`

Evidence正本は同一ディレクトリの `0059_00_contents.md`、R2/R3監査成果物は `docs/99_work/20260914_A3_full_recoding/` 配下を参照する。

# 1. 伝承エントリ基本情報

- `Entry_ID`: `59`
- `伝承エントリ名称`: 深泥池の幽霊タクシー
- `Macro_Category`: 交通・インフラ
- `Entry_Type`: 物語・伝説
- `Version_Scope`: 1969年10月7日付『朝日新聞』京都市内版で公刊確認できる、京都・深泥池に結び付いた「タクシー車内にいたはずの乗客が通常の降車過程なしに消失した」とする一件を中心とする最小核。後代の「雨夜」「若い女性」「濡れた座席」「死者由来」等は初期Scopeの必須条件にしない。

# 2. 分析概念次元

## 2.1. 来歴・流通・提示

### 2.1.1. D01: 生成年代

- Primary Child / Value: `D01.G2` — 1945–1969
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**

1969年10月7日新聞記事が強い同時代確認点であり、京都府立図書館が挙げる後代研究・解説も1960年代以降の流布を整理する。1969年の最古確認日をそのまま生成日に置換せず、年代帯のみIで置く。

**この伝承における現れ方**

深泥池に固定されたタクシー乗客消失譚は、少なくとも1960年代後半までに現代都市交通の中で語り得る形になっていた。

### 2.1.2. D02: 最古確認流通媒体

- Primary Child / Value: `D02.PRT.NEWSPAPER` — 新聞
- Primary Parent: `D02.PRT`
- Secondary: なし
- Status: `D`

**判定根拠**

京都府立図書館が、1969年10月7日付『朝日新聞』京都市内版p.16の記事をマイクロフィルムと朝日新聞クロスサーチで確認している。これより古い実流通媒体は現Evidenceで固定できない。

**この伝承における現れ方**

現在確認できる最初期の公的流通点では、乗客消失の話が地域新聞記事として読者へ提示される。

### 2.1.3. D03: 確認流通媒体ポートフォリオ（Version Scope）

- Primary Child / Value: `D03.PRT.NEWSPAPER` — 新聞
- Primary Parent: `D03.PRT`
- Secondary: なし
- Status: `D`

**判定根拠**

R2 Scopeを1969年新聞確認層の最小核へ限定したため、Scoped Versionで直接確認できる流通媒体は新聞である。後代書籍は変異史・再話比較のH/C Evidenceであり、本ScopeのD03 Secondaryには入れない。

**この伝承における現れ方**

単発の乗客消失報告が新聞地域面を通じて公衆へ可視化される。

### 2.1.4. D04: 生成・変容パターン

- Primary Child / Value: `D04.VAR.RETELLING` — 再話
- Primary Parent: `D04.VAR`
- Secondary: なし
- Status: `I`

**判定根拠**

京都府立図書館のレファレンスが比較する後代資料では、乗車地点、目的地、乗客属性、雨、濡れた座席等が一致しない。一方、深泥池・タクシー・乗客消失という核が再叙述される。どの細部がいつ増補されたかまでは確定しないため、`ACCRETION` より `RETELLING` を採る。

**この伝承における現れ方**

同じ異常核が、異なる語り手・書籍で細部を変えながら再話される。

### 2.1.5. D05: 提示形式

- Primary Child / Value: `D05.DOC.NEWS_STYLE` — 新聞・ニュース風
- Primary Parent: `D05.DOC`
- Secondary: なし
- Status: `D`

**判定根拠**

最小Scopeの確認形は1969年の新聞記事である。後代の地域伝聞形式を初期Scopeへ混ぜない。

**この伝承における現れ方**

「車から乗客が消えた」という届出が新聞の事件・地域記事として提示され、怪談めいた出来事として読者に届けられる。

### 2.1.6. D06: 真実性提示

- Primary Child / Value: `D06.T1` — 直接体験事実
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**

新聞確認層では、タクシー運転手による実際の乗客消失体験・届出として提示される形式を取る。ただし本作業では朝日新聞公式本文全体を直接閲覧していないためStatusはIとする。

**この伝承における現れ方**

「誰かから聞いた怪談」ではなく、運転手が実際に経験して警察へ届けた出来事という形式が本当らしさを支える。

## 2.2. 意味形成

### 2.2.1. D07: 意味形成対象

- Primary Child / Value: `D07.ANO.UNEXPLAINED_EVENT` — 説明不能事象
- Primary Parent: `D07.ANO`
- Secondary: なし
- Status: `D`

**判定根拠**

同時代新聞見出しで直接固定できる異常核は「車から乗客が消えた」ことである。

**この伝承における現れ方**

直前まで存在した乗客が通常の降車過程なしに不在になることが、説明を要求する中心問題となる。

### 2.2.2. D08: 意味形成契機

- Primary Child / Value: `D08.DEX.DIRECT_EVENT` — 出来事への直接遭遇
- Primary Parent: `D08.DEX`
- Secondary: `D08.STY.WITNESS_REPORT` — 目撃証言
- Status: `I`

**判定根拠**

物語内では運転手自身が乗客消失に遭遇することが直接契機である。社会へ流通する際には、その運転手の届出・証言が事案を問題化する手掛かりとなる。

**この伝承における現れ方**

体験者にとっては車内の異常が直接契機であり、読者にとっては体験者の証言がその異常を知る入口になる。

### 2.2.3. D09: 意味付与操作

- Primary Child / Value: `D09.UNK.MYSTERY_LABEL` — 謎として命名
- Primary Parent: `D09.UNK`
- Secondary: なし
- Status: `D`

**判定根拠**

1969年記事見出しは事案を「怪談めいた」届出として枠付ける一方、原因を幽霊等へ確定しない。初期Scopeでは既知怪異類型への確定分類より、不可解な出来事としてラベル化する操作が強い。

**この伝承における現れ方**

乗客消失は説明済みの事件ではなく、「怪談めいた」不可解な出来事として社会的に理解される。

### 2.2.4. D10: 因果源存在論

- Primary Child / Value: なし
- Primary Parent: なし
- Secondary: なし
- Status: `U`

**判定根拠**

1969年最小核は乗客消失を記録するが、その原因を死者霊・妖怪・通常の人間・自然過程のいずれかへ一意に置かない。後代の「幽霊」説明を初期Scopeへ遡及しない。

**この伝承における現れ方**

何が乗客を消失させたのか、そもそも乗客が何者だったのかは開いたまま残る。

## 2.3. 因果・行動モデル

### 2.3.1. D11: 発動・接触条件

- Primary Child / Value: `D11.SOC.MEET_INTERACT` — 会う・会話する
- Primary Parent: `D11.SOC`
- Secondary: なし
- Status: `I`

**判定根拠**

運転手が通常業務で乗客と出会い、客として応対することが異常体験への入口となる。本人が怪異を呼び出す儀式・禁忌はない。

**この伝承における現れ方**

日常的な乗客との相互行為が、そのまま怪異接触へ変わる。

### 2.3.2. D12: 作用対象

- Primary Child / Value: `D12.FOC.PROTAGONIST_EXPERIENCER` — 主人公・体験者
- Primary Parent: `D12.FOC`
- Secondary: なし
- Status: `I`

**判定根拠**

異常を直接経験し、乗客消失の不確実性を引き受ける焦点人物はタクシー運転手である。

**この伝承における現れ方**

運転手の観測と認識を通じて怪異エピソードが成立する。

### 2.3.3. D13: 作用機構

- Primary Child / Value: `D13.MAN.MANIFEST_ONLY` — 顕現のみ
- Primary Parent: `D13.MAN`
- Secondary: なし
- Status: `I`

**判定根拠**

最小Scopeでは、異常な乗客が通常の客として認識され、その後不在になる以上の安定した攻撃・追跡・呪詛作用を必須としない。独立主体をD10で確定しないが、物語上の作用記述としては「現れ、観測される」が最も近い。

**この伝承における現れ方**

異常性は運転手へ危害を加えることより、日常空間に存在した乗客が観測後に消失することにある。

### 2.3.4. D14: 帰結極性

- Primary Child / Value: `D14.NEU` — 中立
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**

最小Scopeでは運転手への身体危害、呪い、利益等を確認できない。出来事の不気味さはあるが、主帰結は未解決性であり、負の実害を必須化しない。

**この伝承における現れ方**

運転手は不可解な事象を経験するが、安定した損失・危害規則は残らない。

### 2.3.5. D15: 帰結領域

- Primary Child / Value: `D15.KNW.UNCERTAINTY_PRESERVED` — 不確実性維持
- Primary Parent: `D15.KNW`
- Secondary: なし
- Status: `I`

**判定根拠**

乗客の正体、消失機構、深泥池との因果関係は解明されない。R3でSecondary候補とした `FEAR_TRAUMA` は、最小ScopeのEvidenceから長期的恐怖・トラウマを独立帰結として固定できないためR4で外す。

**この伝承における現れ方**

「何を乗せたのか」「どうして消えたのか」という問いが未解決のまま終わる。

### 2.3.6. D16: 因果時間構造

- Primary Child / Value: `D16.EVT.SINGLE_EPISODE` — 単一エピソード
- Primary Parent: `D16.EVT`
- Secondary: なし
- Status: `I`

**判定根拠**

乗客との接触、乗車中の移動、消失認識、届出までが一続きの出来事として構成される。単なる瞬間的観測より、一つの乗車エピソードとして捉える。

**この伝承における現れ方**

一回の営業運行の中で怪異との接触から終端までが完結する。

### 2.3.7. D17: 回避・制御方式

- Primary Child / Value: `D17.NON.OBSERVATIONAL_ONLY` — 観測のみ
- Primary Parent: `D17.NON`
- Secondary: なし
- Status: `I`

**判定根拠**

怪異乗車を避ける安定規則、祓い、正答、経路変更等は最小核にない。警察への届出は現実側の事件対応であり、伝承内因果の制御規則ではない。

**この伝承における現れ方**

運転手は怪異を操作・回避できず、起きた出来事を観測して終わる。

### 2.3.8. D18: 作用レイヤー

- Primary Child / Value: `D18.L1=1; D18.L2=0; D18.L3=0`
- Primary Parent: なし
- Secondary: なし
- Status: `I`

**判定根拠**

- L1: 伝承内部で乗客消失という異常事象が体験者へ作用する。
- L2: この話を読む・聞く現実側受容者が、伝承内容上の同じ因果対象になる自己適用規則はない。
- L3: 1969年の警察対応は「乗客が消えたという届出」への個別事件対応であり、**噂・伝承の流通そのものが社会行動・制度等を変えたX Evidenceではない**。したがってR3の `L3=1` をR4で `0` に修正する。

**この伝承における現れ方**

怪異の因果効力は乗車エピソード内部に留まり、伝承を受容するだけで現実側へ作用する構造や、流通による独立した社会効果は現Evidenceで固定しない。

## 2.4. 社会的埋め込み

### 2.4.1. D19: 流通範囲

- Primary Child / Value: `D19.LOC.REGIONAL` — 地域・地方
- Primary Parent: `D19.LOC`
- Secondary: なし
- Status: `I`

**判定根拠**

最古確認は京都市内版新聞で、後代も京都・深泥池の地域怪談として整理される。地域外での再話は確認できるが、1969年最小Scopeへ全国的大衆流通を遡及しない。

**この伝承における現れ方**

京都の具体的場所ロアとして地域性を保持したまま流通する。

### 2.4.2. D20: 特権情報保持者

- Primary Child / Value: `D20.PER.EXPERIENCER` — 体験者本人
- Primary Parent: `D20.PER`
- Secondary: なし
- Status: `I`

**判定根拠**

乗客の様子や消失直前の車内状況について追加情報を持つ中心人物は運転手本人である。運転手が真相を知るわけではないが、一般読者より多くの一次体験情報を保持する。

**この伝承における現れ方**

公衆が知るのは報道・再話された情報であり、直接体験の詳細は当事者へ偏在する。

### 2.4.3. D21: 現実アンカー

- Primary Child / Value: `D21.A2` — 具体的実在対象
- Primary Parent: なし
- Secondary: なし
- Status: `D`

**判定根拠**

京都市北区の実在する深泥池へ具体的に固定される。京都市・文化庁の公式資料で地点の実在性を確認できる。1969年新聞記事も実在記録だが、それを幽霊の実在証明には使わない。

**この伝承における現れ方**

匿名の「どこかの池」ではなく、特定可能な深泥池が怪異の現実感を支える。

# 3. R4 Entry QA

## 3.1. D12 → D13 → D15 causal QA

```text
D12: タクシー運転手が体験者となる
→ D13: 異常な乗客が日常空間に顕現し、観測後に消失する
→ D15: 正体・消失機構が未解決のまま残る
```

**結果: Pass。**

## 3.2. D18 L3 evidence QA

R3では1969年新聞の警察対応をX Evidenceと解釈して `L3=1` とした。しかし現行D18定義は、**噂・伝承の流通により**現実社会で行動・制度・市場等の結果が生じることを要求する。

1969年の警察・パトカー対応は、運転手が報告した一件の乗客消失事案への対応であり、伝承の流通によって生じた社会的二次効果とは区別すべきである。後代に「運転手が深泥池を避けた」等の社会行動があったとしても、本R1 Evidenceでは高品質X Evidenceとして固定していない。

**結果: `L3=0`。R3から修正。Pass。**

## 3.3. U / NA / C QA

- `D10=U`: 初期Scopeで因果源存在論を幽霊へ確定できないため意図的。後代解釈で埋めない。
- `NA`: なし。
- `C`: なし。

**結果: Pass。**

## 3.4. taxonomy gap QA

乗客消失そのものに専用のD13 Childはないが、本Scopeの主作用は体験者への攻撃や対象消去ではなく、怪異の顕現・観測とその後の不在による未解決性であるため、`MANIFEST_ONLY` と `UNCERTAINTY_PRESERVED` の組合せで表現可能と判断する。

**結果: Pass。確定的taxonomy gapなし。**

## 3.5. R3 → R4 QA差分

| 次元 | R3 freeze | R4確定 | 理由 |
|---|---|---|---|
| D15 Secondary | `D15.MND.FEAR_TRAUMA` | なし | 最小Scopeで長期的恐怖・トラウマを独立帰結として固定できない |
| D18 | `L1=1,L2=0,L3=1 / D` | `L1=1,L2=0,L3=0 / I` | 警察対応は個別事案対応であり、伝承流通によるX Evidenceではない |

その他のD01–D21はR3 freezeを維持した。

# 4. 再コーディング前旧10との差分比較

比較対象: `再コーディング前 commit SHA = 4c91bfea47f31b35a5a38799ee145c93f29580f7` の旧 `0059_10_analysis.md`。R3 freeze後にのみ参照した。

| 次元 | 旧判定 | R4確定 | 差分分類 | 要点 |
|---|---|---|---|---|
| D01 | `G2 / I` | 同左 | 一致 | 1960年代流通層を年代帯へIで配置 |
| D02 | `NEWSPAPER / D` | 同左 | 一致 | — |
| D03 | `NEWSPAPER + BOOK / D` | `NEWSPAPER / D` | `Scope mismatch` / `Code-selection mismatch` | R2を1969年最小核へ限定し後代書籍をScope外へ |
| D04 | `ACCRETION + STABILIZED_CANON / I` | `RETELLING / I` | `Scope mismatch` / `Code-selection mismatch` | 増補時点を確定せず、資料間の再叙述差のみ採る |
| D05 | `LOCAL_HEARSAY + NEWS_STYLE / I` | `NEWS_STYLE / D` | `Scope mismatch` / `Code-selection mismatch` / `Status mismatch` | 初期確認形をPrimary化 |
| D06 | `T6 / I` | `T1 / I` | `Scope mismatch` / `Code-selection mismatch` | 超自然解釈の真偽ではなく、初期記事の体験事実提示形式をコード |
| D07 | `UNEXPLAINED_EVENT / D` | 同左 | 一致 | — |
| D08 | `DIRECT_EVENT + PHYSICAL_TRACE / I` | `DIRECT_EVENT + WITNESS_REPORT / I` | `Scope mismatch` / `Code-selection mismatch` | 水濡れ痕跡を初期核から外し、届出・証言をSecondary化 |
| D09 | `TYPE_ASSIGNMENT / I` | `MYSTERY_LABEL / D` | `Scope mismatch` / `Code-selection mismatch` / `Status mismatch` | 幽霊類型への確定より「怪談めいた」未解決ラベルを採る |
| D10 | `GHOST_SPIRIT / I` | `U` | `Scope mismatch` / `Evidence mismatch` / `Prior coding error` | 後代幽霊解釈の初期形への遡及を除去 |
| D11 | `RIDE_BOARD / I` | `MEET_INTERACT / I` | `Code-selection mismatch` | 体験者側の因果入口を通常の乗客との相互行為として取る |
| D12 | `PROTAGONIST_EXPERIENCER / I` | 同左 | 一致 | — |
| D13 | `MANIFEST_ONLY / I` | 同左 | 一致 | — |
| D14 | `NEG / I` | `NEU / I` | `Scope mismatch` / `Code-selection mismatch` | 恐怖を実害として必須化せず未解決観測を中心化 |
| D15 | `UNCERTAINTY_PRESERVED / I` | 同左 | 一致 | — |
| D16 | `SINGLE_OBSERVATION / I` | `SINGLE_EPISODE / I` | `Code-selection mismatch` | 一瞬の観測でなく乗車から届出までの一続きの出来事 |
| D17 | `OBSERVATIONAL_ONLY / I` | 同左 | 一致 | — |
| D18 | `L1=1,L2=0,L3=0 / I` | 同左 | 一致 | R3でL3=1としたがR4 QAで現行定義へ戻した |
| D19 | `REGIONAL + NATIONAL_PUBLIC / I` | `REGIONAL / I` | `Scope mismatch` / `Code-selection mismatch` | 後代全国再話を1969年Scopeへ混ぜない |
| D20 | `COMMON_KNOWLEDGE / I` | `EXPERIENCER / I` | `Code-selection mismatch` | 一般共有範囲ではなく追加情報保持者を問うため |
| D21 | `A2 / D` | 同左 | 一致 | — |

# 5. R4結論

- D12→D13→D15 causal QA: **Pass**
- D18 L3 evidence QA: **Pass / L3=0へ修正**
- U / NA / C QA: **Pass**
- taxonomy gap QA: **Pass / 変更候補なし**
- R3→R4修正: D15 Secondary削除、D18 `L3=1/D → L3=0/I`
- 旧10との差分比較・分類: **完了**
- Coding正本 `0059_10_analysis.md`: **更新済み**
- 旧Excel比較: **R5 Global Reconciliationへ移管**
