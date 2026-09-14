# 0059 深泥池の幽霊タクシー — R3 Independent Recode

- Entry_ID: `0059`
- Version Scope正本: `docs/99_work/20260914_A3_full_recoding/0059_R2_version_scope.md`
- Evidence正本: `docs/10_each_lore/0059_mizorogaike_ghost_taxi/0059_00_contents.md`
- R1監査補助: `docs/99_work/20260914_A3_full_recoding/0059_R1_evidence_audit.md`
- baseline: control plane記載の固定blob
- 旧 `0059_10_analysis.md`: **未参照**
- 旧Excel coding値: **未参照**

# 1. 独立判定

## D01 生成年代
- Value: `D01.G2` — 1945–1969
- Status: `I`
- 根拠: 1969年10月7日新聞記事が直接確認点であり、後代資料も1960年代以降の流布を整理する。ただし1969年を実際の成立年そのものとは断定しないためI。

## D02 最古確認流通媒体
- Primary: `D02.PRT.NEWSPAPER`
- Parent: `D02.PRT`
- Status: `D`
- 根拠: 京都府立図書館が1969年10月7日『朝日新聞』京都市内版の記事をマイクロフィルムと朝日新聞クロスサーチで確認している。これ以前の実流通媒体を直接固定できないため、現在の最古確認媒体を新聞とする。

## D03 確認流通媒体ポートフォリオ
- Primary: `D03.PRT.NEWSPAPER`
- Parent: `D03.PRT`
- Secondary: なし
- Status: `D`
- 根拠: Version Scopeの1969年確認層は新聞地域面で公刊された事案として直接固定できる。後代書籍は変異史確認用であり、Scoped VersionのPrimary媒体にしない。

## D04 生成・変容パターン
- Primary: `D04.VAR.RETELLING`
- Parent: `D04.VAR`
- Status: `I`
- 根拠: 後代資料では「深泥池まで乗せる」「池の脇で乗せる」、乗客年齢、雨、濡れた座席等の細部が異なる。1969年核を保ちながら再話されていることは確認できるが、どの細部が後から増補されたかは確定しない。

## D05 提示形式
- Primary: `D05.DOC.NEWS_STYLE`
- Parent: `D05.DOC`
- Status: `D`
- 根拠: 最小Scopeの確認形は新聞地域面の記事として提示される。

## D06 真実性提示
- Value: `D06.T1` — 直接体験事実
- Status: `I`
- 根拠: 新聞記事はタクシー運転手による乗客消失の届出・体験報告を実際の出来事として扱う形式である。一方、本作業では公式記事全文を直接閲覧していないためI。

## D07 意味形成対象
- Primary: `D07.ANO.UNEXPLAINED_EVENT` — 説明不能事象
- Parent: `D07.ANO`
- Status: `D`
- 根拠: 「車から乗客が消えた」という説明不能事象が新聞見出しに直接固定される。

## D08 意味形成契機
- Primary: `D08.DEX.DIRECT_EVENT` — 出来事への直接遭遇
- Parent: `D08.DEX`
- Secondary: `D08.STY.WITNESS_REPORT` — 目撃証言
- Status: `I`
- 根拠: 物語内の意味形成は運転手自身が乗客消失へ遭遇したことから始まる。同時に、公衆へ流通する際の直接手掛かりは運転手の届出・証言であるためSecondaryとする。

## D09 意味付与操作
- Primary: `D09.UNK.MYSTERY_LABEL` — 謎として命名
- Parent: `D09.UNK`
- Status: `D`
- 根拠: 同時代記事見出し自体が事案を「怪談めいた」ものとして扱い、原因を確定せずミステリーとして枠付ける。

## D10 因果源存在論
- Primary / Parent / Secondary: なし
- Status: `U`
- 根拠: 1969年最小核は乗客消失という現象を記録するが、その原因を死者霊・妖怪・通常の人間・物理過程のいずれかへ一意に置かない。後代の「幽霊」解釈を初期核へ遡及しない。

## D11 発動・接触条件
- Primary: `D11.SOC.MEET_INTERACT` — 会う・会話する
- Parent: `D11.SOC`
- Status: `I`
- 根拠: タクシー運転手が通常業務上の乗客と出会い、乗車サービスの相互行為へ入ることが異常体験への入口となる。儀式・禁忌・特定時刻を必要としない。

## D12 作用対象
- Primary: `D12.FOC.PROTAGONIST_EXPERIENCER` — 主人公・体験者
- Parent: `D12.FOC`
- Status: `I`
- 根拠: 異常な乗客消失を経験し、認識的不確実性を直接負うのはタクシー運転手である。

## D13 作用機構
- Primary: `D13.MAN.MANIFEST_ONLY` — 顕現のみ
- Parent: `D13.MAN`
- Status: `I`
- 根拠: 最小Scopeでは、乗客が通常の客として認識された後に消失する以上の安定した攻撃・追跡・呪詛作用を必須としない。体験者への主作用は怪異の顕現・観測そのものにある。

## D14 帰結極性
- Value: `D14.NEU` — 中立
- Status: `I`
- 根拠: 運転手への身体危害・利益・呪いは最小核に必須でなく、出来事は不気味さと未解決性を残して終わる。

## D15 帰結領域
- Primary: `D15.KNW.UNCERTAINTY_PRESERVED` — 不確実性維持
- Parent: `D15.KNW`
- Secondary: `D15.MND.FEAR_TRAUMA` — 恐怖・トラウマ
- Status: `I`
- 根拠: 最終的に「乗客は何者だったのか／どう消えたのか」が解明されず残ることが主帰結。怪談としての恐怖は重要だが、長期トラウマまで直接固定できないためSecondaryは恐怖一般としてI。

## D16 因果時間構造
- Primary: `D16.EVT.SINGLE_EPISODE` — 単一エピソード
- Parent: `D16.EVT`
- Status: `I`
- 根拠: 乗客との接触から消失認識・届出までが一続きの出来事内で完結する。長期遅延・反復発動は最小核にない。

## D17 回避・制御方式
- Primary: `D17.NON.OBSERVATIONAL_ONLY` — 観測のみ
- Parent: `D17.NON`
- Status: `I`
- 根拠: 最小核には幽霊を避ける安定規則・祓い・正答・経路変更等がない。運転手は異常を認識して警察へ届け出るが、これは伝承内因果を制御する規則ではない。

## D18 作用レイヤー
- `D18.L1=1`
- `D18.L2=0`
- `D18.L3=1`
- Status: `D`
- 根拠:
  - L1: 物語内部で乗客消失という異常事象が体験者に作用する。
  - L2: この話を読む・聞く受容者が同じ異常の因果対象になる自己適用規則はない。
  - L3: 同時代新聞見出しが警察への届出とパトカー対応を直接記録し、異常体験の主張が現実制度上の行動を生じさせたことを確認できる。

## D19 流通範囲
- Primary: `D19.LOC.REGIONAL` — 地域・地方
- Parent: `D19.LOC`
- Status: `I`
- 根拠: 最古確認は京都市内版新聞で、後代も京都・深泥池の地域怪談として整理される。全国型タクシー幽霊への影響は現Evidenceから確定しない。

## D20 特権情報保持者
- Primary: `D20.PER.EXPERIENCER` — 体験者本人
- Parent: `D20.PER`
- Status: `I`
- 根拠: 消失直前の乗客の状態や車内体験について追加情報を持つ中心人物は運転手本人である。警察も調査するが、真相を解明した保持者ではない。

## D21 現実アンカー
- Value: `D21.A2` — 具体的実在対象
- Status: `D`
- 根拠: 京都市北区の実在する深泥池という特定地点へ伝承が固定されている。京都市・文化庁の公式資料で地点の実在性を直接確認できる。

# 2. causal precheck

```text
D11: 日常業務で乗客と会い相互行為へ入る
→ D12: タクシー運転手が体験者となる
→ D13: 異常な乗客が顕現し、通常の降車過程なしに消失する
→ D15: 何者だったのか／どう消えたのかが未解決のまま残る
```

接続は成立する。

# 3. D18 L3 precheck

1969年新聞の見出しに警察届出・パトカー対応が直接確認されるため、X Evidenceありとして `L3=1` を採る。ただし「伝承が広く流布したためタクシー業界が経路変更した」等の後代主張は直接Evidence不足のためL3根拠に使用しない。

# 4. U / NA / C precheck

- D10=`U`: 初期Scopeで原因存在論が確定しないため意図的。
- NA: なし。
- C: なし。

# 5. taxonomy gap precheck

現行Scopeは既存taxonomyで表現可能。確定的な新Child要求なし。

# 6. R3 freeze

本ファイルは旧 `0059_10_analysis.md` および旧Excel coding値を参照せずに作成した独立判定である。以後のR4で初めて旧10を参照する。