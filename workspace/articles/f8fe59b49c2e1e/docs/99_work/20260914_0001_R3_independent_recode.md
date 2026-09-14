# 0001 口裂け女 — R3 全21次元 独立再コーディング

- 実施日: 2026-09-14
- 対象Entry: `0001`
- Evidence正本: `docs/10_each_lore/0001_kuchisake_onna/0001_00_contents.md`（R1確定版）
- Version Scope正本: `docs/99_work/20260914_0001_R2_version_scope.md`
- baseline taxonomy: `20_urban_legend_parent_child_code_system.md` blob `192c2f1593e29eb32292a31d564d21ad4aec2427`
- baseline rules: `30_urban_legend_analysis_coding_rules.md` blob `4fe19cc922840a46367bc6d0162271d2422996a9`
- 独立性: この判定時点で既存 `0001_10_analysis.md` のD01〜D21および全件正本Excel既存コードは参照していない。

# 1. Version Scope

1979年初頭〜同年春に確認できる最小安定共有核、すなわち、**日常社会の中に「口が裂けた女性（口裂け女）」が出現し、主として小中学生・子どもを脅かし／襲うとされる遭遇・危害型の噂**をコード対象とする。

「私、きれい？」、マスク、赤いコート、鎌、高速走行、ポマード、べっこう飴、特定回答、由来説明等はR2でScope外としたため、D07〜D17の判定根拠には使用しない。

# 2. 独立再コード結果

| D | Primary Child / Value | Primary Parent | Secondary | Status | 判定要旨 |
|---|---|---|---|---|---|
| D01 | `D01.G3` 1970年代 | — | — | `I` | 1978年暮れの口頭流通が後代専門家整理で示され、1979年1月には同時代新聞記録がある。成立を1970年代帯へ安全に写像できるが、1978年暮れ自体は回顧整理なのでI。 |
| D02 | `D02.ORL.PEER_ORAL` 友人・仲間内伝承 | `D02.ORL` | — | `I` | 1978年暮れの岐阜での子ども間の噂・反復を専門家整理が明示する。新聞掲載日を起源媒体とはしない。 |
| D03 | `D03.ORL.PEER_ORAL` 友人・仲間内伝承 | `D03.ORL` | `D03.PRT.MAGAZINE`, `D03.PRT.NEWSPAPER` | `I` | Scope成立に子ども同士の対人口承が構造的に重要。1979年1月新聞、3〜4月雑誌による再提示もScope内で確認できる。 |
| D04 | `D04.VAR.ACCRETION` 増補 | `D04.VAR` | `D04.MIG.MEDIUM_SHIFT`, `D04.VAR.ORAL_VARIATION` | `I` | 専門家整理では子ども間の反復で属性・能力・弱点が追加され、口承から新聞・雑誌へ媒体移行した。 |
| D05 | `D05.HRS.SCHOOL_WORK_HEARSAY` 学校・職場伝承 | `D05.HRS` | `D05.PRP.FACT_CLAIM` | `I` | 子ども・学校／塾圏で共有される噂として流通し、「口の裂けた女性が小中学生を襲う」という事実主張型命題としても再提示される。FOAF連鎖は直接確認できないため付与しない。 |
| D06 | — | — | — | `U` | 受容者へ伝達された原発話・採録本文を固定できておらず、共同体既知事実・条件付き信念・真偽未確定のいずれとして提示されたかを一意に決めない。 |
| D07 | `D07.ICT.STRANGER_THREAT` 見知らぬ他者の脅威 | `D07.ICT` | `D07.ICT.ABDUCTION_ASSAULT` | `I` | この伝承がなければ残る不確実性は、日常空間で遭遇する見知らぬ女性が子どもを襲うという対人脅威。Scope内で未知存在の超自然性までは固定しない。 |
| D08 | `D08.STY.COMMUNITY_REPETITION` 共同体反復証言 | `D08.STY` | `D08.CLM.UNSUPPORTED_ASSERTION` | `I` | 子ども集団内で反復される噂自体が問題化の手掛かりであり、初期Scopeでは物的証拠・事件記録より、根拠未提示の共同体反復が中心。 |
| D09 | `D09.AGN.AGENCY_ATTRIBUTION` 主体化 | `D09.AGN` | `D09.CAT.NAMING`, `D09.CAU.DIRECT_CAUSE` | `I` | 都市空間の曖昧な危険を「口裂け女」という名前を持つ女性主体へ集約し、その主体が子どもへの襲撃原因だとする。 |
| D10 | `D10.HUM.INDIVIDUAL_HUMAN` 個人 | `D10.HUM` | — | `I` | R2 Scopeは「口が裂けた女性」を因果主体とするが、幽霊・妖怪・超自然能力を最小核へ含めない。したがって強い超自然存在論を付与せず、保守的に人間個人。 |
| D11 | `D11.PAS.SPONTANEOUS_SELECTION` 偶然選ばれる・遭遇する | `D11.PAS` | — | `I` | 特定儀式・回答・場所進入が再現的発動条件としてScopeに固定されておらず、子どもが非意図的に危険主体へ遭遇する構造。 |
| D12 | `D12.GRP.DEMOGRAPHIC_GROUP` 属性集団 | `D12.GRP` | — | `D` | D13主作用「襲う」の直接対象は同時代雑誌見出しで明示される小・中学生／子ども集団。 |
| D13 | `D13.PHY.PHYSICAL_ATTACK` 物理攻撃 | `D13.PHY` | — | `D` | 1979年4月『週刊新潮』見出しが「口のさけた女性が小・中学生を襲う」と直接示す。具体的損傷・武器はScope外。 |
| D14 | `D14.NEG` 負 | — | — | `D` | 主作用が子どもへの襲撃・脅威であり、帰結極性は負。 |
| D15 | — | — | — | `U` | Scope内で「襲う」ことは確認できるが、最終的な傷害、死亡、失踪等の帰結領域は固定できない。D13の物理攻撃をそのまま重傷・死亡へ読み替えない。 |
| D16 | `D16.EVT.SINGLE_EPISODE` 単一エピソード | `D16.EVT` | — | `I` | 偶然遭遇から襲撃までが一つの連続した遭遇事象として編成される。遅延・周期・長期進行はScope内で確認しない。 |
| D17 | — | — | — | `U` | ポマード、べっこう飴、特定回答等はR2でScope外。最小核に有効な回避法があるとも、回避不能とも断定しない。 |
| D18 | `L1=1`, `L2=0`, `L3=1` | — | — | `D` | L1: 伝承内で女性が子どもへ襲撃作用を持つ。L2: 噂を聞く／読むこと自体が伝承内の発動条件ではない。L3: 1979年6月15日『秋田魁新報』見出しに「子供の登校拒否騒ぎも」とあるX Evidenceを確認。 |
| D19 | `D19.KIN.SCHOOL_YOUTH` 学校・若者集団 | `D19.KIN` | `D19.LOC.REGIONAL` | `D` | 小中学生が中心受容集団であり、1979年3月には京都到達、4月には「各地」と同時代誌面見出しで広域化を確認できる。6月末の全国化はScope後続史として保持する。 |
| D20 | `D20.NON.NO_HIDDEN_TRUTH` 隠れた真相なし | `D20.NON` | — | `I` | 最小Scopeは「危険な女性がいる／襲う」という共有主張で完結し、特定の専門家・内部者だけが持つ追加真相・回避情報を構造上要求しない。後代弱点知識はScope外。 |
| D21 | `D21.A1` 一般的現実背景 | — | — | `I` | 初期流通史は岐阜等の実在地に接続するが、Scoped Content自体は特定地点・制度・史実がなければ成立しない構造ではなく、日常の学校・街路という一般現実背景で成立する。 |

# 3. 因果列予備確認

R4前の予備確認として、主因果列は次のように接続する。

```text
D11: D11.PAS.SPONTANEOUS_SELECTION
→ D12: D12.GRP.DEMOGRAPHIC_GROUP
→ D13: D13.PHY.PHYSICAL_ATTACK
→ D15: U（攻撃後の最終帰結領域はScope内Evidence不足）
```

D15を推測で重傷・死亡へ埋めないことは、現行coding rulesと整合する。

# 4. 独立判定時点のtaxonomy gap候補

R3単独では、既存Childへ収まらない安定構造は確認しない。

留意点として、D19で「全国規模だが受容集団は小中学生」という**社会範囲の広さと人口属性をH3で同時表現する必要**がある。ただしR2 Scopeでは春までを対象とし、`SCHOOL_YOUTH + REGIONAL` で表現できるため、本Entry単独を理由にtaxonomy変更候補とはしない。

# 5. R4への引継ぎ

R4で初めて既存 `0001_10_analysis.md` と全件正本Excel既存値を参照し、差分を `Scope mismatch / Evidence mismatch / Code-selection mismatch / Status mismatch / Taxonomy gap / Prior coding error` に分類する。R3値は旧値へ合わせて変更しない。
