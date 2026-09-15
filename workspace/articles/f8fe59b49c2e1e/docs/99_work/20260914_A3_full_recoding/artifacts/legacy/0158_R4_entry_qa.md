# 0158 R4 Entry QA

- R3 freeze: `aa7968f1cd5d12817409a534247050e5e8bf6cef`
- Old 10 compared after freeze: `0158_10_analysis.md`

## 差分裁定

- D01: `G0/I` 維持。
- D02: 旧Uから `D02.PRT.BOOK/I` へ更新。00および専門研究書誌から、現在固定できる最古流通層を江戸期随筆・奇談書として安全に写像できる。原本未固定のためI。
- D03: `D03.PRT.BOOK/I` を維持。Webは研究・再提示の補助でありSecondaryへ置かない。
- D04: 旧 `DIGITAL_REAMPLIFICATION` より、異国奇談をUFOへ読み替える `D04.REC.CONTEXT_UPDATE` Primary + `ACCRETION` Secondaryが変容の本質に適合。
- D05: 旧地元伝聞より、漂着事件の物語と正体詮議が結合する `NARRATIVE_EXPLANATION`。
- D06: 共同体既知事実と断定せず `T6/I`。
- D07: `UNKNOWN_EXISTENCE` Primaryを維持し、説明不能事象Secondary。
- D08: 江戸期物語内部の直接遭遇をPrimary、現代受容の手掛かりである文書をSecondary。
- D09: `TYPE_ASSIGNMENT` Primary + `DELIBERATE_NONRESOLUTION` Secondary。歴史接続は現実アンカーD21で保持する。
- D10: 未知女性／舟を原因主体と断定せず、`D10.PHN.ANOMALOUS_EXPERIENCE` Primary + `D10.OBJ.ARTIFACT_DEVICE` Secondary、Status `C`。後代UFO解釈との競合を保存。
- D11: `D11.CON.LOCATION_STATE/I`。異形舟が海岸へ漂着し観察可能となる状態を接触条件とする。
- D12: `PROTAGONIST_EXPERIENCER` 維持。
- D13: `MANIFEST_ONLY` 維持。
- D14: NEU維持。
- D15: `UNCERTAINTY_PRESERVED` 維持。
- D16: R3の順次エピソードより、怪異の主作用が一回の漂着遭遇で成立する `D16.EVT.SINGLE_OBSERVATION` を採用。
- D17: 海へ返す行為は危害解除の安定規則ではない。旧 `D17.NON.OBSERVATIONAL_ONLY` を採用。
- D18: L1=1,L2=0,L3=0維持。
- D19: 地域伝承としての基盤と後代の広域流通を分離し、`LOCAL_TRADITION` Primary + `NATIONAL_PUBLIC` Secondary / I。
- D20: 旧地元住民は真相保持Evidence不足。`D20.UNK.NO_ONE_KNOWS/I`。
- D21: A4維持。

## QA

- 初期奇談とUFO解釈を混成しない: Pass。
- 原本未固定をD01/D02 status Iへ反映: Pass。
- L3 independent X Evidenceなし: 0、Pass。
- H3 Primary exactly 1 / Secondary <=2: Pass。
