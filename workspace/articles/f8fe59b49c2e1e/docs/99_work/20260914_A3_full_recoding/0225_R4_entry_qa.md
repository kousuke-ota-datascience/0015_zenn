# 0225 R4 Entry QA

## Result

R3 freeze後に旧 `0225_10_analysis.md` を検索したが存在しなかったため、Prior codingとの比較は不要。R3をEvidence正本とcoding rulesに照らして再監査した。

## Final decisions

- D01=`G3/I`: 昭和50年代頃という後代取材は利用するが、厳密な初出年とはしない。
- D02/D03=`PEER_ORAL/I`: 起源媒体をWebや後代記事へ置き換えない。
- D04=`COMMERCIALIZATION` Primary + `HOAX_REFRAMING` Secondary: ジョーク／都市伝説が実物PR装置へ翻訳された変容を表す。
- D06=`T7/I`: 「一般家庭の水道事実」ではなくジョーク性が知られた再流通。
- D07=`GROUP_STEREOTYPE`: 愛媛＝みかんという地域属性の誇張が意味形成の中心。
- D13=`ENV_OBJECT_MANIPULATION`: 伝承内の装置挙動として水道がジュースを出す。
- D17=`BENIGN_NO_CONTROL`: 脅威・禁忌ではない。
- D18=`1/0/1`: 企業公式に、都市伝説再現のため実物蛇口を制作したという独立X Evidenceがある。
- D21=`A2`: 具体的地域・商品・企業に固定するが、制度史を因果成立条件とはしない。

## QA

- H3 Primary exactly 1: pass.
- Secondary 0–2: pass.
- Parent/Child relation: pass.
- U/NA/C misuse: none.
- L3 evidence: pass.
- Scope leakage: 2008年以後の運営詳細・現営業状況をcoding根拠から除外。

R4: 完了。