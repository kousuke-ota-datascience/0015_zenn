# 0137 R4 Entry QA

- R3 freeze: `5310aca86bb3719fa09703f6693804ad546ef62c`
- Old coding canonical compared only after R3 freeze: `0137_10_analysis.md`

## 差分裁定

- D01/D02/D05/D06/D07/D09/D10/D11/D12/D14/D17/D18: R3と旧10は本質一致。
- D03: 旧10の `WEB.FORUM + PRT.BOOK` を採用。後代書籍流通は00で直接確認できる。
- D04: `VAR.ACCRETION` Primaryは一致。Secondaryは映画化を直接表す `D04.MED.MASS_ADAPTATION` を採用。
- D08: R3 `FOAF_REPORT` より、Evidence正本が強調する「秘密村命題に検証可能証拠が付かない」ことを反映し、旧10の `D08.CLM.UNSUPPORTED_ASSERTION` Primary + `D08.HIS.PLACE_NAME_RUIN` Secondaryを採用。
- D13: 1999年早期型の具体動作は追跡であるため `D13.REL.PURSUIT` Primary、`D13.PHY.PHYSICAL_ATTACK` Secondaryを採用。
- D15: 生還FOAFの共有終端は恐怖であり、重傷は強い異伝。`D15.MND.FEAR_TRAUMA` Primary + `D15.BOD.SEVERE_INJURY` Secondaryを採用。
- D16: 個別訪問の段階進行より、村が恒常的に制度外・排他的であることがEntry-level主張を規定するため `D16.STA.ENDURING_CONDITION` を採用。
- D19: 全国公開Web・映画・書籍まで確認できるため `D19.MAS.NATIONAL_PUBLIC` Primary、地域伝承圏Secondaryを採用。
- D20: 物語構造上、村人／地元側が場所・内部規則の情報優位を持つため `D20.INS.LOCAL_RESIDENT` / `I` を採用。ただし現実の秘密村の実在を意味しない。
- D21: 実在犬鳴地域・旧トンネル等は具体対象アンカーだが、現実史を秘密村因果へ必須統合しないため `D21.A2` を採用。

## QA

- 実在旧集落と都市伝説上の村を分離: Pass。
- H3 Primary exactly 1 / Secondary <=2: Pass。
- L3 independent X Evidence: 不足のため0、Pass。
- U/NA/C: D01=Uのみ。Pass。
- D07 procedural contamination: 旧断片と独立判定は一致したが、最終裁定は00本文の「地図・行政・法制度から切り離された村」記述とコードブック `HIDDEN_VANISHED_PLACE` / `SYSTEM_CAPACITY_FAILURE` 定義に基づく。Reviewerへ注記を残す。

## Coding canonical

既存 `0137_10_analysis.md` は上記最終裁定と一致するため、内容変更を行わず正本として維持する。
