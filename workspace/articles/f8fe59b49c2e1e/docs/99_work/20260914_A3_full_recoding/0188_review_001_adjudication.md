# 0188 Review_001 adjudication

## Scope
- Entry: `0188` 八尺様
- Review: `Review_0188_00_001.md` / `Review_0188_10_001.md`
- SHA gate: PASS。Review対象blobと修正開始時の正本blobが一致。

## 00 findings
- F00-001: ACCEPT。tutorial正規構造へ全面再配置。
- F00-002: ACCEPT。主体・場所・物・条件・帰結・伝承史・Evidence mappingを所定節で追加。
- F00-003: ACCEPT。D11/D17/D15相当の統合分析を00から除去し、各レスで確認できる警告・防御・脱出・帰結をContentとして記録。
- F00-004: ACCEPT。独自Evidence結論を削除し、2008年確認事実と不確実性のみを保持。

## 10 findings
- F10-001: ACCEPT。複合Statusを全廃し単一Statusへ裁定。
- F10-002: ACCEPT。2008年最古確認点を生成時期へ変換せずD01をUへ変更。
- F10-003: ACCEPT。単一観測版から `SINGLE_FIXED` を推定せずD04をUへ変更。
- F10-004: ACCEPT。tutorial正規階層へ復旧し独自QA節を除去。
- F10-005: ACCEPT。D20を地域住民一般から家族・血縁側の知識偏在へ限定し、Kを専門知識のSecondaryとして保持。

## Final decisions
- D01 `U`
- D04 `U`
- D07 `D07.ANO.UNKNOWN_EXISTENCE` + `D07.DCF.UNEXPLAINED_DEATH_LOSS / I`
- D10 `D10.SUP.YOKAI_ENTITY / I`
- D11 `D11.SEN.VISUAL_EXPOSURE` + `D11.CON.LOCATION_STATE / I`
- D13 `D13.REL.TARGETING / I`
- D15 `D15.LIF.FUTURE_CONSTRAINT` + `D15.BEH.AVOIDANCE_ROUTE_CHANGE / D`
- D16 `D16.PRG.STAGED_PROGRESSION / I`
- D17 `D17.RUL.OBEY_TABOO` + `D17.RIT.RELIGIOUS_SPECIALIST / D`
- D18 `1/0/0 / I`
- D20 `D20.PER.FAMILY_BLOODLINE` + `D20.EXP.RELIGIOUS_FOLKLORE / I`
- D21 `D21.A1 / I`

## Status
Coder correction complete. Entry remains `再レビュー待 / 001` pending reviewer approval.