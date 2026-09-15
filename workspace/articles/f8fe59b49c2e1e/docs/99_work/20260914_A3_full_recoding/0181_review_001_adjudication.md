# 0181 Review_001 adjudication

## Scope
- Entry: `0181` コトリバコ
- Review: `Review_0181_00_001.md` / `Review_0181_10_001.md`
- SHA gate: PASS。Review対象blobと修正開始時の正本blobが一致。

## 00 findings
- F00-001: ACCEPT。tutorial正規構造へ全面再配置。
- F00-002: ACCEPT。1.7〜1.10、2.3〜2.4、3.1〜3.3を追加し、主体・場所・物・帰結・不確実性・Evidence mappingを明示。
- F00-003: ACCEPT。独自Evidence結論と生成裁定を除去し、6/6と6/11の差分事実をHistory Evidenceへ限定。

## 10 findings
- F10-001: ACCEPT。全Dimensionを単一Statusへ裁定し、複合Statusを廃止。
- F10-002: ACCEPT。D16を呪物の長期存続から、箱の持込み→異常反応→危険認識→処置→安置という `D16.EVT.SEQUENTIAL_EPISODE / I` へ変更。
- F10-003: ACCEPT。tutorial正規階層・中心質問・判定根拠・現れ方へ復旧し、独自QA節を除去。
- F10-004: ACCEPT。D07は6/6核を尊重し `D07.ANO.UNEXPLAINED_EVENT` をPrimary、死亡説明をSecondaryへ移動。
- F10-005: ACCEPT。D01は最古確認点ではなく、6/6→6/11の直接ログが形成・増補過程そのものを示すことを根拠に `D01.G6 / D` を維持。

## Final decisions
- D01 `D01.G6 / D`
- D04 `D04.VAR.ACCRETION` + `D04.COL.SERIALIZATION / D`
- D07 `D07.ANO.UNEXPLAINED_EVENT` + `D07.DCF.UNEXPLAINED_DEATH_LOSS / I`
- D10 `D10.OBJ.CURSED_OBJECT` + `D10.SUP.IMPERSONAL_CURSE / I`
- D11 `D11.MAN.TAKE_OWN_CARRY` + `D11.CON.LOCATION_STATE / I`
- D15 `D15.BOD.DEATH` + `D15.BOD.SEVERE_INJURY / I`
- D16 `D16.EVT.SEQUENTIAL_EPISODE / I`
- D17 `D17.RIT.RELIGIOUS_SPECIALIST` + `D17.CST.CONTAINMENT / I`
- D18 `1/0/0 / I`
- D20 `D20.EXP.RELIGIOUS_FOLKLORE` + `D20.PER.FAMILY_BLOODLINE / I`
- D21 `D21.A4 / I`

## Status
Coder correction complete. Entry remains `再レビュー待 / 001` pending reviewer approval.