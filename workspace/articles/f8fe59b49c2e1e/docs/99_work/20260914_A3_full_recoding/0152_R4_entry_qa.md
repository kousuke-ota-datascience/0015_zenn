# 0152 R4 Entry QA

- R3 freeze: `684c16cb9c01a5fee5babcb8e415ca775fd65d6b`
- Old coding canonical compared after freeze: `0152_10_analysis.md`

## 差分裁定

- D01/D02/D04/D06/D08/D09/D10/D13/D14/D18/D19/D20: R3と旧10は本質一致。
- D03: 旧10の `TELEVISION` Primary + `WEBSITE` / `BOOK` Secondaryを採用。公開Q&AはEvidenceだがH3 2枠では書籍までの媒体横断を保持する。
- D05: `EXPLANATORY_CLAIM` Primaryに加え `FACT_CLAIM` Secondaryを採用。
- D07: Primary `ENVIRONMENTAL_ANOMALY` は一致。Secondaryは経路構造そのものの異常ではなく場所の危険化を表す `DANGEROUS_PLACE` を採用。
- D11: `ENTER_PASS_CROSS` より、樹海内部という状態条件を表す `D11.CON.LOCATION_STATE` を採用。磁気作用は進入動作そのものではなく所在状態に結び付く。
- D12: 直接作用対象は人ではなく方位磁針であるため `D12.OTD.DEVICE_INFRA` Primary、体験者Secondaryを採用。
- D15: 中心終端はどちらへ進むかという経路選択の崩れであり、旧10の `D15.BEH.AVOIDANCE_ROUTE_CHANGE` を採用。死亡・失踪へ拡張しない。
- D16: `ENDURING_CONDITION` より場所Xと現象Yの静的対応を直接表す `D16.STA.STATIC_ASSOCIATION` を採用。
- D17: 現地測定・科学反証は俗説の真偽検証であり、Version Scope内部の因果を回避・解除する固有規則ではないため `U` を採用。
- D21: 実在森林・溶岩・方位磁針への具体固定で足り、科学資料を伝承内部の必須史実として因果統合しないため `D21.A2` を採用。

## QA

- 強い俗説と局所的磁気影響を分離: Pass。
- 実測反証を伝承消滅とみなさない: Pass。
- L3=1: 2009年大学活動が俗説そのものを現地検証課題として実行した直接Evidenceに限定、Pass。
- H3 Primary/Secondary: Pass。
- U/NA/C: D01=U、D17=U。Pass。

## Coding canonical

既存 `0152_10_analysis.md` は上記最終裁定と一致するため変更せず維持する。
