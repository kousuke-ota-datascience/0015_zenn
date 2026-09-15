# 0169 R4 Entry QA

- R3 freeze: `598f41fc79c2871b347c4981d4d9984057b87ae0`
- Old 10 compared after freeze: `0169_10_analysis.md`

## 差分裁定

- D01/D02/D05/D10/D11/D12/D14/D16/D17/D18/D19: R3/旧10の本質一致。
- D03: 書籍PrimaryはD。テレビは00にあるが個別番組未固定なのでSecondary `TELEVISION/I` を保持。
- D04: 旧STABILIZED_CANONを退け、1998年まで続く解釈更新とメディア展開を `CONTEXT_UPDATE` Primary + `MASS_ADAPTATION` Secondary。
- D06: 歴史・科学的既知事実より「予言を条件付きで信じる」提示が適合するため旧 `D06.T5/I` を採用。
- D07: `FUTURE_SOCIAL_CHANGE` Primary + 旧 `FATE_OMEN` Secondary。危機不足一般より予兆性が直接的。
- D08: 予言詩という `TEXT_DOCUMENT` Primary、五島解釈という `UNSUPPORTED_ASSERTION` Secondary。
- D09: 旧 `D09.PPR.OMEN_FORECAST` Primaryが適切。Secondary `HISTORICAL_ANCHOR` で16世紀詩を現代へ接続。
- D13: `INFORMATION_INDUCED_ACTION` Primary + `MENTAL_INFLUENCE` Secondary。旧 `FATE_FIXING` は予測と原因を混同するため退ける。
- D15: 集団パニックを直接固定する独立Evidence不足。`FEAR_TRAUMA/I` を採用し、社会変化はL3へ上げない。
- D20: 予言内容は大衆出版され一般共有されるため、旧 `D20.NON.COMMON_KNOWLEDGE/I` を採用。解釈者の情報優位を真相特権としない。
- D21: `A1` は弱すぎる。歴史人物、予言詩、1973年出版、1999年という実在記録を予言モデルへ統合するため `A4/I`。

## QA

- 予言の「予測」と破局の「原因」を分離: Pass。
- 1999年不成就を伝承不存在としない: Pass。
- L3=0: 流通規模と独立社会効果を区別、Pass。
- H3制約: Pass。
