# 0275 R4 Entry QA

## Result

R3 freeze後に旧 `0275_10_analysis.md` を検索したが存在しなかったため、Prior codingとの比較は不要。R3をEvidence正本・Version Scope・固定coding rulesに照らして再監査した。

## Final decisions

- D01=`G5/D`: 磐梯山手長足長型の直接確認は1992年。物語内806年や一般的魔魅記述から前近代へ遡及しない。
- D02=`PRT.MAGAZINE/D`: 1992年『あしなか』が最古直接確認媒体。
- D03=`PRT.MAGAZINE` Primary、`INS.OFFICIAL_NOTICE`・`WEB.WEBSITE` Secondary: 研究DBの存在自体ではなく、磐梯町の公的計画・文化施設ページによる実際の再提示を流通Evidenceとして扱う。
- D04=`RETELLING` + `COMMERCIALIZATION` + `ACCRETION`: 容器・調伏者等の細部変異と、2025年展示・肝試し・参加企画への転用を区別して保持。
- D05=`LOCAL_HEARSAY` + `NARRATIVE_EXPLANATION`: 地域伝承としての提示と、公的再話での物語＋由来説明を保持する。原体験談とは扱わない。
- D07=`ENVIRONMENTAL_ANOMALY` + `MISFORTUNE_STREAK`: 天変地異・不作を意味形成対象とする。
- D09=`HIDDEN_CAUSE` + `HISTORICAL_ANCHOR`: 気象被害を怪物へ原因帰属し、弘法大師・磐梯明神へ歴史宗教的に接続する。
- D10=`YOKAI_ENTITY` + `DEITY_DIVINE`: 調伏前の人格怪異と調伏後の神格化を同一Scope内の状態転換として保持する。
- D11=`AMBIENT_EFFECT`: 住民個人の接触ではなく地域環境へ広域作用する。
- D12=`NATURAL_ENVIRONMENT` + `LOCAL_COMMUNITY`: 環境が直接操作され、その影響が地域共同体へ及ぶ。
- D13=`ENV_OBJECT_MANIPULATION` + `LUCK_BENEFIT`: 気象・水の操作が主機構。神格化後の豊作・幸福付与をSecondaryとする。
- D14=`MIX`: 災害・不作と、調伏後の保護・豊作が重要な正負帰結として併存。
- D15=`PROPERTY_DAMAGE` + `DISASTER_AVOIDANCE`: 農作物被害と、その後の災厄回避を分離。
- D16=`SEQUENTIAL_EPISODE`: 災害作用→困窮→介入→封印→神格化という一続きの段階構造。
- D17=`CONTAINMENT` + `RELIGIOUS_SPECIALIST`: 容器への封印と宗教者介入が明示的制御法。
- D18=`1/0/1`: L3は神社・伝承の存在から推定せず、2025年町立資料館の具体的展示・参加催事を独立X Evidenceとして採用。
- D19=`LOCAL_TRADITION` + `REGIONAL`: 磐梯山・会津地方の地域文化圏に固定。
- D20=`NO_HIDDEN_TRUTH`: 弘法大師の調伏能力を「特権情報保持」と誤コードしない。共有核自体は公的に開示される。
- D21=`A4`: 磐梯山・弘法大師・磐梯明神という具体的地理・歴史宗教伝承を物語因果へ統合する。ただし史実性の認定ではない。

## QA

- H3 Primary exactly 1: pass.
- Secondary 0–2: pass.
- Parent/Child relation: pass.
- U/NA/C misuse: none.
- D07→D09→D10 causal interpretation: pass.
- D11 vs D17 separation: pass.
- D12→D13→D15 causal QA: pass.
- D18 L3 evidence gate: pass; 2025年自治体文化施設による展示・催事という独立X Evidenceあり。
- Scope leakage: 鳥海山類話、一般的手長足長図像、2026年以後の創作翻案を除外。
- Prior coding contamination: 旧10不存在、R3 freeze前未参照。

R4: 完了。