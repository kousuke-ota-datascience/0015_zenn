# 0133 将門塚の祟り — R4 Entry QA

- R3 freeze SHA: `058af26a784d134cd6c02e72fac278cb3447f037`
- 旧10 blob: `c1f1b83738dbe977999203dfbf32698b25ce0602`
- QA対象: R3独立判定、旧10差分、causal coherence、L3、U/NA/C、taxonomy、Secondary過剰付与

## 1. 旧10との差分裁定

| Dimension | R3 | 旧10 | R4裁定 | 不一致分類 |
|---|---|---|---|---|
| D04 | event attachment + context update + accretion | accretion + event attachment | R3採用。古い祟りモデルが関東大震災・官庁再建等の実事件へ再接続される変容をPrimaryとし、時代適応・増補をSecondaryで保持。 | Code-selection mismatch |
| D08 | death history + historical event | historical event + place/name/ruin | 旧10採用。平将門の死だけでなく震災・再建・祭祀史を一括する`HISTORICAL_EVENT`をPrimary、実在塚・碑をSecondaryとする方がScope全体を覆う。 | Code-selection mismatch |
| D09 | direct cause + historical anchor + tabooization | direct cause + historical anchor | R3採用。祟り因果から「粗末に扱うな」という規範形成までが意味形成核であり、Secondary上限2の範囲内で両者を保持。 | Code-selection mismatch |
| D10 | ghost + place / D-I | 同じ / I | `I`に統一。将門の御霊は資料に明示されるが、D10としての主副粗視化は分析判断を含む。 | Status mismatch |
| D13 | curse / D-I | curse / D | 旧10採用。祟りによる災厄付与はEvidence正本の共有核に直接対応。 | Status mismatch |
| D16 | static rule | triggered recurrence | 旧10採用。荒廃・改変条件が時代ごとに再成立するたび同じ祟り因果が再利用される反復性がVersion Scopeの重要構造。 | Code-selection mismatch |
| D18 | L3=1 / I | L3=1 / D | `D`。現実の慰霊祭・例祭・史蹟保存会・複数回の保存改修は公式一次資料で確認でき、将門の鎮魂・祭祀・保存を目的とする実践として明示される。超自然因果の実在とは別。 | Status mismatch |
| D19 | cross-generational + local tradition | local tradition + national public | 旧10採用。D19は長期持続より流通共同体の範囲を優先し、地域祭祀圏をPrimary、広域大衆流通をSecondaryとする。 | Code-selection mismatch |
| D20 | religious specialist | religious specialist + institution authority | 旧10採用。神田明神に加え、史蹟将門塚保存会等が保存・改修実務の追加情報を保持するためSecondaryを保持。 | Code-selection mismatch |

その他は実質一致。

## 2. Causal QA

最終因果鎖:

`将門塚の荒廃・冒涜・無礼な撤去／改変` → `平将門の御霊が怒る` → `事故・病気・災厄等の不運` → `祟りと解釈` → `鎮魂・供養・祭祀・保存管理` → `御霊を鎮め場所を維持`

この因果鎖は中世の鎮魂伝承を基底に、関東大震災・官庁再建等の近現代史へ反復適用される。現実の事故・災害を祟りの実証とはしない。

**Causal QA: Pass**

## 3. L3 QA

- L1: 1 — 伝承内では不敬・改変に対し御霊が災厄を与える。
- L2: 0 — 伝承を知るだけでは祟り条件にならない。
- L3: 1 — 神田明神公式が慰霊・例祭・史蹟保存会・保存改修を実際の宗教的／社会的実践として記録する。特に2021年改修は「祈り」「清浄な地」「鎮まり」を明示し、物理工事と祭祀尊重が両立している。

**L3 QA: Pass**

## 4. U / NA / C QA

- D02=U: 中世以来の継承は分かるが、最古の流通媒体を直接固定できない。
- その他はD/Iで十分に決定可能。Cを必要とする主要競合はない。
- NAなし。

**Status QA: Pass**

## 5. Taxonomy / Secondary QA

### Taxonomy gap候補

D11には「供養を怠る」「維持を放棄する」といった**不作為による禁忌違反**を直接表すChildがない。`CREATE_ALTER_MANIPULATE`は撤去・破壊・無礼な改変型には適合するが、荒廃放置型を完全には表現しない。Global Reconciliation候補とする。

### Secondary監査

- D04 `CONTEXT_UPDATE`,`ACCRETION`: 古い怨霊譚の都市開発文脈への更新と具体逸話の増補を区別するため保持。
- D09 `HISTORICAL_ANCHOR`,`TABOOIZATION`: 実在史への接続と行動規範形成の双方が意味形成に不可欠。
- D10 `SPECIFIC_PLACE`: 死者霊作用が将門塚という地点に固定されるため保持。
- D17 `ONGOING_MANAGEMENT`: 一回の鎮魂だけでなく保存・例祭の継続を表すため保持。
- D20 `INSTITUTION_AUTHORITY`: 保存会・関係組織の実務情報を保持するため採用。

**Taxonomy / Secondary QA: Pass（D11不作為条件をGlobal Reconciliationへ送る）**

## 6. 最終判定

R1〜R4 Coder workflowは完了。00正本の変更は不要。10正本をR4裁定で置換し、外部Review_001待ちへ移行する。

**R4 Status: 完了。**