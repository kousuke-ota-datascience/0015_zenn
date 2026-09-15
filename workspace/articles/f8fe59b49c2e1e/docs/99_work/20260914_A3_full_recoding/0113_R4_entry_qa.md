# 0113 井の頭公園のボートに乗ると別れる — R4 Entry QA

- R3 freeze SHA: `7acdb0e50cdfedfc984bd0e14cf45b739eeb90f7`
- 旧10 blob: `841218759790bdc520842e90828f7fdb06ba3305`
- QA対象: R3独立判定、旧10差分、causal coherence、L3、U/NA/C、taxonomy、Secondary過剰付与

## 1. 旧10との差分裁定

| Dimension | R3 | 旧10 | R4裁定 | 不一致分類 |
|---|---|---|---|---|
| D02 | Webサイト / D | U | R3採用。2016年Web記事は「起源媒体」ではないが、現時点で最古に日時付きで確認できる実流通媒体には該当する。 | Prior coding error / Status mismatch |
| D08 | 根拠未提示の主張 | 既存の言い伝え | R3採用。「昔から」等の継承性を示すEvidenceがなく、資料が直接示すのは「噂」。 | Evidence mismatch |
| D09 | 相関規則化＋主体化 | 直接原因化＋吉凶評価 | Primaryは相関規則化。Secondaryに主体化と直接原因化を保持する。 | Code-selection mismatch |
| D10 | 神格 / D | 特定場所＋神格 / C | 旧10のCを採用。共有核は場所固定の裸ジンクスで、神格は主要説明だが必須でない。Primary=特定場所、Secondary=神格。 | Scope mismatch |
| D11 | 乗る＋恋愛関係 | 乗るのみ | R3採用。カップルであることは構造的条件。 | Code-selection mismatch |
| D13 | 不運付与＋運命固定 | 不運付与 | 運命固定は過剰。Primary不運付与のみ。 | Code-selection mismatch |
| D15 | 恋愛成否＋関係破綻 | 関係破綻 | 旧10採用。「別れる」という明示終端を最も直接表す関係破綻をPrimaryとし、恋愛成否は冗長なため付けない。 | Code-selection mismatch |
| D16 | 静的規則 | 遅延 | R3採用。期限・潜伏時間ではなく「乗れば別れる」という条件対応が主題で、coding rule上はSTATIC_RULEが適合。 | Prior coding error |
| D19 | 全国＋地域 / C | 全国＋地域 / I | R3のCを採用。2016年調査は一般向けだが地理的代表性が本文から確定できない。 | Status mismatch |
| D20 | 一般共有 / I | 一般共有 / D | Iに保守化。公開アクセスは直接確認できるが「特権保持者なし／一般共有」の分布判断自体は推論を含む。 | Status mismatch |

その他は実質一致。

## 2. Causal QA

最終因果鎖:

`恋人関係` + `井の頭池でボートへ乗る` → （場所固有ジンクス／弁財天嫉妬説明） → `恋愛上の不運` → `関係破綻`

- D10とD11を混同しない。井の頭池は因果源候補であると同時に条件の場所アンカーだが、D11 Primaryは具体行為「乗る」。
- D13は破局そのものではなく、不運を付与する非物理作用として保持。
- D15は最終状態「関係破綻」。
- D16は時間差そのものを定量規則化していないため静的条件規則。

**Causal QA: Pass**

## 3. L3 QA

- L1: 1 — 伝承内部で乗船した恋人へ破局因果が設定される。
- L2: 0 — 噂を知る／聞くこと自体は破局条件ではない。
- L3: 0 — 認知調査は噂の流通Evidenceであり、噂によってボート利用・観光・制度等が実際に変化したことを示す独立X Evidenceではない。

**L3 QA: Pass**

## 4. U / NA / C QA

- D01=U: 成立年代を固定できないため妥当。
- D10=C: 場所自体が効力を持つ裸ジンクスと、弁財天を主体化する主要説明がScope内で併存。
- D19=C: 地域定着は直接確認できる一方、広域認知の地理範囲を一意に確定できない。
- NAを置く次元なし。

**Status QA: Pass**

## 5. Taxonomy / Secondary QA

- 新規taxonomy gapなし。
- D09 Secondaryは `AGENCY_ATTRIBUTION` と `DIRECT_CAUSE` の2件。弁財天主体化と因果接続を除くと意味形成モデルが変わるため保持。
- D11 Secondaryの恋愛関係は「誰が乗っても発動」ではない点を保存するため保持。
- D13 `FATE_FIXING`、D15 `ROMANCE`、D17 `OBEY_TABOO` はPrimaryと冗長性が高いため除外。

**Taxonomy / Secondary QA: Pass**

## 6. 最終判定

R1〜R4 Coder workflowは完了。00正本の変更は不要。10正本をR4裁定で置換し、外部Review_001待ちへ移行する。

**R4 Status: 完了。**