# 0118 函館山の切れない木 — R4 Entry QA

- R3 freeze SHA: `c17adc9a37c3019e585346f3b8ebbdcdc0346f32`
- 旧10 blob: `f6d5bd343c1869351316c0ea49ac5ec7f311edee`
- QA対象: R3独立判定、旧10差分、causal coherence、L3、U/NA/C、taxonomy、Secondary過剰付与

## 1. 旧10との差分裁定

| Dimension | R3 | 旧10 | R4裁定 | 不一致分類 |
|---|---|---|---|---|
| D02 | Webサイト / D（2016） | Webフォーラム / I（2008転載） | R3採用。2008資料は原ログを直接固定できず、今回のR1ではH補助に限定。最古に直接確認できる2016 Webを採る。 | Evidence mismatch |
| D03 | Webサイト + video + book / D | forum + video + book / I | R3採用。直接確認済み媒体をPrimaryにする。 | Evidence mismatch |
| D04 | medium shift + stabilized | stabilized + cross-media | 旧10採用。複数媒体の存在は確認できるが、時系列的な「媒体移行」そのものを十分に再構成していない。核の安定＋クロスメディア再提示が安全。 | Code-selection mismatch |
| D05 | local hearsay + taboo | taboo + local hearsay | 旧10採用。実践的な識別核は「その木を切ってはいけない」という禁忌規則。 | Code-selection mismatch |
| D07 | dangerous place + misfortune streak | environmental anomaly + dangerous place | 旧10採用。00が明示する意味形成上の問いは「通常なら除去されそうな木がなぜ残るか」という景観異常。 | Code-selection mismatch |
| D08 | physical trace + community repetition | physical trace + unsupported assertion | 旧10採用。反復共同体証言より「伐採すると事故が起きた」という未確認主張の方がEvidenceに直接対応。 | Evidence mismatch |
| D09 | direct cause + tabooization | tabooization + direct cause | 旧10採用。00が「場所に宿る禁止規則」と明示し、因果説明は禁忌形成の根拠として働く。 | Code-selection mismatch |
| D10 | cursed object + impersonal curse / C | specific place + impersonal curse / I | 旧10を基礎に採用。自然物の木を`CURSED_OBJECT`へ入れるのは人工物寄りのChild定義と境界的。特定場所／対象へ固定された非人格的祟りとしてコードし、taxonomy gap候補を記録。 | Taxonomy gap / Code-selection mismatch |
| D16 | sequential episode | delayed | `D16.EVT.SEQUENTIAL_EPISODE`を採用。一定時間後の発現という時間差より、伐採試行→災い→断念→残存という複数段階が意味形成に必要。 | Code-selection mismatch |
| D19 | regional + national / C | local tradition + national / I | Primaryは`LOCAL_TRADITION`を採用、StatusはC。場所固有の地域伝承圏が核だが、広域再流通の実範囲は定量不十分。 | Code-selection mismatch / Status mismatch |
| D20 | U | common knowledge + local resident / I | R3採用。地元住民が追加真相を保持するという直接Evidenceがなく、公開流通していることだけから特権保持構造は決められない。 | Evidence mismatch |
| D21 | A2 / D | A2 / I | `A2 / I`。函館山は実在するが、伝承対象の特定木を公的管理資料で正式同定していない。 | Status mismatch |

その他は実質一致。

## 2. Causal QA

最終因果鎖:

`函館山の特定木を伐採・撤去しようとする` → `場所／木に付着した非人格的な祟り` → `事故・不幸という不運` → `伐採断念` → `木が残る` → `残存景観が伝承を補強`

- D07/D08は「現在なぜ木が残るのか」という景観上の問いと痕跡。
- D09はその景観と事故談を「切ってはいけない」という禁忌へ変換する操作。
- D13は事故種類を推測せず、広い呪詛・不運付与で保持。
- D15も死亡・重傷へ過剰具体化せず一般的不運。
- D16は一続きの因果系列として処理し、根拠のない潜伏時間を付与しない。

**Causal QA: Pass**

## 3. L3 QA

- L1: 1 — 伝承内で伐採試行→災い→断念が成立。
- L2: 0 — 噂を知ること自体は危害条件ではない。
- L3: 0 — 「道路が木を避けた」「工事を現実に中止した」ことを独立に裏付ける工事・行政X Evidenceがない。伝承内部の工事断念を現実効果へ昇格させない。

**L3 QA: Pass**

## 4. U / NA / C QA

- D01=U: 成立年代不明。
- D19=C: 地域伝承圏としての核と、後代の広域メディア流通の実範囲を一意に粗視化できない。
- D20=U: 誰が特権的追加情報を保持するか固定できない。
- D10はCではなくIへ整理。特定場所＋非人格的祟りを主副で記述できるが、自然物の祟り木に完全一致するChild不足はtaxonomy gapとして別記。

**Status QA: Pass**

## 5. Taxonomy / Secondary QA

### Taxonomy gap候補

D10には「自然物そのもの／自然物に宿る固有の霊的効力」を直接表すChildがない。`D10.OBJ.CURSED_OBJECT`は人工物・呪物寄り、`D10.SPC.SPECIFIC_PLACE`は対象物そのものより地点を表す。本Entryでは特定場所をPrimary、非人格的呪力をSecondaryとして暫定処理し、Global Reconciliation候補とする。

### Secondary監査

- D05 Local hearsay: 禁忌の由来が地域怪談として伝聞されるため保持。
- D07 Dangerous place: 景観異常の説明結果として危険属性が付くため保持。
- D08 Unsupported assertion: 事故史未確認というEvidence差を保存するため保持。
- D09 Direct cause: 禁忌化を支える「切る→災い」の因果規則として保持。
- D19 National public: 映像・書籍等の地域外再流通を保存するため保持。

**Taxonomy / Secondary QA: Pass（D10 gapをGlobal Reconciliationへ送る）**

## 6. 最終判定

R1〜R4 Coder workflowは完了。00正本の変更は不要。10正本をR4裁定で置換し、外部Review_001待ちへ移行する。

**R4 Status: 完了。**