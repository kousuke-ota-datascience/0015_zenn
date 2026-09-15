# 0132 八幡の藪知らず — R4 Entry QA

- R3 freeze SHA: `39ca6f738f62972ac58b1a4f5ba875143a5cabf9`
- 旧10 blob: `f29e8cc6f48e04a0b221398090d9ddcc8c8fd9e3`
- QA対象: R3独立判定、旧10差分、causal coherence、L3、U/NA/C、taxonomy、Secondary過剰付与

## 1. 旧10との差分裁定

| Dimension | R3 | 旧10 | R4裁定 | 不一致分類 |
|---|---|---|---|---|
| D01 | G0 / D | G0 / I | `G0 / I`。前近代存在は確実だが、生成時期を史料上の最古確認から推論して年代帯へ写像しているためIが適切。 | Status mismatch |
| D02 | BOOK / D | BOOK / I | `BOOK / I`。今回は原本本文を直接閲覧せず、公的書誌・復刻案内を通じて媒体を固定しているため保守的にI。 | Status mismatch |
| D04 | contested version + accretion + cross-media | accretion + cross-media | R3を基礎にPrimary=`CONTESTED_VERSION`、Secondary=`ACCRETION`,`CROSS_MEDIA`。由来説は単なる時系列増補だけでなく、現在も競合説明として併存する。 | Code-selection mismatch |
| D07 | spatial anomaly + dangerous place / C | dangerous place + spatial anomaly / D | 旧10採用。共有核全体を包むのは禁足・祟りを伴う危険場所で、「出られない」は主要な空間異常型としてSecondary。 | Code-selection mismatch |
| D09 | tabooization + historical anchor + hidden cause | tabooization + historical anchor | 旧10採用。複数由来説は具体的歴史接続で十分記述でき、HIDDEN_CAUSEは冗長。 | Secondary overcoding |
| D10 | specific place + impersonal curse / C | 同じ / I | 旧10採用。場所と祟りは主要異伝の競合というより補完的な存在論記述として扱える。 | Status mismatch |
| D13 | spatial distortion + curse / C | 同じ / D | 旧10採用。両作用は史料・公的解説で伝承内容として直接確認でき、Primaryは特徴的な空間異常、Secondaryに祟りを置ける。 | Status mismatch |
| D15 | disappearance + misfortune / C | misfortune + disappearance / I | 旧10採用。「祟り」は広い共有帰結で、非帰還は強い一異伝。永久失踪を共有核に過剰適用しない。 | Code-selection mismatch |
| D17 | do-not-engage + obey-taboo | obey-tabooのみ | 旧10採用。明示的禁足規則を守ることが制御法で、接触回避を別Secondaryに重複付与しない。 | Secondary overcoding |
| D18 | L3=0 | L3=1 | 旧10採用。江戸期資料は里人が祟りを理由に実際に立入を禁じる行動規範を記録しており、伝承内描写だけでなく現実の場所利用と信念の接続を示す歴史X Evidenceと評価する。 | Evidence interpretation |
| D19 | cross-generational + local tradition | local tradition + national public | 旧10採用。D19は時間持続より流通共同体を主に表すため、地域伝承圏をPrimary、印刷・図像による広域化をSecondaryとする。 | Code-selection mismatch |
| D20 | U | no-one-knows / I | R3採用。複数説が未決着であることは分析上のEvidence gapであり、伝承内で「誰も真相を知らない」と明示されることとは異なる。 | Evidence-Analysis responsibility mismatch |
| D21 | A4 / I | A2 / D | 旧10採用。歴史人物を用いる由来異伝はあるが、Version Scopeの共有核は実在地点への固定で成立し、特定史実の因果統合を必須としない。 | Scope mismatch |

その他は実質一致。

## 2. Causal QA

最終因果鎖:

`八幡不知森という境界` → `内部へ立ち入る（禁忌違反）` → `場所固有の空間異常／祟り` → `脱出困難または災い` → `入らないという規範を再確認`

由来説はこの因果鎖の「なぜ場所が特別なのか」を説明する上流の意味付与であり、日本武尊・将門・入会地等を同一の史実へ統合しない。

**Causal QA: Pass**

## 3. L3 QA

- L1: 1 — 侵入者への迷失・祟り。
- L2: 0 — 知る／読むこと自体は怪異作用条件ではない。
- L3: 1 — 江戸期の記録は「里人が祟りを理由に立入を禁じていた」という社会的行動規範を伝承と結び付けて報告する。これは現代の鳥居・祠・土地管理すべての原因を伝承へ帰属するものではない。

**L3 QA: Pass（歴史的行動接続に限定）**

## 4. U / NA / C QA

- D20=U: 真相保持者を固定できない。`NO_ONE_KNOWS`は伝承内容としての明示がないため採らない。
- D07/D10/D13/D15はPrimary/Secondaryで主要差を保持できるためCを不要とし、DまたはIへ整理。
- NAを置く次元なし。

**Status QA: Pass**

## 5. Taxonomy / Secondary QA

- 新規taxonomy gapなし。
- D04 Secondary 2件は、由来の重層化と媒体横断の二つを除くと長期変容を落とすため保持。
- D07 Spatial-route anomaly、D13 curse、D15 disappearanceは、迷失型／祟り型の差を保存するため保持。
- D09 HIDDEN_CAUSE、D17 DO_NOT_ENGAGEは冗長として除外。

**Taxonomy / Secondary QA: Pass**

## 6. 最終判定

R1〜R4 Coder workflowは完了。00正本の変更は不要。10正本をR4裁定で置換し、外部Review_001待ちへ移行する。

**R4 Status: 完了。**