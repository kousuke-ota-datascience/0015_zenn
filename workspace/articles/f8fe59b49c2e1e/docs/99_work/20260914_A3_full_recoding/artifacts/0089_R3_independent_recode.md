# 0089 日本だるま／だるま女 — R3 Independent Recode

- Entry_ID: `0089`
- Version Scope正本: `docs/99_work/20260914_A3_full_recoding/0089_R2_version_scope.md`
- Evidence正本: `docs/10_each_lore/0089_daruma_onna/0089_00_contents.md`
- R1監査補助: `docs/99_work/20260914_A3_full_recoding/0089_R1_evidence_audit.md`
- baseline: control plane記載の固定blob
- 旧 `0089_10_analysis.md`: **未参照**
- 旧Excel coding値: **未参照**

# 1. 独立判定

## D01 生成年代
- Value: なし
- Status: `U`
- 根拠: 1994年の採録は固定できるが、最初期口承・成立年代を直接確認できない。後代資料の「1980年代」整理だけで年代帯を確定しない。

## D02 最古確認流通媒体
- Primary: `D02.PRT.BOOK`
- Parent: `D02.PRT`
- Status: `D`
- 根拠: 現在直接確認できる最古媒体は、1994年刊の現代伝説集への「日本だるま」収録。これ以前の口承媒体は固定しない。

## D03 確認流通媒体ポートフォリオ
- Primary: `D03.PRT.BOOK`
- Parent: `D03.PRT`
- Secondary: なし
- Status: `D`
- 根拠: Scopeの型が書籍へ収録され受容者へ提示されたことを確認できる。その他の媒体は具体的Evidence不足。

## D04 生成・変容パターン
- Primary: `D04.VAR.LOCALIZATION`
- Parent: `D04.VAR`
- Secondary: `D04.VAR.RETELLING`
- Status: `I`
- 根拠: 海外旅行先が香港、中国、東南アジア、フランス等へ差し替えられ、試着室・市場・店舗などの細部も変わる一方、失踪→後日の四肢切断状態での再発見という核は維持される。

## D05 提示形式
- Primary: `D05.HRS.FOAF`
- Parent: `D05.HRS`
- Secondary: `D05.RUL.WARNING`
- Status: `I`
- 根拠: 「友人の友人／知人の旅行者に起きた実話」として距離を置いて語る形式が核心的で、実践上は海外で単独行動しないという警告へ接続する。

## D06 真実性提示
- Value: `D06.T2` — 近接伝聞事実
- Status: `I`
- 根拠: 語り手自身の直接経験ではなく、身近な人脈上の具体的被害として提示されるFOAF型。

## D07 意味形成対象
- Primary: `D07.ICT.ABDUCTION_ASSAULT` — 誘拐・暴行
- Parent: `D07.ICT`
- Secondary: `D07.ICT.OUTGROUP_THREAT` — 外集団脅威; `D07.ICT.HIDDEN_CRIMINAL_PRACTICE` — 隠れた犯罪慣行
- Status: `I`
- 根拠: 旅行先での失踪・身体暴力が中心不安であり、見知らぬ異文化圏と表から見えない犯罪的搾取が補助的な意味形成対象となる。

## D08 意味形成契機
- Primary: `D08.STY.FOAF_REPORT` — 近接伝聞
- Parent: `D08.STY`
- Secondary: なし
- Status: `I`
- 根拠: 受容者にとっての問題化契機は、知人ネットワークを介した実話らしい被害報告である。

## D09 意味付与操作
- Primary: `D09.AGN.AGENCY_ATTRIBUTION` — 主体化
- Parent: `D09.AGN`
- Secondary: `D09.CAU.DIRECT_CAUSE` — 直接原因化; `D09.NOR.TABOOIZATION` — 禁忌化
- Status: `I`
- 根拠: 海外で人が消える不確実性を、意図をもつ誘拐者・犯罪者の行為へ帰属し、短時間の単独行動を危険原因とし、「離れない」という行動規範へ変換する。

## D10 因果源存在論
- Primary: `D10.HUM.INFORMAL_GROUP` — 非制度的集団・共同体
- Parent: `D10.HUM`
- Secondary: なし
- Status: `I`
- 根拠: 異伝では加害組織の具体性が低いが、複数の人間犯罪者による誘拐・監禁・搾取が想定される。正式組織であるEvidenceはないためORGANIZATIONへ上げない。

## D11 発動・接触条件
- Primary: `D11.PAS.SPONTANEOUS_SELECTION` — 偶然選ばれる・遭遇する
- Parent: `D11.PAS`
- Secondary: `D11.CON.LOCATION_STATE` — 特定場所・状態にいる
- Status: `I`
- 根拠: 海外旅行や試着室利用そのものが再現的に危害を発動する規則ではなく、被害者が犯罪者に選ばれる／遭遇することが直接の入口。同行者から離れた状態は脆弱性の文脈としてSecondary。

## D12 作用対象
- Primary: `D12.OTH.VICTIM_TARGET` — 被害者・標的
- Parent: `D12.OTH`
- Secondary: なし
- Status: `D`
- 根拠: 直接の誘拐・監禁・切断・搾取を受ける女性旅行者が作用対象。

## D13 作用機構
- Primary: `D13.PHY.BODY_TRANSFORMATION` — 身体変容
- Parent: `D13.PHY`
- Secondary: `D13.SOC.COERCE_CONFINE` — 強制・監禁; `D13.REL.TARGETING` — 標的化
- Status: `I`
- 根拠: 終端で物語を最も強く規定する直接作用は四肢切断による身体構造の改変。そこへ至る監禁と標的化をSecondaryに保持する。

## D14 帰結極性
- Value: `D14.NEG` — 負
- Status: `D`
- 根拠: 失踪・監禁・重大身体被害・搾取という一貫した負の帰結。

## D15 帰結領域
- Primary: `D15.BOD.SEVERE_INJURY` — 重傷・障害
- Parent: `D15.BOD`
- Secondary: `D15.LIF.DISAPPEARANCE` — 失踪・消失; `D15.LIF.FUTURE_CONSTRAINT` — 将来制約
- Status: `D`
- 根拠: 四肢切断による重大身体障害が最終的な可視化された主帰結。失踪期間と、その後の人生を拘束する長期的搾取も独立して重要。

## D16 因果時間構造
- Primary: `D16.DLY.DELAYED` — 遅延
- Parent: `D16.DLY`
- Secondary: `D16.PRG.STAGED_PROGRESSION` — 段階進行
- Status: `D`
- 根拠: 分離→失踪→捜索失敗→長い時間経過→再発見という遅延が意味上重要であり、その間に監禁・切断・搾取が段階的に進んだとされる。

## D17 回避・制御方式
- Primary: `D17.AVO.DO_NOT_ENGAGE` — 接触回避
- Parent: `D17.AVO`
- Secondary: `D17.AVO.DISTANCE_ROUTE_CHANGE` — 距離・経路変更
- Status: `I`
- 根拠: 物語が実践的に与える制御は、見知らぬ場所で一人にならない・不審な状況へ入らないという回避。完全な安全保証ではなく警告規則として読む。

## D18 作用レイヤー
- `D18.L1=1`
- `D18.L2=0`
- `D18.L3=0`
- Status: `D`
- 根拠: 伝承内部では人間犯罪者による因果作用がある。話を聞くこと自体が危害条件ではなく、噂の流通による現実社会の行動変化を示す独立X Evidenceも未確認。

## D19 流通範囲
- Primary / Parent / Secondary: なし
- Status: `U`
- 根拠: 日本の現代伝説集への採録は確認できるが、この個別話が当時どの社会範囲まで流通したかを直接測定できるEvidenceが不足する。

## D20 特権情報保持者
- Primary / Parent / Secondary: なし
- Status: `U`
- 根拠: 犯罪者側が内部事情を知る構造は推測できるが、伝承内で真相・回避情報の特権保持者として安定して提示されるわけではない。

## D21 現実アンカー
- Value: `D21.A1` — 一般的現実背景
- Status: `D`
- 根拠: 海外旅行・店舗・人間犯罪という一般的現実背景を用いるが、特定の国・事件・組織へ安定して依存しない。

# 2. causal precheck

```text
D11: 旅行先で犯罪者に偶然選ばれる／遭遇する
→ D12: 女性旅行者が被害標的となる
→ D13: 監禁を経て身体を切断・変容させられる
→ D15: 重大障害と長期的な人生制約へ至る
```

接続は成立する。

# 3. D18 L3 precheck

海外旅行不安や行動抑制を生みそうだという推測だけではL3を付けない。独立X Evidence未確認のため `L3=0`。

# 4. U / NA / C precheck

- D01=`U`: 最初期成立年代未確定。
- D19=`U`: 個別話の流通社会範囲未固定。
- D20=`U`: 特権情報保持者の安定構造なし。
- NA: なし。
- C: なし。

# 5. taxonomy gap precheck

現行Scopeは既存taxonomyで表現可能。新Child要求なし。

# 6. R3 freeze

本ファイルは旧 `0089_10_analysis.md` および旧Excel coding値を参照せずに作成した独立判定である。以後のR4で初めて旧10を参照する。