# 0. INTRODUCTION

## 0.1. 文書の目的

本書は、A3で実施する**パイロット49件の個別伝承文書・コーディング整合化作業のコントロールプレーン**である。

以下を一元管理する。

- パイロット49件の母集団
- task 0〜7の定義と実行順序
- Entry単位の対応状況
- Entry単位のcommit
- 未解消事項
- 最終完了条件

**運用規則:** 1伝承エントリの作業が完了するたびに、本書の該当Entry、集計、必要なremarksを更新する。

個別伝承のcommit規則と衝突させないため、原則として以下の2段階で反映する。

1. 当該Entryの `<Entry_ID>_00_contents.md` と `<Entry_ID>_10_analysis.md` のみを1 commitで反映する。
2. push後検証後、本control planeだけを別commitで更新する。

## 0.2. 対象母集団

パイロット対象は以下の49 Entryである。

```text
0001, 0003, 0005, 0006, 0011, 0019, 0024, 0025,
0059, 0060, 0081, 0089, 0091, 0101, 0112, 0113, 0118,
0132, 0133, 0137, 0152, 0157, 0158, 0169,
0178, 0179, 0180, 0181, 0188, 0198, 0225, 0250, 0275,
0309, 0319, 0349, 0356, 0362, 0363, 0365, 0366,
0384, 0385, 0394, 0403, 0410, 0411, 0412, 0413
```

Entry_ID・全件コードの正本:

- `docs/20_analysis_summary/urban_legend_parent_child_full_application_v1.xlsx`

## 0.3. 基準文書

### 0.3.1. tutorial

ユーザー指定SHAは**tutorial更新commit SHA**である。

```text
0000_00_contents.md 更新commit:
e6bfa448f00af17f82c1e210507d3dbe5aa8c1b5

0000_10_analysis.md 更新commit:
81c567a8012a58beceb239b22e430e603354a68d
```

これらのcommitで生成されたblobは、2026-09-13時点の `main` 上のtutorialと一致する。

```text
0000_00_contents.md blob:
0ed8c22777eb2c213ca0b009016ccc66a01054ce

0000_10_analysis.md blob:
dccc5e4473bc51c484f16ba916d653ea7383f2f5
```

したがってtutorial基準版の不一致はない。

### 0.3.2. 分析・コーディング規則

- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`
- `docs/00_research_overview/90_ducumentation_metadata.md`

## 0.4. status定義

本書のtask statusは以下の4値のみを使用する。

| status | 意味 |
| --- | --- |
| `未` | 対象だが未着手、または完了条件を満たしていない |
| `作業中` | 着手済みだが、commit・push後検証等を含む完了条件を満たしていない |
| `完了` | 当該taskの完了条件を満たし、GitHub反映後の確認まで終わっている |
| `−` | 当該Entryには構造上そのtaskを適用しない |

「文書が存在する」「一度編集した」だけでは `完了` としない。

## 0.5. control plane更新履歴

- 初版作成commit: `2b0e53352240524fdf00ef883ebc415b295a4298`
- 2026-09-13: task 0のtutorial SHA解釈を訂正。指定SHAはcommit SHAであり、現行mainのtutorial blobと整合することを確認。
- 2026-09-13: `0024 幸福の手紙` のtask 2を完了。commit `c6de296c988f50b588db4dd5cd70dbe19aac3750`。

# 1. タスク整理

## 1.0. task 0. 基準版・母集団・正本の固定

**対象:** global task。

以下を固定する。

1. パイロット母集団は49件。
2. Entry_IDは正本Excelを基準とする。
3. 書き込み直前に最新 `main` を取得する。
4. tutorialは0.3.1記載のcommit/blobを基準とする。
5. 理論設計・コード体系・コーディング規則を固定する。
6. 文書増補、Entryコード値変更、コード体系変更を別作業として扱う。

**現状:** `完了`。

## 1.1. task 1. 高密度記載版フォーマットへの変更

**対象:** A2a前半の薄い様式で作成された18件。

```text
0001, 0003, 0005, 0006, 0011, 0019, 0024, 0025,
0112, 0113, 0118, 0132, 0133, 0137, 0152, 0157, 0158, 0169
```

目的:

- `00_contents` を、原典へ戻らず主要内容・展開・条件・帰結・異伝・不確実性を再構成できる密度にする。
- 内容と典拠の対応を追跡可能にする。
- 一次資料、転載・復刻、二次資料、後代解説を区別する。
- `10_analysis` をコード表の転記ではなく、伝承固有の自然言語分析にする。

このtaskではコード値変更を目的としない。値の再判定はtask 3で扱う。

## 1.2. task 2. tutorial記載項目への準拠徹底

**対象:** 既に個別Markdownが存在するEntry。ただし `0180` は基準例として `−` とする。

**A3の直近最優先task。**

`00_contents` と `10_analysis` をtutorialの見出し、フィールド名、章番号、記載順へ揃える。

`10_analysis` では、同義でも独自表記を使用しない。

禁止例:

```text
Primary:
Parent:
現れ方:
Primary/Secondary:
Secondary Parent:
```

原則形:

```text
Primary Child / Value:
Primary Parent:
Secondary:
Status:

**判定根拠**

**この伝承における現れ方**
```

独自の末尾「パイロットメモ・再検討」章へ留保を集約せず、該当D次元内へ配置する。

**重要:** task 2では既存のD01〜D21コード値・Statusを変更しない。再判定はtask 3で行う。

未作成24件はtask 4で最初からtutorial準拠で作成するため、本taskでは `−` とする。

## 1.3. task 3. 「同じものさしでもう一度測り直す」の実行

**対象:** パイロット49件すべて。

高密度化・典拠確認後のEvidenceを使い、分析軸・コード体系・コーディング規則を変更せず、D01〜D21をEntryごとに再判定する。

適用規則:

- `0000_10_analysis.md`
- `10_urban_legend_analysis_axes_theoretical_design.md`
- `20_urban_legend_parent_child_code_system.md`
- `30_urban_legend_analysis_coding_rules.md`

原則:

- 既存Pilot Codingを自動継承しない。
- 証拠不足なら `U` を許容する。
- `D / I / U / NA / C` の根拠を明示する。
- ParentはChildから導出する。
- 最古確認年と生成年代を混同しない。
- 投稿・記録の途絶を死亡・失踪と同一視しない。
- 後代異伝を初期Versionへ遡及しない。

**現状:** task 2を全既存文書で完了するまで、新規のtask 3作業は保留する。`0024` は着手済みだが未commitのため `作業中` を維持する。

## 1.4. task 4. 未作成24件の個別伝承文書作成

**対象:** 以下24件。

```text
0178, 0179, 0181, 0188, 0198, 0225, 0250, 0275,
0309, 0319, 0349, 0356, 0362, 0363, 0365, 0366,
0384, 0385, 0394, 0403, 0410, 0411, 0412, 0413
```

各Entryについて以下を実施する。

1. 正本ExcelからEntry_ID・伝承名を確認。
2. tutorialを再取得。
3. Evidenceを調査。
4. `<Entry_ID>_00_contents.md` を高密度で作成。
5. task 3の規則で `<Entry_ID>_10_analysis.md` を作成。
6. 当該2ファイルのみを1 commit。
7. push後再取得・検証。
8. 本control planeを別commitで更新。

## 1.5. task 5. 再コーディング結果のExcel正本同期

**対象:** 49件すべて。

個別Markdownでtask 3完了後、以下の正本へD01〜D21・Status等を同期する。

- `docs/20_analysis_summary/urban_legend_parent_child_full_application_v1.xlsx`

Parent列はChild→Parentマスターから生成し、独立手入力しない。

## 1.6. task 6. 派生評価資料の再整合

**対象:** global task。

少なくとも以下を監査する。

- `docs/20_analysis_summary/urban_legend_pilot2_evaluation.md`
- `docs/20_analysis_summary/urban_legend_full_application_evaluation_v1.md`

旧評価を歴史的記録として残す必要がある場合は、新versionを作成し、黙って過去版を書き換えない。

## 1.7. task 7. 49件横断QA・reconcile完了判定

**対象:** global task。

最終確認:

- Entry_ID・伝承名・ファイル名・ディレクトリ名
- `00_contents` / `10_analysis` の責務分離
- 高密度記載
- tutorial厳密準拠
- D01〜D21の存在と再測定
- `U / NA / C` の理由記載
- MarkdownとExcel正本の一致
- 派生評価資料の版整合
- Entry単位commitとpush後検証

# 2. task対応状況

## 2.1. 集計

| task | 完了 | 作業中 | 未 | − | 備考 |
| --- | ---: | ---: | ---: | ---: | --- |
| task 1 高密度化 | 11 | 0 | 7 | 31 | 旧様式18件のみ対象 |
| task 2 tutorial準拠 | 8 | 0 | 16 | 25 | 既存文書24件が対象。0180と未作成24件は `−` |
| task 3 再コーディング | 7 | 1 | 41 | 0 | 0024は着手済みだがtask 2優先のため保留 |
| task 4 未作成文書作成 | 0 | 0 | 24 | 25 | 未作成24件のみ対象 |
| task 5 Excel正本同期 | 0 | 0 | 49 | 0 | task 3完了後に同期 |

Global task:

| task | status | remarks |
| --- | --- | --- |
| task 0 基準版固定 | 完了 | tutorial指定SHAはcommit SHA。現行main blobと整合確認済み |
| task 6 派生評価再整合 | 未 | Excel同期後に実施 |
| task 7 横断QA | 未 | 全task後に実施 |

## 2.2. Entry別一覧

| Entry_ID | 伝承 | task 1 | task 2 | task 3 | task 4 | task 5 | remarks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0001 | 口裂け女 | 完了 | 完了 | 完了 | − | 未 | 再測定commit `4e18c1a977c8c223a26865fac0feeafef5d54b37` |
| 0003 | 赤い紙・青い紙／赤マント系 | 完了 | 完了 | 完了 | − | 未 | 再測定commit `5c577eb3d3acfa3f69fd4ca841875888b27305cf` |
| 0005 | 紫の鏡 | 完了 | 完了 | 完了 | − | 未 | 再測定commit `c2a37792f12b6c2aa5b8760536b13764d14c714d` |
| 0006 | メリーさんの電話 | 完了 | 完了 | 完了 | − | 未 | 再測定commit `89b2865013ecd7fca4b39fa914973a0f55eb3b3b` |
| 0011 | こっくりさん | 完了 | 完了 | 完了 | − | 未 | 再測定commit `1b24d197f9cb94241e26a368a841bc1071671d23` |
| 0019 | 小さいおじさん | 完了 | 完了 | 完了 | − | 未 | 再測定commit `e7ada0703530358d46b387b4feb149fc9e1b8ad9` |
| 0024 | 幸福の手紙 | 完了 | 完了 | 作業中 | − | 未 | task 2 commit `c6de296c988f50b588db4dd5cd70dbe19aac3750`。task 3はtask 2全体完了まで保留 |
| 0025 | 不幸の手紙 | 完了 | 未 | 未 | − | 未 | 高密度化済み |
| 0059 | 深泥池の幽霊タクシー | − | 未 | 未 | − | 未 | A2aで高密度化済み。task 2/3監査が必要 |
| 0060 | タクシー幽霊 | − | 未 | 未 | − | 未 | A2aで高密度化済み。task 2/3監査が必要 |
| 0081 | ピアスの白い糸 | − | 未 | 未 | − | 未 | A2aで高密度化済み。task 2/3監査が必要 |
| 0089 | 日本だるま／だるま女 | − | 未 | 未 | − | 未 | A2aで高密度化済み。task 2/3監査が必要 |
| 0091 | ベッドの下の男 | − | 未 | 未 | − | 未 | A2aで高密度化済み。task 2/3監査が必要 |
| 0101 | 海外旅行で臓器を抜かれる | − | 未 | 未 | − | 未 | A2aで高密度化済み。task 2/3監査が必要 |
| 0112 | 事故物件は一度別人を住ませれば告知義務が消える | 完了 | 未 | 未 | − | 未 | 高密度化済み。task 2対象 |
| 0113 | 井の頭公園のボートに乗ると別れる | 完了 | 未 | 未 | − | 未 | 高密度化済み。task 2対象 |
| 0118 | 函館山の切れない木 | 完了 | 完了 | 未 | − | 未 | task 2 commit `aa689d1c860088d26282a29416a58a6c5dd97270` |
| 0132 | 八幡の藪知らず | 未 | 未 | 未 | − | 未 | 旧薄様式。task 2を先行し、その後task 1 |
| 0133 | 将門塚の祟り | 未 | 未 | 未 | − | 未 | 旧薄様式。task 2を先行し、その後task 1 |
| 0137 | 犬鳴村 | 未 | 未 | 未 | − | 未 | 旧薄様式。task 2を先行し、その後task 1 |
| 0152 | 青木ヶ原樹海で方位磁針が狂う | 未 | 未 | 未 | − | 未 | 旧薄様式。task 2を先行し、その後task 1 |
| 0157 | 新郷村キリストの墓 | 未 | 未 | 未 | − | 未 | 旧薄様式。task 2を先行し、その後task 1 |
| 0158 | 虚舟 | 未 | 未 | 未 | − | 未 | 旧薄様式。task 2を先行し、その後task 1 |
| 0169 | ノストラダムスの大予言 | 未 | 未 | 未 | − | 未 | 旧薄様式。task 2を先行し、その後task 1 |
| 0178 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0179 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0180 | きさらぎ駅 | − | − | 完了 | − | 未 | 基準例。再測定commit `4cd8889048060c81c7bdf4ab3689d6f64fa08ea1` |
| 0181 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0188 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0198 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0225 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0250 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0275 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0309 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0319 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0349 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0356 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0362 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0363 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0365 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0366 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0384 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0385 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0394 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0403 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0410 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0411 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0412 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |
| 0413 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task 4対象 |

## 2.3. 未文書化24件の伝承名

未文書化24件は正本Excelを基準とする。task 4開始時に正本Excelから伝承名を取得し、本表へ反映する。名称を推測で補わない。

# 3. 実行順序

## 3.1. 直近の実行順序

ユーザー指示により、**task 2を先に全既存文書で完了させる。**

1. task 2の残り16件をEntry単位で順次完了する。
2. task 1の残り7件を高密度化する。task 2で確定した構造を維持する。
3. 既存文書25件についてtask 3を完了する。すでに完了済みの7件と0180は再作業不要。0024の途中作業を再開する。
4. task 4の未作成24件を1 Entryずつ作成し、作成時にtask 3も同時に完了する。
5. task 5で49件をExcel正本へ同期する。
6. task 6で派生評価資料を再整合する。
7. task 7で49件横断QAを行う。

## 3.2. task 2実施時のEntry単位フロー

```text
最新main取得
→ tutorial再取得
→ 当該Entryの00/10取得
→ コード値・Statusを固定
→ tutorialの章・項目・記載順へ正規化
→ 00/10整合確認
→ 当該Entryの2ファイルだけを1 commit
→ mainへpush
→ GitHubからcommit/ファイルを再取得して検証
→ 本control planeのstatus/remarksを別commitで更新
→ 次Entryへ
```

## 3.3. task 3以降のEntry単位フロー

```text
最新main取得
→ tutorial/理論/コード体系/コーディング規則を再取得
→ Evidence確認・必要な増補
→ D01〜D21再測定
→ 2ファイル整合確認
→ 当該Entryの2ファイルだけを1 commit
→ mainへpush
→ GitHubから再取得・検証
→ 本control plane更新
```

# 4. 完了条件

A3 pilot lore analysis reconcileは、以下をすべて満たした場合のみ完了とする。

1. task 0が完了している。
2. task 1対象18件が高密度記載基準を満たす。
3. 既存個別文書がtutorialの見出し・フィールド名・記載順へ準拠する。
4. 未作成24件を含む49件すべてに個別文書が存在する。
5. 49件すべてについて、同一の理論設計・コード体系・コーディング規則でD01〜D21を再測定済みである。
6. 個別Markdownと正本Excelのコード値・Statusが一致する。
7. 派生評価資料の版整合が取れている。
8. 49件横断QAで未解消の構造不整合がない。
9. 各Entryの作業履歴がcommit単位で追跡可能で、push後検証済みである。

この条件を満たすまでは「49件reconcile完了」と扱わない。