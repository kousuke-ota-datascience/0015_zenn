# 0. INTRODUCTION

本書は、A3で実施する**パイロット49件の個別伝承文書・コーディング整合化作業のコントロールプレーン**である。

**運用規則:** 1伝承エントリが完了するたびに、本書の集計・該当Entry・remarksを更新する。個別Entryは原則として当該Entryのファイルだけを1 commitで反映し、push後検証後に本control planeを別commitで更新する。

## 0.1. 対象母集団

```text
0001, 0003, 0005, 0006, 0011, 0019, 0024, 0025,
0059, 0060, 0081, 0089, 0091, 0101, 0112, 0113, 0118,
0132, 0133, 0137, 0152, 0157, 0158, 0169,
0178, 0179, 0180, 0181, 0188, 0198, 0225, 0250, 0275,
0309, 0319, 0349, 0356, 0362, 0363, 0365, 0366,
0384, 0385, 0394, 0403, 0410, 0411, 0412, 0413
```

Entry_ID・全件コード正本:
- `docs/20_analysis_summary/urban_legend_parent_child_full_application_v1.xlsx`

## 0.2. 基準文書

Tutorial更新commit:
- `0000_00_contents.md`: `e6bfa448f00af17f82c1e210507d3dbe5aa8c1b5`
- `0000_10_analysis.md`: `81c567a8012a58beceb239b22e430e603354a68d`

現行main blob:
- `0000_00_contents.md`: `0ed8c22777eb2c213ca0b009016ccc66a01054ce`
- `0000_10_analysis.md`: `dccc5e4473bc51c484f16ba916d653ea7383f2f5`

分析・コーディング規則:
- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`
- `docs/00_research_overview/90_ducumentation_metadata.md`

## 0.3. status

`未 / 作業中 / 完了 / −` の4値のみを使用する。

# 1. タスク整理

## 1.0. task 0. 基準版・母集団・正本の固定

49件、正本Excel、tutorial、理論設計、コード体系、コーディング規則を固定する。文書増補、Entryコード値変更、コード体系変更を分離する。

**status: 完了**

## 1.1. task 1. 高密度記載版フォーマットへの変更

対象18件:
```text
0001, 0003, 0005, 0006, 0011, 0019, 0024, 0025,
0112, 0113, 0118, 0132, 0133, 0137, 0152, 0157, 0158, 0169
```

`00_contents` を原典へ戻らず主要内容・展開・条件・帰結・異伝・不確実性を再構成できる密度にし、`10_analysis` を伝承固有の自然言語分析にする。コード値変更はtask 3で扱う。

## 1.2. task 2. tutorial記載項目への準拠徹底

**現在の最優先task。** 既存個別文書をtutorialの見出し、フィールド名、章番号、記載順へ揃える。

`10_analysis` の原則形:
```text
Primary Child / Value:
Primary Parent:
Secondary:
Status:

**判定根拠**

**この伝承における現れ方**
```

`Primary:`、`Parent:`、`現れ方:`、`Primary/Secondary:`、`Secondary Parent:` 等の独自表記は使用しない。独自末尾メモの留保は該当D次元へ移す。**task 2ではD01〜D21コード値・Statusを変更しない。** 未作成24件と基準例0180は `−`。

## 1.3. task 3. 「同じものさしでもう一度測り直す」

49件すべてについて、tutorialおよび10/20/30の三文書を固定したままD01〜D21をEvidenceから再判定する。既存Pilot Codingを自動継承せず、証拠不足は `U`、ParentはChildから導出、最古確認年と生成年代を分離、後代異伝を遡及しない。

**task 2全既存文書完了まで新規作業を保留。0024のみ着手済み未commit。**

## 1.4. task 4. 未作成24件の個別伝承文書作成

対象:
```text
0178, 0179, 0181, 0188, 0198, 0225, 0250, 0275,
0309, 0319, 0349, 0356, 0362, 0363, 0365, 0366,
0384, 0385, 0394, 0403, 0410, 0411, 0412, 0413
```

正本ExcelでEntry_ID・名称確認 → Evidence調査 → tutorial準拠の00/10作成 → task 3再測定 → Entry単位commit → push後検証 → control plane更新。

## 1.5. task 5. Excel正本同期

49件のtask 3完了後、`urban_legend_parent_child_full_application_v1.xlsx` へD01〜D21・Status等を同期する。

## 1.6. task 6. 派生評価資料の再整合

- `docs/20_analysis_summary/urban_legend_pilot2_evaluation.md`
- `docs/20_analysis_summary/urban_legend_full_application_evaluation_v1.md`

正本変更後に版整合を監査する。

## 1.7. task 7. 49件横断QA

Entry_ID、命名、00/10責務分離、高密度記載、tutorial準拠、D01〜D21、U/NA/C理由、Markdown/Excel一致、派生評価版、Entry単位commit・push後検証を確認する。

# 2. task対応状況

## 2.1. 集計

| task | 完了 | 作業中 | 未 | − | 備考 |
| --- | ---: | ---: | ---: | ---: | --- |
| task 1 高密度化 | 11 | 0 | 7 | 31 | 旧様式18件のみ対象 |
| task 2 tutorial準拠 | 10 | 0 | 14 | 25 | 既存文書24件対象 |
| task 3 再コーディング | 7 | 1 | 41 | 0 | 0024はtask 2優先のため保留 |
| task 4 未作成文書作成 | 0 | 0 | 24 | 25 | 未作成24件対象 |
| task 5 Excel正本同期 | 0 | 0 | 49 | 0 | task 3後 |

| global task | status | remarks |
| --- | --- | --- |
| task 0 基準版固定 | 完了 | tutorial commit/blob整合確認済み |
| task 6 派生評価再整合 | 未 | Excel同期後 |
| task 7 横断QA | 未 | 全task後 |

## 2.2. Entry別一覧

| Entry_ID | 伝承 | task 1 | task 2 | task 3 | task 4 | task 5 | remarks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0001 | 口裂け女 | 完了 | 完了 | 完了 | − | 未 | `4e18c1a977c8c223a26865fac0feeafef5d54b37` |
| 0003 | 赤い紙・青い紙／赤マント系 | 完了 | 完了 | 完了 | − | 未 | `5c577eb3d3acfa3f69fd4ca841875888b27305cf` |
| 0005 | 紫の鏡 | 完了 | 完了 | 完了 | − | 未 | `c2a37792f12b6c2aa5b8760536b13764d14c714d` |
| 0006 | メリーさんの電話 | 完了 | 完了 | 完了 | − | 未 | `89b2865013ecd7fca4b39fa914973a0f55eb3b3b` |
| 0011 | こっくりさん | 完了 | 完了 | 完了 | − | 未 | `1b24d197f9cb94241e26a368a841bc1071671d23` |
| 0019 | 小さいおじさん | 完了 | 完了 | 完了 | − | 未 | `e7ada0703530358d46b387b4feb149fc9e1b8ad9` |
| 0024 | 幸福の手紙 | 完了 | 完了 | 作業中 | − | 未 | task2 `c6de296c988f50b588db4dd5cd70dbe19aac3750`; task3保留 |
| 0025 | 不幸の手紙 | 完了 | 完了 | 未 | − | 未 | task2 `566dd4b7f0029c04012dbe605463fdd64ab8554c` |
| 0059 | 深泥池の幽霊タクシー | − | 完了 | 未 | − | 未 | task2 `b63ade25f6adb50aea07acd24f87e4bb53000a72`; 00監査済み・10正規化 |
| 0060 | タクシー幽霊 | − | 未 | 未 | − | 未 | task2対象 |
| 0081 | ピアスの白い糸 | − | 未 | 未 | − | 未 | task2対象 |
| 0089 | 日本だるま／だるま女 | − | 未 | 未 | − | 未 | task2対象 |
| 0091 | ベッドの下の男 | − | 未 | 未 | − | 未 | task2対象 |
| 0101 | 海外旅行で臓器を抜かれる | − | 未 | 未 | − | 未 | task2対象 |
| 0112 | 事故物件は一度別人を住ませれば告知義務が消える | 完了 | 未 | 未 | − | 未 | task2対象 |
| 0113 | 井の頭公園のボートに乗ると別れる | 完了 | 未 | 未 | − | 未 | task2対象 |
| 0118 | 函館山の切れない木 | 完了 | 完了 | 未 | − | 未 | task2 `aa689d1c860088d26282a29416a58a6c5dd97270` |
| 0132 | 八幡の藪知らず | 未 | 未 | 未 | − | 未 | task2先行→task1 |
| 0133 | 将門塚の祟り | 未 | 未 | 未 | − | 未 | task2先行→task1 |
| 0137 | 犬鳴村 | 未 | 未 | 未 | − | 未 | task2先行→task1 |
| 0152 | 青木ヶ原樹海で方位磁針が狂う | 未 | 未 | 未 | − | 未 | task2先行→task1 |
| 0157 | 新郷村キリストの墓 | 未 | 未 | 未 | − | 未 | task2先行→task1 |
| 0158 | 虚舟 | 未 | 未 | 未 | − | 未 | task2先行→task1 |
| 0169 | ノストラダムスの大予言 | 未 | 未 | 未 | − | 未 | task2先行→task1 |
| 0178 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0179 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0180 | きさらぎ駅 | − | − | 完了 | − | 未 | 基準例; `4cd8889048060c81c7bdf4ab3689d6f64fa08ea1` |
| 0181 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0188 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0198 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0225 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0250 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0275 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0309 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0319 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0349 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0356 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0362 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0363 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0365 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0366 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0384 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0385 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0394 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0403 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0410 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0411 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0412 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |
| 0413 | （未文書化・正本Excel参照） | − | − | 未 | 未 | 未 | task4 |

# 3. 実行順序

1. task 2の残り14件をEntry単位で順次完了する。
2. task 1残り7件を高密度化する。
3. 既存文書25件のtask 3を完了する。
4. task 4の未作成24件を作成し、同時にtask 3を完了する。
5. task 5 → task 6 → task 7。

## 3.1. task 2 Entryフロー

```text
最新main → tutorial → Entry 00/10取得 → コード値/Status固定
→ tutorial構造へ正規化 → 整合確認 → Entry単位commit
→ push後再取得検証 → control plane更新 → 次Entry
```

# 4. 完了条件

1. task 0完了。
2. task 1対象18件が高密度化済み。
3. 既存個別文書がtutorial準拠。
4. 49件すべてに個別文書が存在。
5. 49件すべてD01〜D21再測定済み。
6. MarkdownとExcel正本が一致。
7. 派生評価資料の版整合。
8. 49件横断QAで未解消構造不整合なし。
9. Entry単位のcommit・push後検証が追跡可能。
