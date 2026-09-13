# 0. INTRODUCTION

本書は、A3で実施する**パイロット49件の個別伝承文書・コーディング整合化作業のコントロールプレーン**である。1伝承エントリ完了ごとに更新する。

Entry_ID・全件コード正本:
- `docs/20_analysis_summary/urban_legend_parent_child_full_application_v1.xlsx`

Tutorial基準:
- `0000_00_contents.md` update commit `e6bfa448f00af17f82c1e210507d3dbe5aa8c1b5` / blob `0ed8c22777eb2c213ca0b009016ccc66a01054ce`
- `0000_10_analysis.md` update commit `81c567a8012a58beceb239b22e430e603354a68d` / blob `dccc5e4473bc51c484f16ba916d653ea7383f2f5`

Coding基準:
- `docs/00_research_overview/10_urban_legend_analysis_axes_theoretical_design.md`
- `docs/00_research_overview/20_urban_legend_parent_child_code_system.md`
- `docs/00_research_overview/30_urban_legend_analysis_coding_rules.md`
- `docs/00_research_overview/90_ducumentation_metadata.md`

statusは `未 / 作業中 / 完了 / −` のみ。

# 1. タスク整理

## 1.0. task 0. 基準版・母集団・正本の固定
49件・正本Excel・tutorial・10/20/30・文書管理規則を固定し、文書増補／Entry再コーディング／コード体系変更を分離する。**完了**。

## 1.1. task 1. 高密度記載版フォーマットへの変更
対象18件: `0001,0003,0005,0006,0011,0019,0024,0025,0112,0113,0118,0132,0133,0137,0152,0157,0158,0169`。00を再構成可能な密度へ、10を伝承固有分析へ増補する。コード値変更はtask 3。

## 1.2. task 2. tutorial記載項目への準拠徹底
**現在の最優先task。** 既存文書24件をtutorialの見出し・フィールド名・章番号・記載順へ揃える。10の標準は `Primary Child / Value` / `Primary Parent` / `Secondary` / `Status` / `判定根拠` / `この伝承における現れ方`。独自末尾メモの留保は該当Dへ移す。**コード値・Statusは変更しない。** 0180と未作成24件は対象外。

## 1.3. task 3. 「同じものさしでもう一度測り直す」
49件すべてを、tutorialと10/20/30を固定したままEvidenceからD01〜D21再判定する。既存Pilotを自動継承しない。証拠不足はU。task 2完了まで新規作業保留。0024のみ着手済み未commit。

## 1.4. task 4. 未作成24件の個別伝承文書作成
対象: `0178,0179,0181,0188,0198,0225,0250,0275,0309,0319,0349,0356,0362,0363,0365,0366,0384,0385,0394,0403,0410,0411,0412,0413`。正本確認→Evidence調査→tutorial準拠00/10→task3→Entry commit→検証→control plane更新。

## 1.5. task 5. Excel正本同期
49件task3後、正本ExcelへD01〜D21・Status等を同期。

## 1.6. task 6. 派生評価資料の再整合
`urban_legend_pilot2_evaluation.md` と `urban_legend_full_application_evaluation_v1.md` を正本変更後に監査。

## 1.7. task 7. 49件横断QA
Entry_ID、命名、00/10責務分離、高密度、tutorial、D01〜D21、U/NA/C理由、Markdown/Excel、評価版、commit/push後検証を横断確認。

# 2. task対応状況

## 2.1. 集計

| task | 完了 | 作業中 | 未 | − |
| --- | ---: | ---: | ---: | ---: |
| task 1 | 11 | 0 | 7 | 31 |
| task 2 | 14 | 0 | 10 | 25 |
| task 3 | 7 | 1 | 41 | 0 |
| task 4 | 0 | 0 | 24 | 25 |
| task 5 | 0 | 0 | 49 | 0 |

Global: task0=`完了`, task6=`未`, task7=`未`。

## 2.2. Entry別一覧

| Entry_ID | 伝承 | task1 | task2 | task3 | task4 | task5 | remarks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0001 | 口裂け女 | 完了 | 完了 | 完了 | − | 未 | `4e18c1a977c8c223a26865fac0feeafef5d54b37` |
| 0003 | 赤い紙・青い紙／赤マント系 | 完了 | 完了 | 完了 | − | 未 | `5c577eb3d3acfa3f69fd4ca841875888b27305cf` |
| 0005 | 紫の鏡 | 完了 | 完了 | 完了 | − | 未 | `c2a37792f12b6c2aa5b8760536b13764d14c714d` |
| 0006 | メリーさんの電話 | 完了 | 完了 | 完了 | − | 未 | `89b2865013ecd7fca4b39fa914973a0f55eb3b3b` |
| 0011 | こっくりさん | 完了 | 完了 | 完了 | − | 未 | `1b24d197f9cb94241e26a368a841bc1071671d23` |
| 0019 | 小さいおじさん | 完了 | 完了 | 完了 | − | 未 | `e7ada0703530358d46b387b4feb149fc9e1b8ad9` |
| 0024 | 幸福の手紙 | 完了 | 完了 | 作業中 | − | 未 | task2 `c6de296c988f50b588db4dd5cd70dbe19aac3750`; task3保留 |
| 0025 | 不幸の手紙 | 完了 | 完了 | 未 | − | 未 | task2 `566dd4b7f0029c04012dbe605463fdd64ab8554c` |
| 0059 | 深泥池の幽霊タクシー | − | 完了 | 未 | − | 未 | task2 `b63ade25f6adb50aea07acd24f87e4bb53000a72`; 00監査済み |
| 0060 | タクシー幽霊 | − | 完了 | 未 | − | 未 | task2 `fee2ae0b4618844e2bdded6e7cef99a59b74a4ea`; 00監査済み |
| 0081 | ピアスの白い糸 | − | 完了 | 未 | − | 未 | task2 `3e3697c22c590e77bd48280ee24e9c88bd65147b`; 00監査済み |
| 0089 | 日本だるま／だるま女 | − | 完了 | 未 | − | 未 | task2 `f992e68b5fc8404abdf8404af8651189739ed72c`; 00監査済み |
| 0091 | ベッドの下の男 | − | 完了 | 未 | − | 未 | task2 `0d795f4ef3d282cb0c82ce9b9cdac8f73634308d`; 00監査済み |
| 0101 | 海外旅行で臓器を抜かれる | − | 未 | 未 | − | 未 | task2対象 |
| 0112 | 事故物件は一度別人を住ませれば告知義務が消える | 完了 | 未 | 未 | − | 未 | task2対象 |
| 0113 | 井の頭公園のボートに乗ると別れる | 完了 | 未 | 未 | − | 未 | task2対象 |
| 0118 | 函館山の切れない木 | 完了 | 完了 | 未 | − | 未 | task2 `aa689d1c860088d26282a29416a58a6c5dd97270` |
| 0132 | 八幡の藪知らず | 未 | 未 | 未 | − | 未 | task2→task1 |
| 0133 | 将門塚の祟り | 未 | 未 | 未 | − | 未 | task2→task1 |
| 0137 | 犬鳴村 | 未 | 未 | 未 | − | 未 | task2→task1 |
| 0152 | 青木ヶ原樹海で方位磁針が狂う | 未 | 未 | 未 | − | 未 | task2→task1 |
| 0157 | 新郷村キリストの墓 | 未 | 未 | 未 | − | 未 | task2→task1 |
| 0158 | 虚舟 | 未 | 未 | 未 | − | 未 | task2→task1 |
| 0169 | ノストラダムスの大予言 | 未 | 未 | 未 | − | 未 | task2→task1 |
| 0178 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0179 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0180 | きさらぎ駅 | − | − | 完了 | − | 未 | 基準例 `4cd8889048060c81c7bdf4ab3689d6f64fa08ea1` |
| 0181 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0188 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0198 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0225 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0250 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0275 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0309 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0319 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0349 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0356 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0362 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0363 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0365 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0366 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0384 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0385 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0394 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0403 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0410 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0411 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0412 | （未文書化） | − | − | 未 | 未 | 未 | task4 |
| 0413 | （未文書化） | − | − | 未 | 未 | 未 | task4 |

# 3. 実行順序

1. task2残り10件を順次完了。
2. task1残り7件。
3. 既存文書25件task3。
4. task4未作成24件＋task3。
5. task5 → task6 → task7。

Task2 Entryフロー:
`最新main → tutorial → 00/10 → 値固定 → 正規化 → Entry commit → push後検証 → control plane更新 → 次Entry`

# 4. 完了条件

task0完了、task1対象18件高密度化、既存文書tutorial準拠、49件全個別文書、49件D01〜D21再測定、Markdown/Excel一致、評価資料版整合、横断QA完了、Entry単位commit/push後検証追跡可能。