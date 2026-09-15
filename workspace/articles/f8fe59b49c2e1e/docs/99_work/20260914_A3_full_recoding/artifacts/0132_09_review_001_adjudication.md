# 0132 Review_001 Coder裁定

- Entry_ID: `0132`
- 対象Review: `Review_0132_00_001.md`, `Review_0132_10_001.md`
- 00判定: 要修正（Major）を受理。
- 10判定: 要修正（Major）を受理。

## 1. 00 Finding裁定

| Finding | 裁定 | 対応 |
|---|---|---|
| F00-001 summaryの意味形成分析 | ACCEPT | 「中心的不可解さ」「意味形成核」「変換」等の分析語を除き、場所・禁足・非帰還・祟り・由来説・光圀譚という資料上の内容へ限定 |
| F00-002 由来説を00で「競合」と裁定 | ACCEPT | 「異なる由来説明」「複数説」と記述し、単一史実へ統合しないことだけをEvidence上の留保として保持 |
| F00-003 比喩用法の因果的接続 | ACCEPT | 比喩用法は後代の独立した語彙的展開として記録し、禁足伝承の意味形成から派生したと因果推定しない |

## 2. 10 Finding裁定

| Finding | 裁定 | 対応 |
|---|---|---|
| F10-001 D04複合Status `D/I` | ACCEPT | 単一 `I` へ裁定 |
| F10-002 D03 `D` 過強 | ACCEPT | 前近代BOOKを公的書誌・復刻案内経由で固定しているためportfolio全体を `I` へ保守化 |
| F10-003 tutorial形式・独自QA節 | ACCEPT | 階層番号・中心質問を復元し、独自QA節を削除 |
| F10-004 D04 `ACCRETION / CROSS_MEDIA` | ACCEPT | 変容過程を直接示さないためSecondaryから撤回。複数の由来Versionが併存する範囲だけ `CONTESTED_VERSION / I` で保持 |

## 3. 維持した判定

- D01 `G0 / I`: 前近代存在確認と起源年代を分離。
- D17 `OBEY_TABOO / D`: trigger反転ではなく、江戸期記録が里人の立入禁止を直接報告するため維持。
- D18 `1/0/1 / I`: L3は祟り信念と現実の立入回避を結ぶ江戸期記録の範囲に限定。
- D20 `U`: 複数説の存在を `NO_ONE_KNOWS` へ変換しない。
- D21 `A2 / D`: 現存地点・石碑・祠により具体的実在対象へ固定。

## 4. 修正結果

- `0132_00_contents.md`
  - 修正commit: `abe0980793d70a106fba8c986059348e343b83d4`
  - 修正後blob: `aa06f63dbb2864bdd8c4f82f095b4be7d89c29c8`
- `0132_10_analysis.md`
  - 修正commit: `64b02dbde98e65baa5a1da541a59a873b7e4ebe8`
  - 修正後blob: `c22521926efbcb285b2196ade6707c091234eb27`

## 5. QA

- 00と10の責務分離: PASS
- Status単一値: PASS
- D03 directness: PASS（`I`）
- D04 History Evidence過拡張抑制: PASS
- D17独立Evidence: PASS
- D18 L3独立X Evidence限定: PASS
- taxonomy gap: 新規なし
- tutorial構造: PASS

## 6. 次状態

`再レビュー待 / 001`。Review_002で00/10双方の再レビューを待つ。