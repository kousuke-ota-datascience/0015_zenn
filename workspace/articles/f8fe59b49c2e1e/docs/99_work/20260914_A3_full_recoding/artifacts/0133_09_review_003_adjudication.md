# 0133 Review_003 Coder Adjudication

- Entry_ID: `0133`
- 伝承: 将門塚の祟り
- Review_Seq: `003`
- 対象Review: `Review_0133_00_003.md`, `Review_0133_10_003.md`
- adjudication日: `2026-09-16`
- Coder結論: Review_003の全Findingを採用し、00/10正本を修正。再レビュー待へ移行する。

## 1. SHA gate

Review_003対象blobと修正前正本blobの一致を確認した。

- 00 target/current before fix: `9f5f3299944442ba702fb51f52707f736eb1dfad`
- 10 target/current before fix: `adb391d59cb453f92e2e8234aac93190a24e9256`

SHA gate: PASS。

## 2. Review_0133_00_003

### F00-001 — 1.6 / 1.9 がtutorial固定形式に一致しない

- 判定: **ACCEPT**
- 理由: Review指摘どおり、修正前00は1.6を文章でまとめ、1.9もtutorial所定の3区分ではなかった。
- 修正: 1.6を `条件 / 規則 / 禁忌 / 対処・回避 / 利用法` へ分解し、1.9を `明示された帰結 / 暗示された帰結 / 確認されない帰結` へ整形した。2.3/2.4もtutorial見出し・列へ統一した。
- correction commit: `cfb529fb7cd96a9754176fdfc13900cd013fcf32`
- corrected blob: `14ec0963d915dd3794e1597c48bdf7b6665dbe94`

### F00-002 — 近現代祟り型のContent traceabilityが粗い

- 判定: **ACCEPT**
- 理由: 個別の工事事故・病気・死亡を高権威資料で十分固定できない状態で共有Contentへ一般化していたため、00→10の再現性が不足していた。
- 修正: 弱い後代事故層を無理に補強せず、共有核を高権威資料で追跡可能な `将門塚の荒廃 → 天変地異 → 将門の御神威への恐れ → 真教上人の鎮魂 → 神田明神への奉祀` へ限定した。個別の工事事故・死亡逸話は未固定派生として明示し、3.1にも未固定として記録した。
- correction commit: `cfb529fb7cd96a9754176fdfc13900cd013fcf32`
- corrected blob: `14ec0963d915dd3794e1597c48bdf7b6665dbe94`

## 3. Review_0133_10_003

### F10-001 — D19 Secondary `NATIONAL_PUBLIC` の流通範囲Evidence不足

- 判定: **ACCEPT**
- 理由: 公的Webの地域外閲覧可能性や来訪者の存在は、全国的大衆への実流通を直接示さない。
- 修正: D19を Primary `D19.LOC.LOCAL_TRADITION / I`、Secondaryなしへ変更し、`NATIONAL_PUBLIC` を撤回した。
- correction commit: `540bc7b4fbe45d57bb92034b564f98d8112e65db`
- corrected blob: `1606fcdd7952dc3715bfbbf15ca4524b4aab5cf2`

### F10-002 — 近現代祟り作用の00→10 traceabilityが不足

- 判定: **ACCEPT**
- 理由: 修正前D11–D15の一部は、00で具体典拠を固定できない近現代事故層へ依存していた。
- 修正: 00の共有核を前近代の鎮魂伝承へ限定した上で10を再接続した。D11は「祭祀対象の荒廃・維持放棄」をtaxonomy gapとして保持、D12は不特定人群、D13は呪詛・不運付与/I、D14はNEG/I、D15は広い不運・災厄/I、D16は荒廃→天変地異→恐れ→鎮魂→奉祀の段階進行/Iへ再裁定した。
- correction commit: `540bc7b4fbe45d57bb92034b564f98d8112e65db`
- corrected blob: `1606fcdd7952dc3715bfbbf15ca4524b4aab5cf2`

## 4. QA

- tutorial 00 fixed fields: PASS
- unsupported modern accident layer excluded from shared core: PASS
- 00→10 traceability: PASS
- D11 taxonomy gap explicit: PASS
- D18 zero-bit handling: PASS (`1/0/1`, Status I)
- D19 actual circulation scope: PASS
- D20 privileged-holder overreach: PASS

## 5. Handoff

Coder correction complete. Reviewer approval前のため `完了` にはせず、`再レビュー待 / 003` とする。