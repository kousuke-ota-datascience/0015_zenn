# 0133 Review_004 adjudication

- Entry_ID: `0133`
- Review cycle: `004`
- Review対象00 blob: `9f5f3299944442ba702fb51f52707f736eb1dfad`
- Review対象10 blob: `adb391d59cb453f92e2e8234aac93190a24e9256`
- 修正開始時現行00 blob: `14ec0963d915dd3794e1597c48bdf7b6665dbe94`
- 修正開始時現行10 blob: `1606fcdd7952dc3715bfbbf15ca4524b4aab5cf2`
- SHA gate: mismatch（Review_004対象より現行正本が後続Review_003修正版へ進んでいたため、旧版へ機械適用せず現行版へFindingを再照合）
- Corrected 00 commit: `dfdd64f2e521ca9026254c4577aac45429059220`
- Corrected 00 blob: `83475f835fabb0a319c08a982bbcf142150b038e`
- 10 canonical change: なし（現行commit `540bc7b4fbe45d57bb92034b564f98d8112e65db` を維持）
- 最終状態: `再レビュー待 / 004`

## Findings adjudication

### Review_0133_00_004
- F00-001: **ACCEPT / 現行版へ再照合**。現行新版では1.6、1.9、chapter 2のtutorial構造はすでに復旧済み。残存していた3.2/3.3の見出し差分のみ、`短い原文引用` / `証拠上の保留事項`へ修正した。
- F00-002: **ACCEPT / 現行版で解消済み**。現行新版は、具体的事故・病気・死亡を伴う後代工事祟り型を共有核から外し、前近代の「荒廃→天変地異→鎮魂→奉祀」を高権威資料へ接続済み。3.1でも個別工事事故・死亡逸話を「未固定」と明示しているため、旧Review対象版のtraceability問題を再生成しない。

### Review_0133_10_004
- F10-001: **ACCEPT / 現行版で解消済み**。現行10はVersion Scopeを前近代の荒廃・天変地異・鎮魂・奉祀へ狭め、D05/D07/D09/D11〜D15をこのScopeへ再接続済み。D13は`CURSE_MISFORTUNE / I`であり、旧対象版の`D`過大評価も解消済み。
- F10-002: **ACCEPT / 現行版で解消済み**。現行D19は`LOCAL_TRADITION / I`のみで、`NATIONAL_PUBLIC` Secondaryは撤回済み。

## QA

- 0133 Review_004は旧checkpoint対象であるため、旧版へ巻き戻していない。
- 現行00/10のVersion Scopeは相互整合し、未固定の近現代事故逸話を共有核へ含めない。
- 10はReview_004 Finding解消済みの現行版を維持し、不要な再commitを行わない。
