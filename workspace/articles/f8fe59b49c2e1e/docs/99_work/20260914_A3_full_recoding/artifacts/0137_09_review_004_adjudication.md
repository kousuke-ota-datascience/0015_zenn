# 0137 Review_004 adjudication

## 0. Summary

- Entry_ID: `0137`
- Review cycle: `004`
- Review target 00 blob: `b2df0647a8d8480b5357b06a6020a171f4fc26ef`
- Review target 10 blob: `42c5d41e1ae8104f71db48d56f2971b537ccd86e`
- Review開始時の現行00 blob: `82bcd114d2bf681bf1554a487bb4a26ffe6d20f4`
- Review開始時の現行10 blob: `77170de70e82b35c2c63c3a85df438245bbad586`
- SHA gate: 00/10ともReview対象と不一致。Review_004は旧checkpointを対象としているため、旧版へ巻き戻さず現行新版へFindingを再照合した。
- Corrected 00 commit: `4caa358574acb66a1187722e2dccb18b83bf7f41`
- 10 current commit: `b31402dd45a0dddde6f2de1fb41eef4045990327`（追加修正なし）
- 最終状態: `再レビュー待 / 004`

## 1. Review_0137_00_004

### F00-001 — tutorial schema差分

**ACCEPT。** 現行新版では1.6、1.9、2.3、2.4、3.1は既に所定構造へ修正済みだった。残存していた3.2/3.3の見出しを `短い原文引用` / `証拠上の保留事項` へ正規化した。

### F00-002 — 1999年原Web未固定なのに一次資料扱い

**ACCEPT / 現行新版で解消済み。** 現行00では1999年層を0.2.2保存転載と0.2.3研究資料へ分離し、0.2.1には福岡県の現実地理資料のみを置いている。原1999年Webページを一次資料として扱わない。

## 2. Review_0137_10_004

### F10-001 — D12/D14のStatus directness

**ACCEPT / 現行新版で解消済み。** 現行10ではD12=`I`、D14=`I`で、1999年原Web未固定というEvidence条件に統一されている。

### F10-002 — D19 NATIONAL_PUBLIC過大評価

**ACCEPT / 現行新版で解消済み。** 現行10ではPrimary=`D19.NET.OPEN_FORUM_WEB`、Secondary=`D19.LOC.LOCAL_TRADITION`、Status=`I`とし、`NATIONAL_PUBLIC`を撤回済み。

## 3. QA

- 1999年早期層は研究・保存転載による間接Evidenceとして維持。
- 実在地域史と秘密村伝承を分離。
- D12/D14を含む1999年Content依存Statusは保守的に統一。
- D19は媒体公開性から全国的大衆流通を推定しない。
- 後代映画・怪異設定を1999年型へ遡及しない。
