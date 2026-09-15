# 0006 Review_001 Coder Adjudication

## 1. 対象

- Entry: `0006` メリーさんの電話
- Reviewer成果物:
  - `docs/99_work/review_10_each_lore/0006/Review_0006_00_001.md`
  - `docs/99_work/review_10_each_lore/0006/Review_0006_10_001.md`
- 修正前00 blob: `b454675ea6425a9c17ee8e7b6ad11c35fb449e95`
- 修正前10 blob: `919bab39a5ed0024a567fc3d7c73d78e843ef1a1`
- 修正後00 blob: `069a34063c3ff822aff09a505ac85ea451e92cc9`
- 修正後10 blob: `e8c57c3cb6c65b93e36f70847a20696b44d430d9`

## 2. 00 Review裁定

| Finding | 裁定 | 対応 |
|---|---|---|
| F00-001 summaryのsense-making分析混入 | 受理 | 「恐怖の中心」「測距装置」「安全圏縮小」「意味形成核」等をEvidence正本から削除し、観察可能な内容だけへ限定 |
| F00-002 1999年直接Evidenceと後代代表形の分離 | 受理 | summary・詳細展開・主体・物・規則・Evidence表で `1999年直接確認核` と `後代代表形` を明示分離 |
| F00-003 人形を捨てる／手放す→電話開始のtraceability | 受理 | 『日本現代怪異事典』に基づく後代代表形の発動前件として独立記載し、Evidence対応表にも専用行を追加 |
| F00-004 保守的留保 | 維持 | 1999年を起源年にしない、後代完成形を遡及しない、リカちゃん起源を断定しない、身体危害を必須化しない |

## 3. 10 Review裁定

| Finding | 裁定 | 対応 |
|---|---|---|
| F10-001 D11 `SPONTANEOUS_SELECTION` | 受理 | 後代代表形では人形の廃棄・放棄・手放しが発動前件。既存D11 Childに適合しないためPrimaryを空欄化し、taxonomy gap `人形の廃棄・放棄・手放しによる発動` として保持 |
| F10-002 Evidence層明示 | 受理 | Version Scope・D07〜D16の根拠で、1999年直接核と後代代表形を明示的に区別 |
| F10-003 D07〜D10 | 維持＋根拠強化 | D07で「捨てた人形がなぜ追跡主体として戻るのか」を中心質問への回答として明示 |
| F10-004 D20 `NO_HIDDEN_TRUTH` | 受理 | 真相・理由・停止法の保持者が確認できないだけなので `U` へ変更 |
| F10-005 D15 `FEAR_TRAUMA` | 受理 | 恐怖を分析者推定で置かず、至近距離到達後を明示しない型に基づき `D15.KNW.UNCERTAINTY_PRESERVED` をPrimaryへ変更 |
| F10-006 書式 | 受理 | `Entry_ID=0006`、0180基準の番号付き見出し・中心質問・固定フィールド・判定根拠・現れ方へ正規化。R4作業節をCoding正本から除去 |

## 4. D11 taxonomy gap

Version Scopeへ含めた後代代表形では、

```text
人形を捨てる／手放す
→ 人形を名乗る電話が始まる
→ 現在地通知を反復しながら接近する
```

が安定した因果開始として整理される。

現行D11には廃棄・放棄を直接表すChildがない。

- `SPONTANEOUS_SELECTION`: 本人の行為なしに対象化されるため不適合
- `TAKE_OWN_CARRY`: 取得・所持で方向が逆
- `CREATE_ALTER_MANIPULATE`: 作成・加工・操作で廃棄・放棄を直接表さない

workflow 9.6に従い、近似Childへ押し込まずコード空欄のtaxonomy gapとしてR5へ送る。

## 5. D15再判定

身体襲撃・死亡はScope外。後代代表形には、至近距離・背後へ到達したところで語りが終了し、その後が明示されない型がある。

資料が受信者の恐怖・トラウマを最終帰結として直接確定していないため、旧 `FEAR_TRAUMA + UNCERTAINTY_PRESERVED` から `UNCERTAINTY_PRESERVED` 単独へ変更した。

## 6. D18再確認

- L1=1: 伝承内部で追跡・接近因果あり
- L2=0: 伝承受容だけで現実側受容者が追跡対象になる自己適用規則なし
- L3=0: X Evidenceなし

L2/L3の0は不在判定を含むため、vector Statusを `D → I` とした。

## 7. 次状態

修正版00/10を `Review_002` の対象とし、Entry Statusを `再レビュー待` とする。00/10双方が承認されるまで `完了` にしない。
