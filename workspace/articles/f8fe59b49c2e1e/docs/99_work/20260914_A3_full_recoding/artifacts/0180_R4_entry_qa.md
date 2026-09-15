# 0180 R4 Entry QA

## 比較対象
- R3 freeze: `24a7d6ec5a50972f27a2a39fff2a57045023aaf4`
- 旧Coding正本 blob: `ed2de0477fbfe02837a37a8f4402577dc43a2524`
- Version Scope: 2004年1月の初期実況のみ

## 差分裁定

### 維持
D01/D02/D03/D05/D06/D07/D12/D13/D14/D16/D21はR3と旧10が実質一致し、そのまま維持する。

### D04
- R3: `LIVE_COCREATION` + `ACCRETION`
- 旧10: `LIVE_COCREATION` のみ
- 裁定: **旧10を採用**。
- 理由: D04自体は変容史を扱うが、今回のVersion Scopeは2004年初期実況に固定され、後代の帰還ルール・異界駅一般化はScope外。初期版の成立形式として直接確認できる実況共同生成のみを保存する。

### D08
- R3: `DIRECT_EVENT` + `MISSING_ALTERED_RECORD`
- 旧10: `DIRECT_EVENT` のみ
- 裁定: **旧10を採用**。
- 理由: 問題化の最初の契機は長時間無停車という直接遭遇で十分特定できる。検索不能は後続の強化手掛かりだが、Primary/Secondaryの情報量に対して冗長。

### D09
- R3: `DELIBERATE_NONRESOLUTION` + `TYPE_ASSIGNMENT`
- 旧10: `MYSTERY_LABEL`
- 裁定: **R3を採用**。
- 理由: 初期実況では異界・あの世・通常運行異常など複数説が並行し、結論が閉じないことが構造上明示的。単に「謎」とラベル化するより、`DELIBERATE_NONRESOLUTION` が具体的である。参加者による異界等への暫定類型化をSecondaryとする。

### D10
- R3: `ANOMALOUS_EXPERIENCE` + `ROUTE_CONNECTION`
- 旧10: `ROUTE_CONNECTION`
- 裁定: **旧10を採用**。
- 理由: D10は因果源存在論であり、体験そのものより、通常路線から未知駅へ接続した経路・接続構造を原因側に置く方がD07/D13との役割分担が明確。

### D11
- R3: `ACCIDENT_INVOLVEMENT` + `RIDE_BOARD`
- 旧10: `SPONTANEOUS_SELECTION` + `RIDE_BOARD`
- 裁定: **R3を採用**。
- 理由: 初期実況に「何者かに選ばれた」という選択主体は示されない。通常乗車中に非意図的な異常事象へ巻き込まれるため `ACCIDENT_INVOLVEMENT` がより保守的。

### D15
- R3: `UNCERTAINTY_PRESERVED` + `FEAR_TRAUMA`
- 旧10: `UNCERTAINTY_PRESERVED`
- 裁定: **旧10を採用**。
- 理由: D15は主帰結領域。恐怖は途中経過として明示されるが、終端の特徴は帰還・所在・真相が不明のまま残ること。

### D17
- R3: `NO_KNOWN_ESCAPE` + `LEGAL_OFFICIAL`
- 旧10: `U`
- 裁定: **`D17.UNA.NO_KNOWN_ESCAPE` 単独を採用**。
- 理由: 参加者助言は互いに矛盾し、有効な帰還法は確立しない。この状態はコードブックの `NO_KNOWN_ESCAPE` に直接対応し、Unknownより具体化できる。110番は試行されたが成功した制御方式ではないためSecondaryにしない。

### D18
- R3: `1/0/1`
- 旧10: `1/0/0`
- 裁定: **旧10を採用**。
- 理由: R3は00にある後代の遠州鉄道コンテンツ展開をL3 Evidenceに使ったが、これは2004年初期実況Scope外。Scope混成になるためL3=0。D18 L3は同一Scopeに対する独立X Evidenceが必要。

### D19
- R3: `OPEN_FORUM_WEB` + `NATIONAL_PUBLIC`
- 旧10: `OPEN_FORUM_WEB`
- 裁定: **旧10を採用**。
- 理由: 全国的大衆化は後代流通であり、2004年初期実況Scopeでは公開掲示板圏のみを直接コードする。

### D20
- R3: `EXPERIENCER` + `NO_ONE_KNOWS`
- 旧10: `NO_ONE_KNOWS`
- 裁定: **旧10を採用**。
- 理由: D20が問うのは真相・追加情報の特権保持者。投稿者は局面情報を持つが、駅の正体・原因・帰還法を知らない。真相保持者は伝承内で特定されない。

## H3 QA
- Primary exactly 1を全H3で確認。
- Secondary 0〜2件を確認。
- ParentはChildから一意に導出。
- U/NA/CをChildとして使用していない。

## causal QA
- D11 `ACCIDENT_INVOLVEMENT` → D13 `SPATIAL_DISTORTION` → D15 `UNCERTAINTY_PRESERVED` の流れに矛盾なし。
- D10で異界・幽霊等を確定因果源にしていない。
- 投稿停止を物理的失踪・死亡へ昇格していない。

## D18 QA
- L1=1: 初期実況内部の空間異常。
- L2=0: 読むこと自体が危害条件ではない。
- L3=0: 初期実況Scope内の独立X Evidenceなし。後代の公式コンテンツ展開はScope外。

## R4結論
Coding正本を上記裁定へ更新し、Statusを `レビュー待` とする。外部00/10 Review前のため `完了` にはしない。
