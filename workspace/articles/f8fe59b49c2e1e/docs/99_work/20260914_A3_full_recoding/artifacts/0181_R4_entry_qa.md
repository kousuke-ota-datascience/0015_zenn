# 0181 R4 Entry QA

## 比較条件
- R3 freeze: `2453520af5a0aa8d99e175f896e4df8f6a160db5`
- 旧 `0181_10_analysis.md`: **存在せず**
- したがってPrior codingとの比較はなく、R3とEvidence正本・coding rules間のQAのみ実施。

## QA結果

### Scope QA
- 2005-06-06初回遭遇譚と、6/11までの早期体系化を同一「初期形成クラスター」に含める。
- ただし初回から存在した要素と後続増補を00/R2で分離済み。
- 6/11以後の長期考察・後代派生はScope外。

### Code QA
- D01/D02/D03: 2005年・公開Web掲示板を直接確認でき、問題なし。
- D04: 初回から由来・等級・歴史体系が数日で追加されるため `VAR.ACCRETION` Primary、専用スレ連続化を `COL.SERIALIZATION` Secondaryとする。
- D05/D06: 初回の一人称回想を形式核とし、客観的真実性とは分離。
- D07/D15: 早期体系化で女性・子どもの死亡が中心危害として語られるため、`UNEXPLAINED_DEATH_LOSS` / `DEATH` を採用。ただし初回Sの死亡を主張しない。
- D09/D21: 実在する隠岐騒動を怪談の由来へ統合する構造をコードする。怪談上の因果接続の史実性は認定しない。
- D10: 箱を `CURSED_OBJECT`、作用を `IMPERSONAL_CURSE` として分離。
- D11: 開封必須ではないため `OPEN_UNSEAL` は不採用。箱との位置関係をPrimary、持参をSecondary。
- D13: 呪いそのものをPrimary、身体症状をSecondary。
- D16: 箱が複数世代にわたり危険性を保持するため `ENDURING_CONDITION`。個々の発症時間は固定しない。
- D17: 神職家系による介入をPrimary、安置・管理をSecondary。
- D20: R3 Secondary `LOCAL_RESIDENT` は広すぎるため不採用。直接Evidenceに合わせ `D20.PER.FAMILY_BLOODLINE` をSecondaryへ変更する。Mの父・祖父という家系内継承が明示される。

### H3 QA
- Primary exactly 1。
- Secondary 0〜2。
- ParentはChildから一意に導出。
- Status値をChildとして使用していない。

### D18 QA
- L1=1: 呪物→人物の伝承内作用。
- L2=0: 初回投稿内で、話すだけでは取り憑かない旨が示される。
- L3=0: Scope内の掲示板拡散・考察は伝承流通であり、独立した社会現実X Evidenceを確認していない。

### Evidence・倫理QA
- 1868年隠岐騒動の実在と、コトリバコ製法伝来説を分離。
- 被差別集落について匿名怪談内の加害・呪術設定を現実属性として一般化しない。
- 2005年以前からコトリバコが実在したと推測しない。

## R4結論
R3をほぼ維持し、D20 Secondaryのみ `D20.PER.FAMILY_BLOODLINE` へ保守化してCoding正本を新規作成する。外部00/10 Review前のためStatusは `レビュー待`。
