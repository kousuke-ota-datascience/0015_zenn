# 0. INTRODUCTION

本書は、A3パイロット49件の全面再分析について、**現在状態を保持するcontrol plane**である。

本書の責務は、対象Entry、baseline、進捗集計、Entry × 正本成果物（00 / 10）の現在Status・Review checkpoint・pre/post-SHA、実行順序、lifecycle履歴を保持することに限定する。

個別伝承分析の作業手順、control planeのフォーマット定義、Status遷移、commit / push単位、Review返却後の修正cycle、SHA整合性判定は本書では定義しない。これらは実行時に指定される標準workflowに従う。

# 1. baseline

- baseline main commit: `9570faea998905f074e59df1691748b9cc54d03d`
- theoretical design blob: `64ed9801ecb17c44f39e04c364c23ce5cb682014`
- code system blob: `192c2f1593e29eb32292a31d564d21ad4aec2427`
- coding rules blob: `4fe19cc922840a46367bc6d0162271d2422996a9`
- workflow blob at baseline: `4d26549329e3e0d5fb20604217ec5e20080f19c0`
- tutorial 00 blob: `0ed8c22777eb2c213ca0b009016ccc66a01054ce`
- tutorial 10 blob: `dccc5e4473bc51c484f16ba916d653ea7383f2f5`

# 2. 進捗集計

## 2.1. Entry単位

| Status | 件数 |
|---|---:|
| 未 | 16 |
| レビュー待 | 0 |
| 要修正 | 18 |
| 再作業中 | 0 |
| 再レビュー待 | 2 |
| 完了 | 13 |
| 対象外 | 0 |

## 2.2. 正本成果物単位

| Status | 00/10成果物件数 |
|---|---:|
| 未 | 32 |
| レビュー待 | 0 |
| 要修正 | 36 |
| 再作業中 | 0 |
| 再レビュー待 | 2 |
| 完了 | 28 |
| 対象外 | 0 |
| 合計 | 98 |

- 完了Entry: `0001/0003/0005/0006/0011/0019/0024/0025/0059/0060/0089/0101/0112`
- 再作業中Entry: なし
- 再レビュー待Entry: `0081/0091`
- 要修正Entry: `0113/0118/0132/0133/0137/0152/0157/0158/0169/0178/0179/0180/0181/0188/0198/0225/0250/0275`
- 未着手Entry: 16件
- R5/R6/R7: 未着手

# 3. Entry別進捗

以下の表を、Entry × 正本成果物（00 / 10）の現在状態の正本とする。列定義・状態遷移・更新方法は標準workflowに従う。

| Entry_ID | 伝承 | 成果物 | Status | 最新レビュー版 | pre-SHA | post-SHA | remarks |
|---|---|---|---|---|---|---|---|
| 0001 | 口裂け女 | 00 | 完了 | `003` | `4e18c1a977c8c223a26865fac0feeafef5d54b37` | `9a565a4879688a9a07e6c60a617813e026d46959` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0001 | 口裂け女 | 10 | 完了 | `003` | `9a565a4879688a9a07e6c60a617813e026d46959` | `134a9ce8394ed0e2e5a7f491c6a52b19cb236c0c` | legacy移行: 旧R3→R4 checkpoint。Review Pass |
| 0003 | 赤い紙・青い紙／赤マント系 | 00 | 完了 | `004` | `5c577eb3d3acfa3f69fd4ca841875888b27305cf` | `7e08f01ca3119e71268292513504ff79f6987d60` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0003 | 赤い紙・青い紙／赤マント系 | 10 | 完了 | `004` | `7e08f01ca3119e71268292513504ff79f6987d60` | `c0d5fac7edad8edd31f4d768277ce4ae6a11bda3` | legacy移行: 旧R3→R4 checkpoint。Review Pass |
| 0005 | 紫の鏡 | 00 | 完了 | `002` | `c2a37792f12b6c2aa5b8760536b13764d14c714d` | `74c8f1367a53020c2ff02b3ae5072202ad74e7e9` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0005 | 紫の鏡 | 10 | 完了 | `002` | `74c8f1367a53020c2ff02b3ae5072202ad74e7e9` | `658b95e6fe5ea8f8b0a3be83314cc851a788e016` | legacy移行: 旧R3→R4 checkpoint。Review Pass |
| 0006 | メリーさんの電話 | 00 | 完了 | `004` | `89b2865013ecd7fca4b39fa914973a0f55eb3b3b` | `92bbf26600e52525c2c11398c55d32082f0b3d0e` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0006 | メリーさんの電話 | 10 | 完了 | `004` | `92bbf26600e52525c2c11398c55d32082f0b3d0e` | `6a36aebbc047908679f4884d4202da5d7a2efc5b` | legacy移行: 旧R3→R4 checkpoint。Review Pass |
| 0011 | こっくりさん | 00 | 完了 | `003` | `1b24d197f9cb94241e26a368a841bc1071671d23` | `4254c58c1dc3fab6315f62e2846e31401c89f3fb` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0011 | こっくりさん | 10 | 完了 | `003` | `4254c58c1dc3fab6315f62e2846e31401c89f3fb` | `0b0f861577ceae5f546d0d3796387e27aa65dfe2` | legacy移行: 旧R3→R4 checkpoint。Review Pass |
| 0019 | 小さいおじさん | 00 | 完了 | `002` | `e7ada0703530358d46b387b4feb149fc9e1b8ad9` | `b3f91feaba652b5b3b8eeca202bf2bddb4449c34` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0019 | 小さいおじさん | 10 | 完了 | `002` | `b3f91feaba652b5b3b8eeca202bf2bddb4449c34` | `1f180c1d0fde5100b0cf71d6bc184ad29a562f0d` | legacy移行: 旧R3→R4 checkpoint。Review Pass |
| 0024 | 幸福の手紙 | 00 | 完了 | `002` | `fc120f41ab98bc1f150f624dcafaaa016e2c6d39` | `41e27cc1c43d79fa231f32b9e0ffdf80fb296811` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0024 | 幸福の手紙 | 10 | 完了 | `002` | `41e27cc1c43d79fa231f32b9e0ffdf80fb296811` | `362878efa061cd073f48b2cf09d9e006c2af6ba9` | legacy移行: 旧R3→R4 checkpoint。Review Pass |
| 0025 | 不幸の手紙 | 00 | 完了 | `002` | `aee0942ab1c70a42008c786670c0c0d5b6e04a1e` | `7ea7647a2b29cf93687c71109adce9e5637f3e7f` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0025 | 不幸の手紙 | 10 | 完了 | `002` | `7ea7647a2b29cf93687c71109adce9e5637f3e7f` | `443b8f6a87afd69f2a0276b8677b8f093b7266e2` | legacy移行: 旧R3→R4 checkpoint。Review Pass |
| 0059 | 深泥池の幽霊タクシー | 00 | 完了 | `003` | `4c91bfea47f31b35a5a38799ee145c93f29580f7` | `67db4d07c94341c71b219b7b6469a4208ea3d39f` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0059 | 深泥池の幽霊タクシー | 10 | 完了 | `003` | `67db4d07c94341c71b219b7b6469a4208ea3d39f` | `60ff1c2a50d75f95f74170a5a9085ee1fb2b5aa3` | legacy移行: 旧R3→R4 checkpoint。Review Pass |
| 0060 | タクシー幽霊 | 00 | 完了 | `005` | `01bb8e996a50304ba6689d6a01f5bf957fd85042` | `8b8b4be57a11d36e246599ef03f82cca4856af2b` | Review_005 Pass。正本変更なし。 |
| 0060 | タクシー幽霊 | 10 | 完了 | `005` | `22b8b547d7ba1d5ffcb22d0d7c473575b6957b5b` | `a33c7f83f1a8c98d5e511a4f2c0ce58995cc06d2` | Review_005 Pass。正本変更なし。 |
| 0081 | ピアスの白い糸 | 00 | 完了 | `005` | `410c9c4bd69a957e0fc0374541444b38ecdfaf73` | `1492f1c0b7e70815674df6dd4d4d44d2d6f6e7e5` | Review_005 Pass。正本変更なし。 |
| 0081 | ピアスの白い糸 | 10 | 再レビュー待 | `005` | `56e3c2af2a40dafe3c339f8262600dfaf9b5e9f4` | `c4f81113bf317f7b38246e5f7a221bf97e18cb73` | Review_005指摘対応。D07 Secondary `D07.BHF.MEDICAL_RISK` を除外し、`BODY_ANOMALY` のみへ保守化。 |
| 0089 | 日本だるま／だるま女 | 00 | 完了 | `005` | `89a4a64c08e191d410f938c919e8715130790139` | `f0e3abb21018117629c6b817043cb5861e42cf81` | Review_005 Pass。正本変更なし。 |
| 0089 | 日本だるま／だるま女 | 10 | 完了 | `005` | `46ae27129bced8b8752e00a87677ab191d26449b` | `cdee7caeb58edbf5269ca3340b96bf853d74e59f` | Review_005 Pass。正本変更なし。 |
| 0091 | ベッドの下の男 | 00 | 完了 | `005` | `693ba24ff2f8e04bfc4b3957c8479a6df4607a92` | `afc764c69fb9eb0dfeb50aaced7d20e75a912ec9` | Review_005 Pass。正本変更なし。 |
| 0091 | ベッドの下の男 | 10 | 再レビュー待 | `005` | `31075c0f7bbbc84cbea0cdd8762b23a0e9586c37` | `05c2acbdce1a361262c06c4f4642a16dd2e730b8` | Review_005指摘対応。D20をUへ保守化し、警察の役割をD17の制度的介入へ限定。 |
| 0101 | 海外旅行で臓器を抜かれる | 00 | 完了 | `005` | `2aefa37f5d3343a15d3436823f9001db4a43ff40` | `04db54f073ea0c5f736cce96f69038797dba6315` | Review_005 Pass。正本変更なし。 |
| 0101 | 海外旅行で臓器を抜かれる | 10 | 完了 | `005` | `9f3a2368fedbdd771fe97fcc6464b7e426e5168b` | `d7e67837fcd9fdddb583141668a34de953561ff8` | Review_005 Pass。正本変更なし。 |
| 0112 | 事故物件は一度別人を住ませれば告知義務が消える | 00 | 完了 | `005` | `85111550bf6370368616a7483692e3f1ef27786a` | `2748bf99dd88cf594125c54a76084c635e523512` | Review_005 Pass。正本変更なし。 |
| 0112 | 事故物件は一度別人を住ませれば告知義務が消える | 10 | 完了 | `005` | `b0441eb452661dc4d9457a7e17bd0151c3b07269` | `c213bbd821e368708e49010cacca974d9ba95138` | Review_005 Pass。正本変更なし。 |
| 0113 | 井の頭公園のボートに乗ると別れる | 00 | 要修正 | `001` | `b778af5f1b7bb38d10323dc563191f6d67fe1174` | `7acdb0e50cdfedfc984bd0e14cf45b739eeb90f7` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0113 | 井の頭公園のボートに乗ると別れる | 10 | 要修正 | `001` | `7acdb0e50cdfedfc984bd0e14cf45b739eeb90f7` | `5be227c1b03e2a1aeabcbb255822778c166a0fd0` | legacy移行: 旧R3→R4 checkpoint。D01=U、L3=0。 |
| 0118 | 函館山の切れない木 | 00 | 要修正 | `001` | `cfe370ce21791dcaaa815da1d636ff41572c11c6` | `c17adc9a37c3019e585346f3b8ebbdcdc0346f32` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0118 | 函館山の切れない木 | 10 | 要修正 | `001` | `c17adc9a37c3019e585346f3b8ebbdcdc0346f32` | `63c122601d2fe7bf975f047f5f8b7dd9b1fe2b3a` | legacy移行: 旧R3→R4 checkpoint。D10 taxonomy gap、L3=0。 |
| 0132 | 八幡の藪知らず | 00 | 要修正 | `001` | `04a86cc9c6fbf4405bd6123aa5425a0e0b0bdd97` | `39ca6f738f62972ac58b1a4f5ba875143a5cabf9` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0132 | 八幡の藪知らず | 10 | 要修正 | `001` | `39ca6f738f62972ac58b1a4f5ba875143a5cabf9` | `9d662fe4e36baa4cfda390b9a4744260467aaba4` | legacy移行: 旧R3→R4 checkpoint。D01=G0/I、L3=1限定。 |
| 0133 | 将門塚の祟り | 00 | 要修正 | `001` | `54f22c7712abcd2205231ed76a2fe1885a8ca911` | `058af26a784d134cd6c02e72fac278cb3447f037` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0133 | 将門塚の祟り | 10 | 要修正 | `001` | `058af26a784d134cd6c02e72fac278cb3447f037` | `ae74f13a5c03563f8c556987e17fae78343f3b28` | legacy移行: 旧R3→R4 checkpoint。D01=G0/I、L3=1。 |
| 0137 | 犬鳴村 | 00 | 要修正 | `001` | `b240224bcfdb7b215ef2ccf1bd45ad9e551cc8a9` | `5310aca86bb3719fa09703f6693804ad546ef62c` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0137 | 犬鳴村 | 10 | 要修正 | `001` | `5310aca86bb3719fa09703f6693804ad546ef62c` | `399880d335a6d9cc55b5d9004887e34dcc2a7c32` | legacy移行: 旧R3→R4 checkpoint。D01=U、L3=0。 |
| 0152 | 青木ヶ原樹海で方位磁針が狂う | 00 | 要修正 | `001` | `b52c65cc5f1a7910784c571d1d9d9abf759318ea` | `684c16cb9c01a5fee5babcb8e415ca775fd65d6b` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0152 | 青木ヶ原樹海で方位磁針が狂う | 10 | 要修正 | `001` | `684c16cb9c01a5fee5babcb8e415ca775fd65d6b` | `ed326e171be81b02f57026b429456142e48998c7` | legacy移行: 旧R3→R4 checkpoint。D17=U、大学検証L3=1。 |
| 0157 | 新郷村キリストの墓 | 00 | 要修正 | `001` | `4933870b3e3062c8ce16ee4eec7aa0ee76b0b693` | `68f901f86daa9a2f0e53d00a80ac791bc9b3fa3e` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0157 | 新郷村キリストの墓 | 10 | 要修正 | `001` | `68f901f86daa9a2f0e53d00a80ac791bc9b3fa3e` | `afcc03d239e668f141f10cd6c3d83b655358ace5` | legacy移行: 旧R3→R4 checkpoint。D01=G1/I、L3=1。 |
| 0158 | 虚舟 | 00 | 要修正 | `001` | `90f08e7968e188540022762a88926b7ea69586fc` | `aa7968f1cd5d12817409a534247050e5e8bf6cef` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0158 | 虚舟 | 10 | 要修正 | `001` | `aa7968f1cd5d12817409a534247050e5e8bf6cef` | `6f9e7370a626d64544e630302a06e93bcaa5c4e0` | legacy移行: 旧R3→R4 checkpoint。D01=G0/I、L3=0。 |
| 0169 | ノストラダムスの大予言 | 00 | 要修正 | `001` | `f0632eee87636fe2f584511c3544cd443d9290d0` | `598f41fc79c2871b347c4981d4d9984057b87ae0` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0169 | ノストラダムスの大予言 | 10 | 要修正 | `001` | `598f41fc79c2871b347c4981d4d9984057b87ae0` | `50f54b95bb478e739dd93aef58b248828433b285` | legacy移行: 旧R3→R4 checkpoint。D01=G3/D、D16=DEADLINE、L3=0。 |
| 0178 | 猿夢 | 00 | 要修正 | `001` | `1a85ab52869944c1f8c2ee3a6852fdba832884b6` | `e7f31217b47a73cf8c226e991b57cbe4c237e914` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0178 | 猿夢 | 10 | 要修正 | `001` | `e7f31217b47a73cf8c226e991b57cbe4c237e914` | `64429af4d65cfc5b941e6db878f0f6a81779ba01` | legacy移行: 旧R3→R4 checkpoint。2000年原型。D17覚醒離脱taxonomy gap、L3=0。 |
| 0179 | くねくね | 00 | 要修正 | `001` | `b6aac89cc94e864063d2e9cdff332bfa78a61153` | `c1e1c625fa13c2f1d164fc67703bcc7e09397888` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0179 | くねくね | 10 | 要修正 | `001` | `c1e1c625fa13c2f1d164fc67703bcc7e09397888` | `55e44c086c579b9b4a97d21df109addafb828def` | legacy移行: 旧R3→R4 checkpoint。2001原型→2003増補。D11=UNDERSTAND_RECOGNIZE、D13認知災害、L3=0。 |
| 0180 | きさらぎ駅 | 00 | 要修正 | `001` | `390a59182cffc7c6982c45af037e732c514c4fc0` | `24a7d6ec5a50972f27a2a39fff2a57045023aaf4` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0180 | きさらぎ駅 | 10 | 要修正 | `001` | `24a7d6ec5a50972f27a2a39fff2a57045023aaf4` | `0ef39d4cb61802c34fe4e271b3cdc1aa867feeb9` | legacy移行: 旧R3→R4 checkpoint。D09=非解決維持、D11=ACCIDENT_INVOLVEMENT、D17=NO_KNOWN_ESCAPE、L3=0。 |
| 0181 | コトリバコ | 00 | 要修正 | `001` | `15e4277b3be93a189cc168fa20f62cfb5a63f048` | `2453520af5a0aa8d99e175f896e4df8f6a160db5` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0181 | コトリバコ | 10 | 要修正 | `001` | `2453520af5a0aa8d99e175f896e4df8f6a160db5` | `afc96e6404a9fb7ce0e4731052d5da5598bc6c57` | legacy移行: 旧R3→R4 checkpoint。2005-06-06→06-11早期増補、D21=A4、L3=0。 |
| 0188 | 八尺様 | 00 | 要修正 | `001` | `4d762489c220d89837df9edf603d84780b2195f9` | `9d59272c2d141ef05a8e4f9ec80633ceb3fb7a21` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0188 | 八尺様 | 10 | 要修正 | `001` | `9d59272c2d141ef05a8e4f9ec80633ceb3fb7a21` | `94d7be41f4d6e2e61a7f1546e11e77eb773e1d4b` | legacy移行: 旧R3→R4 checkpoint。D15=FUTURE_CONSTRAINT、L3=0。 |
| 0198 | 一人かくれんぼ | 00 | 要修正 | `001` | `ed97ce65ee6856da024c91a0a14bbbc202353306` | `ebc2fc8400e97feaa85c5bfecb5be7b39ae821e3` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0198 | 一人かくれんぼ | 10 | 要修正 | `001` | `ebc2fc8400e97feaa85c5bfecb5be7b39ae821e3` | `37452dff345a613dc030eab5ce3aa296462efdb7` | legacy移行: 旧R3 freeze checkpoint（00単独commitではない）。 |
| 0225 | 蛇口からポンジュース | 00 | 要修正 | `001` | `73568a3f4cbab96cb708e4b7ce5cae1cfb7300c0` | `3c3e87323619418878c3ad7a239bfc5658054d5f` | legacy移行: post-SHAは旧R3 freeze checkpoint（00単独commitではない）。 |
| 0225 | 蛇口からポンジュース | 10 | 要修正 | `001` | `3c3e87323619418878c3ad7a239bfc5658054d5f` | `36f1860f8490eeaec62d0f81f1ea1d73139c6037` | legacy移行: 旧R3→R4 checkpoint。昭和50年代頃/I、2007年企業再現、D18=1/0/1。 |
| 0250 | 七人ミサキ | 00 | 要修正 | `001` | `93f71046e0cdb61ed94b97ea525f57a9a4693110` | `b79c45fe6f256e134967725be6281b874a0dab3e` | 新運用: Evidence正本単独commit。Review_001要修正。 |
| 0250 | 七人ミサキ | 10 | 要修正 | `001` | `eb27237da162253685c49800d686e251590b3c39` | `29013e64175e96fd2d58dd30688b428487b7cbf4` | 新運用: Coding正本単独commit。R4 Final decisions反映。Review_001要修正。 |
| 0275 | 磐梯山の手長足長 | 00 | 要修正 | `001` | `c3afafad641a2aa2a12897b951f0e1027208540d` | `76b7c0755d8ffd2a981cbbc8d66375249b831625` | 新運用: Evidence正本単独commit。1992年直接採録＋2025年自治体文化施設X Evidence。Review_001要修正。 |
| 0275 | 磐梯山の手長足長 | 10 | 要修正 | `001` | `47340c42411e2c05b6fc1143f634172ba7ceca41` | `d7f2d5d7513d4b9cac9c3c4f5d04ea4816e44656` | 新運用: Coding正本単独commit。D01=G5、D18=1/0/1、D21=A4。Review_001要修正。 |
| 0309 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0309 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0319 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0319 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0349 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0349 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0356 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0356 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0362 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0362 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0363 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0363 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0365 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0365 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0366 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0366 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0384 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0384 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0385 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0385 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0394 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0394 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0403 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0403 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0410 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0410 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0411 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0411 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0412 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0412 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |
| 0413 | 名称未確認 | 00 | 未 | － | － | － | R1でinventory確認 |
| 0413 | 名称未確認 | 10 | 未 | － | － | － | 00確定後に作成 |

# 4. 現在のReview状態

- 完了13件は00/10双方の既存最終Review Passを維持する。
- `0081/0091` は00が`完了 / 005`、10はReview_005指摘対応済みで`再レビュー待 / 005`。
- `0113/0118/0132/0133/0137/0152/0157/0158/0169/0178/0179/0180/0181/0188/0198/0225/0250/0275` は `要修正 / 001`。

# 5. 実行順序

`0001 → 0003 → 0005 → 0006 → 0011 → 0019 → 0024 → 0025 → 0059 → 0060 → 0081 → 0089 → 0091 → 0101 → 0112 → 0113 → 0118 → 0132 → 0133 → 0137 → 0152 → 0157 → 0158 → 0169 → 0178 → 0179 → 0180 → 0181 → 0188 → 0198 → 0225 → 0250 → 0275 → 0309 → 0319 → 0349 → 0356 → 0362 → 0363 → 0365 → 0366 → 0384 → 0385 → 0394 → 0403 → 0410 → 0411 → 0412 → 0413`

次の新規Entry: `0309`

# 6. lifecycle変更履歴

- 2026-09-14: baseline固定。
- 2026-09-15: 先行9件Review Cycle完了。
- 2026-09-15: 0060〜0169を順次Coder R1〜R4完了・レビュー待へ移行。
- 2026-09-15: 0178 猿夢、0179 くねくね、0180 きさらぎ駅、0181 コトリバコ、0188 八尺様を順次レビュー待へ移行。
- 2026-09-15: 0198 一人かくれんぼをレビュー待へ移行。2006説を採らず2007年4月を直接定点とし、別参加者の実践行動を独立X EvidenceとしてD18 L3=1。
- 2026-09-15: 0225 蛇口からポンジュースをレビュー待へ移行。地域ジョークから2007年の企業による実物再現までをScope化し、D18 L3=1。
- 2026-09-15: 0250 七人ミサキをレビュー待へ移行。00/10を新運用の成果物単独commitで確定。
- 2026-09-15: 0275 磐梯山の手長足長をレビュー待へ移行。1992年直接採録を年代定点とし、2025年町立資料館の展示・参加型催事を独立X EvidenceとしてD18 L3=1。
- 2026-09-15: control planeをtask別進捗から正本成果物（00/10）単位の進捗管理へ変更。成果物ごとのcommit→push、pre/post-SHA、Review Statusを主キー化。
- 2026-09-15: `最新レビュー版` 列を追加。`－`=未レビュー、3桁連番=最後に完了したReview版として固定。
- 2026-09-15: 0060〜0275の24件について00/10のReview_001返却を反映し、48成果物を`要修正 / 001`へ同期。
- 2026-09-15: 0081・0089のReview_001指摘対応を完了し、00/10の4成果物を`再レビュー待 / 001`へ移行。
- 2026-09-15: 0091・0101・0112のReview_001指摘対応を完了し、00/10の6成果物を`再レビュー待 / 001`へ移行。
- 2026-09-15: 0060・0081・0089のReview_002返却を反映し、00/10の6成果物を`要修正 / 002`へ同期。
- 2026-09-15: 0060・0081・0089のReview_002指摘対応を完了し、00/10の6成果物を`再レビュー待 / 002`へ移行。
- 2026-09-15: Section 6にReview駆動の更新プロトコルを追加。`post-SHA` とReview mdの`対象commit SHA`をexact match / ancestor / divergedで検証し、不一致時は作業開始を禁止するゲートを明文化。Review返却・修正開始・修正完了・再Reviewにおけるpre/post-SHA更新規則とSection 5同期規則を固定。
- 2026-09-15: Review Cycleの実行手順と進捗表更新規則を責務分離。SHA整合性判定・修正→再Reviewの手番をSection 2.2へ移し、Section 6はSection 2.2の状態遷移を表へ反映する規則に限定。
- 2026-09-15: 0091・0101・0112のReview_002指摘対応を完了し、00/10の6成果物を`再レビュー待 / 002`へ移行。0091_10はReview md記載の対象blob SHAに誤記があったが、対象commit SHAとcommit graph・実blobの不変性を照合してReview対象版を確定した。
- 2026-09-15: 0060・0081・0089のReview_003返却を反映。3件の00はPassとして`完了 / 003`、10は修正要求として採用。
- 2026-09-15: 0060・0081・0089のReview_003指摘対応を完了。10の3成果物を`再レビュー待 / 003`へ移行し、次ReviewをReview_004とする。
- 2026-09-15: 0091・0101のReview_003返却を反映。両Entryの00はPassとして`完了 / 003`、10は修正要求として採用。
- 2026-09-15: 0091・0101のReview_003指摘対応を完了。10の2成果物を`再レビュー待 / 003`へ移行し、次ReviewをReview_004とする。
- 2026-09-15: 0112_00のReview_003（Minor）指摘を反映。0.2.3見出しをtutorial正規名称へ戻し、`再レビュー待 / 003`へ移行。
- 2026-09-15: 0112_10のReview_003（Moderate）指摘を反映。D12 Secondary `SPECIFIC_OTHER` とD13 Secondary `CONCEAL_SUPPRESS` を除外し、`再レビュー待 / 003`へ移行。
- 2026-09-15: control planeの責務を現在状態の保持へ限定。作業手順、Status遷移、commit/push規則、Review修正cycle、SHA整合性判定を標準workflowへ移管。
- 2026-09-15: 対象6 EntryのReview_004返却を反映。6本の00は`完了 / 004`、6本の10は`再作業中 / 004`として修正cycleを開始。
- 2026-09-15: 0060_10のReview_004指摘対応を完了し、`再レビュー待 / 004`へ移行。
- 2026-09-15: 0081_10のReview_004指摘対応を完了し、`再レビュー待 / 004`へ移行。後続の現行taxonomy再照合でD19を`D19.KIN.SCHOOL_YOUTH`へ正規化しpost-SHAを更新。
- 2026-09-15: 0089_10のReview_004指摘対応を完了し、`再レビュー待 / 004`へ移行。
- 2026-09-15: 0091_10のReview_004指摘対応を完了し、直接作用列と現行taxonomyを再正規化して`再レビュー待 / 004`へ移行。
- 2026-09-15: 0101_10のReview_004指摘対応を完了し、`再レビュー待 / 004`へ移行。
- 2026-09-15: 0112_10のReview_004指摘対応を完了。D10/D13を中心規則に合わせてtaxonomy gapへ再判定し、D07/D11の派生Secondaryを除外、D14を中立、D17をUへ保守化して`再レビュー待 / 004`へ移行。
- 2026-09-15: 対象6 EntryのReview_005返却を反映。0060/0089/0101/0112は00/10ともPassで完了。0081/0091は00 Pass、10修正要求として採用。
- 2026-09-15: 0081_10のReview_005指摘対応を完了。D07 Secondary `D07.BHF.MEDICAL_RISK` を除外し、`再レビュー待 / 005`へ移行。
- 2026-09-15: 0091_10のReview_005指摘対応を完了。D20をUへ保守化し、警察の確認役割をD17制度的介入へ限定して`再レビュー待 / 005`へ移行。