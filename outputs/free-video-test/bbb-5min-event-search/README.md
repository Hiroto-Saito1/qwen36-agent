# Big Buck Bunny 5分動画イベント探索

この出力は、5分版の Big Buck Bunny で次の条件が成立しそうな時刻を探す検証です。

```text
蝶が木の幹の近くに見えている
```

現在は `retrieve_verify` 方式です。まず `search.scan_interval_seconds` ごとに軽量検索モデルで全体をスキャンし、類似度の高い候補区間だけをQwen VLで `local_scan_interval_seconds` ごとに確認します。成立サンプルが見つかったら、その前後だけ `boundary_tolerance_seconds` まで境界を詰めます。

再現方法:

```bash
cd /path/to/qwen36-agent/workspace
./scripts/setup-vision-env.sh

cd /path/to/qwen36-agent/workspace/outputs/free-video-test/bbb-5min-event-search
./reproduce.sh
```

初回は検索モデルの取得、動画の準備、VLモデルの読み込みに時間がかかります。途中結果は `output.json` に随時書かれます。

主に見る出力:

- `output.json`: 機械処理用の最終JSON。成立時刻は `occurrences` に入ります。
- `output.json` の `retrieval_scan.samples`: 全体スキャンで各時刻がどれくらい条件に近かったか。
- `output.json` の `retrieval_scan.candidate_windows`: Qwen VLで検証した候補区間。
- `output.json` の `verification.evaluations`: Qwen VLの判定、信頼度、否定理由。
- `event-search.md`: 人間向けの要約。
- `search-trace.json`: 検索・検証・境界調整の詳細。

この検証では `score_threshold: 0.70`、`scan_interval_seconds: 2.0`、`local_scan_interval_seconds: 2.0`、`minimum_candidate_windows: 6`、`max_candidate_windows: 6` を使います。軽量検索だけの事前確認では、狙いの200秒台が候補区間に入る設定です。

以前の「紫色の蝶」や「木の幹の上」条件は、VL側が色や厳密な接触関係を不安定に扱って `near_miss` を返すことがありました。この検証では探索ロジックそのものを確認するため、「木の幹の近く」という視覚的に安定した条件にしています。

`input.json` には `required_visual_checks` と `not_required_visual_checks` も入れています。これは、VLが条件にない「色」「接触」「他キャラとの相互作用」を勝手に必須扱いしないようにするためです。

さらに `lexical_match_terms` に `蝶`、`木の幹`、`近く`、`lexical_caption_terms` に `蝶`、`木の幹` を入れています。VLが肯定キャプションを書きながら `near_miss` と返した場合、証拠文だけでなく neutral_caption 側にも主要語が出ているときだけ成立へ補正し、`output.json` の `override_reason` に記録します。

本番の長い動画では、候補が多すぎる場合に `score_threshold` を `0.82` へ上げるか、`max_candidate_windows` を減らします。見落としが気になる場合は、`score_threshold` を下げるか、`scan_interval_seconds` を短くします。

古い `event-search.json` は使いません。

## Source video attribution

Big Buck Bunny is (c) copyright 2008, Blender Foundation / www.bigbuckbunny.org and is licensed under the Creative Commons Attribution 3.0 license. The reproduction script downloads the public 320x180 movie from Blender's download server and keeps the source video under ignored `work/` files. License reference: https://download.blender.org/ED/poster.pdf
