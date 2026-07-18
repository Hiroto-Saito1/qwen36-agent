# Big Buck Bunny 5分動画イベント探索

この出力は、5分版の Big Buck Bunny で次の条件が成立しそうな時刻を探す検証です。

```text
木の幹に矢が刺さっている
```

現在は `retrieve_verify` 方式です。まず `condition` から検索用クエリを内部生成し、`search.scan_interval_seconds` ごとに軽量検索モデルで全体をスキャンします。その後、類似度の高い候補区間だけをQwen VLで `local_scan_interval_seconds` ごとに確認します。Qwen VLは、対象フレーム1枚の一次判定後、関係ありそうな時刻だけ前後3枚で確定します。成立サンプルが見つかったら、その前後だけ `boundary_tolerance_seconds` まで境界を詰めます。

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
- `output.json` の `verification.evaluations`: Qwen VLの一次判定、3枚確認、信頼度、否定理由。
- `output.json` の `verification`: VLリクエスト数、送信画像数、縮小後の入力サイズ。
- `event-search.md`: 人間向けの要約。
- `search-trace.json`: 検索・検証・境界調整の詳細。内部生成された検索クエリもここで確認できます。

この検証では `score_threshold: 0.70`、`scan_interval_seconds: 10.0`、`local_scan_interval_seconds: 2.0`、`candidate_padding_seconds: 8.0`、`minimum_candidate_windows: 3`、`max_candidate_windows: 3`、`minimum_positive_samples: 1`、`max_evaluations: 80`、`verification_image_max_edge_pixels: 960` を使います。5分動画でも、1時間動画用テンプレートに近い軽めの探索設定で通るかを確認するためです。1時間動画用の初期設定は `outputs/video-event-search-template/input.example.json` を使います。

以前の「紫色の蝶」や「蝶が木の幹の近く」条件は、VL側が小さい対象や厳密な位置関係を不安定に扱って `near_miss` を返すことがありました。この検証では探索ロジックそのものを確認するため、「木の幹に矢が刺さっている」という視覚的に安定した条件にしています。

`input.json` で人間が書く意味入力は `condition` だけです。補助クエリ、必須チェック、不要チェック、語彙補正は使いません。検索用の短いクエリは起動時に自動生成され、`config.snapshot.json` と `search-trace.json` の `retrieval_query_plan` に記録されます。

本番の長い動画では、候補が多すぎる場合に `score_threshold` を `0.82` へ上げるか、`max_candidate_windows` を減らします。VL検証が重い場合は `verification_image_max_edge_pixels` を `640` へ下げます。見落としが気になる場合は、`score_threshold` を下げるか、`scan_interval_seconds` を短くします。

古い `event-search.json` は使いません。

## Source video attribution

Big Buck Bunny is (c) copyright 2008, Blender Foundation / www.bigbuckbunny.org and is licensed under the Creative Commons Attribution 3.0 license. The reproduction script downloads the public 320x180 movie from Blender's download server and keeps the source video under ignored `work/` files. License reference: https://download.blender.org/ED/poster.pdf
