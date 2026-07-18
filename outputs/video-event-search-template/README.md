# JSON駆動の動画イベント探索テンプレート

このテンプレートは、1時間程度のローカル動画から「指定した条件が成立する時刻」を探すためのものです。

現在の標準方式は `retrieve_verify` です。まず `condition` から検索用の短い英語クエリを内部生成し、軽量な画像テキスト検索モデルで動画全体を細かくスキャンします。その後、条件に近い候補区間だけをQwen VLで検証します。Qwen VL検証は、対象フレーム1枚の一次判定後、関係ありそうな時刻だけ前後3枚で確定します。最終判定は、生成クエリではなく元の `condition` 本文だけを基準にします。

この方式では、昔のように30秒区間をいきなり二分探索しません。最初の粗いフレームに該当シーンがぴったり写っていなくても、検索モデルの類似度が高い区間を拾えるため、短いイベントに少し強くなります。

## 初回セットアップ

イベント探索には、VLサーバーとは別に軽量検索モデル用のPython環境が必要です。

```bash
cd /path/to/qwen36-agent/workspace
./scripts/setup-vision-env.sh
```

このコマンドは `../vision-env/` を作り、`../models/retrieval/` に `google/siglip2-base-patch16-224` を取得します。

## 使い方

1. `input.example.json` をコピーするか、同じ形のJSONを用意します。
2. `video_path` に調べたい動画ファイルを指定します。
3. `condition` に探したい条件をすべて書きます。
4. `search.scan_interval_seconds` に全体スキャンの間隔を指定します。
5. `search.score_threshold` に候補選択とVL成立判定で共通に使うしきい値を指定します。
6. `output_directory` に結果を置く場所を指定します。

実行例:

```bash
cd /path/to/qwen36-agent/workspace/outputs/video-event-search-template
./reproduce.sh input.example.json
```

`output_directory` や `video_path` が相対パスの場合は、入力JSONが置かれているディレクトリを基準に解釈します。

## 入力JSONの意味

| キー | 意味 |
| --- | --- |
| `video_path` | 調べたいローカル動画ファイル |
| `condition` | 探したい条件。視覚条件、除外条件、緩めたい条件はここにすべて書きます |
| `output_directory` | 結果の保存先 |
| `language` | キャプションと根拠の言語 |
| `search.strategy` | 探索方式。現在は `retrieve_verify` |
| `search.scan_interval_seconds` | 動画全体を軽量検索する間隔。1時間動画ならまず `10.0` |
| `search.score_threshold` | 共通しきい値。検索候補の選択と、VLが成立とみなす最低信頼度に使います |
| `search.minimum_event_duration_seconds` | 想定する最短イベント長。短いイベントを探すなら下げます |
| `search.local_scan_interval_seconds` | 候補区間内をQwen VLで見る間隔 |
| `search.boundary_tolerance_seconds` | 成立区間の前後境界を詰める最終粒度 |
| `search.retrieval_model` | 軽量検索モデル。標準は `google/siglip2-base-patch16-224` |
| `search.retrieval_content_filter` | 標準は `none`。このテンプレートは追加フィルタを入れません |
| `search.candidate_padding_seconds` | 検索で拾った時刻の前後を何秒広げてVL検証するか |
| `search.minimum_candidate_windows` | しきい値を超える候補が少ない場合でも最低何区間は検証するか |
| `search.max_candidate_windows` | VL検証する候補区間の上限 |
| `search.minimum_positive_samples` | 1イベントとして採用するために必要な成立サンプル数 |
| `search.verification_image_max_edge_pixels` | Qwen VLへ送る画像の長辺上限。標準は `640`。`null` なら縮小しません |
| `search.max_evaluations` | Qwen VLリクエスト数の上限。1時間動画の初回は `120` などで止めどころを作ります。`null` なら上限なし |

検索用クエリは、起動時に `condition` から自動生成されます。生成クエリは `output.json` には出しません。確認したい場合は `config.snapshot.json` または `search-trace.json` の `retrieval_query_plan` を見ます。クエリ生成に失敗した場合は、`condition` 本文をそのまま検索クエリとして使い、理由を `retrieval_query_plan.fallback_reason` に残します。

旧方式の `caption_interval_seconds`、`candidate_threshold`、`match_threshold`、`fallback_top_intervals`、`minimum_interval_seconds`、`query_texts`、`required_visual_checks`、`not_required_visual_checks`、`lexical_match_terms`、`lexical_caption_terms` は使いません。後方互換はありません。

## 調整の考え方

1時間動画では、まず `scan_interval_seconds: 10.0`、`score_threshold: 0.82`、`candidate_padding_seconds: 8.0`、`max_candidate_windows: 6`、`local_scan_interval_seconds: 2.0`、`verification_image_max_edge_pixels: 640`、`max_evaluations: 120` から始めます。この設定なら、360本前後の検索フレームから候補を選び、VL確認は多くても120リクエストで止まります。

VL検証が重い場合は、`verification_image_max_edge_pixels` をさらに `480` へ下げるか、`max_candidate_windows` を減らします。細部が潰れて誤判定する場合は `960` や `1280` へ上げるか、`null` で元解像度を送ります。見落としが気になる場合は、`score_threshold` を `0.75` へ下げる、`max_candidate_windows` を増やす、または `scan_interval_seconds` を `5.0` へ下げます。候補が多すぎる場合は、`score_threshold` を上げるか `max_candidate_windows` を減らします。

研究用途で「拒否なく検索したい」前提に合わせ、軽量検索段階にはキーワード拒否、NSFW分類器、マスク、ブロックリストなどを入れていません。ただし、検索モデル自体の学習データが完全に無加工である保証ではありません。

## 出力

| ファイル/ディレクトリ | 内容 |
| --- | --- |
| `output.json` | 最終結果JSON。実行中も評価ごとに完全なJSONとして上書き更新されます |
| `captions.md` | Qwen VLで確認したフレームの説明 |
| `captions.json` | Qwen VL確認結果 |
| `event-search.md` | 見つかった成立時刻の人間向けまとめ |
| `search-trace.json` | 検索スコア、候補区間、境界調整の記録 |
| `config.snapshot.json` | 実行時の設定 |
| `evidence/` | 各成立区間の前・代表・後フレーム |

`output.json` には `condition` 本文や内部生成された検索クエリは入れません。条件本文は `input.json` と `config.snapshot.json`、検索クエリは `config.snapshot.json` と `search-trace.json` を見ます。

`output.json` の `retrieval_scan.samples` には、全体スキャンした各時刻の `retrieval_raw_score` と `retrieval_likelihood_score` が入ります。どの区間がなぜQwen VL検証へ進んだかは `retrieval_scan.candidate_windows` と `selected_for_verification` を見ます。

`output.json` の `verification` には、一次判定数、3枚確認数、VLリクエスト数、送信画像数、使った画像縮小上限が入ります。各評価の `primary_verification` は1枚判定、`confirmation_verification` は3枚確認です。`occurrences` に入るのは、3枚確認で成立した時刻だけです。VLモデルには `is_match` を自己申告させず、`event_phase: match`、`score_threshold`、肯定的な `evidence` の有無からスクリプト側で成立を決めます。

同じ出力先で再実行すると、古い生成物は上書きされます。`input.json`、`reproduce.sh`、README、未知の手元ファイルは残します。
