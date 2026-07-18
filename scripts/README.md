# scripts README

このディレクトリには、ローカルモデルサーバーの起動、Qwen Code起動、動画キャプション生成、再現スクリプト共通処理、動画イベント探索用の軽量検索環境セットアップを置いています。

詳細な動画イベント探索本体は `outputs/video-event-search-template/video_event_search.py` にあります。`scripts/` 側は、主にサーバー起動と汎用キャプション生成を担当します。

## `setup-vision-env.sh`

動画イベント探索の軽量検索段階で使うPython環境を作ります。

実行例:

```bash
cd /path/to/qwen36-agent/workspace
./scripts/setup-vision-env.sh
```

内部処理:

- `PROJECT_ROOT/vision-env/` がなければ作成します。
- `torch`、`transformers`、`pillow`、`safetensors`、`sentencepiece` をインストールします。
- `PROJECT_ROOT/models/retrieval/` に検索モデルをキャッシュします。
- 既定の検索モデルは `google/siglip2-base-patch16-224` です。

環境変数:

| 変数 | 意味 |
| --- | --- |
| `PYTHON_BIN` | venv作成に使うPython。既定値は `python3` |
| `RETRIEVAL_MODEL` | 事前取得する検索モデル。既定値は `google/siglip2-base-patch16-224` |

注意:

- この検索モデルは非生成の画像テキスト検索器です。チャットの拒否文は出しません。
- テンプレート側では追加の拒否リスト、NSFW分類器、マスク処理を入れていません。
- 「uncensored」はこのパイプライン上で拒否フィルタを足さないという意味で、学習データが完全に無加工である保証ではありません。

## `start-coder-server.sh`

Qwen Codeから使うコード用モデルを `llama-server` で起動します。

実行例:

```bash
cd /path/to/qwen36-agent/workspace
./scripts/start-coder-server.sh
```

内部処理:

- スクリプト自身の場所から `WORKSPACE_ROOT` と `PROJECT_ROOT` を決めます。
- モデルは `../models/Qwen2.5-Coder-14B-Instruct-abliterated-Q4_K_M.gguf` を使います。
- モデルファイルがない場合は、エラーを出して終了します。
- `llama-server` を次の条件で起動します。
  - alias: `qwen2.5-coder-14b-abliterated-q4km`
  - context: `32768`
  - parallel: `-np 1`
  - GPU layers: `-ngl 99`
  - host: `127.0.0.1`
  - port: `8080`
  - web UIなし

注意:

- このコマンドはサーバーとして起動し続けます。
- 別ターミナルで `start-qwen-code.sh` を実行します。

## `start-qwen-code.sh`

`workspace/` をカレントディレクトリにして Qwen Code を起動します。

実行例:

```bash
cd /path/to/qwen36-agent/workspace
./scripts/start-qwen-code.sh
```

内部処理:

- `WORKSPACE_ROOT` を決めて、そこへ移動します。
- `QWEN_SANDBOX=sandbox-exec` を設定します。
- `SEATBELT_PROFILE=permissive-open` を設定します。
- 最後に `qwen` を実行します。

注意:

- 事前に `start-coder-server.sh` でコード用モデルサーバーを起動しておきます。
- Qwen Code側の設定が `http://127.0.0.1:8080/v1` を見る前提です。

## `start-vl-server.sh`

動画や画像を読むVLモデルを `llama-server` で起動します。

実行例:

```bash
cd /path/to/qwen36-agent/workspace
./scripts/start-vl-server.sh
```

内部処理:

- `PROJECT_ROOT/models/` から次の2ファイルを使います。
  - `Qwen2.5-VL-7B-Instruct-abliterated.Q4_K_M.gguf`
  - `Qwen2.5-VL-7B-Instruct-abliterated.mmproj-f16.gguf`
- モデル本体またはmmprojがない場合は、エラーを出して終了します。
- `llama-server` を次の条件で起動します。
  - alias: `qwen2.5-vl-7b-instruct-abliterated-q4km`
  - mmproj: f16版
  - image min tokens: `1024`
  - context: `16384`
  - parallel: `-np 1`
  - GPU layers: `-ngl 99`
  - host: `127.0.0.1`
  - port: `8081`
  - web UIなし

注意:

- 動画キャプションやイベント探索は、このVLサーバーを使います。
- 再現用 `reproduce.sh` は、必要に応じてこのサーバーを自動起動します。

## `vl-repro-common.sh`

VL検証系の `reproduce.sh` から読み込まれる共通部品です。単体で直接実行するものではありません。

提供している主な処理:

- `VL_ENDPOINT`
  - 既定値は `http://127.0.0.1:8081/v1`。
  - 環境変数で上書きできます。
- `VL_MODEL_ALIAS`
  - 既定値は `qwen2.5-vl-7b-instruct-abliterated-q4km`。
  - 環境変数で上書きできます。
- `require_command`
  - 必要なコマンドが入っているか確認します。
- `vl_endpoint_alive`
  - 指定エンドポイントの `/models` が応答するか確認します。
- `vl_model_available`
  - 指定エンドポイントで期待モデルが出ているか確認します。
- `ensure_vl_server`
  - すでに期待モデルが動いていればそのまま使います。
  - 8081番などに別モデルが動いている場合は、安全のため停止せずにエラーにします。
  - 何も動いていない場合は `start-vl-server.sh` をバックグラウンド起動します。
  - 起動ログは `../logs/vl-server-repro.log` に保存します。
  - 最大180秒待ち、準備できなければログ末尾を表示して終了します。
- `stop_owned_vl_server`
  - `ensure_vl_server` が自分で起動したVLサーバーだけを終了します。
  - もともと動いていたサーバーは止めません。

## `caption-video.py`

ローカル動画から一定間隔でフレームを抽出し、VLモデルで通常キャプションを付けるPythonスクリプトです。

実行例:

```bash
cd /path/to/qwen36-agent/workspace
./scripts/caption-video.py work/free-video-test/big_buck_bunny_5min.mp4 \
  --interval 30 \
  --out-dir outputs/video-captions
```

主な引数:

| 引数 | 既定値 | 意味 |
| --- | --- | --- |
| `video` | 必須 | 入力動画パス |
| `--interval` | `5.0` | 何秒ごとにフレームを切り出すか |
| `--out-dir` | `outputs/video-captions` | `captions.md` と `captions.json` の保存先 |
| `--frame-dir` | 自動決定 | 抽出フレームの保存先 |
| `--endpoint` | `http://127.0.0.1:8081/v1` | OpenAI互換VLエンドポイント |
| `--model` | `qwen2.5-vl-7b-instruct-abliterated-q4km` | VLモデルalias |
| `--language` | `Japanese` | キャプション言語 |

内部処理:

1. 入力動画が存在するか確認します。
2. `ffmpeg` が入っているか確認します。
3. 出力先とフレーム保存先を作成します。
4. 既存の `frame_*.jpg` を消してから、新しくフレームを抽出します。
5. `ffmpeg -vf fps=...` で一定間隔のJPEGフレームを作ります。
6. 各フレームをbase64のdata URLに変換します。
7. `/chat/completions` へ画像つきリクエストを送り、事実ベースの短いキャプションを作ります。
8. 結果を `captions.json` と `captions.md` に保存します。

出力:

- `captions.json`
  - `start_time`
  - `end_time`
  - `frame_path`
  - `caption_ja`
- `captions.md`
  - 人間が読みやすい箇条書きのキャプション

補足:

- `default_frame_dir` は、出力先が `outputs/` 配下なら対応する `work/.../frames` を自動的に選びます。
- このスクリプトは「動画全体の粗い説明」を作る用途です。
- 条件に合う時刻を探索する用途では、`outputs/video-event-search-template/video_event_search.py` を使います。

## 動画イベント探索本体について

`video_event_search.py` は `outputs/video-event-search-template/` に置いています。成果物ごとに `reproduce.sh` と `input.json` を持たせるため、テンプレート本体も `outputs/` 側にあります。

現在の探索方式は `retrieve_verify` です。`scan_interval_seconds` ごとに軽量検索モデルで全体を見て、スコア上位の候補区間だけをQwen VLで検証します。古い `candidate_binary`、`caption_interval_seconds`、`candidate_threshold`、`match_threshold` は使いません。

詳しい使い方は次を見てください。

`outputs/video-event-search-template/README.md`
