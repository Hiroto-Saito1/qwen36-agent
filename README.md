# qwen36-agent workspace

この `workspace/` は、Qwen Code とローカルVL検証の実作業ディレクトリです。

`qwen36-agent` 直下はモデル・ログ・Python環境を置く実行環境ルート、`workspace/` はREADME、スクリプト、成果物、一時作業を置く作業ルートとして使います。

## まず使うもの

### コード用モデルサーバー

```bash
cd /path/to/qwen36-agent/workspace
./scripts/start-coder-server.sh
```

別ターミナルで Qwen Code を起動します。

```bash
cd /path/to/qwen36-agent/workspace
./scripts/start-qwen-code.sh
```

### 動画/VL用モデルサーバー

```bash
cd /path/to/qwen36-agent/workspace
./scripts/start-vl-server.sh
```

### 動画イベント探索の軽量検索環境

```bash
cd /path/to/qwen36-agent/workspace
./scripts/setup-vision-env.sh
```

`vision-env/` と検索モデルのキャッシュは `workspace/` の外に作ります。巨大な依存関係やモデルをGit管理しないためです。

### Big Buck Bunny 5分動画のイベント探索を再現

```bash
cd /path/to/qwen36-agent/workspace/outputs/free-video-test/bbb-5min-event-search
./reproduce.sh
```

結果は同じディレクトリの `output.json` と `event-search.md` を見ます。機械処理用の結果JSONは `output.json` だけです。実行中も評価ごとに更新されます。古い `event-search.json` は使いません。
1時間程度の動画では、テンプレートはVLへ送る画像を既定で長辺640pxへ縮小します。画質を優先したい場合は入力JSONの `search.verification_image_max_edge_pixels` を `960` や `1280` へ上げ、速度を優先したい場合は `480` などへ下げます。

## ディレクトリ構成

```text
qwen36-agent
├─ models/          # GGUFモデル。workspace外なのでGit管理しない
├─ logs/            # llama-serverなどの実行ログ
├─ hf-env/          # Hugging Face取得用のPython環境
├─ vision-env/      # 動画イベント探索の軽量検索モデル用Python環境
└─ workspace/
   ├─ QWEN.md       # Qwen Code向けの作業ルール
   ├─ scripts/      # サーバー起動・動画キャプション用スクリプト
   ├─ work/         # 一時ファイル、動画素材、抽出フレーム、キャッシュ
   └─ outputs/      # 残したい成果物と再現用テンプレート
```

## 詳細ドキュメント

- `scripts/README.md`: 各シェルスクリプトと `caption-video.py` の用途、引数、内部処理。
- `work/README.md`: `work/` の役割、キャッシュ、削除してよいもの。
- `outputs/README.md`: 成果物ディレクトリの一覧。
- `outputs/video-event-search-template/README.md`: JSON駆動の動画イベント探索テンプレート。

## 運用ルール

- 残したい結果は `outputs/` に置きます。
- 動画素材、抽出フレーム、途中キャッシュは `work/` に置きます。
- `models/` は `workspace/` の外に置き、巨大なGGUFをGit管理しません。
- 動画イベント探索では、軽量検索モデルで候補区間を拾い、Qwen VLで候補区間だけを検証します。
- Qwen VL検証は、まず対象フレーム1枚で一次判定し、関係ありそうな時刻だけ前後を含む3枚で確定します。
- 入力JSONと同じ出力ディレクトリに `output.json`、`event-search.md`、`search-trace.json` が生成されます。
- 同じ出力ディレクトリで再実行すると、古い生成結果は上書きされます。`input.json`、`reproduce.sh`、README、未知の手元ファイルは残します。
