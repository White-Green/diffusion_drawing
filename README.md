# Diffusion Drawing
[Krita](https://krita.org/) plugin for Drawing Illustration "together with" AI.

画像生成AI「と」絵を描くための[Krita](https://krita.org/)プラグイン

## setup
### ComfyUI（shadow / light 生成用）
- https://github.com/comfyanonymous/ComfyUI からComfyUIをセットアップする
- カスタムノードとして https://github.com/Fannovel16/comfyui_controlnet_aux 及び https://github.com/White-Green/diffusion_drawing_custom_nodes を使えるようにする
- https://civitai.com/models/260267?modelVersionId=403131 のModelとVAE両方、及び https://civitai.com/models/441432 をComfyUIから使えるようにする

### Krita Plugin
[Releases](https://github.com/White-Green/diffusion_drawing/releases/latest) から
`diffusion_drawing-windows-x64.zip` をダウンロードし、Krita の Python プラグイン
インポーターで読み込む。Windows x64、CPython 3.10 を組み込んだ Krita が対象。別の Python minor 版ではその版向けに ZIP をビルドする。

lineart の Gen は、同梱された ONNX モデルを ONNX Runtime で推論する。
Windows ZIP では DirectML（DirectX 12、既定の GPU）を優先し、CPU provider も利用できる。
学習コードは JAX / Flax NNX で、Krita には JAX を同梱しない。
lineart の **Gen はトグル**で、ON にすると一度生成し、その後は描画が止まってから更新する。
ペン／マウスを離して約 250ms 待ち、続けて描き始めた場合は待ち直す。
押したまま停止している間や、Krita がストロークを処理している間は入力取得・生成を開始しない。
生成中に描き足した場合は古い結果を捨て、描画が止まってから最新の入力で生成する。
入力と設定が同じなら再推論せず、何も操作していない間は画像の取り出しも行わない。
Undo / Redo やレイヤー操作でも入力を確認する。生成結果の再描画は次の生成を起動しない。
Gen を OFF にすると更新を止める。実行中の推論は完了まで待つが、その結果は適用しない。
ドキュメント切り替え・クローズ・エラー時にも自動的に OFF になる。
Strength・Denoise steps・Seed は ON のまま変更でき、変更が止まってから再生成する。
Transfer や shadow / light 生成を使うときは Gen を OFF にして実行中の推論の完了を待つ。
生成後のログに `Lineart backend: onnxruntime: DmlExecutionProvider, CPUExecutionProvider` を表示する。
初回生成には ONNX セッションと GPU の初期化時間がかかる。
ZIP 更新時は Krita を再起動し、Gen を押す前に上書きインポートする。
インポート後にもう一度 Krita を再起動して、新しいランタイムを読み込む。
線画生成には ComfyUI、Python パッケージの追加インストール、モデルのダウンロードは不要。
scribble（カラーラベル 1）と描き途中の線画（カラーラベル 2）を使い、
結果は専用の描画レイヤーに反映する。Strength、Denoise steps（1〜20）、Seed は
パネルから指定できる。同じデバイス上では、同じ入力・設定・Seed で再現できる。
runtime 0.2.1 以降は画像とノイズ強度マップを各解像度へ縮小してから各段にノイズを加える。
denoise の各ステップは段ごとの状態を引き継ぐ。多段推論の結果は旧 runtime から変わる。
GPU やドライバーが異なる場合は浮動小数点演算の差が生じることがある。
画像サイズは各辺 16 ピクセル以上で、2 のべき乗以外のサイズにも対応する。

入力は元のドキュメントを複製して 8-bit sRGB に変換し、BGRA バイト列として渡す。
下描き（カラーラベル 1）は合成後の透明度だけを使う。透明な部分は白、不透明な部分は黒、半透明な部分は濃淡として扱い、描画色は参照しない。
白く塗った不透明な部分も黒い線として扱う。レイヤーの不透明度とマスクは合成結果に反映される。
描き途中の線画（カラーラベル 2）は RGB の明るさと透明度を使って下描きに重ねる。
モデルの読み込みと推論をワーカースレッドで実行し、Krita のレイヤー操作は UI スレッドで行う。
Krita の整数 RGBA のチャンネル順序については
[Node API](https://api.kde.org/legacy/krita/html/classNode.html) を参照。

## Windows x64 ビルド

`lineartgen/` submodule の JAX rewrite を使う。学習とモデル変更の手順は
その README を参照。配布に必要なのは `packages/lineartgen-runtime` のみ。
Python は配布先 Krita の minor 版と一致させる（CI は 3.10 x64）。

```powershell
git submodule update --init --recursive
uv build --project lineartgen/packages/lineartgen-runtime --wheel --out-dir wheels
uv export --project lineartgen/packages/lineartgen-runtime --frozen --extra directml --no-dev --no-emit-project --output-file runtime-requirements.txt
python -m pip download --only-binary=:all: --dest wheels --require-hashes -r runtime-requirements.txt
python -m pip install PyQt5==5.15.11
python -m unittest discover -s tests -v
python scripts/package.py --wheel-dir wheels --provider directml --output dist/diffusion_drawing-windows-x64.zip
```

uv、対象 Python と pip が必要。`wheels/` は空のディレクトリから始める。
`package.py` は wheelhouse から依存関係込みで一時領域へインストールし、
モデル推論を実行してから ZIP 化する。ONNX、NumPy、ONNX Runtime と各ライセンスを同梱する。
ランタイム自体は pure Python wheel だが、NumPy と ORT は OS・CPU・Python minor に依存する。
ZIP 内の `runtime.json` に対象版を記録し、不一致はモデル読み込み前に通知する。

Linux の CPU パッケージでは export の extra と package の provider を `cpu` に変える。
Windows DirectML の依存解決・推論確認は Windows 上で実行する。
Python 3.10 では実際に対応 wheel のある ORT CPU 1.23.2 / DirectML 1.22.0 を固定している。
CPU と DirectML を同じ環境へ両方インストールしない（同じ import 名を使う）。

runtime 0.2.2 は、JAXで100,000更新学習した最大候補の基準UNetを同梱する。
channels `[4,10,20,40]`、各block 3 conv、126,765パラメータ。
実験の主seed 42（data seed 47）のEMA重み `seed42_baseline` を使い、
出自とSHA256を `lineartgen_runtime/assets/model-source.json` に記録している。
品質の評価結果は [lineartgenの最終報告](lineartgen/docs/completion-results-20260930.md)を参照。
別の学習済みモデルに変更する場合は lineartgen の export コマンドで
`packages/lineartgen-runtime/src/lineartgen_runtime/assets/model.onnx` に書き出して再ビルドする。
モデルを変更した場合は `model-source.json` の出自・hashも更新する。
入出力・操作は維持するが、同梱重みが変わるため旧版と同じSeedでも生成結果は変わる。

GitHub Actions は main・`feat/**`・`rewrite/**` への push、PR、手動実行で
Windows ZIP をビルドし、同梱された依存関係だけで推論できること、DirectML provider、
パディング・再帰・固定 Seed・入力検証を確認する。
Artifacts からの再ダウンロード後、従来通り Krita 5.2.9 の公式インポーターでも検証する。
ローカルでは次を使う（Qt が必要、Krita GUI は不要）。

```powershell
python scripts/check_krita_import.py dist/diffusion_drawing-windows-x64.zip
```

main への push だけが Release を作る。作業ブランチでは Release を作らない。
この rewrite のローカル検証結果は lineartgen の `docs/validation.md` に記録する。

### lineartgen の更新

1. lineartgen 側で変更を commit して **先に push** する。
2. このリポジトリの `lineartgen/` でその commit を checkout する。
3. 親リポジトリで `git add lineartgen` して commit・push する。

CI は親リポジトリに記録された commit をビルドする。lineartgen 側の変更が
未コミットまたは未 push の状態では GitHub Actions から利用できない。
