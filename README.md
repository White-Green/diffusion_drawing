# Diffusion Drawing
[Krita](https://krita.org/) plugin for Drawing Illustration "together with" AI.

画像生成AI「と」絵を描くための[Krita](https://krita.org/)プラグイン

## setup
### ComfyUI（shadow / light 生成用）
- https://github.com/comfyanonymous/ComfyUI からComfyUIをセットアップする
- カスタムノードとして https://github.com/Fannovel16/comfyui_controlnet_aux 及び https://github.com/White-Green/diffusion_drawing_custom_nodes を使えるようにする
- https://civitai.com/models/260267?modelVersionId=403131 のModelとVAE両方、及び https://civitai.com/models/441432 をComfyUIから使えるようにする

### Krita Plugin
[Actions](https://github.com/White-Green/diffusion_drawing/actions?query=branch%3Arewrite%2Fgraph-preview-20261019) の
成功したビルドの Artifacts から `diffusion_drawing-graph-preview-windows-x64-py3.13.zip` を
ダウンロードし、Krita の Python プラグインインポーターで読み込む。
配布対象は Python 3.13 を内蔵した Windows x64 版 Krita。
main のビルドは [Releases](https://github.com/White-Green/diffusion_drawing/releases/latest) にも公開する。

Python の版は Krita の Scripter で `import sys; print(sys.version)` を実行して確認できる。
PC に別途インストールした Python の版ではなく、Krita 内蔵版に合わせる。
旧 `diffusion_drawing-windows-x64.zip` は Python 3.10 専用で、現在は配布を更新しない。
別の Python minor 版では、その版向けに ZIP をビルドする。

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
Strength・Lineart passes は ON のまま変更でき、変更が止まってから再生成する。
まずStrength 0.5・Lineart passes 3で試す。
Lineart passes は1〜6段、既定3段。今回のモデルは決定的なのでSeedは無効。
Transfer や shadow / light 生成を使うときは Gen を OFF にして実行中の推論の完了を待つ。
生成後のログに `Lineart backend: onnxruntime graph: DmlExecutionProvider, CPUExecutionProvider` を表示する。
初回生成には ONNX セッションと GPU の初期化時間がかかる。
ZIP 更新時は Krita を再起動し、Gen を押す前に上書きインポートする。
インポート後にもう一度 Krita を再起動して、新しいランタイムを読み込む。
線画生成には ComfyUI、Python パッケージの追加インストール、モデルのダウンロードは不要。
scribble（カラーラベル 1）と描き途中の線画（カラーラベル 2）を使い、
結果は専用の描画レイヤーに反映する。StrengthとLineart passesをパネルから指定できる。
今回のgraphモデルは下描きから線の位置・接続・幅・塗りを予測し、不透明な線分として描く。
画素値の平均を線画として出力する方式ではなく、線の縁にはアンチエイリアスが残る。
入力を局所的な濃さで正規化してから解像度ごとに縮小し、小さい段の占有予測を次の段へ渡す。
ガウシアンノイズは加えない。Strengthは学習した条件として渡し、0では入力を保つ。
描き途中の線画の不透明な画素は保護する。同じデバイス・入力・設定では決定的に動く。
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
Python は配布先 Krita の minor 版と一致させる（CI・以下の手順は 3.13 x64）。

```powershell
git submodule update --init --recursive
uv build --project lineartgen/packages/lineartgen-runtime --wheel --out-dir wheels
uv export --project lineartgen/packages/lineartgen-runtime --frozen --extra directml --no-dev --no-emit-project --output-file runtime-requirements.txt
python -m pip download --only-binary=:all: --dest wheels --require-hashes -r runtime-requirements.txt
python -m pip install PyQt5==5.15.11
python -m unittest discover -s tests -v
python scripts/package.py --wheel-dir wheels --provider directml --output dist/diffusion_drawing-graph-preview-windows-x64-py3.13.zip
```

uv、対象 Python と pip が必要。`wheels/` は空のディレクトリから始める。
`package.py` は wheelhouse から依存関係込みで一時領域へインストールし、
モデル推論を実行してから ZIP 化する。ONNX、NumPy、ONNX Runtime と各ライセンスを同梱する。
通常のDirectMLビルドではDirectMLでの実推論を必須とする。GPUを使えないビルドホストでは
`--allow-cpu-fallback` を指定すると、DirectML版の同梱を確認したうえでCPUで推論検証できる。
この指定は検証時だけに適用し、配布プラグインは引き続きDirectMLを優先する。
ランタイム自体は pure Python wheel だが、NumPy と ORT は OS・CPU・Python minor に依存する。
ZIP 内の `runtime.json` に対象版を記録し、不一致はモデル読み込み前に通知する。

Linux の CPU パッケージでは export の extra と package の provider を `cpu` に変える。
Windows DirectML の依存解決・推論確認は Windows 上で実行する。
CPU と DirectML を同じ環境へ両方インストールしない（同じ import 名を使う）。

runtime 0.3.0 は22,725パラメータのgraphモデルを使う。
16,200更新の重みから84,000更新追加し、累計100,200更新のEMAを同梱した。
学習更新数、checkpointと6個のONNXのSHA256は
`lineartgen_runtime/assets/graph-model.json`、詳しい出自は`model-source.json`に記録する。
合成下描きだけで学習した試用版で、実際の絵での品質はまだ評価が必要。
一部の幅や曖昧な重ね描きでは、余分な線・点・端の閉ループや局所的な太りが残る。
新しい寸法のベタ塗りが穴だらけになる、または中心線だけになる失敗もあり、塗りの復元には適さない。
未使用16例の実出力と失敗例は、submodule内の
[graph試用版の評価](lineartgen/docs/graph-preview-20261019.md)に保存した。
旧UNetとは構造・学習目的が異なり、出力も変わる。
6個のONNX、manifest、独立したJAX参照出力を同じcheckpointから作り直してから再ビルドする。

GitHub Actions は main・`feat/**`・`rewrite/**` への push、PR、手動実行で
Python 3.13 用の Windows ZIP をビルドし、Python 3.13 で
同梱された依存関係だけで推論できること、DirectML版の同梱、
6段すべての独立JAX参照との一致、パディング・再帰・保護領域・入力検証を確認する。
CIは `--allow-cpu-fallback` を使う。ランナーでDirectMLを初期化できない場合は
CPUで全推論検証を行い、その旨をログに残す。GPU実機の動作確認とは区別する。
Artifacts からの再ダウンロード後、従来通り Krita 5.2.9 の公式インポーターでも検証する。
ローカルでは次を使う（Qt が必要、Krita GUI は不要）。

```powershell
python scripts/check_krita_import.py dist/diffusion_drawing-graph-preview-windows-x64-py3.13.zip
```

main への push だけが Release を作る。作業ブランチでは Release を作らない。
この rewrite のローカル検証結果は lineartgen の `docs/validation.md` に記録する。

### lineartgen の更新

1. lineartgen 側で変更を commit して **先に push** する。
2. このリポジトリの `lineartgen/` でその commit を checkout する。
3. 親リポジトリで `git add lineartgen` して commit・push する。

CI は親リポジトリに記録された commit をビルドする。lineartgen 側の変更が
未コミットまたは未 push の状態では GitHub Actions から利用できない。
