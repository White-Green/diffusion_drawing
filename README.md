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
インポーターで読み込む。Windows x64、CPython 3.10 以降を組み込んだ Krita が対象。

lineart の Gen は、同梱された lineartgen のモデルを wgpu バックエンドで推論する。
Windows では DirectX 12 を使い、利用可能な高性能 GPU を優先して選択する。
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
生成後のログに `Lineart backend: wgpu: デバイス名 (種別, Dx12)` を表示する。
初回生成には GPU 初期化とシェーダーのコンパイル時間がかかる。
ZIP 更新時は Krita を再起動し、Gen を押す前に上書きインポートする。
インポート後にもう一度 Krita を再起動して、新しい拡張を読み込む。
線画生成には ComfyUI、Python パッケージの追加インストール、モデルのダウンロードは不要。
scribble（カラーラベル 1）と描き途中の線画（カラーラベル 2）を使い、
結果は専用の描画レイヤーに反映する。Strength、Denoise steps（1〜20）、Seed は
パネルから指定できる。同じデバイス上では、同じ入力・設定・Seed で再現できる。
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

このリポジトリは `lineartgen/` に Rust ソースを submodule として保持する。
通常の利用者は配布 ZIP だけでよく、以下は開発者向けの手順。

```powershell
git clone --recurse-submodules https://github.com/White-Green/diffusion_drawing.git
cd diffusion_drawing
python -m pip install "maturin>=1.15,<2"
python -m maturin build --release --locked --no-default-features --features wgpu --target x86_64-pc-windows-msvc --manifest-path lineartgen/crates/lineartgen-native/Cargo.toml --out wheels
python -m unittest discover -s tests -v
python scripts/package.py --wheel-dir wheels --output dist/diffusion_drawing-windows-x64.zip
```

64-bit CPython 3.10 以降、Rust、および Visual Studio の C++ ビルドツールが必要。
既存 checkout では先に `git submodule update --init --recursive` を実行する。
`wheels/` には対象環境向けの lineartgen-native wheel を1つだけ置く。
`package.py` は一時ディレクトリへ拡張をインストールして実モデルの推論を検証し、
ZIP に同梱する。Rust ソースやビルドキャッシュ、`.git` は ZIP に含めない。

GitHub Actions は main・`feat/**` への push、pull request、手動実行で
Windows x64 のビルドとテストを行い、Artifacts に
`diffusion_drawing-windows-x64.zip` を保存する。
ダウンロードした ZIP をそのまま Krita にインポートできる。
CI はアップロード後の ZIP を再ダウンロードして同一性を確認し、
Krita 5.2.9 の公式インポーターで全ファイルがインストールされることも検証する。
CI の推論テストでは wgpu を使っていることと、パディング・再帰処理・固定 Seed を検証する。
CI ランナーでソフトウェアアダプターが選ばれた場合、実 GPU での動作・速度は別途確認する。
この検証ではコミットと SHA-256 を固定したインポーターを GitHub から取得する。
ローカルでも `python scripts/check_krita_import.py dist/diffusion_drawing-windows-x64.zip`
で確認できる（Krita の GUI は不要）。
main への push のみ、ビルド成功後に GitHub Release も作成する。
手動実行や作業ブランチのビルドでは Release は作成しない。

lineartgen は公開リポジトリとして、通常の recursive submodule checkout で取得する。
追加の Deploy Key や Actions Secret は不要。

### lineartgen の更新

1. lineartgen 側で変更を commit して **先に push** する。
2. このリポジトリの `lineartgen/` でその commit を checkout する。
3. 親リポジトリで `git add lineartgen` して commit・push する。

CI は親リポジトリに記録された commit をビルドする。lineartgen 側の変更が
未コミットまたは未 push の状態では GitHub Actions から利用できない。
