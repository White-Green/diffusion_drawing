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

lineart の Gen は、同梱された lineartgen のモデルで CPU 推論する。
線画生成には ComfyUI、Python パッケージの追加インストール、モデルのダウンロードは不要。
scribble（カラーラベル 1）と描き途中の線画（カラーラベル 2）を使い、
結果は専用の描画レイヤーに反映する。Strength、Denoise steps（1〜20）、Seed は
パネルから指定できる。同じ入力・設定・Seed なら同じ結果になる。
画像サイズは各辺 16 ピクセル以上で、2 のべき乗以外のサイズにも対応する。

入力は元のドキュメントを複製して 8-bit sRGB に変換し、BGRA バイト列として渡す。
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
python -m maturin build --release --locked --target x86_64-pc-windows-msvc --manifest-path lineartgen/crates/lineartgen-native/Cargo.toml --out wheels
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
`diffusion-drawing-windows-x64` を保存する。
main への push のみ、ビルド成功後に GitHub Release も作成する。
手動実行や作業ブランチのビルドでは Release は作成しない。

### lineartgen の更新

1. lineartgen 側で変更を commit して **先に push** する。
2. このリポジトリの `lineartgen/` でその commit を checkout する。
3. 親リポジトリで `git add lineartgen` して commit・push する。

CI は親リポジトリに記録された commit をビルドする。lineartgen 側の変更が
未コミットまたは未 push の状態では GitHub Actions から利用できない。
