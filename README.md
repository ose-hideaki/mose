# microevent-tools

重力マイクロレンズの最適化・MCMC・結果比較を、イベント単位の設定ファイルから実行するツール集。
個人サーバーに依存していたスクリプトを、共通のデータ読み込み・設定・実行記録を持つパッケージに整理した。

## 対応する処理

- PSPL / FSPL、それぞれのannual parallax付きモデルの最適化と局所NUTS
- 外部autogridの探索結果の取り込み、binary多点初期化最適化
- 改善度判定、static binary / annual parallax binaryの局所NUTSと候補選択
- 必要に応じたpointモデル再フィット、比較表・ライトカーブ・caustic図
- 同一条件の完了工程の再利用、工程別ログ、Grid Engineへの投入
- 独立したVBMicrolensing + emceeフィッター

**実行完了は、MCMCの収束や全ての縮退解を回収したことを意味しない。**
特にNUTSは最適化解周辺の有限範囲で実行する。物理的な事前分布を設定した大域的事後分布とは区別する。

## インストール

Python 3.11以上、C++コンパイラー、CMakeを使用する。既存の解析環境と分けた仮想環境を推奨する。
JAXバックエンドは、同梱ソースとVBMicrolensing内部APIの組み合わせを固定している。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[jax,native,test]'
python -m pip install --no-build-isolation --no-deps -e vendor/autodiff-vbm
autodiff-vbm-build-native
```

MacとLinuxのCPUで使用する構成。コンパイル済みライブラリーは同梱しない。
VBMicrolensing 5.5以外のバージョンへ変更する場合は、nativeビルドと数値一致を再検証する。
同梱第三者コードのライセンス・由来は [vendor/autodiff-vbm/NOTICE](vendor/autodiff-vbm/NOTICE) を参照。

## 合成データで試す

```bash
microevent plan examples/point.toml
microevent doctor examples/point.toml
microevent run examples/point.toml
microevent status examples/point.toml
```

例には実観測データを含めていない。2系列のPSPL合成データを使い、4モデルを比較する。
`examples/demo/analysis/2026/DEMO/v1/` にログ・状態・結果を保存する。
例のMCMCは500 warmup、1000 draws、2 chains。動作確認だけなら別設定・別versionに小さい値を指定する。

同じコマンドを再実行すると、入力・コード・出力のハッシュが一致した完了工程を再利用する。
実行途中で停止したMCMCのチェーンを延長する機能ではなく、未完了工程を最初から実行し直す。
条件変更時は**新しいversion**を使用する。`--force` は既存の追跡済み結果を上書きし得るため、通常は使用しない。

## 自分のイベントを用意する

```text
project/
  data/2026/EVENT/RADec
  data/2026/EVENT/v1/list
  data/2026/EVENT/v1/EVENT-0
  data/2026/EVENT/v1/obs/band.flux
  analysis/                         # 自動作成
```

`examples/point.toml` をコピーし、`event.name/year/version/project_root` を設定する。
`project_root` などの相対パスは、設定ファイルのある場所を基準に解決する。
`list` は `obs/band.flux OBSERVATORY BAND` のように、1行1データセット。
測光ファイルの先頭3列は `time flux sigma`。sigmaが正の有限行だけを使用する。
**時間は全てJD/HJD − 2450000の日単位に揃える。自動変換しない。**
初期値は同梱例の`#BEST`形式、RADecは同梱例の赤経・赤緯形式を使用する。
角度・parallaxの規約は [docs/scientific-scope.md](docs/scientific-scope.md) を必ず確認する。

## binary解析

`examples/full.toml` が設定例。autogrid本体・増光マップは別途用意する。
既存の探索結果を指定するか、`grid.command` に外部プログラムの引数配列を指定する。
完全性を`grid.expected`または`grid.parameters`で検査し、不完全な結果を黙って解析しない。
外部プログラムが非同期にジョブを投入する場合、その終了後にパイプラインを再実行する。
外部コマンド側には重複投入を避ける設定を使用する。

詳細は [docs/workflow.md](docs/workflow.md) を参照。

## Grid Engineで実行

設定の`[scheduler]`でqueueとparallel environmentを自分の環境に合わせる。
既定値は2 slots、nice 19。内部数値計算のスレッド数もslotsに合わせる。

```bash
microevent plan event.toml
microevent submit event.toml
```

`submit`を実行した場合にのみqsubする。設定で使用するPython環境を指定できる。
外部autogridが追加投入するジョブの資源は、外部コマンド側で別途制限する。
ログはversion別の`logs/`、投入用スクリプトは`scheduler/`に保存する。

## テスト

```bash
python -m unittest discover -s tests -v
```

設定・データ選択・再実行・資源設定、native増光・短いemcee、JAX増光と勾配・短いparallax NUTSを検査する。
JAXまたはnative依存がない場合は対応する試験をskipする。短いチェーンの試験は収束試験ではない。

## 由来と公開範囲

元スクリプトとの対応と既知の修正は [docs/migration.md](docs/migration.md)、元ファイルのハッシュは
[docs/source-provenance.json](docs/source-provenance.json) に記録した。
実データ・認証情報・サーバーの個人パス・結果・既存のバイナリーは配布対象に含めない。
主プロジェクトのライセンスは未設定。第三者コードのライセンスは各ディレクトリのものを維持する。

検証した環境・範囲は [docs/validation.md](docs/validation.md) に記録している。
