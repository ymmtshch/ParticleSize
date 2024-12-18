![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)
![Streamlit](https://img.shields.io/badge/streamlit-%E2%AC%9B-orange)
![Last Commit](https://img.shields.io/github/last-commit/ymmtshch/ParticleSize)
![GitHub Issues](https://img.shields.io/github/issues/ymmtshch/ParticleSize)
![GitHub Stars](https://img.shields.io/github/stars/ymmtshch/ParticleSize?style=social)
![GitHub Forks](https://img.shields.io/github/forks/ymmtshch/ParticleSize?style=social)
![Maintenance](https://img.shields.io/badge/maintenance-active-brightgreen)

# 粒径測定アプリケーション

このアプリは、画像から粒子を検出し、粒径を計測するためのツールです。Streamlitを使用して、インタラクティブなインターフェースを提供します。粒子の重なりを検出し、計測対象外の粒子を自動的に除外する機能も搭載しています。

## 📄 概要

本アプリケーションは以下の機能を備えています：
- 画像からの粒子検出と粒径測定
- 粒子の重なり検出と除外
- インタラクティブに粒子を除外または復帰させる機能
- 粒径データのCSV出力
- 測定結果の可視化（ヒストグラム、箱ひげ図）

## 🚀 アプリの使用方法

Streamlit Cloud上で動作するアプリを直接使用できます。以下のリンクからアクセスしてください：

[粒径測定アプリ（Streamlit Cloud）](https://particlesize-fuegcxbepwtam9wgmcewkq.streamlit.app/)

1. **画像のアップロード**  
   粒径を測定したい画像をアップロードします。（JPEG、PNG形式に対応）

2. **除外粒子の選択**  
   粒子が重なっている場合や除外したい粒子をインタラクティブに選択できます。

3. **結果の保存**  
   粒径データをCSVファイルに保存できます。

## 💻 ローカルでの実行方法

アプリをローカル環境で動作させる場合は以下の手順を実行してください：

### 1. 環境のセットアップ

以下のコマンドで必要なライブラリをインストールしてください。

```bash
pip install streamlit opencv-python-headless numpy pillow matplotlib seaborn
```

### 2. アプリの起動
以下のコマンドでStreamlitアプリを起動します。
```bash
streamlit run app.py
```

### 3. 画像のアップロードと処理
1. アプリ上で画像（JPG/PNG形式）を複数アップロードします。
2. 粒子検出後、除外する粒子を選択できます。
3. 結果をCSVファイルに保存可能です。

### 4. 粒径データの可視化（CSVアップロード）
1. 計測結果のCSVファイルをアップロードします。
2. 粒径分布のヒストグラムを生成。
3. 平均値、標準偏差、変動係数（CV）を自動計算します。

### 主要関数の説明
#### `detect_circles(image)`
画像から粒子を検出する関数。Hough Circle Transformを用いて円形粒子を検出します。

#### `check_overlap(circles)`
粒子間の重なりを判定し、重なっている下の粒子を除外対象とする関数。

#### `draw_circles(image, circles, excluded_indices)`
検出した粒子を画像に描画する関数。除外粒子は赤、計測対象粒子は緑で表示されます。

#### `process_image(image)`
粒子検出、重なり判定、スケールバー検出を行い、計測対象粒子を選別する関数。

### 出力例
#### CSVファイル構造
```csv
Image,Particle Index,Diameter (pixels)
sample1.jpg,0,15.23
sample1.jpg,1,22.45
...
```

### 粒子検出結果
* 緑の円：計測対象粒子
- 赤の円：除外された粒子
+ 粒子サイズ分布のヒストグラム
* 横軸: 粒子サイズ（nm）
- 縦軸: 密度

### 注意事項
* 粒子の重なり判定は、粒子のy座標を基準に、上に位置する粒子を優先的に計測対象とします。
- スケールバーを検出する代わりに、標準PS粒子の計測データを用います。倍率などが異なる場合は、別途、手動でスケール係数を設定してください。

### ライセンス
このプロジェクトのライセンスはMITライセンスです、詳細はLICENSE.txtをご覧ください。
