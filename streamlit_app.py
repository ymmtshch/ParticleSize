### 粒子サイズ計測 ###

import streamlit as st
import cv2
import numpy as np
import csv
from PIL import Image
import tempfile
import io

# グローバル変数
excluded_indices = []
circles = []

# 画像処理
def detect_circles(image):
    global circles
    height, _ = image.shape[:2]
    cropped_image = image[0:int(height * 7 / 8), :]
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    detected_circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=30, minRadius=5, maxRadius=50
    )
    if detected_circles is not None:
        circles = detected_circles[0, :]  # 円情報を保持
        return cropped_image, circles
    else:
        circles = []
        return cropped_image, None

def draw_circles(image, circles, excluded_indices):
    output_image = image.copy()
    for i, (x, y, r) in enumerate(circles):
        color = (0, 0, 255) if i in excluded_indices else (0, 255, 0)
        cv2.circle(output_image, (int(x), int(y)), int(r), color, 2)
        cv2.putText(output_image, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    return output_image

# Streamlit アプリ
st.title("粒子検出・サイズ測定アプリ")
st.sidebar.header("設定")

# 複数画像のアップロード
uploaded_files = st.file_uploader("JPG/PNG画像をアップロードしてください", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# 結果の CSV を保持するためのリスト
csv_data = []

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        # アップロードされたファイルを一時保存して処理
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_image_path = temp_file.name

        # 画像を読み込む
        original_image = cv2.imread(temp_image_path)
        st.subheader(f"{uploaded_file.name} - 検出結果")
        processed_image, detected_circles = detect_circles(original_image)

        if detected_circles is not None and len(detected_circles) > 0:
            st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="元画像")

            # 画像に粒子番号を表示した画像を作成
            annotated_image = draw_circles(processed_image, detected_circles, excluded_indices)
            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="粒子番号付き画像")

            # 粒子番号の選択 UI (複数選択可能)
            particle_count = len(detected_circles)

            if particle_count > 0:
                # `multiselect`を使用して複数選択、キーで一意に識別
                selected_particles = st.multiselect(
                    "選択または解除したい粒子を選択してください", 
                    options=list(range(particle_count)), 
                    default=excluded_indices,  # 現在除外されている粒子をデフォルト選択
                    key=f"select_particles_{idx}"  # 各画像に一意のキーを設定
                )

                # `excluded_indices` を選択した粒子に合わせて更新
                excluded_indices = selected_particles

                # 更新された画像の表示
                st.image(cv2.cvtColor(draw_circles(processed_image, detected_circles, excluded_indices), cv2.COLOR_BGR2RGB),
                         caption="粒子選択後")

                # 除外された粒子をCSVデータに反映（除外された粒子はCSVに追加しない）
                for i, (x, y, r) in enumerate(detected_circles):
                    # `excluded_indices` に含まれるインデックスを除外してCSVに追加
                    if i not in excluded_indices:
                        diameter = 2 * r
                        csv_data.append([uploaded_file.name, i, f"{diameter:.2f}"])

        else:
            st.warning(f"{uploaded_file.name} では円が検出されませんでした。")
else:
    st.info("JPGまたはPNG画像をアップロードしてください。")

# CSVダウンロードボタン
if csv_data:
    # CSVデータを文字列に変換
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Image", "Particle Index", "Diameter (pixels)"])  # ヘッダー行
    writer.writerows(csv_data)

    # ダウンロード用のCSVファイルを提供
    st.download_button(
        label="結果をCSVとしてダウンロード",
        data=output.getvalue(),
        file_name="output_results.csv",
        mime="text/csv"
    )

### 粒子サイズグラフ化 ###
import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import io

# スタイル設定
sns.set(style="whitegrid")

# CSVファイルのアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

if uploaded_file:
    # アップロードされたCSVファイルを読み込む
    data = pd.read_csv(uploaded_file)

    # "ps"を含む行と含まない行に分割
    ps = data[data['Image'].str.contains("ps")]
    others = data[~data['Image'].str.contains("ps")]

    # "Image"列を "sample" と "num" に分割
    others[['sample', 'num']] = others['Image'].str.split('-', expand=True)

    # スケーリングに使う定数
    atai = 1.984375

    # PSデータの直径にスケーリングを適用し、平均を計算
    ps['scaled'] = ps['Diameter (pixels)'] * atai
    scaling_factor = ps['scaled'].mean(skipna=True)

    # スケーリング値を計算
    scaling = 152 / scaling_factor

    # 最終スケーリング係数を計算
    coefficient = scaling * atai

    # 全データにスケーリングを適用
    others['scaled'] = others['Diameter (pixels)'] * coefficient

    # 結果の表示
    st.subheader("スケーリング結果")
    st.write(others[['sample', 'num', 'scaled']])

    # 可視化
    st.subheader("スケーリング後の粒子サイズの分布")
    g = sns.FacetGrid(others, col="sample", col_wrap=1, height=4, aspect=1.5)
    g.map_dataframe(sns.histplot, x="scaled", binwidth=2, kde=False, stat="density", color="skyblue")
    g.set_axis_labels("PID size [nm]", "Density")
    g.set_titles("{col_name}")
    st.pyplot(g)

    # 集計とCVの計算
    result = others.groupby('sample').agg(
        平均値=('scaled', 'mean'),
        標準偏差=('scaled', 'std'),
    )

    # CVの計算と結果の結合
    result['CV'] = result['標準偏差'] / result['平均値'] * 100

    # 結果の表示
    st.subheader("集計結果（平均値、標準偏差、CV）")
    st.write(result)

else:
    st.info("CSVファイルをアップロードしてください。")
