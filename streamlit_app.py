### 粒径検出 ###

import streamlit as st
import cv2
import numpy as np
import csv
from PIL import Image
import tempfile

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

output_csv = st.sidebar.text_input("結果のCSVファイル名", "output_results.csv")

# ファイルアップロード
uploaded_files = st.file_uploader("JPG/PNG画像を複数アップロードしてください", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        # アップロードされたファイルを一時保存して処理
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_image_path = temp_file.name

        # 画像を読み込む
        original_image = cv2.imread(temp_image_path)
        st.subheader(f"検出結果: {uploaded_file.name}")
        processed_image, detected_circles = detect_circles(original_image)

        if detected_circles is not None:
            st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="元画像")

            # 粒子検出結果を描画
            annotated_image = draw_circles(processed_image, detected_circles, excluded_indices)
            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="粒子検出後")

            # 粒子番号を使った選択/解除
            selected_particle = st.text_input(f"{uploaded_file.name} - 粒子番号を入力してください (例: 0,1,2):", "")
            if st.button("選択を更新", key=uploaded_file.name):
                if selected_particle:
                    try:
                        indices = [int(i) for i in selected_particle.split(",")]
                        for i in indices:
                            if i < len(detected_circles):
                                if i in excluded_indices:
                                    excluded_indices.remove(i)
                                else:
                                    excluded_indices.append(i)
                        st.success("選択が更新されました。")
                    except ValueError:
                        st.error("入力が正しくありません。数字をカンマ区切りで入力してください。")

            # 更新後の画像を再描画
            updated_image = draw_circles(processed_image, detected_circles, excluded_indices)
            st.image(cv2.cvtColor(updated_image, cv2.COLOR_BGR2RGB), caption="更新後の粒子検出結果")

            # 保存用データ収集
            diameters = [2 * r for i, (_, _, r) in enumerate(detected_circles) if i not in excluded_indices]
            for idx, diameter in enumerate(diameters):
                results.append([uploaded_file.name, idx, f"{diameter:.2f}"])
        else:
            st.warning(f"{uploaded_file.name}: 円が検出されませんでした。")

    # 結果をCSVに保存
    if results:
        if st.button("結果を保存"):
            with open(output_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Image", "Particle Index", "Diameter (pixels)"])
                writer.writerows(results)
            st.success(f"結果が {output_csv} に保存されました。")
else:
    st.info("JPGまたはPNG画像をアップロードしてください。")





### グラフ化 ###
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
