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

# 複数画像のアップロード
uploaded_files = st.file_uploader("JPG/PNG画像をアップロードしてください", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

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
                # `multiselect`を使用して複数選択
                selected_particles = st.multiselect("選択または解除したい粒子を選択してください", options=list(range(particle_count)), default=[])

                # `excluded_indices` を選択した粒子に合わせて更新
                excluded_indices = selected_particles

                # 更新された画像の表示
                st.image(cv2.cvtColor(draw_circles(processed_image, detected_circles, excluded_indices), cv2.COLOR_BGR2RGB),
                         caption="粒子選択後")

                # 保存オプション
                if st.button(f"{uploaded_file.name} の結果を保存"):
                    diameters = [2 * r for i, (_, _, r) in enumerate(detected_circles) if i not in excluded_indices]
                    with open(output_csv, mode='a', newline='') as file:  # 追記モードに変更
                        writer = csv.writer(file)
                        if file.tell() == 0:  # ファイルが空の場合、ヘッダーを書き込む
                            writer.writerow(["Image", "Particle Index", "Diameter (pixels)"])
                        for idx, diameter in enumerate(diameters):
                            writer.writerow([uploaded_file.name, idx, f"{diameter:.2f}"])
                    st.success(f"{uploaded_file.name} の結果が {output_csv} に保存されました。")
        else:
            st.warning(f"{uploaded_file.name} では円が検出されませんでした。")
else:
    st.info("JPGまたはPNG画像をアップロードしてください。")
