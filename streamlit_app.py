import streamlit as st
import cv2
import numpy as np
import os
import csv
from PIL import Image

# グローバル変数
excluded_indices = []
circles = []

# フォルダとファイル操作
def get_jpg_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

# 画像処理
def detect_circles(image_path):
    global circles
    image = cv2.imread(image_path)
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

# Streamlitでクリックをエミュレート
def display_image_with_click(image, width=700):
    st.image(image, caption="クリックして粒子を選択/解除してください", use_column_width=False, width=width)
    coords = st.text_input("クリック座標 (例: 100,200):", "")
    if coords:
        try:
            x, y = map(int, coords.split(","))
            return {"x": x, "y": y}
        except ValueError:
            st.error("座標の形式が正しくありません。x,y の形式で入力してください。")
    return None

# Streamlit アプリ
st.title("粒子検出・サイズ測定アプリ")
st.sidebar.header("設定")

folder_path = st.sidebar.text_input("画像フォルダのパス", ".")
output_csv = st.sidebar.text_input("結果のCSVファイル名", "output_results.csv")

# フォルダ内の画像リストを取得
if folder_path and os.path.exists(folder_path):
    image_files = get_jpg_files(folder_path)
else:
    image_files = []

if image_files:
    selected_image = st.sidebar.selectbox("画像を選択", image_files)
    image_path = os.path.join(folder_path, selected_image)

    # 画像を処理
    st.subheader("検出結果")
    original_image, detected_circles = detect_circles(image_path)
    if detected_circles is not None:
        st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="元画像")

        # Streamlitでのクリック処理
        annotated_image = draw_circles(original_image, detected_circles, excluded_indices)
        coords = display_image_with_click(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

        if coords:
            for i, (cx, cy, r) in enumerate(detected_circles):
                # 円の内側クリックでトグル処理
                if (cx - coords["x"])**2 + (cy - coords["y"])**2 <= r**2:
                    if i in excluded_indices:
                        excluded_indices.remove(i)
                    else:
                        excluded_indices.append(i)

        st.image(cv2.cvtColor(draw_circles(original_image, detected_circles, excluded_indices), cv2.COLOR_BGR2RGB),
                 caption="粒子検出後")

        # 保存オプション
        if st.button("結果を保存"):
            diameters = [2 * r for i, (_, _, r) in enumerate(detected_circles) if i not in excluded_indices]
            with open(output_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Image", "Particle Index", "Diameter (pixels)"])
                for idx, diameter in enumerate(diameters):
                    writer.writerow([selected_image, idx, f"{diameter:.2f}"])
            st.success(f"結果が {output_csv} に保存されました。")
    else:
        st.warning("円が検出されませんでした。")
else:
    st.warning("有効な画像フォルダパスを指定してください。")
