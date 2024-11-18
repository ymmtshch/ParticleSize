import streamlit as st
import cv2
import numpy as np
import csv
from io import BytesIO

# Streamlit アプリの設定
st.title("Particle Size Detection Web App")
st.write("Upload SEM images to detect particles and calculate their diameters.")

# ファイルアップロード
uploaded_files = st.file_uploader("Upload your SEM images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

def process_image(image):
    """
    Processes an image to detect particles and calculate their diameters.
    """
    height, width = image.shape[:2]
    cropped_image = image[0:int(height * 7 / 8), :]
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    detected_circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=30, minRadius=5, maxRadius=50
    )

    if detected_circles is not None:
        circles = detected_circles[0, :]
        diameters = [2 * r for x, y, r in circles]
    else:
        diameters = []

    return diameters

def process_uploaded_files(files):
    """
    Processes all uploaded images and returns their diameters.
    """
    results = []
    for uploaded_file in files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        diameters = process_image(image)
        results.append((uploaded_file.name, diameters))
    return results

# 処理ボタン
if st.button("Process Images"):
    if uploaded_files:
        results = process_uploaded_files(uploaded_files)
        
        # 結果を表示
        st.write("### Results")
        for file_name, diameters in results:
            st.write(f"**{file_name}**: {len(diameters)} particles detected")
            st.write(", ".join([f"{d:.2f} pixels" for d in diameters]))
        
        # CSVに保存
        output = BytesIO()
        writer = csv.writer(output)
        writer.writerow(["Image", "Particle Index", "Diameter (pixels)"])
        for file_name, diameters in results:
            for idx, diameter in enumerate(diameters):
                writer.writerow([file_name, idx, f"{diameter:.2f}"])
        
        output.seek(0)
        st.download_button(
            label="Download Results as CSV",
            data=output,
            file_name="particle_results.csv",
            mime="text/csv",
        )
    else:
        st.warning("Please upload images before processing.")
