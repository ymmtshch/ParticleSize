$ pip install opencv-python
$ pip install matplotlib

import cv2
import os
import numpy as np
import csv
import matplotlib.pyplot as plt

# Title of the Streamlit app
st.title("Particle Size Distribution Analyzer")

# Upload the CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)
    
    st.write("### Uploaded Data Preview:")
    st.write(data.head())

    # Perform basic preprocessing and show data info
    st.write("### Data Information:")
    st.write(data.describe())

    # Assuming the file contains a column named 'Particle Size'
    if 'Particle Size' in data.columns:
        particle_size = data['Particle Size']
        
        st.write("### Summary Statistics:")
        st.write(particle_size.describe())

        # Plot the histogram of particle sizes
        st.write("### Particle Size Distribution:")
        fig, ax = plt.subplots()
        ax.hist(particle_size, bins=20, color='blue', alpha=0.7)
        ax.set_title("Particle Size Histogram")
        ax.set_xlabel("Particle Size (µm)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Example: Compute D10, D50, and D90
        D10 = np.percentile(particle_size, 10)
        D50 = np.percentile(particle_size, 50)
        D90 = np.percentile(particle_size, 90)
        
        st.write("### Percentile Analysis:")
        st.write(f"D10: {D10:.2f} µm")
        st.write(f"D50: {D50:.2f} µm")
        st.write(f"D90: {D90:.2f} µm")
        
    else:
        st.error("The dataset must contain a 'Particle Size' column.")
else:
    st.info("Awaiting CSV file upload.")

# Footer information
st.markdown("---")
st.markdown("Powered by Streamlit")
