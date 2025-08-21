import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tensorflow.keras.models import load_model

# -------------------------
# Load Pretrained Models
# -------------------------
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
lda = joblib.load("lda.pkl")

models = {
    "Logistic Regression": joblib.load("log_reg.pkl"),
    "Decision Tree": joblib.load("dt.pkl"),
    "Random Forest": joblib.load("rf.pkl"),
    "Support Vector Classifier": joblib.load("svc.pkl"),
    "K-Nearest Neighbors": joblib.load("knn.pkl"),
    "Naive Bayes": joblib.load("nb.pkl"),
    "XGBoost": joblib.load("xgb.pkl")
}

ffnn = load_model("ffnn.h5")

# -------------------------
# Streamlit UI
# -------------------------
st.title("🚗 Engine Fault Prediction System")
st.markdown("Upload sensor values or enter manually to predict engine condition.")

# Sidebar for input
option = st.sidebar.radio("Choose Input Method:", ["Manual Entry", "Upload CSV"])

if option == "Manual Entry":
    feature1 = st.number_input("Feature 1", value=1.15)
    feature2 = st.number_input("Feature 2", value=0.93)
    feature3 = st.number_input("Feature 3", value=74.20)
    feature4 = st.number_input("Feature 4", value=0.90)
    feature5 = st.number_input("Feature 5", value=1851.19)
    feature6 = st.number_input("Feature 6", value=3.34)
    feature7 = st.number_input("Feature 7", value=8.36)
    feature8 = st.number_input("Feature 8", value=39.69)
    feature9 = st.number_input("Feature 9", value=0.96)
    feature10 = st.number_input("Feature 10", value=143.08)
    feature11 = st.number_input("Feature 11", value=13.45)
    feature12 = st.number_input("Feature 12", value=0.53)
    feature13 = st.number_input("Feature 13", value=1.00)
    feature14 = st.number_input("Feature 14", value=14.73)

    input_data = np.array([[feature1,feature2,feature3,feature4,feature5,
                            feature6,feature7,feature8,feature9,feature10,
                            feature11,feature12,feature13,feature14]])

elif option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your input CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:", df.head())
        input_data = df.values
    else:
        input_data = None

# -------------------------
# Prediction
# -------------------------
if input_data is not None and st.button("Predict Engine Fault"):
    # Preprocessing
    noisy_input = input_data + np.random.normal(0, 0.01, input_data.shape)
    scaled_input = scaler.transform(noisy_input)
    pca_input = pca.transform(scaled_input)
    lda_input = lda.transform(pca_input)

    st.subheader("Predictions from Models:")
    for name, model in models.items():
        pred = model.predict(lda_input)[0]
        st.write(f"🔹 {name}: **{pred}**")

    # FFNN Prediction
    ffnn_pred = np.argmax(ffnn.predict(lda_input), axis=1)[0]
    st.write(f"🔹 Feed-Forward Neural Network: **{ffnn_pred}**")

# -------------------------
# GPS Integration
# -------------------------
st.subheader("📍 GPS Location Tracking")
gps = st.checkbox("Enable GPS Simulation")
if gps:
    lat = np.random.uniform(12.8, 13.1)  # Simulated Bangalore range
    lon = np.random.uniform(77.5, 77.7)
    st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))
    st.write(f"Current Location: **({lat:.4f}, {lon:.4f})**")
