import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os
import gdown

# ---- Page Config ----
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# ---- Custom CSS - Green Gradient Theme ----
st.markdown("""
<style>
/* Full page green gradient background */
.stApp {
    background: radial-gradient(ellipse at 30% 50%, #a8d878 0%, #4a9e5c 25%, #1a6b3a 55%, #0d3d22 100%);
    min-height: 100vh;
}

/* Hide default streamlit header */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Main container */
.block-container {
    padding-top: 3rem;
    max-width: 680px;
}

/* Title styling */
h1 {
    color: #ffffff !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    text-shadow: 0px 2px 12px rgba(0,0,0,0.4);
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem !important;
}

/* Subtitle text */
p {
    color: rgba(255,255,255,0.85) !important;
    font-size: 1rem !important;
}

/* Upload box */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.12) !important;
    border: 2px dashed rgba(255,255,255,0.4) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    backdrop-filter: blur(10px);
}

[data-testid="stFileUploader"] label {
    color: white !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}

[data-testid="stFileUploader"] small {
    color: rgba(255,255,255,0.7) !important;
}

/* Upload button */
[data-testid="stFileUploader"] button {
    background: rgba(255,255,255,0.2) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.5) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploader"] button:hover {
    background: rgba(255,255,255,0.35) !important;
}

/* Image display */
[data-testid="stImage"] {
    border-radius: 14px !important;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Success box */
[data-testid="stAlert"] {
    background: rgba(255,255,255,0.15) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 12px !important;
    color: white !important;
    backdrop-filter: blur(10px);
}

/* Spinner */
[data-testid="stSpinner"] {
    color: white !important;
}

/* Metric / result text */
.stSuccess {
    background: rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    border-left: 4px solid #90ee90 !important;
    color: white !important;
}

div[data-testid="stMarkdownContainer"] p {
    color: rgba(255,255,255,0.9) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ---- Download model from Google Drive if not present ----
MODEL_PATH = "plant_model.keras"
FILE_ID = "1NLEl_BLYK_z05ZM3Efp2PEoysAxgEH-T"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model... please wait ⏳"):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

# ---- Load model and class names ----
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_class_names():
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return class_names

model = load_model()
CLASS_NAMES = load_class_names()

IMG_SIZE = (224, 224)

# ---- Image Preprocessing ----
def preprocess_image(image):
    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---- UI ----
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image and the model will predict the disease.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)

    with st.spinner("Analyzing leaf... 🔍"):
        preds = model.predict(img_array)
        pred_index = np.argmax(preds[0])
        pred_class = CLASS_NAMES[pred_index]
        confidence = float(np.max(preds[0]) * 100)

    st.success(f"🌱 **Predicted Disease:** {pred_class}")
    st.write(f"🔍 Confidence: `{confidence:.2f}%`")
