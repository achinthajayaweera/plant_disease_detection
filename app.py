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
