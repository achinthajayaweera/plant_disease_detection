import streamlit as st
from PIL import Image
import tensorflow as tf
import json

# ---- Page Config ----
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# ---- Load model and class names ----
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_model.keras")
    return model

@st.cache_resource
def load_class_names():
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return class_names

model = load_model()
CLASS_NAMES = load_class_names()

# ---- UI ----
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image and the model will predict the disease.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.info("Image uploaded successfully! Prediction coming soon...")
