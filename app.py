import streamlit as st
from PIL import Image

# ---- Page Config ----
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# ---- UI ----
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image and the model will predict the disease.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.info("Model prediction will appear here soon...")
