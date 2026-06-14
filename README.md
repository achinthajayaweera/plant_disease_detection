# 🌿 Plant Disease Detection using Deep Learning

A web-based application that detects plant leaf diseases using a trained deep learning model.
Users can upload a plant leaf image, and the system predicts the disease with confidence.

## 📌 Project Overview

This project aims to build an end-to-end plant disease detection system using:
- A deep learning model trained on plant leaf images
- A user-friendly web interface for image upload and prediction

## 🎯 Goals

- Detect plant diseases from leaf images
- Support multiple plant species and disease types
- Provide confidence scores for predictions
- Deploy as a web application

## 🗂 Planned Project Structure

```
plant_disease_detection/
│── app.py                  # Streamlit web app
│── plant_model.keras       # Trained deep learning model
│── class_names.json        # Disease class labels
│── requirements.txt        # Python dependencies
│── training_notebook.ipynb # Model training notebook
│── README.md               # Project documentation
```

## 📚 Dataset

We will use the **New Plant Diseases Dataset (Augmented)** from Kaggle:
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

## 🧠 Planned Model

- Architecture: MobileNetV2 (transfer learning)
- Input Size: 224 × 224 × 3
- Output: 26 disease classes

## 👨‍💻 Author

Deep Learning & Web App Developer
