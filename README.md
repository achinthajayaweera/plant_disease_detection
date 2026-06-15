🌿 Plant Disease Detection using Deep Learning

A web-based application that detects plant leaf diseases using a trained MobileNetV2 deep learning model.
Users can upload a plant leaf image, and the system predicts the disease with confidence.

📌 Features

🔍 Image upload interface (Streamlit UI)

🤖 Deep Learning model (MobileNetV2) trained on
New Plant Diseases Dataset (Augmented)

📊 Predicts 26 different plant diseases

📈 Shows confidence scores and top-3 predictions

🌐 Ready for deployment (Streamlit Cloud / local)

🧠 Model Details

Architecture: MobileNetV2 (transfer learning)

Input Size: 224 × 224 × 3

Output Classes: 26

Training Framework: TensorFlow / Keras

Dataset:
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

🚀 How to Run Locally
1️⃣ Clone the repository
git clone https://github.com/dmrs2003/plant_disease_detection.git
cd plant_disease_detection

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit app
python -m streamlit run app.py


The app will open at:

👉 http://localhost:8501

🗂 Project Structure
plant_disease_detection/
│── app.py                # Streamlit UI
│── plant_model.keras     # Trained deep learning model
│── class_names.json      # List of class labels
│── requirements.txt      # Python dependencies
│── training_notebook.ipynb # Model training notebook
│── README.md             # Project documentation

🖥️ Streamlit UI Preview

Upload a plant leaf image

Model preprocesses and predicts the disease

Displays:

Disease name

Confidence percentage

Top 3 probable classes

📚 How the Model Was Trained

Dataset loaded using image_dataset_from_directory()

Applied augmentation: rotation, zoom, translation, flipping

Rescaled pixel values (1./255)

Used MobileNetV2 with:

base.trainable = False


Added:

GlobalAveragePooling2D

Dropout

Dense softmax layer (26 units)

Trained for 10 epochs

Achieved:

🟩 Training accuracy: ~91%

🟦 Validation accuracy: ~89%

✨ Future Improvements

Add Grad-CAM heatmaps

Support for multiple leaves

Deploy on Streamlit Cloud

Add disease treatment suggestions

👨‍💻 Author

dmrs2003
Deep Learning & Web App Developer
GitHub: https://github.com/dmrs2003
