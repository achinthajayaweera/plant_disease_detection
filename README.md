# 🌿 Plant Disease Detection — Project Documentation

## Overview

Plant Disease Detection is an AI-powered web application designed to help farmers, agricultural researchers, and plant enthusiasts identify diseases in plant leaves using deep learning. By simply uploading a photograph of a plant leaf, the application analyzes the image in real time and returns a predicted disease classification along with a confidence score. The system is built on a transfer learning model trained on thousands of labeled plant images, and it is deployed as a fully accessible web application using Streamlit Cloud — meaning no installation, no setup, and no technical knowledge is required from the end user.

This project sits at the intersection of computer vision, machine learning, and modern web deployment, making it both technically rich and practically meaningful. It demonstrates how cutting-edge AI technology can be packaged into a simple, beautiful interface and made available to anyone with a browser.

---

## Problem Statement

Agricultural crop losses due to plant diseases represent one of the most significant global challenges. According to the Food and Agriculture Organization (FAO), plant diseases account for approximately 20–40% of global crop losses every year. Early and accurate detection of plant diseases is critical — the sooner a disease is identified, the sooner intervention can occur, and the greater the chance of saving a crop.

Traditional methods of disease diagnosis rely on trained agronomists physically inspecting crops, which is time-consuming, expensive, and not scalable — especially in remote or developing regions where agricultural expertise is scarce. This project addresses that gap by providing an automated, AI-powered tool that can perform disease classification instantly from a simple leaf photo, with no expert required on-site.

---

## Project Scope

The scope of this project covers the full machine learning lifecycle and web deployment pipeline:

**Data Collection and Preparation** — Sourcing a large, labeled dataset of plant disease images, downloading and extracting it programmatically, and organizing it into training and validation sets.

**Data Preprocessing and Augmentation** — Normalizing image pixel values, resizing images to a uniform size, and applying geometric augmentations such as flipping, rotation, zooming, and translation to increase the diversity of training data and improve model generalization.

**Model Architecture Design** — Selecting and configuring a pretrained deep learning model (MobileNetV2) as a feature extractor, then adding a custom classification head tailored to the specific disease categories in the dataset.

**Model Training and Evaluation** — Training the model on GPU hardware using Google Colab, monitoring training and validation accuracy and loss across epochs, and evaluating final model performance.

**Model Persistence** — Saving the trained model in Keras format and exporting class label mappings to JSON for use during inference.

**Web Application Development** — Building a user-facing Streamlit application with a custom UI that handles image upload, preprocessing, model inference, and result display.

**Cloud Deployment** — Deploying the application to Streamlit Cloud with automatic model downloading from Google Drive, making the app publicly accessible via a shareable URL.

**UI/UX Design** — Designing a visually distinctive interface using custom CSS, background gradient imagery, decorative botanical assets, and smooth CSS animations to create a professional and thematic presentation.

---

## Dataset

The dataset used for training is the **New Plant Diseases Dataset** sourced from Kaggle (uploaded by user `vipoooool`). It is one of the most widely used benchmark datasets in plant pathology research and contains:

- Over **87,000 images** of plant leaves
- **38 different classes** covering both healthy and diseased states
- Images across **14 crop species** including Apple, Tomato, Corn, Grape, Potato, Peach, Strawberry, Cherry, Orange, Squash, and Pepper
- Images collected under controlled conditions with consistent lighting and backgrounds
- A pre-split **train/validation structure** for reproducible model evaluation

For this project, the dataset was filtered to focus specifically on **26 disease classes**, excluding the healthy class variants to focus the model on pathological identification.

---

## Tech Stack

### Machine Learning & Deep Learning

**TensorFlow 2.x / Keras**
The core deep learning framework used for model building, training, and inference. TensorFlow provides the computational graph engine while Keras offers the high-level API for defining model architecture, compiling with loss functions and optimizers, and running training loops.

**MobileNetV2 (Transfer Learning)**
Rather than training a convolutional neural network from scratch — which would require massive amounts of data and compute — this project uses MobileNetV2, a lightweight and efficient CNN architecture pretrained on ImageNet (a dataset of 1.4 million images across 1,000 classes). The pretrained base acts as a powerful visual feature extractor, and only the custom classification head is trained on the plant disease data. This approach drastically reduces training time and data requirements while achieving high accuracy.

MobileNetV2 was specifically chosen because:
- It is optimized for mobile and edge deployment with low parameter count
- It achieves strong accuracy relative to its size
- It uses depthwise separable convolutions which are computationally efficient
- It works well as a feature backbone for transfer learning tasks

**Keras ImageDataGenerator / `image_dataset_from_directory`**
Used to load images directly from directory structures, infer labels automatically, apply batching, shuffling, and preprocessing pipelines.

**Data Augmentation**
Applied using Keras preprocessing layers including `RandomFlip`, `RandomRotation`, `RandomZoom`, and `RandomTranslation`. Augmentation artificially expands the effective size of the training set and teaches the model to be invariant to common real-world image variations.

**Adam Optimizer with Categorical Cross-Entropy Loss**
The model is compiled with the Adam optimizer, which adaptively adjusts learning rates during training, and categorical cross-entropy as the loss function — the standard choice for multi-class classification problems.

### Development Environment

**Google Colaboratory (Colab)**
Model training was performed on Google Colab using a free T4 GPU, which accelerates matrix operations in deep learning by orders of magnitude compared to CPU training. Colab also provides convenient integration with Google Drive for storing datasets and saving model checkpoints.

**Google Drive**
Used as cloud storage for both the raw dataset (downloaded via Kaggle API) and the trained model file (`plant_model.keras`). The model is also served from Google Drive at inference time using a public sharing link.

**Kaggle API**
Used programmatically within the notebook to download the dataset directly to the Colab environment using environment variable–based authentication.

### Web Application

**Streamlit**
Streamlit is a Python framework for building interactive data science and machine learning web applications with minimal frontend code. It converts Python scripts into shareable web apps automatically. In this project, Streamlit handles:
- File upload UI
- Image display
- Spinner/loading indicators
- Result display with success alerts
- Caching of model loading using `@st.cache_resource` to avoid reloading the model on every user interaction

**gdown**
A Python library for downloading files from Google Drive using a file ID. Used in the app to automatically download the trained model (`plant_model.keras`) on first launch if it is not already present on the server.

**Pillow (PIL)**
Python Imaging Library used for opening uploaded images, converting color modes, resizing to the model's expected input dimensions (224×224), and processing decorative PNG assets by removing black backgrounds programmatically.

**NumPy**
Used for converting PIL images to NumPy arrays, normalizing pixel values to the [0, 1] range, and interpreting model prediction outputs using `np.argmax`.

### Frontend & UI/UX

**Custom CSS via `st.markdown(unsafe_allow_html=True)`**
Streamlit's default UI is styled using injected CSS to completely override the default appearance. Custom styles include:
- Full-page background image using base64-encoded gradient PNG
- Fixed-position decorative branch images in page corners using base64-encoded PNGs with black backgrounds removed
- CSS `@keyframes` animation for the floating leaf icon
- Glassmorphism-style upload card with `backdrop-filter: blur`
- Custom typography, text shadows, and color palette
- Hiding default Streamlit header elements

**Adobe Photoshop**
Used to design the visual layout and export the background gradient and botanical branch assets as PNG files, which were then processed and embedded directly into the application.

**Base64 Image Embedding**
All image assets are base64-encoded and embedded directly into the Python source file. This eliminates the need for a separate static file server and ensures all assets load correctly in the cloud deployment environment without any file path issues.

### Deployment

**Streamlit Community Cloud**
The application is deployed on Streamlit's free community hosting platform, which connects directly to the GitHub repository and automatically redeploys whenever changes are pushed. The live app is accessible at a public URL without any server management required.

**GitHub**
Version control and source hosting for the project. The repository contains all source code files including the Streamlit app, notebook, class names JSON, and requirements. The commit history demonstrates the progressive development of the project across 13 meaningful versions.

**`requirements.txt`**
Specifies all Python package dependencies with pinned versions to ensure reproducible builds on Streamlit Cloud:
```
streamlit
tensorflow==2.17.1
pillow
numpy
gdown
```

---

## Model Architecture (Detailed)

```
Input: (224, 224, 3) RGB image

↓ MobileNetV2 Base (pretrained on ImageNet, frozen)
  - 154 layers
  - Depthwise separable convolutions
  - Linear bottlenecks with residual connections
  - Output shape: (7, 7, 1280)

↓ GlobalAveragePooling2D
  - Reduces spatial dimensions to a single vector
  - Output shape: (1280,)

↓ Dropout(0.3)
  - Randomly zeroes 30% of neurons during training
  - Prevents overfitting

↓ Dense(26, activation='softmax')
  - 26 output neurons, one per disease class
  - Softmax converts logits to probability distribution

Output: Probability vector across 26 disease classes
        Predicted class = argmax of output vector
```

Total trainable parameters: ~26,000 (classification head only)
Total non-trainable parameters: ~2,257,984 (frozen MobileNetV2 base)

---

## Disease Classes

The model is trained to classify 26 plant disease conditions across multiple crop species:

| Crop | Disease |
|---|---|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus |
| Corn (Maize) | Cercospora Leaf Spot / Gray Leaf Spot, Common Rust, Northern Leaf Blight |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight |
| Potato | Early Blight, Late Blight |
| Peach | Bacterial Spot |
| Cherry | Powdery Mildew |
| Strawberry | Leaf Scorch |
| Orange | Haunglongbing (Citrus Greening) |
| Squash | Powdery Mildew |
| Pepper (Bell) | Bacterial Spot |

---

## Application Flow

1. **App Launch** — Streamlit app starts; checks if `plant_model.keras` exists locally. If not, downloads it from Google Drive using `gdown`.

2. **Model Loading** — Model and class names JSON are loaded into memory and cached using `@st.cache_resource`, so subsequent interactions don't reload them.

3. **Image Upload** — User uploads a JPG or PNG image through the file uploader widget.

4. **Preprocessing** — The uploaded image is opened with PIL, converted to RGB, resized to 224×224 pixels, converted to a NumPy array, normalized to [0, 1], and expanded to a batch of 1.

5. **Inference** — The preprocessed image tensor is passed through the model. The model outputs a probability vector of length 26.

6. **Result Display** — The class with the highest probability is selected via `np.argmax`. The predicted disease name and confidence percentage are displayed to the user.

---

## Key Learnings & Challenges

**Transfer Learning Efficiency** — Using MobileNetV2 as a frozen feature extractor demonstrated how pretrained models can be repurposed for specialized tasks with minimal data and compute, achieving strong performance without training from scratch.

**Cloud Deployment Constraints** — Deploying to Streamlit Cloud revealed real-world constraints around Python version compatibility (TensorFlow requires Python ≤ 3.11) and model file size (GitHub's 100MB limit required hosting the model on Google Drive).

**Asset Embedding** — Serving static image assets in a cloud-hosted Streamlit app required base64-encoding all images directly into the Python source, since Streamlit Cloud does not serve local files as static assets.

**CSS in Streamlit** — Overriding Streamlit's default styling required deep knowledge of its HTML/CSS structure and the use of `unsafe_allow_html=True` to inject custom styles targeting internal data attributes.

---

## Future Improvements

- **Expand disease classes** to cover all 38 classes in the dataset including healthy plant detection
- **Fine-tune the MobileNetV2 base** by unfreezing the top layers for higher accuracy
- **Add treatment recommendations** for each detected disease
- **Support real-time camera input** via Streamlit's camera widget
- **Add a probability distribution chart** showing confidence scores across top-5 predictions
- **Multilingual support** for accessibility in non-English speaking farming communities
- **Mobile-optimized layout** for field use on smartphones
- **Batch image processing** to analyze multiple leaves at once
- **Export report functionality** to generate PDF summaries of detected diseases

---

## Live Demo

🌐 **[aj-plant-disease-detection.streamlit.app](https://aj-plant-disease-detection.streamlit.app)**

## Repository

📁 **[github.com/achinthajayaweera/plant_disease_detection](https://github.com/achinthajayaweera/plant_disease_detection)**