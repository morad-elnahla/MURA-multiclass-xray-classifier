# 🦴 BONIFY — AI-Powered Fracture & Abnormality Detection

<div align="center">

![BONIFY Banner](assets/sample_images.png)

<br>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-MURA_v1.1-0ea5e9?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/cjinny/mura-v11)

<br>

**Deep learning system for automated bone fracture and abnormality detection**  
**from musculoskeletal X-ray images — 14-class classification across 7 body regions**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Results](#-results)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Training Strategy](#-training-strategy)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [References](#-references)

---

## 🔍 Overview

BONIFY is a multi-class deep learning classifier built on **DenseNet169** that simultaneously identifies the **body part** and detects **abnormalities** from X-ray radiographs. Trained on the Stanford MURA v1.1 dataset using a **3-phase transfer learning** strategy.

| Property | Value |
|----------|-------|
| **Task** | Multi-class Classification (14 classes) |
| **Model** | DenseNet169 — Transfer Learning + Fine-Tuning |
| **Dataset** | MURA v1.1 — Stanford ML Group |
| **Image Size** | 320 × 320 px |
| **Framework** | TensorFlow / Keras |
| **Deployment** | Streamlit Web App |

### Class Mapping

| Label | Class | Label | Class |
|-------|-------|-------|-------|
| 0 | ELBOW_Normal | 1 | ELBOW_Abnormal |
| 2 | FINGER_Normal | 3 | FINGER_Abnormal |
| 4 | FOREARM_Normal | 5 | FOREARM_Abnormal |
| 6 | HAND_Normal | 7 | HAND_Abnormal |
| 8 | HUMERUS_Normal | 9 | HUMERUS_Abnormal |
| 10 | SHOULDER_Normal | 11 | SHOULDER_Abnormal |
| 12 | WRIST_Normal | 13 | WRIST_Abnormal |

---

## 🎬 Demo

```bash
streamlit run app.py
```

Upload any X-ray image and BONIFY will:
- 🦴 Identify the body region
- ✅ / ❌ Detect normal or abnormal
- 📊 Show confidence score + Top-5 probability distribution

---

## 📊 Results

### Overall Metrics — Test Set (3,197 images)

| Metric | Score |
|--------|-------|
| **Top-1 Accuracy** | **78.26%** |
| **Top-3 Accuracy** | **98.65%** |
| **Cohen's Kappa** | **0.7641** |
| **F1 Score (Macro)** | **0.7744** |
| **F1 Score (Weighted)** | **0.7791** |

### Accuracy by Body Region

| Region | Accuracy | Test Samples |
|--------|----------|-------------|
| 🦾 Elbow | **83.4%** | 465 |
| ⌚ Wrist | **83.0%** | 659 |
| 🦴 Humerus | **78.5%** | 288 |
| ✋ Hand | **76.7%** | 460 |
| 🩻 Shoulder | **76.0%** | 563 |
| ☝️ Finger | **75.5%** | 461 |
| 💪 Forearm | **70.4%** | 301 |

### Training Curves

![Training Curves](assets/training_curves.png)

### Evaluation Results

![Evaluation Results](assets/evaluation_results.png)

### Correct vs Wrong Predictions

![Correct vs Wrong](assets/correct_vs_wrong.png)

---

## 📦 Dataset

**MURA v1.1 — Musculoskeletal Radiographs**  
Stanford ML Group · [Kaggle Dataset](https://www.kaggle.com/datasets/cjinny/mura-v11)

| Split | Images |
|-------|--------|
| Train | ~36,808 |
| Validation (15% stratified) | ~6,496 |
| Test (MURA valid folder) | 3,197 |

### Exploratory Data Analysis

![EDA Overview](assets/eda_overview.png)

Key observations:
- **59.6%** Normal / **40.4%** Abnormal — mild class imbalance addressed with balanced class weights
- WRIST (9,752) and SHOULDER (8,379) are the most represented regions
- HUMERUS (1,272) and FOREARM (1,825) are underrepresented

---

## 🧠 Model Architecture

```
Input (320×320×3)
    └── DenseNet169 (pretrained on ImageNet)
        └── GlobalAveragePooling2D
            └── BatchNormalization
                └── Dense(512, ReLU)
                    └── Dropout(0.4)
                        └── Dense(14, Softmax)
```

**Loss Function:** Categorical Focal Loss (γ=2.0, α=0.25)  
Focuses training on hard examples — effective for class imbalance.

```python
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        ce     = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * ce, axis=-1))
    return focal_loss
```

---

## 🚀 Training Strategy

3-phase progressive training for stable convergence:

```
Phase 1 — Transfer Learning    LR: 1e-4   Epochs: 10   (frozen base)
Phase 2 — Fine-Tuning          LR: 5e-6   Epochs: 20   (all layers)
Phase 3 — Deep Fine-Tuning     LR: 1e-6   Epochs: 15   (all layers)
```

**Callbacks:** `EarlyStopping` · `ReduceLROnPlateau` · `ModelCheckpoint`  
**Augmentation:** Horizontal flip · Vertical flip · Brightness · Contrast · Saturation

---

## 📁 Project Structure

```
BONIFY/
│
├── app.py                              # Streamlit web application
├── mura-multi-class.ipynb              # Full training notebook
│
├── densenet169_multiclass_final.keras  # Trained model (157 MB)
│
├── assets/
│   ├── eda_overview.png                # Exploratory data analysis plots
│   ├── training_curves.png             # Training history (3 phases)
│   ├── evaluation_results.png          # Confusion matrix + metrics
│   ├── sample_images.png               # Sample X-rays per class
│   └── correct_vs_wrong.png            # Correct vs wrong predictions
│
├── results/
│   ├── test_predictions.csv            # Full test set predictions
│   ├── test_probs.npy                  # Softmax probabilities (3197×14)
│   ├── test_preds.npy                  # Predicted labels (3197,)
│   └── true_labels.npy                 # Ground truth labels (3197,)
│
├── docs/
│   ├── BONIFY_Documentation.docx       # Full model documentation
│   ├── BONIFY_References.pdf           # References & citations
│   └── BONIFY_Data.pdf                 # Dataset documentation
│
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/BONIFY.git
cd BONIFY

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```
tensorflow>=2.12
streamlit>=1.28
opencv-python>=4.8
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
```

---

## 🖥️ Usage

### Run the Web App

```bash
streamlit run app.py
```

Make sure `densenet169_multiclass_final.keras` is in the same folder as `app.py`.

### Run Inference in Python

```python
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma)
        return K.sum(weight * cross_entropy, axis=1)
    return focal_loss

BODY_PARTS  = ['ELBOW','FINGER','FOREARM','HAND','HUMERUS','SHOULDER','WRIST']
CLASS_NAMES = [f'{bp}_{s}' for bp in BODY_PARTS for s in ['Normal','Abnormal']]

model = tf.keras.models.load_model(
    'densenet169_multiclass_final.keras',
    custom_objects={'categorical_focal_loss': categorical_focal_loss}
)

def predict(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320)) / 255.0
    probs = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    pred_idx = np.argmax(probs)
    print(f'Prediction : {CLASS_NAMES[pred_idx]}')
    print(f'Confidence : {probs[pred_idx]*100:.1f}%')

predict('xray.png')
```

### Load Saved Results

```python
import numpy as np

test_probs = np.load('results/test_probs.npy')   # (3197, 14) — softmax probabilities
test_preds = np.load('results/test_preds.npy')   # (3197,)   — predicted class indices
true_labels = np.load('results/true_labels.npy') # (3197,)   — ground truth labels
```

---

## ⚠️ Disclaimer

> This project is for **research and educational purposes only**.  
> BONIFY is **not a medical device** and should **not be used for clinical diagnosis**.  
> Always consult a qualified medical professional for radiological interpretation.

---

## 📚 References

1. Rajpurkar, P. et al. (2018). MURA: Large Dataset for Abnormality Detection in Musculoskeletal Radiographs. *arXiv:1712.06957*
2. Huang, G. et al. (2017). Densely Connected Convolutional Networks. *CVPR 2017*
3. Lin, T.Y. et al. (2017). Focal Loss for Dense Object Detection. *ICCV 2017*

Full references available in [`docs/BONIFY_References.pdf`](docs/BONIFY_References.pdf)

---

<div align="center">

Made with ❤️ · BONIFY v1.0 · 2025

</div>
