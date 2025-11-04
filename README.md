# Diabetic_Retinopathy

# 🩺 Diabetic Retinopathy Detection

This project detects **Diabetic Retinopathy** from retinal images using **EfficientNetB7** deep learning model.

## 🚀 Overview
- Model: EfficientNetB7 (Transfer Learning)
- Framework: TensorFlow / Keras
- Dataset: Kaggle - Diabetic Retinopathy
- Output: 5 classes (No DR → Proliferative DR)

## 📊 Model Info
Model file: `efficientnetb7_finetuned.keras_04`

## 🧩 Steps
1. Preprocessed images and applied data augmentation.
2. Fine-tuned EfficientNetB7 on retinal images.
3. Evaluated using accuracy, confusion matrix, and AUC.
4. Saved best model as `.keras_04`.

## 🖼️ Prediction Example
```python
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2

model = load_model('model/efficientnetb7_finetuned.keras_04')
img = cv2.imread('sample.jpg')
img = cv2.resize(img, (512, 512))
img = np.expand_dims(img/255.0, axis=0)

pred = model.predict(img)
print("Predicted class:", np.argmax(pred))
```

## 🧠 Classes
| Label | Description |
|:------:|-------------|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

---

## ⚙️ Requirements
See `requirements.txt`


