# 🤘✨ Real-Time Sign Language Recognition with CNN

## 💡 Introduction

This project provides a robust solution for the *real-time recognition of static sign language gestures* using a Convolutional Neural Network (CNN). By analyzing individual images, the model can accurately classify alphabet signs from various sign languages. It is optimized for high-accuracy, image-based recognition, offering a streamlined approach to sign language translation without needing temporal data.

-----

## 📄 Dataset

  - *Name*: Kaggle Sign Language Dataset
  - *Source*: Publicly available dataset from Kaggle.
  - *Classes*: 61
  - *Samples*:
      - 112,000+ labeled images
      - Each image is preprocessed and resized for the model.

-----

## ⚙ How It Works

### 🎯 Model Architecture

  - *Input*: Grayscale image (shape determined by model, e.g., 64x64)
  - *Core*: A deep Convolutional Neural Network (CNN) to perform feature extraction and classification.
  - *Output*: Softmax over 61 classes.
  - *Training*:
      - 25 epochs
      - 70:15:15 split (train:validation:test)
      - Optimizer: Adam

-----

### ⚡ Real-Time Pipeline

1.  *Image Capture & Preprocessing*: The webcam captures a frame, which is then converted to grayscale and resized.
2.  *Hand Detection*: An object detector identifies the hand region within the frame.
3.  *Prediction*: The CNN classifies the sign in the processed image, showing the result if the prediction confidence exceeds a set threshold.

-----

## 📊 Results

| Metric    | Score   |
|------------|---------|
| Accuracy | 96.0%   |
| Precision | 0.968   |
| Recall    | 0.960   |
| F1-Score | 0.962   |

### ✅ Key Observations

  - Achieves very high accuracy on isolated and clearly framed static signs.
  - Performance is robust to variations in lighting conditions due to preprocessing.
  - Minimal overfitting observed, with validation accuracy closely tracking training accuracy.

-----

## 🚀 Quick Start

### 🔧 Install Requirements
```bash
pip install mediapipe tensorflow opencv-python


