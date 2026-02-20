# Alzheimer-MRI-Stage-Classification-XAI
Deep Learning classification of Alzheimer's stages using Transfer Learning (MobileNetV2) and Explainable AI (Grad-CAM) to visualize brain atrophy.
# Alzheimer’s Disease Stage Classification & Clinical Explainability (XAI)

## 📋 Executive Summary
This project focuses on the multi-class classification of Alzheimer’s Disease (AD) stages using Deep Learning. By leveraging **Transfer Learning** and **Gradient-weighted Class Activation Mapping (Grad-CAM)**, the model provides not just a diagnosis, but a visual justification for its decisions.

---

## 🏗️ Technical Architecture
### 1. The "Resource-Optimized" Pivot
During development, environmental constraints (session timeouts) required a strategic shift from heavy architectures to **MobileNetV2**. 
* **Benefit:** High-speed inference and lower memory footprint, making it ideal for edge deployment in clinical settings.
* **Weights:** Initialized with ImageNet for robust low-level feature extraction.

### 2. Global Average Pooling & Regularization
Instead of flattening high-dimensional feature maps, I utilized **Global Average Pooling (GAP)**. This reduces the number of trainable parameters, mitigating the risk of overfitting on the specific textures of the training MRI set.



---

## 🔍 Explainable AI (XAI)
Medical AI cannot be a "Black Box." To ensure clinical validity, I implemented **Grad-CAM**. 

By calculating the gradients of the target class (e.g., *Moderate Demented*) with respect to the last convolutional layer (`out_relu`), the model generates a heatmap. This allows a neurologist to verify if the AI is focusing on **cortical thinning** or **ventricular enlargement**.



---

## 📊 Evaluation & Insights
- **Recall Priority:** In dementia screening, the cost of a "False Negative" is significantly higher than a "False Positive." The model was tuned to maximize sensitivity for early-stage detection.
- **Error Analysis:** The confusion matrix reveals a logical overlap between "Very Mild" and "Non-Demented" stages, highlighting the physical similarity in early disease progression.



---

## 🛠️ Repository Structure
- `alzheimer_classification.ipynb`: Full end-to-end pipeline (EDA, Training, XAI).
- `images/`: Visual assets (Confusion Matrix, Heatmaps).
- `requirements.txt`: Environment dependencies.
