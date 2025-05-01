# Deep Learning Final Project - Code Folder

This folder contains all the code components for our final project on **Melanoma Classification using Deep Learning**, implemented in PyTorch and Streamlit.

The goal is to classify skin lesion images into one or more of nine diagnostic categories using a fine-tuned ResNet18 model trained on the **ISIC 2019 dataset**.

## Code Execution Flow

Follow the order below to reproduce the project pipeline:

### 1. `labels_generator.py`  
- **Purpose**: Creates a multi-label NumPy array (`labels.npy`) based on the metadata CSV.
- **Run this first** to ensure you have the label tensor ready for training.

---

### 2. `train.py` / `DL.py`  
- **Purpose**: Loads the images, applies preprocessing (cropping, CLAHE, Ben Graham), performs data augmentation (rotation, flipping, color jitter), and trains the modified ResNet18 model.
- **Includes**:
  - Data augmentation (train/val transforms)
  - Model setup with binary cross-entropy loss
  - GPU training support
  - Learning rate scheduler (ReduceLROnPlateau)
- **Output**: `model_weights.pt` – saved trained weights

---

### 3. `model.py`  
- **Purpose**: Contains the architecture definition for the modified ResNet18 model used during training and inference.
- Custom classification head: 512 → 256 → 9 with dropout and sigmoid activation.

---

### 4. `app.py` (Streamlit Web App)  
- **Purpose**: Loads the trained model and provides a user interface for classifying uploaded skin lesion images.
- **Features**:
  - Upload and predict image category
  - Adjustable threshold slider
  - Interactive bar chart (Altair)
  - Confidence filter for low-certainty predictions

---

## Requirements

Make sure to install the dependencies listed in `requirements.txt` before running the scripts.

```bash
pip install -r requirements.txt
```

---

## Notes

- Dataset used: [ISIC 2019 Challenge Data](https://challenge2019.isic-archive.com/)
- Trained weights: `model_weights.pt`
- Label file: `labels.npy`
- If training from scratch, ensure all image paths and metadata are configured properly in `train.py`.

---

## Contributors

- Rasika Nilatkar – Data augmentation, model setup, Streamlit UI  
- Amrutha Jayachandradhara – Preprocessing, training pipeline
