
# Classification of Skin Lesion Melanoma Detection (Group 5)

This repository contains the implementation of our deep learning final project focused on **multi-label classification of dermoscopic images** using a ResNet18-based convolutional neural network. The project includes complete preprocessing, model training, evaluation, and a Streamlit demo for interactive testing.

## Project Overview

Early detection of skin cancer, particularly melanoma, is critical for patient survival. We leverage **deep learning** techniques on the **HAM10000** dataset to classify dermoscopic images into **nine skin lesion categories**:

| Class Name | Description                  |
|------------|------------------------------|
| MEL        | Melanoma                     |
| NV         | Melanocytic nevus            |
| BCC        | Basal cell carcinoma         |
| AK         | Actinic keratosis            |
| BKL        | Benign keratosis             |
| DF         | Dermatofibroma               |
| VASC       | Vascular lesion              |
| SCC        | Squamous cell carcinoma      |
| UNK        | Unknown/None of the above    |

## Folder Structure

```
Final-Project-Group5/
│
├── Code/
│   ├── data/
│   │   ├── load_data.py          # Label generation from metadata
│   │   └── preprocess.py         # Preprocessing (cropping, CLAHE, Ben Graham)
│   ├── models/
│   │   └── model.py              # create_model() using ResNet18
│   ├── result/
│   │   └── resultviz.py          # MetricTracker for training loss/accuracy plotting
│   ├── app.py                    # Streamlit demo application
│   ├── main.py                   # Model training and evaluation
│   ├── metadata.csv              # Metadata from ISIC
│   ├── labels.npy                # Saved label matrix
│   ├── model_weights.pt          # Trained PyTorch model weights
│   └── README.md                 # Code-specific execution guide
│
├── Group-Report.pdf              # Final team report (Group 5)
├── Individual-Final-Project-Report 
└── README.md                     # You are here!
```

## Technologies Used

- **Language**: Python
- **Frameworks**: PyTorch, Streamlit
- **Libraries**: OpenCV, NumPy, Pandas, Altair, scikit-learn, tqdm

## Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/Deep_Learning.git
cd Deep_Learning/
```

### 2. Setup Environment

```bash
python -m venv env
source env/bin/activate   # Mac/Linux
env\Scripts\activate      # Windows

pip install -r requirements.txt
```

### 3. Run Training

```bash
cd Code
python main.py
```

This trains the model for 6 epochs using a ResNet18 architecture.

### 4. Launch Streamlit App

```bash
streamlit run app.py
```

Upload a skin lesion image and view predicted class probabilities interactively.

## 📊 Highlights

- **Preprocessing**: Cropping gray borders, CLAHE contrast enhancement, and Ben Graham sharpening.
- **Augmentation**: Horizontal/vertical flips, color jitter, random crop & rotation.
- **Model**: ResNet18 + Custom Head (FC → Dropout → Sigmoid for 9-class multi-label output).
- **Evaluation**: Training/Validation accuracy and loss curves with MetricTracker.
- **UI**: Streamlit-powered interactive prediction interface.

## Contributors (Group 5)

- **Rasika Nilatkar** – Streamlit app, model setup, data augmentation  
- **Amrutha Jayachandradhara** – Preprocessing, training pipeline, CLAHE/BenGraham methods  

## Future Scope

- Replace ResNet18 with more advanced models like EfficientNet or Vision Transformers  
- Implement skin lesion segmentation as preprocessing  
- Train with larger ISIC 2020 dataset  
- Host Streamlit app on cloud for public access  
- Incorporate Grad-CAM for visual interpretability

## References

1. ISIC Archive: [https://www.isic-archive.com](https://www.isic-archive.com)
2. BCN20000: Dermoscopic Lesions in the Wild (arXiv:1908.02288)
3. Deep Learning based Malignant Melanoma Detection (IEEE Xplore)
4. [Kiel Dang, Medium](https://medium.com/@kiell.dang/deep-learning-skin-cancer)
