
# Final Project – Group 5: Skin Lesion Classification

This repository contains the final deep learning project for Group 5, focused on multi-label classification of skin lesions using dermoscopic images from the HAM10000 dataset. The project includes preprocessing, model training, evaluation, and a Streamlit-based interactive demo.

---

## 🗂 Project Structure

```
Final-Project-Group5/
├── Code/
│   ├── train.py
│   ├── model.py
│   ├── preprocess.py
│   └── README.md
│
├── streamlit_app/
│   ├── app.py
│   ├── model.py
│   ├── model_weights_15.pt
│   ├── labels.npy
│   └── requirements.txt
│
├── Final_Report.pdf
└── README.md
```

---

## 🚀 Streamlit Web App

We built an interactive demo using [Streamlit](https://streamlit.io/) to showcase our trained melanoma classification model.

### To Run Locally

1. Navigate to the streamlit app folder:
   ```bash
   cd Final-Project-Group5/streamlit_app
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

### Files in this folder:
- `app.py` – Streamlit frontend for inference
- `model_weights_15.pt` – Trained ResNet18 model weights
- `model.py` – Custom ResNet model definition
- `labels.npy` – Class labels used in classification
- `requirements.txt` – Dependency list for the app

---

## 📊 Results

See our full experimental pipeline, metrics, and training visualizations in the final report.

---

## 🤝 Contributors

- Rasika Nilatkar
- Amrutha Jayachandradhara

---

## 📄 License & References

- Dataset: [ISIC Archive](https://challenge.isic-archive.com/)
- References and code inspiration listed in `Final_Report.pdf`
