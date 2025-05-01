import streamlit as st
import os
import gdown
if not os.path.exists("model_weights.pt"):
    url = "https://drive.google.com/file/d/1oOxjSwphbIz_aZa470t1MD3MS1vne7ma/view?usp=sharing"
    gdown.download(url, "model_weights.pt", quiet=False)
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="Melanoma Classifier",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Sidebar info
with st.sidebar:
    st.header("📘 About")
    st.markdown("""
    - **Model**: ResNet18 + Custom Head  
    - **Classes**: MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK  
    - **Dataset**: HAM10000  
    - **Created by**: *Rasika Nilatkar, Amrutha Jayachandradhara*
    """)

    with st.sidebar:
        st.header("⚙️ Settings")
        threshold = st.slider("Prediction threshold", 0.0, 1.0, 0.5, 0.01)

# Class names
class_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

@st.cache_resource
def load_model():
    class CustomResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = models.resnet18(weights=None)
            self.backbone.fc = nn.Identity()  # remove original classification layer
            self.fc1 = nn.Linear(512, 256)  # 512 comes from ResNet18 output
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(256, 9)

        def forward(self, x):
            x = self.backbone(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            return torch.sigmoid(x)

    model = CustomResNet()
    model.load_state_dict(torch.load("model_weights_15.pt", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    if image.mode != "RGB":
        image = image.convert("RGB")
    return transform(image).unsqueeze(0)

# App main interface
st.title("Melanoma Classification Demo")
st.markdown("Upload a skin lesion image to classify it.")

#threshold = st.slider(" Set prediction threshold", 0.0, 1.0, 0.5, 0.01)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Reject non-RGB or too-small images
    if image.mode != "RGB" or image.size[0] < 100 or image.size[1] < 100:
        st.error("🚫 Invalid image: Please upload a high-quality RGB skin lesion image.")
        st.stop()
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            probs = output.squeeze().tolist()

            max_prob = max(probs)
            if max_prob < 0.55:
                st.error("⚠️ Model confidence is too low. This may not be a valid lesion image.")
                st.stop()

        st.subheader("Prediction Results:")
        predicted_labels = [(class_names[i], prob) for i, prob in enumerate(probs) if prob > threshold]

        if predicted_labels:
            for label, prob in predicted_labels:
                st.success(f"✅ **{label}** — {prob:.2%} confidence")
        else:
            st.warning("⚠️ No class was confidently predicted.")

        # Bar chart of top predictions
        df = pd.DataFrame({
            "Class": class_names,
            "Probability": probs
        }).sort_values(by="Probability", ascending=False).head(5)

        st.subheader("📊 Top Predictions")
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X("Probability", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("Class", sort="-x"),
            color="Class"
        )
        st.altair_chart(chart, use_container_width=True)

        st.caption(" Model: ResNet18 | Trained on: HAM10000 | Version: v1.0")

        # Expandable full probabilities
        with st.expander("See full class probabilities"):
            for i, prob in enumerate(probs):
                st.write(f" **{class_names[i]}**: {prob:.2%}")
