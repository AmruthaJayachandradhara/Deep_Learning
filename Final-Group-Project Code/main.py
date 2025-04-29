import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

from DeepLearning_project.src.data.load_data import create_labels
from DeepLearning_project.src.data.preprocess import load_ben_color
from DeepLearning_project.src.models.model import create_model

#config
DATA_DIR = "DeepLearning_project/data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
METADATA_CSV = os.path.join(DATA_DIR, "metadata.csv")
LABELS_NPY = os.path.join(DATA_DIR, "labels.npy")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 9


class SkinLesionDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        self.image_paths = dataframe['image'].values
        self.labels = dataframe.drop(columns=['image']).values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(IMAGE_DIR, img_name + '.jpg')

        image = cv2.imread(img_path)
        image = load_ben_color(image)

        label = torch.tensor(self.labels[idx])
        return image, label


#CREATE LABELS
if not os.path.exists(LABELS_NPY):
    create_labels(METADATA_CSV, DATA_DIR)

#LOAD DATA
df = pd.read_csv(METADATA_CSV)
labels = np.load(LABELS_NPY)

df_combined = pd.DataFrame({'image': df['image']})
label_columns = [f'label_{i}' for i in range(NUM_CLASSES)]
df_labels = pd.DataFrame(labels, columns=label_columns)
df_combined = pd.concat([df_combined, df_labels], axis=1)

train_df, val_df = train_test_split(df_combined, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


#MODEL SETUP

model = create_model(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#TRAINING
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch+1} Training Loss: {train_loss/len(train_loader):.4f}")

#EVALUATION
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs > 0.5
        correct += (preds == labels.bool()).sum().item()
        total += labels.numel()
print(f"Validation Accuracy: {correct/total:.4f}")

# Save trained weights
torch.save(model.state_dict(), os.path.join(abspath_curr, "model_weights.pt"))
print("Model weights saved at:", os.path.join(abspath_curr, "model_weights.pt"))


