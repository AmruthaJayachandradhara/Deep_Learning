import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.data.load_data import create_labels
from src.data.preprocess import load_ben_color
from src.models.model import create_model
from src.results.resultviz import MetricTracker

#config
DATA_DIR = "ISIC_Dataset"
IMAGE_DIR = os.path.join(DATA_DIR, "ISIC_2019_Training_Input")
METADATA_CSV = os.path.join(DATA_DIR, "ISIC_2019_Training_GroundTruth.csv")
LABELS_NPY = os.path.join(DATA_DIR, "labels.npy")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 6
NUM_CLASSES = 9

class SkinLesionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
        self.image_paths = dataframe['image'].values
        self.labels = dataframe.drop(columns=['image']).values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(IMAGE_DIR, img_name + '.jpg')

        image = cv2.imread(img_path)
        image = load_ben_color(image, sigmaX=2)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx])
        return image, label
        
# Training transforms with augmentations
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Validation transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


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
train_dataset = SkinLesionDataset(train_df, transform=train_transform)
val_dataset = SkinLesionDataset(val_df, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)




#MODEL SETUP

model = create_model(num_classes=NUM_CLASSES).to(DEVICE)
tracker = MetricTracker()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)


#training
from torch.nn.functional import sigmoid
def train():
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        all_train_preds = []
        all_train_labels = []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = (outputs > 0.5).int().cpu()
            all_train_preds.append(preds)
            all_train_labels.append(labels.cpu().int())

        # Train metrics
        all_train_preds = torch.cat(all_train_preds).numpy()
        all_train_labels = torch.cat(all_train_labels).numpy()

        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_precision = precision_score(all_train_labels, all_train_preds, average='macro', zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds, average='macro', zero_division=0)

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f}")
        tracker.log_train_loss(avg_train_loss)


    # Evaluation (Validation accuracy after each epoch)
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (outputs > 0.5).int().cpu()
                all_val_preds.append(preds)
                all_val_labels.append(labels.cpu().int())

        # Val metrics
        all_val_preds = torch.cat(all_val_preds).numpy()
        all_val_labels = torch.cat(all_val_labels).numpy()

        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_precision = precision_score(all_val_labels, all_val_preds, average='macro', zero_division=0)
        val_recall = recall_score(all_val_labels, all_val_preds, average='macro', zero_division=0)

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")
        tracker.log_val_loss(avg_val_loss)

        # Update LR scheduler
        scheduler.step(avg_val_loss)



    # Save trained weights
    MODEL_SAVE_PATH = os.path.join(DATA_DIR, "model_weights_15.pt")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model weights saved at:", MODEL_SAVE_PATH)

    tracker.plot()



if __name__ == "__main__":
    train()

