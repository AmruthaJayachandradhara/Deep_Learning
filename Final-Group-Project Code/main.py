
from google.colab import drive
import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Mount Google Drive
drive.mount('/content/drive')

# Paths
abspath_curr = '/content/drive/My Drive/Colab Notebooks/DeepLearning_project/'
DATA_CSV = os.path.join(abspath_curr, 'metadata.csv')
LABELS_NPY = os.path.join(abspath_curr, 'labels.npy')
IMAGE_DIR = os.path.join(abspath_curr, 'img100')

# ===========================
# CONFIGURATION
# ===========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# PREPROCESSING FUNCTIONS
# ===========================
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def clahe_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))   # <-- convert tuple to list here
    clahe = cv2.createCLAHE(clipLimit=1.0)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb


def load_ben_color(image, sigmaX):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = clahe_lab(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image

# ===========================
# DATASET CLASS
# ===========================
class SkinLesionDataset(Dataset):
    def __init__(self, df, transform=None, preprocess_sigmaX=2):
        self.df = df
        self.transform = transform
        self.image_paths = df['image'].values
        self.labels = df.drop(columns=['image']).values.astype(np.float32)
        self.preprocess_sigmaX = preprocess_sigmaX

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.image_paths[idx]
        if not filename.endswith('.jpg'):
            filename += '.jpg'

        image_path = os.path.join(IMAGE_DIR, filename)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Apply Ben Graham preprocessing
        image = load_ben_color(image, sigmaX=self.preprocess_sigmaX)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx])
        return image, label

# ===========================
# TRANSFORMS
# ===========================
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

# ===========================
# DATA LOADING
# ===========================
df = pd.read_csv(DATA_CSV)
labels = np.load(LABELS_NPY)

df_combined = pd.DataFrame({'image': df['image']})
label_columns = [f'label_{i}' for i in range(NUM_CLASSES)]
df_labels = pd.DataFrame(labels, columns=label_columns)
df_combined = pd.concat([df_combined, df_labels], axis=1)

train_df, val_df = train_test_split(df_combined, test_size=0.2, random_state=42)

train_dataset = SkinLesionDataset(train_df, transform=train_transform)
val_dataset = SkinLesionDataset(val_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ===========================
# MODEL
# ===========================
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, NUM_CLASSES),
    nn.Sigmoid()
)
model = model.to(DEVICE)

# ===========================
# TRAINING
# ===========================
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1, verbose=True)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Train Loss: {train_loss / len(train_loader):.4f}")

# ===========================
# EVALUATION
# ===========================
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        predicted = outputs > 0.5
        correct += (predicted == labels.bool()).sum().item()
        total += labels.numel()
print(f"Validation Accuracy: {correct / total:.4f}")


val_accuracy = correct / total
scheduler.step(val_accuracy)

# Save trained weights
torch.save(model.state_dict(), os.path.join(abspath_curr, "model_weights.pt"))
print("Model weights saved at:", os.path.join(abspath_curr, "model_weights.pt"))
