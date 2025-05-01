import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

IMG_SIZE = 224  # You can change if needed

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        img = img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if mask.any():
            img = img[np.ix_(mask.any(1), mask.any(0))]
    return img

def clahe_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img

def debug_preprocessing(image_path):
    # Step 0: Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    images = []
    titles = []

    images.append(img)
    titles.append("Original")

    # Step 1: Crop
    img_crop = crop_image_from_gray(img)
    images.append(img_crop)
    titles.append("After Crop")

    # Step 2: CLAHE
    img_clahe = clahe_lab(img_crop)
    images.append(img_clahe)
    titles.append("After CLAHE")

    # Step 3: Ben Graham Preprocessing
    img_resized = cv2.resize(img_clahe, (IMG_SIZE, IMG_SIZE))
    img_blur = cv2.GaussianBlur(img_resized, (0, 0), 10)
    img_final = cv2.addWeighted(img_resized, 4, img_blur, -4, 128)
    images.append(img_final)
    titles.append("After Ben Graham")

    # Plotting all steps side-by-side
    n = len(images)
    plt.figure(figsize=(5 * n, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i], fontsize=14)
        plt.axis("off")
    plt.suptitle("Preprocessing Journey", fontsize=18)
    plt.tight_layout()
    plt.show()

    print("✅ Debug visualization completed!")
