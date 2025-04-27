import pandas as pd
import numpy as np
import torch
import os

def create_labels(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    label_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    labels_np = df[label_cols].values
    labels = torch.tensor(labels_np, dtype=torch.uint8)
    
    np.save(os.path.join(output_dir, 'labels.npy'), labels.numpy())
    torch.save(labels, os.path.join(output_dir, 'labels.pt'))
    
