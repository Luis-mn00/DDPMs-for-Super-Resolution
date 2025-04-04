import os
import torch
from skimage import transform
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

from utils import *
from datasets import KolmogorovFlowDataset, LucaFlowDataset
from network_swinir import SwinIR

def save_dataset(low_res_images, high_res_images, dataset_path):
    np.save(os.path.join(dataset_path, 'low_res_images.npy'), low_res_images)
    np.save(os.path.join(dataset_path, 'high_res_images.npy'), high_res_images)

# Function to load dataset from disk
def load_dataset(dataset_path):
    low_res_images = np.load(os.path.join(dataset_path, 'low_res_images.npy'))
    high_res_images = np.load(os.path.join(dataset_path, 'high_res_images.npy'))
    return low_res_images, high_res_images

class KolmogorovFlowDatasetSuperRes(Dataset):
    def __init__(self, low_res_images, high_res_images, transform=None):
        self.low_res_images = low_res_images
        self.high_res_images = high_res_images
        self.transform = transform

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        low_res = self.low_res_images[idx]
        high_res = self.high_res_images[idx]

        if self.transform:
            low_res = self.transform(low_res)
            high_res = self.transform(high_res)

        return low_res, high_res

# Define a simple transformation for the dataset
transform = transforms.Compose([
    lambda x: np.transpose(x, (1, 2, 0)) if x.shape[0] == 3 else x,  # Asegurar (H, W, C)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizar a [-1, 1]
])

def denormalize(img):
    return img * 0.5 + 0.5

# Generate data using the KolmogorovFlowDataset and interpolation
seed = 1234
dataset = KolmogorovFlowDataset(seed=seed)
Y = dataset.data  # High resolution images
print(Y.shape)
Nx = Y.shape[-2]
Ny = Y.shape[-1]
perc = 0.05
interp = "nearest"

#X, sampled_ids = interpolate_dataset(Y, perc, method=interp)  # Blurred (low-resolution) images
#save_dataset(X, Y, "dataset_5")
X, Y = load_dataset("dataset_5")

# Split dataset into training, validation, and test sets
train_X, val_X, test_X = X[:3000], X[3000:3240], X[3240:]  # First 3000 for training, next 240 for validation, last 1000 for testing
train_Y, val_Y, test_Y = Y[:3000], Y[3000:3240], Y[3240:]

# Create DataLoaders
def create_dataloader(low_res_images, high_res_images, batch_size):
    dataset = KolmogorovFlowDatasetSuperRes(low_res_images, high_res_images, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = create_dataloader(train_X, train_Y, batch_size=1)
val_loader = create_dataloader(val_X, val_Y, batch_size=1)
test_loader = create_dataloader(test_X, test_Y, batch_size=1)

# Inicializar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

upscale = 1
window_size = 8
model = SwinIR(
    upscale=upscale,
    img_size=(256, 256),
    window_size=window_size,
    img_range=1.,
    depths=[6, 6, 6, 6],
    embed_dim=60,
    num_heads=[6, 6, 6, 6],
    mlp_ratio=2,
    upsampler=None  # No pixel shuffle, since we are not upsampling
)
model = model.to(device)
print(next(model.parameters()).shape)


# Configurar optimizador y función de pérdida
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Entrenamiento
checkpoint_dir = "checkpoints_swin_5"
os.makedirs(checkpoint_dir, exist_ok=True)

num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()

    # Compute validation loss
    model.eval()  # Switch model to evaluation mode
    total_val_loss = 0
    with torch.no_grad():  # Turn off gradient calculation to save memory
        for batch in val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            output = model(x)
            val_loss = criterion(output, y)
            total_val_loss += val_loss.item()

    # Print the training and validation loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_train_loss/len(train_loader):.6f}, Validation Loss: {total_val_loss/len(val_loader):.6f}")
    
    # Guardar el modelo cada 10 épocas
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"swinir_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)