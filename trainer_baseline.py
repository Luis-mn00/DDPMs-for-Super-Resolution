import os
import torch
from skimage import transform
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

from utils import *
from networks_baseline import Model_base
from datasets import KolmogorovFlowDataset, LucaFlowDataset
from residuals import ResidualOp

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

# Generate data using the KolmogorovFlowDataset and interpolation
seed = 1234
dataset = KolmogorovFlowDataset(seed=seed)
Y = dataset.data  # High resolution images
print(Y.shape)
Nx = Y.shape[-2]
Ny = Y.shape[-1]
perc = 0.05
interp = "nearest"

#X, sampled_ids = interpolate_dataset(Y, perc, method=interp)
#save_dataset(X, Y, "dataset_5")
X, Y = load_dataset("dataset_5")

# Split dataset into training, validation, and test sets
train_X, val_X, test_X = X[:3000], X[3000:3240], X[3240:]  # First 3000 for training, next 240 for validation, last 1000 for testing
train_Y, val_Y, test_Y = Y[:3000], Y[3000:3240], Y[3240:]

# Create DataLoaders
def create_dataloader(low_res_images, high_res_images, batch_size):
    dataset = KolmogorovFlowDatasetSuperRes(low_res_images, high_res_images, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = create_dataloader(train_X, train_Y, batch_size=8)
val_loader = create_dataloader(val_X, val_Y, batch_size=8)
test_loader = create_dataloader(test_X, test_Y, batch_size=8)

# Model config (example)
class Config:
    class model:
        ch = 64
        out_ch = 3 
        ch_mult = [1, 1, 1, 2]
        num_res_blocks = 1
        attn_resolutions = [16, ]
        dropout = 0.0
        in_channels = 3  
        padding_mode = 'reflect'
        resamp_with_conv = True
    
    class data:
        image_size = 256

# Initialize model
config = Config()
model = Model_base(config)
print("Model initialized")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 1000
coef = 1e-2 # Coefficient for physics loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

checkpoint_dir = 'checkpoints_5'
os.makedirs(checkpoint_dir, exist_ok=True)

# Training and Validation Loop
residual_op = ResidualOp(device=device, Nx=Nx, Ny=Ny)
print("Training started")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    index = 1
    for low_res, high_res in train_loader:
        index += 1
        low_res, high_res = low_res.to(device), high_res.to(device)
        
        # Forward pass
        output = model(low_res)
        loss_data = criterion(output, high_res)
        
        #eq_residual, _ = residual_op(output)
        #loss_physics = torch.sqrt(torch.mean(eq_residual**2))
        
        loss = loss_data #+ coef * loss_physics
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Data Loss: {avg_loss:.4f}")
    
    # Validation Step
    if epoch % 10 == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for low_res, high_res in val_loader:
                low_res, high_res = low_res.to(device), high_res.to(device)
                output = model(low_res)
                loss = criterion(output, high_res)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")
        
        # Save the model every 10 epochs
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved at {checkpoint_path}")

# You can test the model with the test_loader at the end if needed
model.eval()
test_loss = 0.0
with torch.no_grad():
    for low_res, high_res in test_loader:
        low_res, high_res = low_res.to(device), high_res.to(device)
        output = model(low_res)
        loss = criterion(output, high_res)
        test_loss += loss.item()
print(f"Test Loss: {test_loss / len(test_loader):.4f}")
