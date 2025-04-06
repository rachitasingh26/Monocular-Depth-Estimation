import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from models_baseline import SimpleUNet  # Import the model from models_baseline.py

# Simple Dataset for NYU Depth V2
class NYUDepthDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.rgb_paths = []
        self.depth_paths = []
        
        # Load dataset paths
        split = 'train' if is_train else 'test'
        rgb_dir = os.path.join(root_dir, 'rgb', split)
        depth_dir = os.path.join(root_dir, 'depth', split)
        
        for file_name in os.listdir(rgb_dir):
            if file_name.endswith('.png') or file_name.endswith('.jpg'):
                rgb_path = os.path.join(rgb_dir, file_name)
                depth_path = os.path.join(depth_dir, file_name)
                
                if os.path.exists(depth_path):
                    self.rgb_paths.append(rgb_path)
                    self.depth_paths.append(depth_path)

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        # Load RGB image
        rgb_path = self.rgb_paths[idx]
        rgb_image = Image.open(rgb_path).convert('RGB')
        
        # Load depth image
        depth_path = self.depth_paths[idx]
        depth_image = Image.open(depth_path)
        
        # Convert to numpy arrays
        rgb_np = np.array(rgb_image)
        depth_np = np.array(depth_image).astype(np.float32)
        
        # Normalize depth to [0, 1]
        if np.max(depth_np) > 0:
            depth_np = depth_np / np.max(depth_np)
        
        # Apply transformations
        if self.transform:
            rgb_np = self.transform(rgb_np)
            depth_np = self.transform(depth_np, is_depth=True)
        
        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb_np.transpose(2, 0, 1)).float()
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).float()
        
        return rgb_tensor, depth_tensor

# Simple transforms
class SimpleTransform:
    def __init__(self, input_size=(256, 256)):
        self.input_size = input_size
    
    def __call__(self, image, is_depth=False):
        # Resize image
        interpolation = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
        image = cv2.resize(image, self.input_size, interpolation=interpolation)
        return image

# Simple loss function
class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss, self).__init__()
    
    def forward(self, pred, target):
        # L1 loss
        l1_loss = F.l1_loss(pred, target)
        
        # Gradient loss
        dy_true, dx_true = self._gradient(target)
        dy_pred, dx_pred = self._gradient(pred)
        grad_loss = F.l1_loss(dy_pred, dy_true) + F.l1_loss(dx_pred, dx_true)
        
        # Combined loss
        loss = l1_loss + 0.5 * grad_loss
        return loss
    
    def _gradient(self, x):
        # Compute gradients
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        return dy, dx

# Training function
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10, save_dir='./models'):
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            
            # Iterate over data
            for inputs, targets in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f}')
            
            # Save best model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                print(f'Saved best model with loss {best_loss:.4f}')
        
        print()
    
    return model

# Visualization function
def visualize_results(model, dataloader, device, num_samples=5):
    model.eval()
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_samples:
                break
                
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Convert tensors to numpy for visualization
            input_np = inputs[0].cpu().permute(1, 2, 0).numpy()
            target_np = targets[0, 0].cpu().numpy()
            output_np = outputs[0, 0].cpu().numpy()
            
            # Normalize for visualization
            input_np = (input_np * 255).astype(np.uint8)
            
            # Create figure
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.title('Input RGB')
            plt.imshow(input_np)
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.title('Ground Truth Depth')
            plt.imshow(target_np, cmap='viridis')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.title('Predicted Depth')
            plt.imshow(output_np, cmap='viridis')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'result_{i}.png')
            plt.close()

# Main function
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Configuration
    config = {
        'dataset_path': './data/nyu',
        'batch_size': 8,
        'num_workers': 4,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'input_size': (256, 256),
        'save_dir': './models',
    }
    
    # Create transforms
    transform = SimpleTransform(input_size=config['input_size'])
    
    # Create datasets
    train_dataset = NYUDepthDataset(
        root_dir=config['dataset_path'],
        transform=transform,
        is_train=True
    )
    
    val_dataset = NYUDepthDataset(
        root_dir=config['dataset_path'],
        transform=transform,
        is_train=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    # Create model
    model = SimpleUNet(pretrained=True).to(device)
    
    # Create loss function
    criterion = DepthLoss()
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Train the model
    model = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=config['num_epochs'],
        save_dir=config['save_dir']
    )
    
    # Visualize results
    visualize_results(model, val_loader, device)

if __name__ == '__main__':
    main()
