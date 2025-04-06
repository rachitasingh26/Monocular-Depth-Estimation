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
import torchvision.models as models
import cv2
from torch.quantization import quantize_dynamic
from models import DepthEstimationStudent, DepthEstimationTeacher, CombinedLoss

# Custom Dataset for NYU Depth V2
class NYUDepthDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.rgb_paths = []
        self.depth_paths = []
        self._load_dataset()

    def _load_dataset(self):
        # NYU Depth V2 dataset structure
        split = 'train' if self.is_train else 'test'
        split_file = os.path.join(self.root_dir, f'{split}.txt')
        
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                for line in f:
                    rgb_path, depth_path = line.strip().split(' ')
                    self.rgb_paths.append(os.path.join(self.root_dir, rgb_path))
                    self.depth_paths.append(os.path.join(self.root_dir, depth_path))
        else:
            # Alternative structure if split file doesn't exist
            rgb_dir = os.path.join(self.root_dir, 'rgb', split)
            depth_dir = os.path.join(self.root_dir, 'depth', split)
            
            if os.path.exists(rgb_dir) and os.path.exists(depth_dir):
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
        
        # Create numpy arrays
        rgb_np = np.array(rgb_image)
        depth_np = np.array(depth_image).astype(np.float32)
        
        # Normalize depth to [0, 1]
        if np.max(depth_np) &gt; 0:
            depth_np = depth_np / np.max(depth_np)
        
        # Create confidence map (1 for valid depth, 0 for invalid)
        confidence_map = (depth_np &gt; 0).astype(np.float32)
        
        # Apply transformations
        if self.transform:
            # Apply same transform to both RGB and depth
            transformed = self.transform(image=rgb_np, mask=depth_np, confidence=confidence_map)
            rgb_np = transformed['image']
            depth_np = transformed['mask']
            confidence_map = transformed['confidence']
        
        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb_np.transpose(2, 0, 1)).float()
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).float()
        confidence_tensor = torch.from_numpy(confidence_map).unsqueeze(0).float()
        
        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'confidence': confidence_tensor,
            'rgb_path': rgb_path,
            'depth_path': depth_path
        }

# Simple data augmentation transforms
class Transforms:
    def __init__(self, input_size=(480, 640), is_train=True):
        self.input_size = input_size
        self.is_train = is_train

    def __call__(self, image, mask, confidence):
        # Resize to input size
        image = cv2.resize(image, self.input_size[::-1])
        mask = cv2.resize(mask, self.input_size[::-1], interpolation=cv2.INTER_NEAREST)
        confidence = cv2.resize(confidence, self.input_size[::-1], interpolation=cv2.INTER_NEAREST)
        
        if self.is_train:
            # Random brightness and contrast
            if np.random.rand() &gt; 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                contrast = np.random.uniform(0.8, 1.2)
                image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
            
            # Random flipping
            if np.random.rand() &gt; 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
                confidence = np.fliplr(confidence).copy()
        
        return {'image': image, 'mask': mask, 'confidence': confidence}

# Visualization function to check model outputs during training
def visualize_predictions(student_model, dataloader, device, epoch, save_dir='./visualizations', num_samples=4):
    """
    Visualize predictions from the student model during training
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set model to evaluation mode
    student_model.eval()
    
    # Get a batch of data
    batch = next(iter(dataloader))
    inputs = batch['rgb'].to(device)
    targets = batch['depth'].to(device)
    
    # Only process a subset of the batch
    inputs = inputs[:num_samples]
    targets = targets[:num_samples]
    
    # Forward pass
    with torch.no_grad():
        outputs, _ = student_model(inputs)
    
    # Create visualization grid
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        # Convert tensors to numpy arrays
        input_np = inputs[i].cpu().permute(1, 2, 0).numpy()
        target_np = targets[i, 0].cpu().numpy()
        output_np = outputs[i, 0].cpu().numpy()
        
        # Normalize RGB image for display
        input_np =