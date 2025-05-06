import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import argparse
from models import DepthEstimationStudent

# Custom Dataset for inference
class InferenceDataset(Dataset):
    def __init__(self, image_dir, transform=None, input_size=(480, 640)):
        self.image_dir = image_dir
        self.transform = transform
        self.input_size = input_size
        self.image_paths = []
        
        # List all image files (jpg, jpeg, png)
        valid_extensions = ['.jpg', '.jpeg', '.png']
        for file in os.listdir(image_dir):
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                self.image_paths.append(os.path.join(image_dir, file))
                
        # Sort for consistent order
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load RGB image
        image_path = self.image_paths[idx]
        rgb_image = Image.open(image_path).convert('RGB')
        
        # Convert to numpy array
        rgb_np = np.array(rgb_image)
        
        # Apply preprocessing
        if self.transform:
            processed = self.transform(rgb_np)
            rgb_np = processed['image']
        
        # Convert to tensor
        rgb_tensor = torch.from_numpy(rgb_np.transpose(2, 0, 1)).float()
        
        return {
            'rgb': rgb_tensor,
            'path': image_path
        }

# Simple preprocessing transforms (based on the training code)
class InferenceTransform:
    def __init__(self, input_size=(480, 640)):
        self.input_size = input_size

    def __call__(self, image):
        # Resize to input size
        image = cv2.resize(image, self.input_size[::-1])
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = image.astype(np.float32)
        
        return {'image': image}

# Visualization function (adapted from train.py)
def visualize_predictions(model, inputs, outputs, paths, save_dir='./predictions', batch_idx=0):
    """
    Visualize predictions from the model
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = inputs.shape[0]
    
    # Create visualization grid
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5*batch_size))
    
    # Handle single image case
    if batch_size == 1:
        axes = axes.reshape(1, 2)
    
    for i in range(batch_size):
        # Convert tensors to numpy arrays
        input_np = inputs[i].cpu().permute(1, 2, 0).numpy()
        output_np = outputs[i, 0].cpu().numpy()
        
        # Normalize RGB image for display
        input_np = np.clip((input_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])), 0, 1)
        
        # Get filename for saving
        filename = os.path.basename(paths[i])
        
        # Display images
        axes[i, 0].imshow(input_np)
        axes[i, 0].set_title('Input RGB')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(output_np, cmap='plasma')
        axes[i, 1].set_title('Predicted Depth')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/batch_{batch_idx}.png")
    plt.close()
    
    # Save individual depth maps as well
    for i in range(batch_size):
        output_np = outputs[i, 0].cpu().numpy()
        filename = os.path.basename(paths[i])
        base_name = os.path.splitext(filename)[0]
        
        # Create a colormap for better visualization
        cm = plt.cm.get_cmap('plasma')
        colored_depth = cm(output_np)
        colored_depth = (colored_depth[:, :, :3] * 255).astype(np.uint8)
        
        # Save depth image
        cv2.imwrite(f"{save_dir}/{base_name}_depth.png", cv2.cvtColor(colored_depth, cv2.COLOR_RGB2BGR))

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Depth Estimation Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='Directory to save predictions')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--img_size', type=str, default='480,640', help='Image size (height,width)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()
    
    # Parse image size
    height, width = map(int, args.img_size.split(','))
    input_size = (height, width)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = DepthEstimationStudent(pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Handle both DataParallel and non-DataParallel state dicts
    if 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']
    
    if list(checkpoint.keys())[0].startswith('module.'):
        # Remove 'module.' prefix from DataParallel
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith('module.') else k  # remove 'module.'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Create transforms
    transform = InferenceTransform(input_size=input_size)
    
    # Create dataset and dataloader
    dataset = InferenceDataset(args.image_dir, transform=transform, input_size=input_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Found {len(dataset)} images in {args.image_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run inference
    total_time = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
            # Move data to device
            inputs = batch['rgb'].to(device)
            paths = batch['path']
            
            # Record start time
            start_time = time.time()
            
            # Forward pass
            outputs = model(inputs)
            
            # Synchronize CUDA operations
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Record end time
            end_time = time.time()
            
            # Calculate batch inference time
            batch_time = end_time - start_time
            batch_size = inputs.shape[0]
            total_time += batch_time
            total_samples += batch_size
            
            # Visualize predictions
            visualize_predictions(model, inputs, outputs, paths, save_dir=args.output_dir, batch_idx=batch_idx)
            
            # Print batch stats
            print(f"Batch {batch_idx+1}/{len(dataloader)}: {batch_time:.4f} seconds, {batch_size/batch_time:.2f} images/second")
            
    # Print total stats
    print(f"\nTotal inference time: {total_time:.4f} seconds")
    print(f"Average time per image: {total_time/total_samples:.4f} seconds")
    print(f"Average throughput: {total_samples/total_time:.2f} images/second")
    print(f"Processed {total_samples} images")
    print(f"Predictions saved to {args.output_dir}")

if __name__ == "__main__":
    main()