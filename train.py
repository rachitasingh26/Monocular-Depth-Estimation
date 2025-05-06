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
import pandas as pd
from torch.optim.lr_scheduler import OneCycleLR
from torch.quantization import quantize_dynamic
from torch.cuda.amp import autocast, GradScaler
import torch.nn.parallel.data_parallel as DataParallel
from models import DepthEstimationStudent, DepthEstimationTeacher, CombinedLoss, LightweightDepthModel, DepthLoss, apply_pruning, remove_pruning, gradual_pruning, print_sparsity, compute_model_sparsity
import torch.nn.utils.prune as prune
from torch.nn.utils import parameters_to_vector, vector_to_parameters

# Custom Dataset for NYU Depth V2
class NYUDepthDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.rgb_paths = []
        self.depth_paths = []
        
        # NYU Depth V2 dataset structure
        split = 'train' if is_train else 'test'
        df = pd.read_csv(f"{root_dir}/data/nyu2_{split}.csv")
        
        for index, row in df.iterrows():
            rgb_path = f"{root_dir}/{row[0]}"
            depth_path = f"{root_dir}/{row[1]}"
            if os.path.exists(rgb_path) and os.path.exists(depth_path):
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
        if np.max(depth_np) > 0:
            depth_np = depth_np / np.max(depth_np)
        
        # Create confidence map (1 for valid depth, 0 for invalid)
        confidence_map = (depth_np > 0).astype(np.float32)
        
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
            if np.random.rand() > 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                contrast = np.random.uniform(0.8, 1.2)
                image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
            
            # Random flipping
            if np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
                confidence = np.fliplr(confidence).copy()
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = image.astype(np.float32)
        
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
        outputs = student_model(inputs)
    
    # Create visualization grid
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        # Convert tensors to numpy arrays
        input_np = inputs[i].cpu().permute(1, 2, 0).numpy()
        target_np = targets[i, 0].cpu().numpy()
        output_np = outputs[i, 0].cpu().numpy()
        
        # Normalize RGB image for display
        input_np = np.clip((input_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])), 0, 1)
        
        # Display images
        axes[i, 0].imshow(input_np)
        axes[i, 0].set_title('Input RGB')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(target_np, cmap='plasma')
        axes[i, 1].set_title('Ground Truth Depth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(output_np, cmap='plasma')
        axes[i, 2].set_title('Predicted Depth')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch}.png")
    plt.close()

# Calculate evaluation metrics
def calculate_metrics(pred, target, mask=None):
    """
    Calculate standard depth estimation metrics
    Args:
        pred: Predicted depth map
        target: Ground truth depth map
        mask: Optional mask for valid depth values
    Returns:
        Dictionary of metrics
    """
    if mask is None:
        mask = torch.ones_like(target)
    
    # Apply mask to predictions and targets
    pred = pred * mask
    target = target * mask
    
    # Number of valid pixels
    n_valid = torch.sum(mask)
    
    # Absolute relative error
    abs_rel = torch.sum(torch.abs(pred - target) / (target + 1e-10)) / n_valid
    
    # Squared relative error
    sq_rel = torch.sum(((pred - target) ** 2) / (target + 1e-10)) / n_valid
    
    # Root mean squared error
    rmse = torch.sqrt(torch.sum((pred - target) ** 2) / n_valid)
    
    # Log RMSE
    log_rmse = torch.sqrt(torch.sum((torch.log(pred + 1e-10) - torch.log(target + 1e-10)) ** 2) / n_valid)
    
    # Thresholded accuracy
    thresh_1 = torch.sum((torch.max(pred / (target + 1e-10), target / (pred + 1e-10)) < 1.25).float()) / n_valid
    thresh_2 = torch.sum((torch.max(pred / (target + 1e-10), target / (pred + 1e-10)) < 1.25 ** 2).float()) / n_valid
    thresh_3 = torch.sum((torch.max(pred / (target + 1e-10), target / (pred + 1e-10)) < 1.25 ** 3).float()) / n_valid
    
    return {
        'abs_rel': abs_rel.item(),
        'sq_rel': sq_rel.item(),
        'rmse': rmse.item(),
        'log_rmse': log_rmse.item(),
        'thresh_1': thresh_1.item(),
        'thresh_2': thresh_2.item(),
        'thresh_3': thresh_3.item()
    }

# Training function
def train_model(student_model, teacher_model, train_loader, val_loader, criterion, optimizer, 
                scheduler, device, num_epochs=30, save_dir='./checkpoints', 
                use_teacher=True, visualize_every=5,
                pruning_start_epoch=50, pruning_end_epoch=150, final_sparsity=0.5):
    """
    Training function for the depth estimation model
    """
    # Create directories for checkpoints and visualizations
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/visualizations", exist_ok=True)
    
    # Best validation metrics
    best_abs_rel = float('inf')
    best_epoch = 0
    
    scaler = GradScaler()

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'abs_rel': [],
        'rmse': [],
        'thresh_1': [],
        'sparsity': []  # Add tracking for sparsity
    }

    torch.backends.cudnn.benchmark = True
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        if epoch >= pruning_start_epoch and epoch <= pruning_end_epoch:
            # Calculate current sparsity using cubic schedule
            progress = (epoch - pruning_start_epoch) / (pruning_end_epoch - pruning_start_epoch)
            current_sparsity = 0.0 + (final_sparsity - 0.0) * (1.0 - (1.0 - progress) ** 3)
            
            print(f"Applying pruning with sparsity {current_sparsity:.4f}")
            
            # Remove any existing pruning
            for name, module in student_model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if hasattr(module, 'weight_mask'):
                        prune.remove(module, 'weight')
            
            # Apply new level of pruning
            for name, module in student_model.named_modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    prune.l1_unstructured(module, name='weight', amount=current_sparsity)
            
            # Record current sparsity
            current_sparsity_actual = compute_model_sparsity(student_model)
            history['sparsity'].append(current_sparsity_actual)
            print(f"Current model sparsity: {current_sparsity_actual:.4f}")
        
        # Training phase
        student_model.train()
        if teacher_model is not None:
            teacher_model.eval()  # Teacher model is always in eval mode
            
        train_loss = 0.0
        si_loss_avg = 0.0
        grad_loss_avg = 0.0
        distill_loss_avg = 0.0
        
        # Progress bar
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch in progress_bar:
            # Move data to device
            inputs = batch['rgb'].to(device)
            targets = batch['depth'].to(device)
            masks = batch['confidence'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Use mixed precision for forward pass
            with autocast():
                # Forward pass through student model (now returns only depth)
                student_output = student_model(inputs)
                
                # Get teacher output if using knowledge distillation
                teacher_output = None
                if use_teacher and teacher_model is not None:
                    with torch.no_grad():
                        teacher_output = teacher_model(inputs)
                
                # Calculate loss
                loss, si_loss, l1_loss, distill_loss = criterion(student_output, targets, teacher_output, masks)
            
            # Check if loss is valid
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss value: {loss.item()}, skipping batch")
                continue
                
            # Backward pass and optimization with gradient scaling
            scaler.scale(loss).backward()
            
            # Add gradient clipping to prevent exploding gradients (works with AMP)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            
            # Step with scaler
            scaler.step(optimizer)
            scaler.update()
            
            # Update metrics
            train_loss += loss.detach().item()
            si_loss_avg += si_loss.detach().item()
            l1_loss_avg += l1_loss.detach().item()  # Changed from grad_loss_avg
            distill_loss_avg += distill_loss.detach().item() if distill_loss != 0 else 0

            # del si_loss, grad_loss, distill_loss
            # del student_output, teacher_output
            
            # Update progress bar description
            progress_bar.set_description(f"Training Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        # Calculate average metrics for the epoch
        train_loss /= len(train_loader)
        si_loss_avg /= len(train_loader)
        grad_loss_avg /= len(train_loader)
        distill_loss_avg /= len(train_loader)
        
        # Print training metrics
        print(f"Train Loss: {train_loss:.4f}, SI Loss: {si_loss_avg:.4f}, "
            f"L1 Loss: {l1_loss_avg:.4f}, Distill Loss: {distill_loss_avg:.4f}") 

        if epoch % 30 == 0 or epoch >= pruning_start_epoch:
            print_sparsity(student_model)
        
        # Validation phase
        student_model.eval()
        val_loss = 0.0
        metrics_avg = {
            'abs_rel': 0.0,
            'sq_rel': 0.0,
            'rmse': 0.0,
            'log_rmse': 0.0,
            'thresh_1': 0.0,
            'thresh_2': 0.0,
            'thresh_3': 0.0
        }
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                # Move data to device
                inputs = batch['rgb'].to(device)
                targets = batch['depth'].to(device)
                masks = batch['confidence'].to(device)
                
                # Forward pass
                student_output = student_model(inputs)
                
                # Calculate loss
                loss, _, _, _ = criterion(student_output, targets, None, masks)
                val_loss += loss.item()
                
                # Calculate metrics
                metrics = calculate_metrics(student_output[0], targets, masks)
                
                # Update metrics
                for k, v in metrics.items():
                    metrics_avg[k] += v
        
        # Calculate average validation metrics
        val_loss /= len(val_loader)
        for k in metrics_avg:
            metrics_avg[k] /= len(val_loader)
        
        # Print validation metrics
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Abs Rel: {metrics_avg['abs_rel']:.4f}, RMSE: {metrics_avg['rmse']:.4f}, "
              f"δ < 1.25: {metrics_avg['thresh_1']:.4f}")
        
        # Update best model if current one is better
        if metrics_avg['abs_rel'] < best_abs_rel:
            best_abs_rel = metrics_avg['abs_rel']
            best_epoch = epoch
            
            # Save best model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_model.module.state_dict() if isinstance(student_model, torch.nn.DataParallel) else student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics_avg
            }, f"{save_dir}/best_model.pth")
            
            print(f"New best model saved at epoch {epoch+1}!")
        
        # Save regular checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': student_model.module.state_dict() if isinstance(student_model, torch.nn.DataParallel) else student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics_avg
        }, f"{save_dir}/checkpoint_epoch_{epoch+1}.pth")
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(metrics_avg['abs_rel'])  # Use abs_rel as the metric to monitor
        
        # Visualize predictions
        if (epoch + 1) % visualize_every == 0 or epoch == 0:
            visualize_predictions(student_model, val_loader, device, epoch+1, save_dir=f"{save_dir}/visualizations")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['abs_rel'].append(metrics_avg['abs_rel'])
        history['rmse'].append(metrics_avg['rmse'])
        history['thresh_1'].append(metrics_avg['thresh_1'])
        
        # Save training history
        np.save(f"{save_dir}/history.npy", history)
        
        # Plot learning curves
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(history['abs_rel'], label='Abs Rel Error')
        plt.plot(history['rmse'], label='RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(history['thresh_1'], label='δ < 1.25')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/learning_curves.png")
        plt.close()
    
        torch.cuda.empty_cache()
        import gc; gc.collect()
    print(f"Training completed! Best model saved at epoch {best_epoch+1} with Abs Rel: {best_abs_rel:.4f}")
    return history

# Evaluate function for testing
def evaluate_model(model, test_loader, device, save_dir='./results'):
    """
    Evaluate model on test set
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    metrics_avg = {
        'abs_rel': 0.0,
        'sq_rel': 0.0,
        'rmse': 0.0,
        'log_rmse': 0.0,
        'thresh_1': 0.0,
        'thresh_2': 0.0,
        'thresh_3': 0.0
    }
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Move data to device
            inputs = batch['rgb'].to(device)
            targets = batch['depth'].to(device)
            masks = batch['confidence'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate metrics
            metrics = calculate_metrics(outputs, targets, masks)
            
            # Update metrics
            for k, v in metrics.items():
                metrics_avg[k] += v
    
    # Calculate average metrics
    for k in metrics_avg:
        metrics_avg[k] /= len(test_loader)
    
    # Print test metrics
    print("Test Results:")
    print(f"Abs Rel: {metrics_avg['abs_rel']:.4f}")
    print(f"Sq Rel: {metrics_avg['sq_rel']:.4f}")
    print(f"RMSE: {metrics_avg['rmse']:.4f}")
    print(f"Log RMSE: {metrics_avg['log_rmse']:.4f}")
    print(f"δ < 1.25: {metrics_avg['thresh_1']:.4f}")
    print(f"δ < 1.25^2: {metrics_avg['thresh_2']:.4f}")
    print(f"δ < 1.25^3: {metrics_avg['thresh_3']:.4f}")
    
    # Save results to file
    with open(f"{save_dir}/test_results.txt", 'w') as f:
        for k, v in metrics_avg.items():
            f.write(f"{k}: {v:.4f}\n")
    
    return metrics_avg

# Function to create quantized model for deployment
def create_quantized_model(model_path, save_path):
    """
    Create quantized model for deployment
    """
    # Load the best model
    checkpoint = torch.load(model_path)
    model = DepthEstimationStudent(pretrained=False)
    # Handle both DataParallel and non-DataParallel state dicts
    if 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
        # Remove 'module.' prefix
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] # remove 'module.'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Quantize the model
    quantized_model = quantize_dynamic(
        model,
        {nn.Conv2d, nn.Linear},
        dtype=torch.qint8
    )
    
    # Save the quantized model
    torch.save(quantized_model.state_dict(), save_path)
    
    return quantized_model

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Hyperparameters
    batch_size = 16
    learning_rate = 5e-5
    num_epochs = 20
    input_size = (480, 640)  # Height, Width
    use_teacher = True  # Whether to use knowledge distillation
    
    # Dataset and dataloader
    data_root = "/projectnb/dl4ds/materials/datasets/monocular-depth-estimation/nyuv2/nyu_data"  # Path to NYU Depth V2 dataset
    
    # Create transforms
    train_transform = Transforms(input_size=input_size, is_train=True)
    val_transform = Transforms(input_size=input_size, is_train=False)
    
    # Create datasets
    train_dataset = NYUDepthDataset(root_dir=data_root, transform=train_transform, is_train=True)
    val_dataset = NYUDepthDataset(root_dir=data_root, transform=val_transform, is_train=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create models
    student_model = DepthEstimationStudent(pretrained=True)
    student_model = student_model.to(device)
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs for training!")
    #     student_model = torch.nn.DataParallel(student_model)

    teacher_model = None
    if use_teacher:
        teacher_model = DepthEstimationTeacher()
        teacher_model = teacher_model.to(device)
        # if torch.cuda.device_count() > 1:
        #     teacher_model = torch.nn.DataParallel(teacher_model)
    
    # Create loss function
    criterion = CombinedLoss(
        si_weight=1.0,     # Scale-invariant loss
        l1_weight=1.0,     # L1 loss (replaced gradient loss)
        distill_weight=0.5 if use_teacher else 0.0  # Knowledge distillation
    )
    
    # Create optimizer
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Create learning rate scheduler
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=3, verbose=True
    # )
    total_steps = len(train_loader) * num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=0.3,  # Spend 30% of training warming up
        div_factor=25,  # Initial learning rate = max_lr/25
        final_div_factor=10000,  # Final learning rate = max_lr/10000
        anneal_strategy='cos'  # Use cosine annealing
    )
    
    # Create directory for results
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Train the model
    history = train_model(
        student_model=student_model,
        teacher_model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        save_dir=results_dir,
        use_teacher=use_teacher,
        visualize_every=5
    )
    
    # Load the best model for evaluation
    best_model_path = f"{results_dir}/best_model.pth"
    checkpoint = torch.load(best_model_path)
    student_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_metrics = evaluate_model(student_model, val_loader, device, save_dir=results_dir)
    
    # Create quantized model for deployment
    quantized_model = create_quantized_model(
        model_path=best_model_path,
        save_path=f"{results_dir}/quantized_model.pth"
    )
    
    print("Training, evaluation, and model quantization completed!")

if __name__ == "__main__":
    main()