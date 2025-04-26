# Import necessary libraries for pruning
import torch.nn.utils.prune as prune
from torch.nn.utils import remove_parameters

# Function to apply unstructured pruning to model
def apply_unstructured_pruning(model, amount=0.5):
    """
    Apply unstructured pruning to the model
    Args:
        model: The PyTorch model to prune
        amount: The fraction of parameters to prune (between 0 and 1)
    """
    for name, module in model.named_modules():
        # Apply pruning to convolutional and linear layers
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            
    return model

# Function to apply structured pruning (channel pruning)
def apply_structured_pruning(model, amount=0.3):
    """
    Apply structured pruning (channel pruning) to the model
    Args:
        model: The PyTorch model to prune
        amount: The fraction of channels to prune (between 0 and 1)
    """
    # Apply to specific parts of the encoder
    for idx, module in enumerate(model.encoder):
        if isinstance(module, nn.Conv2d) and module.out_channels > 8:  # Ensure we don't prune too many channels
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)  # Prune output channels
    
    # Apply to decoder blocks
    for idx, block in enumerate(model.decoder):
        for layer in block:
            if isinstance(layer, nn.Conv2d) and layer.out_channels > 8:
                prune.ln_structured(layer, name='weight', amount=amount, n=2, dim=0)
    
    return model

# Function to make pruning permanent and remove the pruning buffers
def make_pruning_permanent(model):
    """
    Make pruning permanent by removing the pruning buffers
    Args:
        model: The pruned PyTorch model
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
            if hasattr(module, 'bias_mask'):
                prune.remove(module, 'bias')
    
    return model

# Function for gradual pruning during training
def gradual_pruning_schedule(initial_amount, final_amount, begin_step, end_step, current_step):
    """
    Calculate pruning amount for gradual pruning
    Args:
        initial_amount: Initial pruning percentage (e.g., 0.05 for 5%)
        final_amount: Final pruning percentage (e.g., 0.85 for 85%)
        begin_step: Step to begin pruning
        end_step: Step to end pruning
        current_step: Current training step
    Returns:
        Current pruning amount
    """
    if current_step < begin_step:
        return initial_amount
    
    if current_step >= end_step:
        return final_amount
    
    # Linear pruning schedule
    pruning_amount = initial_amount + (final_amount - initial_amount) * \
                    (current_step - begin_step) / (end_step - begin_step)
    
    return pruning_amount

# Update the training function to include pruning
def train_model_with_pruning(student_model, teacher_model, train_loader, val_loader, criterion, optimizer, 
                scheduler, device, num_epochs=30, save_dir='./checkpoints', 
                use_teacher=True, visualize_every=5, 
                pruning_config=None):
    """
    Training function for the depth estimation model with pruning
    Args:
        pruning_config: Dictionary containing pruning configuration
            - method: 'unstructured', 'structured', or 'gradual'
            - amount: Pruning amount (for non-gradual)
            - initial_amount: Initial pruning amount (for gradual)
            - final_amount: Final pruning amount (for gradual)
            - begin_epoch: Epoch to begin pruning (for gradual)
            - end_epoch: Epoch to end pruning (for gradual)
            - frequency: Pruning frequency in epochs (for gradual)
    """
    # Create directories for checkpoints and visualizations
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/visualizations", exist_ok=True)
    
    # Best validation metrics
    best_abs_rel = float('inf')
    best_epoch = 0
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'abs_rel': [],
        'rmse': [],
        'thresh_1': [],
        'pruning_amount': []
    }
    
    # Global step counter for gradual pruning
    global_step = 0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Apply pruning based on configuration
        if pruning_config is not None:
            pruning_method = pruning_config.get('method', 'none')
            
            if pruning_method == 'unstructured':
                # Apply one-shot unstructured pruning at the specified epoch
                if epoch == pruning_config.get('start_epoch', 0):
                    print(f"Applying unstructured pruning with amount {pruning_config['amount']}")
                    student_model = apply_unstructured_pruning(student_model, pruning_config['amount'])
            
            elif pruning_method == 'structured':
                # Apply one-shot structured pruning at the specified epoch
                if epoch == pruning_config.get('start_epoch', 0):
                    print(f"Applying structured pruning with amount {pruning_config['amount']}")
                    student_model = apply_structured_pruning(student_model, pruning_config['amount'])
            
            elif pruning_method == 'gradual':
                # Apply gradual pruning at the specified frequency
                begin_epoch = pruning_config.get('begin_epoch', 5)
                end_epoch = pruning_config.get('end_epoch', num_epochs - 5)
                frequency = pruning_config.get('frequency', 2)
                
                if epoch >= begin_epoch and epoch <= end_epoch and (epoch - begin_epoch) % frequency == 0:
                    # Calculate current pruning amount based on schedule
                    current_amount = gradual_pruning_schedule(
                        pruning_config.get('initial_amount', 0.05),
                        pruning_config.get('final_amount', 0.7),
                        begin_epoch, end_epoch, epoch
                    )
                    
                    print(f"Applying gradual pruning step with amount {current_amount}")
                    if pruning_config.get('pruning_type', 'unstructured') == 'unstructured':
                        student_model = apply_unstructured_pruning(student_model, current_amount)
                    else:
                        student_model = apply_structured_pruning(student_model, current_amount)
                    
                    # Make pruning permanent before the next pruning step
                    student_model = make_pruning_permanent(student_model)
                    
                    # Record pruning amount
                    history['pruning_amount'].append((epoch, current_amount))
        
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
            global_step += 1
            
            # Move data to device
            inputs = batch['rgb'].to(device)
            targets = batch['depth'].to(device)
            masks = batch['confidence'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass through student model
            student_output = student_model(inputs)
            
            # Get teacher output if using knowledge distillation
            teacher_output = None
            if use_teacher and teacher_model is not None:
                with torch.no_grad():
                    teacher_output = teacher_model(inputs)
            
            # Calculate loss
            loss, si_loss, grad_loss, distill_loss = criterion(student_output, targets, teacher_output, masks)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            si_loss_avg += si_loss.item()
            grad_loss_avg += grad_loss.item()
            distill_loss_avg += distill_loss.item() if distill_loss != 0 else 0
            
            # Update progress bar description
            progress_bar.set_description(f"Training Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        # Calculate average metrics for the epoch
        train_loss /= len(train_loader)
        si_loss_avg /= len(train_loader)
        grad_loss_avg /= len(train_loader)
        distill_loss_avg /= len(train_loader)
        
        # Print training metrics
        print(f"Train Loss: {train_loss:.4f}, SI Loss: {si_loss_avg:.4f}, "
              f"Grad Loss: {grad_loss_avg:.4f}, Distill Loss: {distill_loss_avg:.4f}")
        
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
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics_avg
            }, f"{save_dir}/best_model.pth")
            
            # Also save a pruned version where pruning is made permanent
            pruned_model = make_pruning_permanent(copy.deepcopy(student_model))
            torch.save({
                'epoch': epoch,
                'model_state_dict': pruned_model.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics_avg
            }, f"{save_dir}/best_model_pruned.pth")
            
            print(f"New best model saved at epoch {epoch+1}!")
        
        # Save regular checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': student_model.state_dict(),
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
        
        # If we have pruning data, create a pruning plot
        if len(history.get('pruning_amount', [])) > 0:
            plt.figure(figsize=(10, 5))
            pruning_epochs, pruning_amounts = zip(*history['pruning_amount'])
            plt.plot(pruning_epochs, pruning_amounts, marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Pruning Amount')
            plt.title('Pruning Schedule')
            plt.grid(True)
            plt.savefig(f"{save_dir}/pruning_schedule.png")
            plt.close()
    
    # At the end of training, make pruning permanent for the final model
    if pruning_config is not None:
        final_pruned_model = make_pruning_permanent(copy.deepcopy(student_model))
        torch.save({
            'epoch': num_epochs - 1,
            'model_state_dict': final_pruned_model.state_dict(),
            'metrics': metrics_avg
        }, f"{save_dir}/final_pruned_model.pth")
    
    print(f"Training completed! Best model saved at epoch {best_epoch+1} with Abs Rel: {best_abs_rel:.4f}")
    return history

# Calculate model sparsity to monitor pruning effect
def calculate_sparsity(model):
    """
    Calculate the sparsity percentage of a model
    Args:
        model: PyTorch model
    Returns:
        Sparsity percentage
    """
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    
    sparsity = 100.0 * zero_params / total_params if total_params > 0 else 0
    return sparsity

# Function to analyze model efficiency metrics
def analyze_model_efficiency(model, input_shape=(1, 3, 480, 640), device='cpu'):
    """
    Analyze model efficiency metrics like parameter count, MACs, inference time
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to run analysis on
    """
    from thop import profile
    import time
    
    # Set model to evaluation mode
    model.eval()
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Compute MACs and FLOPs
    input_tensor = torch.randn(input_shape).to(device)
    macs, params = profile(model, inputs=(input_tensor,))
    
    # Measure inference time
    warmup_runs = 10
    benchmark_runs = 50
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(benchmark_runs):
            _ = model(input_tensor)
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / benchmark_runs * 1000  # Convert to ms
    
    # Calculate model sparsity
    sparsity = calculate_sparsity(model)
    
    # Print results
    print(f"Model Efficiency Analysis:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model sparsity: {sparsity:.2f}%")
    print(f"MACs: {macs / 1e6:.2f} M")
    print(f"Average inference time: {avg_inference_time:.2f} ms")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'sparsity': sparsity,
        'macs': macs,
        'inference_time': avg_inference_time
    }

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Hyperparameters
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 30
    input_size = (480, 640)  # Height, Width
    use_teacher = True  # Whether to use knowledge distillation
    
    # Pruning configuration
    pruning_config = {
        'method': 'gradual',           # 'none', 'unstructured', 'structured', or 'gradual'
        'pruning_type': 'unstructured', # For gradual pruning: 'unstructured' or 'structured'
        'amount': 0.5,                 # For one-shot pruning
        'start_epoch': 5,              # For one-shot pruning
        'initial_amount': 0.05,        # For gradual pruning
        'final_amount': 0.7,           # For gradual pruning
        'begin_epoch': 5,              # For gradual pruning
        'end_epoch': 25,               # For gradual pruning
        'frequency': 2                 # For gradual pruning (apply every N epochs)
    }
    
    # Dataset and dataloader
    data_root = "./nyu_depth_v2"  # Path to NYU Depth V2 dataset
    
    # Create transforms
    train_transform = Transforms(input_size=input_size, is_train=True)
    val_transform = Transforms(input_size=input_size, is_train=False)
    
    # Create datasets
    train_dataset = NYUDepthDataset(root_dir=data_root, transform=train_transform, is_train=True)
    val_dataset = NYUDepthDataset(root_dir=data_root, transform=val_transform, is_train=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create models
    student_model = DepthEstimationStudent(pretrained=True)
    student_model = student_model.to(device)
    
    teacher_model = None
    if use_teacher:
        teacher_model = DepthEstimationTeacher()
        teacher_model = teacher_model.to(device)
    
    # Create loss function
    criterion = CombinedLoss(
        si_weight=1.0,
        grad_weight=0.5,
        distill_weight=1.0 if use_teacher else 0.0,
        affinity_weight=0.5
    )
    
    # Create optimizer
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Create directory for results
    results_dir = "./results_pruned"
    os.makedirs(results_dir, exist_ok=True)
    
    # Analyze model before pruning
    print("Analyzing model before pruning:")
    pre_pruning_metrics = analyze_model_efficiency(student_model, device=device)
    
    # Train the model with pruning
    history = train_model_with_pruning(
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
        visualize_every=5,
        pruning_config=pruning_config
    )
    
    # Load the best pruned model for evaluation
    best_pruned_model_path = f"{results_dir}/best_model_pruned.pth"
    checkpoint = torch.load(best_pruned_model_path)
    
    # Create a new model instance
    pruned_model = DepthEstimationStudent(pretrained=False)
    pruned_model.load_state_dict(checkpoint['model_state_dict'])
    pruned_model = pruned_model.to(device)
    
    # Analyze model after pruning
    print("Analyzing model after pruning:")
    post_pruning_metrics = analyze_model_efficiency(pruned_model, device=device)
    
    # Print comparison
    print("\nModel Size Reduction:")
    print(f"Pre-pruning parameters: {pre_pruning_metrics['total_params']:,}")
    print(f"Post-pruning parameters: {post_pruning_metrics['total_params']:,}")
    print(f"Parameter reduction: {100 * (1 - post_pruning_metrics['total_params'] / pre_pruning_metrics['total_params']):.2f}%")
    print(f"Sparsity: {post_pruning_metrics['sparsity']:.2f}%")
    print(f"Inference time reduction: {100 * (1 - post_pruning_metrics['inference_time'] / pre_pruning_metrics['inference_time']):.2f}%")
    
    # Evaluate on test set
    test_metrics = evaluate_model(pruned_model, val_loader, device, save_dir=results_dir)
    
    # Create quantized model from the pruned model
    quantized_pruned_model = create_quantized_model(
        model_path=best_pruned_model_path,
        save_path=f"{results_dir}/quantized_pruned_model.pth"
    )
    
    # Analyze quantized pruned model
    print("Analyzing quantized pruned model:")
    quantized_metrics = analyze_model_efficiency(quantized_pruned_model, device='cpu')  # Quantized models often run on CPU
    
    # Print final optimized model stats
    print("\nFinal optimized model (pruned + quantized):")
    print(f"Parameters: {quantized_metrics['total_params']:,}")
    print(f"Sparsity: {quantized_metrics['sparsity']:.2f}%")
    print(f"MACs: {quantized_metrics['macs'] / 1e6:.2f} M")
    print(f"Inference time: {quantized_metrics['inference_time']:.2f} ms")
    print(f"Size reduction from original: {100 * (1 - quantized_metrics['total_params'] / pre_pruning_metrics['total_params']):.2f}%")
    
    print("Training, pruning, quantization, and evaluation completed!")

if __name__ == "__main__":
    main()