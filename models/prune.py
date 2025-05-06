import torch.nn.utils.prune as prune
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch


def apply_pruning(model, pruning_amount=0.5):
    """
    Apply unstructured weight-based pruning to convolutional and linear layers
    Args:
        model: PyTorch model
        pruning_amount: Amount of weights to prune
    """
    for name, module in model.named_modules():
        # Apply pruning to convolutional and linear layers
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)


def remove_pruning(model):
    """
    Remove pruning reparameterization and make pruning permanent
    Args:
        model: PyTorch model with pruning applied
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')


def print_sparsity(model):
    """
    Print the sparsity of each layer in the model
    Args:
        model: PyTorch model
    """
    total_params = 0
    zero_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            tensor = module.weight.data
            nz_count = torch.count_nonzero(tensor)
            total = tensor.numel()
            total_params += total
            zero_params += (total - nz_count)
            print(f"{name}: sparsity {100 * (total - nz_count) / total:.2f}%")
    
    print(f"Global sparsity: {100 * zero_params / total_params:.2f}%")


def gradual_pruning(model, initial_sparsity, final_sparsity, start_epoch, end_epoch, current_epoch):
    """
    Apply gradual pruning during training
    Args:
        model: PyTorch model
        initial_sparsity: Initial sparsity ratio
        final_sparsity: Target sparsity ratio
        start_epoch: Epoch to start pruning
        end_epoch: Epoch to end pruning
        current_epoch: Current training epoch
    """
    if current_epoch < start_epoch or current_epoch > end_epoch:
        return
    
    # Calculate current sparsity ratio using cubic schedule
    # From "To Prune, or Not to Prune: Exploring the Efficacy of Pruning for Model Compression" (Zhu & Gupta)
    progress = (current_epoch - start_epoch) / (end_epoch - start_epoch)
    current_sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * (1.0 - (1.0 - progress) ** 3)
    
    print(f"Epoch {current_epoch}: Applying pruning with sparsity {current_sparsity:.4f}")
    
    # Remove existing pruning
    remove_pruning(model)
    
    # Apply new pruning
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=current_sparsity)


def compute_model_sparsity(model):
    """
    Compute the global sparsity of the model
    Args:
        model: PyTorch model
    Returns:
        float: Sparsity ratio (percentage of zero weights)
    """
    total_params = 0
    zero_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            tensor = module.weight.data
            nz_count = torch.count_nonzero(tensor)
            total = tensor.numel()
            total_params += total
            zero_params += (total - nz_count)
    
    if total_params == 0:
        return 0.0
    
    return float(zero_params) / total_params
