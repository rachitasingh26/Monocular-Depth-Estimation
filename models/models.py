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
from transformers import AutoModelForDepthEstimation

class DepthEstimationStudent(nn.Module):
    def __init__(self, pretrained=True):
        super(DepthEstimationStudent, self).__init__()
        # Use MobileNetV3-Small as the encoder backbone
        self.encoder = models.mobilenet_v3_small(pretrained=pretrained).features
        
        # Create the decoder using depth-wise separable convolutions
        self.decoder = nn.ModuleList([
            # Upsampling blocks with skip connections
            self._make_dsconv_block(576, 96, 3, 1),
            self._make_dsconv_block(96, 48, 3, 1),
            self._make_dsconv_block(48, 24, 3, 1),
        ])

        self.skip_projections = nn.ModuleList([
            nn.Conv2d(16, 576, kernel_size=1),  # For layer 3
            nn.Conv2d(40, 96, kernel_size=1),   # For layer 8
            nn.Conv2d(576, 48, kernel_size=1),  # For layer 11
        ])
        
        # Affinity maps at different scales
        self.affinity_maps = nn.ModuleList([
            nn.Conv2d(96, 1, kernel_size=1),
            nn.Conv2d(48, 1, kernel_size=1),
            nn.Conv2d(24, 1, kernel_size=1),
        ])
        
        # Final output layer
        self.final_conv = nn.Conv2d(24, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid() # Output between 0 and 1 for relative depth

    def _make_dsconv_block(self, in_channels, out_channels, kernel_size, stride):
        """Create a block with depthwise separable convolution for efficiency"""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                    stride=stride, padding=kernel_size//2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
            # Upsample
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        # Store input size for later upsampling
        original_size = (x.shape[2], x.shape[3])
        # Normalize input
        # x = x / 255.0
        
        # Extract features at different scales
        features = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in [3, 8, 11]: # Save features from different scales for skip connections
                features.append(x)
        
        # Decode and upsample with affinity maps
        affinity_outputs = []
        for i, (decoder_block, affinity_map) in enumerate(zip(self.decoder, self.affinity_maps)):
            # if i < len(features):
            #     # Add skip connection if available
            #     x = x + features[len(features) - i - 1]
            x = decoder_block(x)
            affinity_outputs.append(affinity_map(x))
        
        # Final convolution and sigmoid for depth values between 0-1
        depth = self.final_conv(x)
        depth = self.sigmoid(depth)
        if (depth.shape[2], depth.shape[3]) != original_size:
            depth = F.interpolate(depth, size=original_size, mode='bilinear', align_corners=True)
        
        return depth, affinity_outputs

# Define the Teacher Model - DepthAnything V2
class DepthEstimationTeacher(nn.Module):
    def __init__(self):
        super(DepthEstimationTeacher, self).__init__()
        self.model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf")
        
    def forward(self, x):
        with torch.no_grad():
            # Get predicted depth from model
            depth = self.model(x).predicted_depth
            
            # Ensure output is a proper 4D tensor [B, 1, H, W]
            if len(depth.shape) == 3:
                depth = depth.unsqueeze(1)
                
            # Normalize depth to [0, 1] range for consistency with student model
            batch_min = depth.view(depth.shape[0], -1).min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
            batch_max = depth.view(depth.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
            normalized_depth = (depth - batch_min) / (batch_max - batch_min + 1e-6)
            
            return normalized_depth

# Scale-invariant loss function as described in Eigen et al.
class ScaleInvariantLoss(nn.Module):
    def __init__(self, alpha=10.0):
        super(ScaleInvariantLoss, self).__init__()
        self.alpha = alpha # Weight for the variance term

    def forward(self, pred, target, mask=None):
        """
        Scale-invariant loss as described in Eigen et al.
        Args:
            pred: Predicted depth map
            target: Ground truth depth map
            mask: Optional mask for valid depth values
        """
        if mask is None:
            mask = torch.ones_like(target)

        # Apply mask
        pred = pred * mask
        target = target * mask

        # Log space for scale invariance
        d = torch.log(pred + 1e-6) - torch.log(target + 1e-6)
        
        # Count valid pixels
        valid_pixels = torch.sum(mask > 0.5) + 1e-8

        # Mean term
        mean_term = torch.sum(d ** 2 * mask) / valid_pixels

        # Variance term - modified to ensure positive loss value
        variance_term = (torch.sum(d * mask) / valid_pixels) ** 2
        
        # Scale-invariant loss - note we ADD the variance term instead of subtracting
        loss = mean_term - self.alpha * variance_term

        return loss

# Gradient matching loss
class GradientMatchingLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(GradientMatchingLoss, self).__init__()
        self.alpha = alpha

        # Sobel filters for gradient computation
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)

    def forward(self, pred, target, mask=None):
        """
        Gradient matching loss to preserve edges
        Args:
            pred: Predicted depth map
            target: Ground truth depth map
            mask: Optional mask for valid depth values
        """
        if mask is None:
            mask = torch.ones_like(target)

        # Move Sobel filters to the same device as inputs
        device = pred.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)

        # Compute gradients
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)

        # Apply mask
        pred_grad_x = pred_grad_x * mask
        pred_grad_y = pred_grad_y * mask
        target_grad_x = target_grad_x * mask
        target_grad_y = target_grad_y * mask

        # Gradient matching loss
        loss_x = F.l1_loss(pred_grad_x, target_grad_x, reduction='sum') / mask.sum()
        loss_y = F.l1_loss(pred_grad_y, target_grad_y, reduction='sum') / mask.sum()

        return self.alpha * (loss_x + loss_y)

# Knowledge Distillation Loss
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=3.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_output, teacher_output, target=None, mask=None):
        """
        Knowledge distillation loss
        Args:
            student_output: Output from the student model
            teacher_output: Output from the teacher model
            target: Ground truth depth map (optional)
            mask: Optional mask for valid depth values
        """
        if mask is None:
            mask = torch.ones_like(student_output)
            
        # Resize teacher output to match student output size if they differ
        if teacher_output.shape != student_output.shape:
            teacher_output = F.interpolate(
                teacher_output, 
                size=(student_output.shape[2], student_output.shape[3]), 
                mode='bilinear', 
                align_corners=True
            )

        # Apply mask
        student_output = student_output * mask
        teacher_output = teacher_output * mask

        # MSE loss between student and teacher outputs
        mse_loss = F.mse_loss(student_output, teacher_output, reduction='sum') / mask.sum()
        
        return mse_loss

# Pairwise Affinity Loss
class AffinityLoss(nn.Module):
    def __init__(self):
        super(AffinityLoss, self).__init__()
        
    def forward(self, affinity_maps, target, mask=None):
        """
        Loss for affinity maps
        Args:
            affinity_maps: List of affinity maps from student
            target: Ground truth depth map
            mask: Optional mask for valid depth values
        """
        if mask is None:
            mask = torch.ones_like(target)
            
        total_loss = 0
        for aff_map in affinity_maps:
            # Resize target to match affinity map size
            resized_target = F.interpolate(target, size=aff_map.shape[2:], mode='bilinear', align_corners=True)
            resized_mask = F.interpolate(mask, size=aff_map.shape[2:], mode='nearest')
            
            # Apply mask
            aff_map = aff_map * resized_mask
            resized_target = resized_target * resized_mask
            
            # MSE loss
            loss = F.mse_loss(aff_map, resized_target, reduction='sum') / resized_mask.sum()
            total_loss += loss
            
        return total_loss / len(affinity_maps)

# Combined loss function
class CombinedLoss(nn.Module):
    def __init__(self, si_weight=1.0, grad_weight=1.0, distill_weight=1.0, affinity_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.si_loss = ScaleInvariantLoss()
        self.grad_loss = GradientMatchingLoss()
        self.distill_loss = DistillationLoss()
        self.affinity_loss = AffinityLoss()
        self.si_weight = si_weight
        self.grad_weight = grad_weight
        self.distill_weight = distill_weight
        self.affinity_weight = affinity_weight

    def forward(self, student_output, target, teacher_output=None, mask=None):
        """
        Combined loss function
        Args:
            student_output: Tuple of (depth, affinity_maps) from student
            target: Ground truth depth map
            teacher_output: Output from the teacher model (optional)
            mask: Optional mask for valid depth values
        """
        # Unpack student output
        student_depth, student_affinity = student_output
        
        # Scale-invariant loss
        si_loss = self.si_loss(student_depth, target, mask)
        
        # Gradient matching loss
        grad_loss = self.grad_loss(student_depth, target, mask)
        
        # Knowledge distillation loss
        distill_loss = 0
        if teacher_output is not None:
            # Ensure teacher_output is not None and is a proper tensor
            if isinstance(teacher_output, torch.Tensor):
                distill_loss = self.distill_loss(student_depth, teacher_output, target, mask)
            else:
                print("Warning: teacher_output is not a tensor. Skipping distillation loss.")
        
        # Affinity loss
        affinity_loss = self.affinity_loss(student_affinity, target, mask)
        
        # Combined loss
        total_loss = (
            self.si_weight * si_loss +
            self.grad_weight * grad_loss +
            self.distill_weight * distill_loss +
            self.affinity_weight * affinity_loss
        )
        
        return total_loss, si_loss, grad_loss, distill_loss