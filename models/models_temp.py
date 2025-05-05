import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class LightweightDepthModel(nn.Module):
    def __init__(self, pretrained=True, dropout_rate=0.2):
        super(LightweightDepthModel, self).__init__()
        # Use MobileNetV3-Small as the encoder backbone
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        # Use first part of mobilenet as encoder (before pooling)
        self.encoder = mobilenet.features
        # Add dropout rate as a parameter
        self.dropout_rate = dropout_rate
        
        # Reduce number of decoder stages for simplicity and efficiency
        self.decoder = nn.ModuleList([
            self._make_decoder_block(576, 96),
            self._make_decoder_block(96, 48),
            self._make_decoder_block(48, 24)
        ])
        
        # Final convolution to get depth
        self.final_conv = nn.Conv2d(24, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid() # Output between 0 and 1 for relative depth
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Create a simplified decoder block with fewer layers and dropout"""
        return nn.Sequential(
            # Use standard convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Add dropout after activation
            nn.Dropout2d(p=self.dropout_rate),  # Spatial dropout for feature maps
            # Upsample
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
    
    def forward(self, x):
        # Store original size for resizing output
        original_size = (x.shape[2], x.shape[3])
        
        # Encoder (MobileNetV3-Small features)
        features = self.encoder(x)
        
        # Apply dropout after encoder (before decoder)
        features = F.dropout(features, p=self.dropout_rate, training=self.training)
        
        # Decoder (simplified)
        for decoder_block in self.decoder:
            features = decoder_block(features)
        
        # Final convolution and sigmoid
        depth = self.final_conv(features)
        depth = self.sigmoid(depth)
        
        # Resize to original input size if needed
        if depth.shape[2] != original_size[0] or depth.shape[3] != original_size[1]:
            depth = F.interpolate(depth, size=original_size, mode='bilinear', align_corners=True)
        
        return depth



class DepthLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.1):
        """
        Combined loss function for depth estimation
        Args:
            alpha: Weight for scale-invariant term
            beta: Weight for L1 term
            gamma: Weight for edge-preserving term
        """
        super(DepthLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Define Sobel filters for edge detection
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                  dtype=torch.float32).reshape(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                  dtype=torch.float32).reshape(1, 1, 3, 3)
    
    def forward(self, pred, target, mask=None):
        if mask is None:
            mask = torch.ones_like(target)
            
        # Apply mask
        pred = pred * mask
        target = target * mask
        
        # Count valid pixels
        valid_pixels = torch.sum(mask) + 1e-8
        
        # 1. Scale-invariant term
        log_diff = torch.log(pred + 1e-8) - torch.log(target + 1e-8)
        log_diff = log_diff * mask
        
        term1 = torch.sum(log_diff ** 2) / valid_pixels
        term2 = (torch.sum(log_diff) / valid_pixels) ** 2
        si_loss = term1 - self.alpha * term2
        
        # 2. L1 loss for direct depth supervision
        l1_loss = torch.sum(torch.abs(pred - target) * mask) / valid_pixels
        
        # 3. Edge-preserving term (gradient matching)
        # Move Sobel filters to the same device as inputs
        device = pred.device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        
        # Compute gradients
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_grad_x = F.conv2d(target, self.sobel_x, padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1)
        
        # Apply mask to gradients
        pred_grad_x = pred_grad_x * mask
        pred_grad_y = pred_grad_y * mask
        target_grad_x = target_grad_x * mask
        target_grad_y = target_grad_y * mask
        
        # Edge loss (L1 on gradients)
        edge_loss = (torch.sum(torch.abs(pred_grad_x - target_grad_x)) + 
                    torch.sum(torch.abs(pred_grad_y - target_grad_y))) / valid_pixels
        
        # Combine losses
        total_loss = si_loss + self.beta * l1_loss + self.gamma * edge_loss
        
        return total_loss