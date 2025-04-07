import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SimpleUNet(nn.Module):
    def __init__(self, pretrained=True):
        super(SimpleUNet, self).__init__()
        
        # Encoder (use pretrained ResNet18)
        resnet = models.resnet18(pretrained=pretrained)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64 channels
        self.pool1 = resnet.maxpool
        self.encoder2 = resnet.layer1  # 64 channels
        self.encoder3 = resnet.layer2  # 128 channels
        self.encoder4 = resnet.layer3  # 256 channels
        self.encoder5 = resnet.layer4  # 512 channels
        
        # Decoder
        self.upconv5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder5 = self._make_decoder_block(512, 256)
        
        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder4 = self._make_decoder_block(256, 128)
        
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = self._make_decoder_block(128, 64)
        
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = self._make_decoder_block(96, 32)
        
        self.conv_final = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Normalize input
        x = x / 255.0
        
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        
        # Decoder with skip connections
        d5 = self.upconv5(e5)
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.decoder5(d5)
        
        d4 = self.upconv4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2(d2)
        
        out = self.conv_final(d2)
        out = self.sigmoid(out)
        out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=True)
        return out
