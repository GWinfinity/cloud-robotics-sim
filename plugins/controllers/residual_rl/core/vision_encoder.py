"""
Vision Encoders for Visual Observations

支持ResNet18和自定义CNN编码器
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict


class ResNet18Encoder(nn.Module):
    """
    ResNet18视觉编码器
    
    使用预训练的ResNet18作为视觉 backbone
    """
    
    def __init__(
        self,
        output_dim: int = 256,
        pretrained: bool = True,
        freeze_layers: bool = False
    ):
        super().__init__()
        
        # 加载预训练ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # 特征投影层
        self.fc = nn.Linear(512, output_dim)
        
        # 冻结层 (可选)
        if freeze_layers:
            for param in self.features.parameters():
                param.requires_grad = False
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        编码图像
        
        Args:
            image: [batch, 3, 224, 224]
            
        Returns:
            features: [batch, output_dim]
        """
        x = self.features(image)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CustomVisionEncoder(nn.Module):
    """
    自定义轻量级CNN编码器
    
    用于资源受限的场景
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 256,
        image_size: int = 224
    ):
        super().__init__()
        
        # 计算经过卷积后的特征图大小
        # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        
        self.conv_layers = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 5: 14x14 -> 7x7
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 全局平均池化: 7x7 -> 1x1
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.conv_layers(image)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        return x


class MultiViewEncoder(nn.Module):
    """
    多视角编码器
    
    融合多个相机视角的信息
    """
    
    def __init__(
        self,
        num_views: int = 2,
        output_dim: int = 256,
        fusion_method: str = 'concat'  # 'concat', 'attention', 'average'
    ):
        super().__init__()
        
        self.num_views = num_views
        self.fusion_method = fusion_method
        
        # 每个视角的编码器
        self.encoders = nn.ModuleList([
            CustomVisionEncoder(output_dim=output_dim)
            for _ in range(num_views)
        ])
        
        # 融合层
        if fusion_method == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(output_dim * num_views, output_dim),
                nn.ReLU()
            )
        elif fusion_method == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(output_dim * num_views, num_views),
                nn.Softmax(dim=-1)
            )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码多视角图像
        
        Args:
            images: [batch, num_views, 3, H, W]
            
        Returns:
            features: [batch, output_dim]
        """
        batch_size = images.size(0)
        
        # 编码每个视角
        view_features = []
        for i, encoder in enumerate(self.encoders):
            view_img = images[:, i]
            feat = encoder(view_img)
            view_features.append(feat)
        
        # 融合
        if self.fusion_method == 'concat':
            fused = torch.cat(view_features, dim=-1)
            output = self.fusion(fused)
        
        elif self.fusion_method == 'attention':
            stacked = torch.stack(view_features, dim=1)  # [batch, num_views, dim]
            concat = torch.cat(view_features, dim=-1)
            weights = self.attention(concat)  # [batch, num_views]
            weights = weights.unsqueeze(-1)  # [batch, num_views, 1]
            output = (stacked * weights).sum(dim=1)  # [batch, dim]
        
        else:  # average
            output = torch.stack(view_features, dim=0).mean(dim=0)
        
        return output
