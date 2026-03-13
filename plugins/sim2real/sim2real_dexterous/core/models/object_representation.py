"""
Hybrid Object Representation

多模态物体表示: 视觉 + 点云 + 几何
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class VisualEncoder(nn.Module):
    """视觉编码器 (ResNet18风格)"""
    
    def __init__(self, output_dim: int = 256):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(512, output_dim)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: [B, 3, H, W]"""
        x = self.conv_layers(image)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class PointNetEncoder(nn.Module):
    """点云编码器 (PointNet简化版)"""
    
    def __init__(self, num_points: int = 512, output_dim: int = 128):
        super().__init__()
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        point_cloud: [B, N, 3]
        """
        # 转置为 [B, 3, N]
        x = point_cloud.transpose(1, 2)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        
        # 全局最大池化
        x = torch.max(x, 2, keepdim=False)[0]
        
        return self.fc(x)


class GeometricEncoder(nn.Module):
    """几何特征编码器 (3D边界框)"""
    
    def __init__(self, output_dim: int = 32):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(13, 64),  # pos(3) + quat(4) + size(3) + vel(3)
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, position, orientation, size, velocity):
        """
        编码几何特征
        position: [B, 3]
        orientation: [B, 4] (quat)
        size: [B, 3]
        velocity: [B, 3]
        """
        features = torch.cat([position, orientation, size, velocity], dim=-1)
        return self.encoder(features)


class ModalityFusion(nn.Module):
    """模态融合 (注意力机制)"""
    
    def __init__(self, visual_dim: int = 256, point_dim: int = 128, geo_dim: int = 32):
        super().__init__()
        
        total_dim = visual_dim + point_dim + geo_dim
        
        # 注意力权重计算
        self.attention = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # 3种模态
            nn.Softmax(dim=-1)
        )
    
    def forward(self, visual_feat, point_feat, geo_feat):
        """
        融合多模态特征
        """
        concat = torch.cat([visual_feat, point_feat, geo_feat], dim=-1)
        weights = self.attention(concat)  # [B, 3]
        
        # 加权融合
        visual_weight = weights[:, 0:1]
        point_weight = weights[:, 1:2]
        geo_weight = weights[:, 2:3]
        
        fused = visual_weight * visual_feat + point_weight * point_feat + geo_weight * geo_feat
        
        return fused, weights


class HybridObjectRepresentation(nn.Module):
    """
    混合物体表示
    
    整合视觉、点云、几何三种表示
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        
        # 各模态编码器
        if config['visual']['enabled']:
            self.visual_encoder = VisualEncoder(config['visual']['output_dim'])
        
        if config['point_cloud']['enabled']:
            self.point_encoder = PointNetEncoder(
                config['point_cloud']['num_points'],
                config['point_cloud']['output_dim']
            )
        
        if config['geometric']['enabled']:
            self.geo_encoder = GeometricEncoder(config['geometric']['output_dim'])
        
        # 融合
        if config['fusion_method'] == 'attention':
            self.fusion = ModalityFusion(
                config['visual']['output_dim'],
                config['point_cloud']['output_dim'],
                config['geometric']['output_dim']
            )
    
    def forward(self, observation: Dict) -> torch.Tensor:
        """
        编码多模态观测
        
        observation: {
            'image': [B, 3, H, W],
            'point_cloud': [B, N, 3],
            'position': [B, 3],
            'orientation': [B, 4],
            'size': [B, 3],
            'velocity': [B, 3]
        }
        """
        features = []
        
        # 视觉特征
        if self.config['visual']['enabled'] and 'image' in observation:
            visual_feat = self.visual_encoder(observation['image'])
            features.append(visual_feat)
        
        # 点云特征
        if self.config['point_cloud']['enabled'] and 'point_cloud' in observation:
            point_feat = self.point_encoder(observation['point_cloud'])
            features.append(point_feat)
        
        # 几何特征
        if self.config['geometric']['enabled']:
            geo_feat = self.geo_encoder(
                observation['position'],
                observation['orientation'],
                observation['size'],
                observation['velocity']
            )
            features.append(geo_feat)
        
        # 融合
        if len(features) == 1:
            return features[0]
        elif self.config['fusion_method'] == 'concat':
            return torch.cat(features, dim=-1)
        else:  # attention
            return self.fusion(*features)[0]
    
    def augment_visual(self, image: torch.Tensor) -> torch.Tensor:
        """视觉数据增强"""
        aug_config = self.config.get('augmentation', {}).get('visual', {})
        
        # 亮度
        if 'brightness' in aug_config:
            brightness = np.random.uniform(*aug_config['brightness'])
            image = image * brightness
        
        # 对比度
        if 'contrast' in aug_config:
            contrast = np.random.uniform(*aug_config['contrast'])
            mean = image.mean(dim=[2, 3], keepdim=True)
            image = (image - mean) * contrast + mean
        
        # 高斯噪声
        if 'gaussian_noise' in aug_config:
            noise = torch.randn_like(image) * aug_config['gaussian_noise']
            image = image + noise
        
        return torch.clamp(image, 0, 1)
    
    def augment_point_cloud(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """点云数据增强"""
        aug_config = self.config.get('augmentation', {}).get('point_cloud', {})
        
        # 抖动
        if 'jitter' in aug_config:
            jitter = torch.randn_like(point_cloud) * aug_config['jitter']
            point_cloud = point_cloud + jitter
        
        # 随机dropout
        if 'dropout' in aug_config:
            mask = torch.rand(point_cloud.shape[:2]) > aug_config['dropout']
            point_cloud = point_cloud * mask.unsqueeze(-1)
        
        return point_cloud
