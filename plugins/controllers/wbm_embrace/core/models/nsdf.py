"""
Neural Signed Distance Field (NSDF)

提供准确连续的几何感知，用于全身拥抱任务中的接触意识
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class NSDF(nn.Module):
    """
    神经符号距离场
    
    输入: 3D空间坐标 (x, y, z)
    输出: 到最近表面的有符号距离
         - 正值: 在物体外部
         - 负值: 在物体内部
         - 0: 在表面上
    """
    
    def __init__(
        self,
        hidden_dims: list = [256, 256, 128, 128],
        activation: str = 'softplus',
        bounds: Dict = None,
        grid_resolution: int = 64
    ):
        super().__init__()
        
        self.bounds = bounds or {
            'x': [-1.0, 1.0],
            'y': [-1.0, 1.0],
            'z': [0.0, 2.0]
        }
        self.grid_resolution = grid_resolution
        
        # 激活函数
        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
        
        # SIREN风格的网络 (适合隐式表示)
        layers = []
        prev_dim = 3  # 输入: 3D坐标
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        # 最后一层输出距离
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化 (SIREN初始化)
        self._init_siren()
        
        # 缓存的网格 (用于快速查询)
        self.cached_grid = None
        self.cached_positions = None
    
    def _init_siren(self):
        """SIREN初始化"""
        with torch.no_grad():
            for i, layer in enumerate(self.network):
                if isinstance(layer, nn.Linear):
                    if i < len(self.network) - 2:  # 隐藏层
                        layer.weight.uniform_(-np.sqrt(6.0 / layer.in_features), 
                                               np.sqrt(6.0 / layer.in_features))
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        查询NSDF
        
        Args:
            positions: [batch, 3] 3D坐标 (x, y, z)
            
        Returns:
            distances: [batch, 1] 有符号距离
        """
        # 归一化坐标到[-1, 1]
        normalized_pos = self._normalize_positions(positions)
        
        # 查询网络
        distances = self.network(normalized_pos)
        
        return distances
    
    def _normalize_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """将位置归一化到[-1, 1]"""
        x_norm = 2 * (positions[:, 0] - self.bounds['x'][0]) / (self.bounds['x'][1] - self.bounds['x'][0]) - 1
        y_norm = 2 * (positions[:, 1] - self.bounds['y'][0]) / (self.bounds['y'][1] - self.bounds['y'][0]) - 1
        z_norm = 2 * (positions[:, 2] - self.bounds['z'][0]) / (self.bounds['z'][1] - self.bounds['z'][0]) - 1
        
        return torch.stack([x_norm, y_norm, z_norm], dim=-1)
    
    def get_gradient(self, positions: torch.Tensor) -> torch.Tensor:
        """
        计算距离场的梯度 (即表面法向)
        
        Args:
            positions: [batch, 3] 查询点
            
        Returns:
            gradients: [batch, 3] 梯度方向
        """
        positions.requires_grad_(True)
        distances = self.forward(positions)
        
        gradients = torch.autograd.grad(
            outputs=distances,
            inputs=positions,
            grad_outputs=torch.ones_like(distances),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 归一化
        gradients = F.normalize(gradients, p=2, dim=-1)
        
        return gradients
    
    def get_contact_points(self, robot_body_positions: torch.Tensor, threshold: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取潜在接触点
        
        Args:
            robot_body_positions: [num_bodies, 3] 机器人身体部位位置
            threshold: 距离阈值
            
        Returns:
            contact_mask: [num_bodies] 是否在接触
            distances: [num_bodies] 到表面的距离
        """
        with torch.no_grad():
            distances = self.forward(robot_body_positions).squeeze(-1)
            contact_mask = distances < threshold
        
        return contact_mask, distances
    
    def compute_proximity_reward(self, robot_body_positions: torch.Tensor) -> torch.Tensor:
        """
        计算接近奖励
        
        鼓励机器人身体部位接近物体表面 (但不穿透)
        """
        distances = self.forward(robot_body_positions).squeeze(-1)
        
        # 奖励: 距离接近0但不负太多
        # 理想情况: 距离略负 (轻微穿透) 或接近0
        reward = torch.exp(-torch.abs(distances) * 20)  # 在0附近峰值
        
        # 惩罚过度穿透
        penetration = torch.clamp(distances, max=0)
        penalty = torch.abs(penetration) * 10
        
        return reward - penalty
    
    def reconstruct_mesh(self, resolution: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        从NSDF重建网格 (使用Marching Cubes)
        
        Returns:
            vertices: [N, 3] 顶点
            faces: [M, 3] 面
        """
        # 创建网格
        x = np.linspace(self.bounds['x'][0], self.bounds['x'][1], resolution)
        y = np.linspace(self.bounds['y'][0], self.bounds['y'][1], resolution)
        z = np.linspace(self.bounds['z'][0], self.bounds['z'][1], resolution)
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
        
        # 批量查询
        batch_size = 10000
        sdf_values = []
        
        with torch.no_grad():
            for i in range(0, len(points), batch_size):
                batch_points = torch.FloatTensor(points[i:i+batch_size]).to(next(self.parameters()).device)
                batch_sdf = self.forward(batch_points).cpu().numpy()
                sdf_values.append(batch_sdf)
        
        sdf_values = np.concatenate(sdf_values, axis=0).reshape(resolution, resolution, resolution)
        
        # Marching Cubes (简化版，实际需要skimage或pymcubes)
        # 这里返回占位符
        return sdf_values, None
    
    def update_from_object(self, object_mesh_vertices: torch.Tensor, object_mesh_faces: Optional[torch.Tensor] = None):
        """
        从物体网格更新NSDF (在线适应)
        
        Args:
            object_mesh_vertices: [N, 3] 物体顶点
            object_mesh_faces: [M, 3] 物体面 (可选)
        """
        # 采样点用于训练
        num_samples = 100000
        
        # 表面点
        surface_indices = np.random.choice(len(object_mesh_vertices), num_samples // 2)
        surface_points = object_mesh_vertices[surface_indices]
        surface_sdf = torch.zeros(len(surface_points), 1)
        
        # 空间随机点
        random_points = torch.rand(num_samples // 2, 3)
        random_points[:, 0] = random_points[:, 0] * (self.bounds['x'][1] - self.bounds['x'][0]) + self.bounds['x'][0]
        random_points[:, 1] = random_points[:, 1] * (self.bounds['y'][1] - self.bounds['y'][0]) + self.bounds['y'][0]
        random_points[:, 2] = random_points[:, 2] * (self.bounds['z'][1] - self.bounds['z'][0]) + self.bounds['z'][0]
        
        # 计算这些点的SDF (简化: 到最近顶点的距离)
        random_sdf = self._compute_sdf_from_points(random_points, object_mesh_vertices)
        
        # 合并数据
        train_points = torch.cat([surface_points, random_points], dim=0)
        train_sdf = torch.cat([surface_sdf, random_sdf], dim=0)
        
        # 快速训练几步 (在线适应)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        for _ in range(100):
            pred_sdf = self.forward(train_points)
            loss = F.mse_loss(pred_sdf, train_sdf)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def _compute_sdf_from_points(self, query_points: torch.Tensor, surface_points: torch.Tensor) -> torch.Tensor:
        """计算查询点到表面的有符号距离 (简化版)"""
        # 计算到最近表面点的距离
        distances = torch.cdist(query_points, surface_points)
        min_distances = distances.min(dim=1)[0].unsqueeze(-1)
        
        # 简化的符号判断 (实际应该使用表面法向)
        # 这里假设所有查询点都在外部
        return min_distances


class MultiObjectNSDF(nn.Module):
    """
    多物体NSDF
    
    处理场景中的多个物体
    """
    
    def __init__(self, max_objects: int = 5, **nsdf_kwargs):
        super().__init__()
        
        self.max_objects = max_objects
        
        # 为每个物体创建NSDF
        self.object_nsdfs = nn.ModuleList([
            NSDF(**nsdf_kwargs) for _ in range(max_objects)
        ])
        
        # 物体存在掩码
        self.object_active = torch.zeros(max_objects, dtype=torch.bool)
    
    def query_scene(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        查询整个场景
        
        Returns:
            min_distances: [batch, 1] 到最近物体的距离
            object_indices: [batch] 最近物体的索引
        """
        all_distances = []
        
        for i, nsdf in enumerate(self.object_nsdfs):
            if self.object_active[i]:
                dist = nsdf(positions)
                all_distances.append(dist)
        
        if len(all_distances) == 0:
            return torch.full((len(positions), 1), float('inf')), torch.zeros(len(positions), dtype=torch.long)
        
        all_distances = torch.cat(all_distances, dim=-1)  # [batch, num_objects]
        
        min_distances, object_indices = all_distances.min(dim=-1, keepdim=True)
        
        return min_distances, object_indices
