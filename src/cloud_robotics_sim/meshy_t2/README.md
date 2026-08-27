# Meshy T2 复现（arXiv:2607.28675）

对论文 **"Meshy T2: Fast Native Mesh Generation with Flow Matching"**（Meshy AI,
官方代码/权重尚未发布）的纯 PyTorch 复现，已适配本项目的依赖（torch / trimesh /
scipy / numpy）与代码规范（ruff / black / mypy / pytest）。

## 架构对照（论文章节 → 代码）

| 论文 | 实现 |
| --- | --- |
| Sec. 2.1 顶点集 Mesh VAE | `models/mesh_vae.py`（编码器：稀疏体素点上下文 + Fourier 顶点查询 + 图注意力/自注意力交错，全部 3D RoPE；解码器：共享 trunk + 顶点分支 + 拓扑分支） |
| Sec. 2.1 边预测（时空对数几率，Eq. 1/2） | `models/mesh_vae.py::edge_logits` + `losses.py::edge_loss`（分块计算，类别平衡 softplus BCE） |
| Sec. 2.1 面预测（半边后继 + NULL 扩展 + Sinkhorn，Eq. 3/4/5） | `geometry/halfedge.py`（π 提取/重建）、`models/mesh_vae.py::face_scores`、`losses.py::face_loss`（按度数分组） |
| Sec. 2.1 网格装配（线性分配 + 单扇环约束） | `geometry/assembly.py` |
| Sec. 2.1 非流形修复（边/顶点分裂） | `geometry/mesh_utils.py::repair_nonmanifold` |
| Sec. 2.2 Voxel VAE（64³→8×16³，pixel-shuffle 解码） | `models/voxel_vae.py` |
| Sec. 2.2 Stage I 体素流（DiT + 3D RoPE + 图像交叉注意力 + AdaLN） | `models/voxel_flow.py` |
| Sec. 2.3 Stage II 网格流（单流 DiT、存在通道、计数条件、条件丢弃/CFG） | `models/mesh_flow.py` |
| Sec. 2.3/3.1 Sobol OT 位置编码（含 Morton 消融） | `sobol_ot.py` |
| Rectified Flow / logit-normal 时间步 / Euler 采样 / CFG | `flow.py` |
| 端到端图生网格 / 重拓扑 | `pipeline.py::MeshyT2Pipeline` |
| 四个阶段训练入口 | `train.py`（`python -m cloud_robotics_sim.meshy_t2 train <stage>`） |

## 与论文的差异（复现说明）

- 图像条件默认使用内置的 `TinyViTImageEncoder`（离线可跑）；有网络时可用
  `DINOv3ImageEncoder` 加载论文所用的 DINOv3。
- 论文 CSR 体素池化用 `index_add` 等价实现（功能一致，吞吐略低）。
- 训练循环为单卡简化版（无 FSDP2 / token 打包）；支持 `--data synthetic`
  程序化基元数据集做无资产冒烟训练。
- 默认配置为论文尺寸（width 1024、28/24 层等）；CPU 测试使用 `*.tiny()` 配置。

## 快速开始

```bash
# 冒烟训练（合成数据，CPU 可跑）
uv run python -m cloud_robotics_sim.meshy_t2 train voxel-vae --tiny --max-steps 50
uv run python -m cloud_robotics_sim.meshy_t2 train mesh-vae  --tiny --max-steps 50
uv run python -m cloud_robotics_sim.meshy_t2 train voxel-flow --tiny --max-steps 50
uv run python -m cloud_robotics_sim.meshy_t2 train mesh-flow  --tiny --max-steps 50

# 图生网格（面数预算 → 顶点预算，欧拉关系 F ≈ 2V - 4）
uv run python -m cloud_robotics_sim.meshy_t2 generate ref.png --num-faces 4000 --out out.obj

# 高模重拓扑（使用 GT 体素 scaffold）
uv run python -m cloud_robotics_sim.meshy_t2 retopo dense.obj --num-vertices 2000

# 测试
uv run python -m pytest tests/meshy_t2/
```
