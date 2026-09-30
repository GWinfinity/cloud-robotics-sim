# Sparse Voxel Volumes (`cloud_robotics_sim.vdb`)

块稀疏体素体积模块，双后端（仿 `cfd_coupling` 的 `backend` 选项语义）：

| 后端 | 类 | 依赖 | 特点 |
|---|---|---|---|
| `torch` | `TorchSparseVolume` | torch（必有） | 固定逻辑域（`nx×ny×nz`，必须为块大小整数倍），动态块拓扑（activate/deactivate + swap-remove），设备无关（CUDA/MUSA/CPU） |
| `xuvdb` | `XuvdbVolume` | `xuvdb` extra | 无界稀疏域（dict 叶根 + 稠密叶块），SDF/CSG 编辑、DDA 射线、粒子 splat、`.xuvdb`/OpenVDB `.vdb` 原生互转 |

安装 xuvdb 后端：

```bash
pip install cloud-robotics-sim[xuvdb]   # 或 uv sync --extra xuvdb
```

`import cloud_robotics_sim.vdb` 在无 xuvdb 环境下安全（惰性导入）；只有实例化
`XuvdbVolume` / `create_volume("xuvdb")` 才会抛出指明 extra 的 `ImportError`。
`create_volume("auto")` 在 xuvdb 可用时选 xuvdb，否则回落 torch。

```python
from cloud_robotics_sim.vdb import create_volume, to_xuvdb, from_xuvdb

vol = create_volume("auto", nx=64, ny=64, nz=64, block_size=8, background=0.0)

# xuvdb 后端可用时：SDF CSG + 射线 + 文件互转
if vol.backend == "xuvdb":
    vol.stamp_sphere((0.3, 0.2, 0.1), radius=0.25)   # min-union SDF stamp
    hit = vol.ray_surface_hit((0.3, 0.2, 2.0), (0, 0, -1))
```

后端互转（经 dense numpy 往返，非热路径；避免叶序转置问题）：

```python
from cloud_robotics_sim.vdb import from_xuvdb, to_xuvdb
from cloud_robotics_sim.vdb.core.vdb_xuvdb import (
    load_xuvdb, read_openvdb, save_xuvdb, write_openvdb,
)

xvol = to_xuvdb(torch_vol, voxel_size=0.1, grid_class="fog volume")
back = from_xuvdb(xvol)                      # 域向上对齐到块大小整数倍
save_xuvdb("scene.xuvdb", [xvol])            # 自有格式（zlib + CRC32）
grids = load_xuvdb("scene.xuvdb")
write_openvdb("scene.vdb", [xvol])           # 真 OpenVDB 流（Houdini/Blender 可读）
one = read_openvdb("scene.vdb", grid_name="shield")
```

注意：

- xuvdb 的 `to_dense()` 返回 **active bbox 紧致窗口** `(dense, ijk_min)`（索引可为负），
  而 `TorchSparseVolume.to_dense()` 返回整个逻辑域；互转时窗口起点都归零，
  `ijk_min` 偏移被丢弃（需要时从 `xvol.to_dense()` 取）。
- xuvdb 值语义：SDF `stamp_sphere` 为 min-union（= CSG union），fog stamp 覆盖写，
  `scatter_*` 累加；`band`/`h` 以体素计，`radius`/`background` 以世界单位计。
- `xuvdb.read_vdb` 返回网格列表（README 示例的单个赋值是误导），
  `read_openvdb()` 已处理并取第一个/按名选择。

测试：`pytest tests/vdb/`（torch 测试常跑；xuvdb 测试在缺包时优雅 skip，
由 `tests/optional_deps.py` 的 `xuvdb_only` 守护）。
