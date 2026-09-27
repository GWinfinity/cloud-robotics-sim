# articulation_gen — 关节资产自动标注

抽屉开合 / 柜门铰链类**关节资产**的自动标注模块：分割 → 几何关节估计 → URDF 导出。
与 `plugins/envs/maniskill` 的 ReplicaCAD URDF 关节加载链路对接（生成的 URDF
可被 `ReplicaCADSceneBuilder` 同款流程加载为 Genesis articulation）。

## 为什么是这个设计（关于"97% 自动标注"）

诚实的答案是：**2D 检测/分割可以，端到端关节参数不行**。

| 环节 | YOLO26 + SAM 能达到的自动化水平 |
| --- | --- |
| 2D 部件检测（drawer/door/handle 框） | 在家具窄域上微调后 mAP 95%+ 可达，接近 97% |
| 2D mask 分割（SAM/SAM2） | 边界质量高，但**不含任何 3D/关节信息** |
| 关节类型 / 轴向 / 原点 / 限位 | 2D 信息本质不足，单靠 YOLO+SAM 无法做到 97% 正确——这也是 DIPO、Articulate AnyMesh 等论文仍在解决的开问题 |

所以本模块的分工是：

1. **分割**（YOLO+SAM，可替换）：只负责"哪个像素属于 drawer/door"；
2. **关节参数**（纯几何估计器，本模块核心）：从 3D 部件做 OBB 拟合、
   外露面选择、碰撞扫描验证——这一步对"方正"家具是确定性的、可解释的；
3. **置信度 + 复核队列**：低置信度不静默放过，`batch_annotate` 落盘
   `review.jsonl` 供人工确认；
4. **Genesis 闭环验证**（`GenesisVerifier`，可选）：把生成的 URDF 加载为
   Genesis articulation，逐关节开环驱动到限位，判定跟踪误差 + 非有限值
   + **几何穿透检查**（相邻 link 默认无自碰撞，镜像铰链侧这类错误必须
   显式比对子 link 世界位姿与基座 AABB 才能抓住）。未通过的关节降级进
   复核队列，reason 带仿真读数。

整体接受率（自动 + 复核后）可逼近 100%，但**全自动率**在复杂资产上应
预期 80–90% 量级，把 97% 当全自动承诺是不现实的。

## 闭环验证（core/verify.py）

```python
from plugins.predictors.articulation_gen import GenesisVerifier, batch_annotate

verifier = GenesisVerifier()          # gs.init 每进程一次，懒触发
summary = batch_annotate(items, out_dir, verifier=verifier)
# summary 增加 sim_rejected：被仿真判否的关节数（验证器异常单列，不计入）
```

验证逻辑：URDF → `gs.morphs.URDF(fixed=True)` → 逐关节 ramp 到 `upper` →
判定：① 跟踪误差 < max(2% 行程, 1e-3)；② 全程有限值；③ 子 link 碰撞盒
角点（世界系）不与基座 AABB 穿透（2mm 容差）。每个关节单独驱动，避免
门与抽屉同时打开互相碰撞造成假阴性。

## 三条输入路径

```python
from plugins.predictors.articulation_gen import (
    annotate_labeled_mesh,    # 路径 1: GLB/USD 命名节点（离线，推荐起点）
    annotate_from_images,     # 路径 2: YOLO + SAM 图像反投影（需微调权重）
    annotate_parts,           # 路径 3: 已有部件顶点（PartNet 式标签）
    estimate_joints, result_to_urdf, validate_urdf,
)
```

**路径 1（完全离线，零推理依赖）**：asset_gen 产物或人工 rig 的 GLB，
节点名按子串映射语义（`drawer_*`→drawer、`door_*`→door、
`*_body`/`*_cabinet`→base …）：

```python
result, parts = annotate_labeled_mesh("cabinet.glb")
urdf = result_to_urdf(result, parts, object_name="cabinet")
```

**路径 2（图像）**：需要本地 `ultralytics`（任意 YOLO checkpoint，建议用
自标注家具部件集微调的 YOLO26 权重——公开权重不覆盖 drawer/door 类别）
与 `sam2`（或 ultralytics FastSAM 退化）：

```python
from plugins.predictors.articulation_gen.core.pipeline import Camera, annotate_from_images
from plugins.predictors.articulation_gen.core.segment import YoloPartDetector, SamMaskRefiner

detector = YoloPartDetector("runs/detect/train/weights/best.pt")
refiner = SamMaskRefiner()
result, parts = annotate_from_images(mesh, images, cameras, detector, refiner)
```

2D mask 经多视角投票反投影回 mesh 顶点，切分部件后走同一几何估计器。

**批量与复核队列**：

```python
summary = batch_annotate([("cab_a", parts_a), ("cab_b", parts_b)], "outputs/articulation_gen")
# → results.json + review.jsonl + auto_acceptance_rate
```

## 几何估计器做什么（core/estimate.py）

1. 基座 = OBB 体积最大的部件；
2. 外露面 = 面中心沿法向**超出基座 OBB 表面**（proud）且顶点最聚集的面；
3. 类型：`part.kind`/节点名/YOLO 类别优先，否则按外露面高宽比猜；
4. 抽屉 → prismatic：轴向 = 外露面法向（方向由"按部件深度平移能否脱离
   基座"判定），行程 = 深度 × 0.95（封顶 0.8 m）；
5. 柜门 → revolute：候选铰链为两条竖直棱，旋转扫描取"无碰撞自由转角"
   最大的棱，限位 = 碰撞前最大角（上限 110°）；
6. 置信度 = 面平面性 × 运动自由度达成率的加权和，低于阈值进复核队列。

## 已知局限

- 只支持竖直铰链（revolute）与直线滑轨（prismatic)；翻盖、旋转托盘
  （水平轴）会进复核队列；
- 几何启发式针对"方正"家具；圆润/有机造型的置信度会自然降低（这是特性）;
- 合成部件（如两个同心盒）在几何上本就歧义，应靠语义标签而非放宽阈值；
- 导出的 URDF link 质量为占位值，接入仿真前应按材质/体积换算
  （参考 `replica_cad_scene.py` 在 `scene.build()` 后 `set_mass()` 的做法）。

## 测试

```bash
uv run python -m pytest plugins/predictors/articulation_gen/tests/ -q
```

15 个离线测试（numpy + trimesh，无网络/GPU/genesis）：合成柜体
（2 抽屉 + 1 柜门）的关节类型/轴向/限位断言、复核队列、URDF 结构校验、
GLB 往返、batch 落盘、可选依赖 guard。
