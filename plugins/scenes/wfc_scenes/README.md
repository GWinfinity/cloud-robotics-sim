# WFC 中式家居场景生成插件（wfc_scenes）

基于**波函数坍缩（Wave Function Collapse, WFC）**的中式住宅场景自动生成插件。
针对 ReplicaCAD 数据集的两个痛点：

1. **生活方式不符** —— ReplicaCAD 是欧美公寓（西式厨房、沙发朝向、动线）。
   本插件内置一套"整理好的"中国生活方式规则：客餐厨一体、封闭式厨房、
   玄关入户、生活阳台、圆餐桌、双人床+衣柜、鱼缸/字画/绿植等。
2. **场景数量太少** —— ReplicaCAD 本质上只有一套公寓原型的变体。
   本插件只要给定随机种子，即可**无限生成**满足中式居住规则的不同户型。

与 `art_scenes`（春节主题的单个样板间）互补：`art_scenes` 是"节日的家"，
本插件生成"日常的家"。

## 快速开始

```bash
cd <仓库根目录>

# 打印一张户型图（ASCII，上北下南）
uv run python -m plugins.scenes.wfc_scenes --grid 4x3 --seed 42 --ascii

# 生成一套完整场景：布局 JSON + 俯视 SVG 户型图 + 整体 GLB 模型
uv run python -m plugins.scenes.wfc_scenes --grid 5x4 --seed 7 \
    --out-dir outputs/wfc_scenes --name home07

# 批量生成 12 套不同户型
uv run python -m plugins.scenes.wfc_scenes --batch 12 --grid 4x3 \
    --out-dir outputs/wfc_scenes

# 春节装饰（红灯笼/福字）可选
uv run python -m plugins.scenes.wfc_scenes --grid 4x3 --festival --ascii
```

```python
from plugins.scenes.wfc_scenes import ChineseHomeScene

home = ChineseHomeScene(grid=(4, 3), seed=42)
print(home.ascii_map())                       # ASCII 户型图
# 行列不等宽（方案 2）：客厅 4m、卫生间 2.4m…
home2 = ChineseHomeScene(grid=(4, 3), seed=42, variable_size=True)
home.export_layout("home.json")               # 可复放的布局
home.topdown_svg("plan.svg")                  # 俯视户型图（无需 Genesis）
home.export_glb("home.glb")                   # 整体模型（可作静态舞台）

genesis_scene = home.build(headless=True, device="cpu")   # 实体场景
for _ in range(100):
    genesis_scene.step()
```

从已有布局复放（跳过 WFC）：

```python
home = ChineseHomeScene(layout="home.json")
```

## 整理好的规则（core/rules_cn.py）

### 模块目录（每格默认 3.2 m）

| 模块 | 中文名 | 默认墙模式 | 权重 | 数量上限 | 家具 |
|------|--------|-----------|------|---------|------|
| living | 客厅 | 沙发墙+电视墙（对面或转角） | 1.0 | - | 布艺沙发、茶几+茶具、电视柜+壁挂电视、地毯、鱼缸(随机)、字画、绿植(随机) |
| dining | 餐厅 | 四边开放 | 0.75 | 2 | 圆餐桌+转盘+四椅、碗碟 |
| kitchen | 厨房 | 三面台面+一面开口 | 0.35 | 2 | U型橱柜、双灶灶台、抽油烟机、水槽、冰箱 |
| bedroom | 卧室 | 床头墙+衣柜墙（对面或转角） | 0.55 | 4 | 双人床、床头柜×2+台灯、衣柜 |
| study | 书房 | 书桌墙+书架墙（对面或转角） | 0.35 | 1 | 书桌+座椅+显示器、书架+书 |
| entry | 玄关 | 鞋柜墙+入户门外墙 | 1.0 | 1 | 鞋柜、换鞋凳、地垫、挂衣架、穿衣镜 |
| balcony | 阳台 | 一面外栏杆 | 0.5 | 2 | 洗衣机、晾衣架、储物柜、绿植×2 |
| tea | 茶室 | 茶台墙（或转角） | 0.25 | 1 | 茶台+茶具、坐墩×4、博古架、字画 |
| bathroom | 卫生间 | 三面墙+门洞 | 0.22 | 2 | 马桶、洗手台+镜子、淋浴 |
| corridor | 过道 | 四边开放 | 0.15 | 2 | 长地毯、绿植 |

**转角墙模式**：客厅/卧室/书房/茶室除默认的"对墙"模式外，还有"相邻两面墙"
的转角模式（真实户型的转角房间），家具布置随模式切换（如转角客厅的电视墙贴
侧墙、面向沙发）。

### 格子尺寸（行列不等宽）

默认全屋统一 3.2m 正方形格子。开启 `variable_size=True`（CLI `--variable`）
后每列宽度/每行深度独立随机抽取（`min_cell`~`max_cell`，默认 2.4~4.2m），并
按入住模块的**最小边长**（`TileSpec.min_edge`，客厅 3.2 / 卧室 3.0 / 厨房 2.8 /
卫生间 2.2 …见模块目录）抬升——"厅大卧小卫小"由此而来。家具全部按格子实际
尺寸贴墙布置：台面/地毯/晾衣杆等长度自适应，沙发/圆桌椅子随跨度收放。
布局 JSON 记录 `col_widths`/`row_heights`（v2 格式，兼容读取 v1 统一格式）。

### 边插座与邻接规则

每格四条边（N/E/S/W）带"插座"，旋转随模块整体旋转：

| 插座 | 含义 |
|------|------|
| `wall` | 实墙/家具靠墙 |
| `open` | 开放通行（动线、门洞、厨房开口） |
| `counter` | 厨房台面沿 |
| `rail` | 阳台栏杆（仅外墙） |
| `door` | 入户门（仅外墙） |

邻接规则（对称）：

- `wall ⇆ wall`：两间房背靠背共用隔墙
- `open ⇆ open`：开放动线连通
- `counter ⇆ counter`：台面跨格延续（同一间大厨房）
- `counter ⇆ wall`：台面顶到邻居隔墙

### 边界规则（外墙）

只有 `wall / counter / rail / door` 允许朝向室外 —— 住宅永远是封闭外壳，
因此**入户门和阳台栏杆只能出现在外墙上**（由 WFC 约束天然保证）。

### 全局规则（边约束表达不了的，由校验器 + 预设种子格保证）

1. **玄关恰好 1 个**，进门动线至少通向客厅/过道/餐厅
2. **客厅 ≥ 1，厨房 ≥ 1**；面积 ≥ 8 格时**阳台 ≥ 1**
3. **厨房开口必须邻餐厅/客厅**（现代户型客餐厨一体）
4. **阳台必须邻客厅/卧室/餐厅/茶室**
5. 每类模块数量不超过上表上限
6. 卧室数量 ≥ `min_bedrooms`（默认 ≥ 8 格的户型 1 间）

存在性/动线规则通过**预设种子格**（WFC 标准技巧）构造性保证：每轮坍缩前，
随机把某个边缘格约束为玄关、某相邻格对约束为"厨房→开口→餐厅"、阳台格约束
其一个内侧邻格为居室；其余格子正常 WFC 坍缩。数量上限作为坍缩期硬约束参与
传播。校验失败的种子自动重试（`max_attempts`）。

## 产物

| 产物 | 说明 |
|------|------|
| `*.json` | 布局（格坐标、模块、墙模式、旋转、种子），可复放 |
| `*.svg` | 俯视户型图（房间填色 + 家具投影 + 中文标签），无需 Genesis |
| `*.glb` | 整体模型（结构+家具合并），可导入 DCC 或作静态舞台 |
| ASCII | 终端即时的户型检查（`L^ K>` … 字符=模块，箭头=朝向） |
| Genesis | 全部家具为独立刚体（可交互），墙体/楼板 fixed |

## 与其他插件集成

DreamDojo / maniskill 环境想在自定义户型里采数据时，在创建 `gs.Scene` 之后、
`scene.build()` 之前调用：

```python
from plugins.scenes.wfc_scenes import populate_genesis_scene, load_layout

layout = load_layout("home.json")
populate_genesis_scene(scene, layout)   # 在 build() 前调用
scene.build()
```

## 网格尺寸建议

推荐 **3x2 ~ 6x5**（6~30 格）。4x3 / 5x4 接近 100% 成功率；6x4/6x5 亦稳定；
更大的网格（如 8x6）预设约束与数量上限的组合会显著降低成功率（自动放大重试
次数部分缓解）。小户型（< 8 格）不强制阳台与卧室。

## 模块结构

```
wfc_scenes/
├── core/wfc.py         # 通用 WFC 求解器（传播/熵坍缩/回溯/预设格/数量上限）
├── core/rules_cn.py    # 中式家居规则（模块目录/插座/邻接/边界/全局校验）
├── assets/furniture_cn.py  # 家具原语库（Trimesh/Genesis 双后端）
├── scene.py            # ChineseHomeScene：WFC→JSON/SVG/GLB/Genesis
├── __main__.py         # CLI
└── tests/              # 离线测试（无网络；Genesis 冒烟测试按需跳过）
```

## 测试

```bash
uv run python -m pytest plugins/scenes/wfc_scenes/tests/    # 44 项，全部离线
```
