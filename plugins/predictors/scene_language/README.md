# scene_language Plugin

来源: [genesis-scene-language](../../genesis-scene-language)

Scene Language: Representing Scenes with Programs, Words, and Embeddings (CVPR 2025)

使用大语言模型（LLM）将文本描述转换为3D场景的程序化生成框架。

## 核心功能

### 1. 文本到3D生成
将自然语言描述转换为3D场景：
```python
from cloud_robotics_sim.plugins.predictors.scene_language import SceneGenerator

# 创建场景生成器
generator = SceneGenerator(
    renderer='mitsuba',  # 或 'minecraft'
    lm_model='claude-3-sonnet'
)

# 生成场景
scene = generator.generate(
    description="a chessboard with a full set of chess pieces"
)

# 渲染
scene.render(output_path='chessboard.gif')
```

### 2. 程序合成
LLM生成Python代码来程序化创建场景：
```python
# LLM生成的代码示例
def create_chessboard():
    board = Chessboard(size=8, cell_size=1.0)
    for piece_type, position in INITIAL_SETUP:
        piece = ChessPiece(type=piece_type, color='white' if position[1] < 4 else 'black')
        board.place_piece(piece, position)
    return board
```

### 3. 多种渲染器支持
| 渲染器 | 特点 | 适用场景 |
|--------|------|----------|
| Mitsuba | 物理渲染，高质量光照 | 真实感渲染 |
| Minecraft | 体素风格，程序化建筑 | 游戏场景 |
| 3DGS | 神经渲染 (即将支持) | 新视角合成 |

### 4. 层次化场景表示
场景被表示为层次化的结构：
```
Scene
├── Level 0: 整体场景
│   └── 城市
├── Level 1: 主要组件
│   ├── 建筑物A
│   ├── 建筑物B
│   └── 道路
└── Level 2: 细节组件
    ├── 窗户
    ├── 门
    └── 车辆
```

## 快速开始

### 基础使用

```python
from cloud_robotics_sim.plugins.predictors.scene_language import SceneGenerator

# 配置API密钥 (在engine/key.py中设置)
# ANTHROPIC_API_KEY = 'your-key'

# 创建生成器
generator = SceneGenerator(renderer='mitsuba')

# 文本生成
tasks = [
    "a detailed cylindrical medieval tower",
    "a witch's house in Halloween",
    "a Roman Colosseum",
]

for task in tasks:
    scene = generator.generate(task)
    scene.render(f"outputs/{task}.gif")
```

### 图像条件生成

```python
# 基于参考图像生成3D场景
scene = generator.generate(
    description="similar 3D scene",
    condition_image="reference.png",
    temperature=0.8
)
```

### Minecraft渲染

```python
# 使用Minecraft风格渲染
generator = SceneGenerator(renderer='minecraft')
scene = generator.generate("a Greek temple")

# 导出为Minecraft JSON
scene.export('greek_temple.json')

# 在浏览器中查看
# python viewers/minecraft/run.py
# 打开 http://127.0.0.1:5001
```

### 导出到Genesis

```python
# 生成场景
scene = generator.generate("a detailed model of robot arm")

# 导出为Genesis格式
scene.export_to_genesis('robot_arm_genesis.py')

# 在Genesis中加载
import genesis as gs
from robot_arm_genesis import load_scene

scene = load_scene()
```

## 算法原理

### 场景语言表示

```
文本描述
    ↓ (LLM理解)
场景程序 (Python代码)
    ├── 几何定义 (形状、尺寸)
    ├── 材质属性 (颜色、纹理)
    ├── 空间布局 (位置、姿态)
    └── 层次结构 (父子关系)
    ↓ (执行)
3D场景表示
    ↓ (渲染)
图像/视频输出
```

### LLM程序合成

```
系统提示: 你是一个3D场景生成专家...

用户输入: "a chessboard with pieces"

LLM输出Python代码:
```python
def create_chessboard():
    # 创建棋盘
    board = Cube(size=[8, 1, 8])
    board.material = CheckerboardPattern(colors=['white', 'black'])
    
    # 添加棋子
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                piece = create_piece(type='pawn', color='white')
                board.add_child(piece, position=[i, 1, j])
    
    return board
```
```

### 层次化分解

复杂场景被分解为层次化的组件：
```python
# 自动层次化
city = Scene()
city.add_level('districts', [district1, district2, ...])
city.add_level('buildings', [building1, building2, ...])
city.add_level('rooms', [room1, room2, ...])
city.add_level('furniture', [table, chair, ...])
```

## 配置参数

### LLM配置

```yaml
# constants.yaml
language_model:
  default: claude-3-sonnet
  
  models:
    claude-3-sonnet:
      provider: anthropic
      model_id: claude-3-sonnet-20240229
      temperature: 0.7
      max_tokens: 4096
    
    gpt-4:
      provider: openai
      model_id: gpt-4-turbo-preview
      temperature: 0.7
      max_tokens: 4096
    
    llama-2-70b:
      provider: local
      model_path: models/llama-2-70b
      temperature: 0.8
```

### 渲染配置

```yaml
# renderer_config.yaml
mitsuba:
  integrator: path
  samples_per_pixel: 128
  resolution: [512, 512]
  
minecraft:
  block_size: 1.0
  max_blocks: 10000
  color_palette: default
```

## API密钥设置

在 `engine/key.py` 中设置API密钥：

```python
# Anthropic API key (for Claude models)
ANTHROPIC_API_KEY = 'YOUR_ANTHROPIC_API_KEY'

# OpenAI API key (for GPT models)
OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'

# Aliyun Qwen API key
ALIYUN_QWEN_API_KEY = 'YOUR_ALIYUN_QWEN_API_KEY'
```

## 示例

见 [examples/](examples/) 目录：

- `basic_usage.py` - 基础使用示例
- `ab_test.py` - A/B 测试框架

## 高级功能

### 自反思与MoE

```bash
# 使用自反思和多专家(MoE)生成更复杂的场景
python scripts/run_self_reflect_with_moe.py --tasks "a large-scale city"
```

### 层次化导出

```bash
# 导出场景的层次化部分
python scripts/postprocess/export.py --exp-patterns "run_*/*/0"
```

### 物理仿真

```bash
# 导出到物理仿真器
python scripts/experimental/simulate_pybullet.py
```

## 与其他方法对比

| 方法 | 输入 | 输出 | 可控性 | 质量 |
|------|------|------|--------|------|
| 传统建模 | 手动 | 3D模型 | 高 | 依赖技能 |
| 生成模型 | 文本/图像 | 3D表示 | 低 | 中高 |
| Scene Language | 文本 | 程序化场景 | 高 | 高 |

## 引用

```bibtex
@inproceedings{zhang2025scene,
  title={The Scene Language: Representing Scenes with Programs, Words, and Embeddings},
  author={Zhang, Yunzhi and Li, Zizhang and Zhou, Matt and Wu, Shangzhe and Wu, Jiajun},
  booktitle={CVPR},
  year={2025}
}
```

## 相关链接

- [论文](https://arxiv.org/abs/2410.16770)
- [项目页面](https://ai.stanford.edu/~yzzhang/projects/scene-language/)
- [Colab Notebook](https://colab.research.google.com/github/zzyunzhi/scene-language/blob/main/colab/text_to_scene.ipynb)

## Changelog

- **2026-03-13**: 从 genesis-scene-language 迁移到 genesis-cloud-sim plugins
- **2025**: CVPR 2025 发表论文
