"""
Scene Language Plugin - 基础使用示例

演示如何使用 Scene Language 将文本描述转换为3D场景
"""

import numpy as np
from pathlib import Path

# 导入插件
try:
    from cloud_robotics_sim.plugins.predictors.scene_language import (
        SceneGenerator, ProgramExecutor,
        MitsubaRenderer, MinecraftRenderer
    )
except ImportError:
    # 直接导入 (开发模式)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    # 注意: 实际导入路径可能需要根据项目结构调整


def example_text_to_3d():
    """示例1: 文本到3D生成"""
    print("=" * 60)
    print("示例1: 文本到3D生成")
    print("=" * 60)
    
    print("""
Scene Language的核心功能:
将自然语言描述转换为3D场景

流程:
  1. 用户提供文本描述
     例如: "a chessboard with a full set of chess pieces"
  
  2. LLM理解描述并生成Python代码
     - 分析场景组成 (棋盘、棋子)
     - 确定几何形状 (立方体、圆柱体)
     - 设定空间布局 (8x8网格)
  
  3. 执行代码创建3D场景
     - 实例化几何对象
     - 应用材质和纹理
     - 设置光照和相机
  
  4. 渲染输出
     - Mitsuba物理渲染
     - 生成图像或视频

支持的描述类型:
  - 具体物体: "a red apple on a wooden table"
  - 建筑场景: "a medieval castle with towers"
  - 抽象概念: "a scene inspired by Egon Schiele"
  - 复杂场景: "a large-scale city with skyscrapers"
""")
    
    print("\n代码示例:")
    print("""
  from scene_language import SceneGenerator
  
  # 创建生成器
  generator = SceneGenerator(renderer='mitsuba')
  
  # 生成场景
  scene = generator.generate(
      "a detailed cylindrical medieval tower"
  )
  
  # 渲染
  scene.render('tower.gif')
  """)
    
    print("\n✓ 文本到3D生成说明完成\n")


def example_program_synthesis():
    """示例2: 程序合成"""
    print("=" * 60)
    print("示例2: 程序合成")
    print("=" * 60)
    
    print("""
LLM程序合成原理:

不同于直接生成3D模型，Scene Language让LLM生成Python代码。

优势:
  1. 可解释性: 代码可以被理解和修改
  2. 可控性: 用户可以编辑生成的代码
  3. 组合性: 可以组合多个代码片段
  4. 泛化性: 代码可以参数化，生成变体

示例代码结构:
  ```python
  def create_medieval_tower():
      # 创建塔基
      base = Cylinder(radius=5, height=2)
      base.material = StoneTexture()
      
      # 创建塔身
      body = Cylinder(radius=4, height=15)
      body.position = [0, 8.5, 0]
      
      # 创建塔顶
      roof = Cone(radius=4.5, height=5)
      roof.position = [0, 18, 0]
      
      # 添加窗户
      for i in range(4):
          window = create_arched_window()
          angle = i * np.pi / 2
          window.position = [3*np.cos(angle), 10, 3*np.sin(angle)]
      
      return Group([base, body, roof, windows])
  ```

代码执行:
  - 在安全的沙箱环境中执行
  - 自动处理依赖和导入
  - 捕获3D场景对象
""")
    
    print("\n程序合成的层次:")
    levels = [
        ("高层描述", "a medieval tower", "整体概念"),
        ("中层分解", "base, body, roof, windows", "组件拆分"),
        ("底层几何", "Cylinder, Cone, Box", "具体形状"),
        ("材质纹理", "StoneTexture, WoodTexture", "视觉属性"),
    ]
    for level, example, desc in levels:
        print(f"  {level}: '{example}' ({desc})")
    
    print("\n✓ 程序合成说明完成\n")


def example_renderers():
    """示例3: 多种渲染器"""
    print("=" * 60)
    print("示例3: 多种渲染器")
    print("=" * 60)
    
    print("""
Scene Language支持多种渲染器，适应不同应用场景:

1. Mitsuba渲染器
   ├── 特点: 物理渲染，高质量光照
   ├── 材质: PBR材质，支持反射/折射
   ├── 光照: 全局光照，软阴影
   └── 适用: 真实感渲染，照片级效果

2. Minecraft渲染器
   ├── 特点: 体素风格，程序化建筑
   ├── 方块: 标准Minecraft方块类型
   ├── 风格: 像素化，游戏风格
   └── 适用: 游戏场景，快速原型

3. 3D Gaussian Splatting (即将支持)
   ├── 特点: 神经渲染，新视角合成
   ├── 表示: 3D高斯点云
   ├── 速度: 实时渲染
   └── 适用: 新视角合成，VR/AR

渲染器选择:
  ```python
  # Mitsuba - 高质量
  generator = SceneGenerator(renderer='mitsuba')
  
  # Minecraft - 游戏风格
  generator = SceneGenerator(renderer='minecraft')
  
  # 3DGS - 神经渲染
  generator = SceneGenerator(renderer='3dgs')
  ```
""")
    
    print("\n渲染对比:")
    renderers = [
        ("Mitsuba", "真实感", "高质量", "慢"),
        ("Minecraft", "体素", "中等", "快"),
        ("3DGS", "神经", "中高", "实时"),
    ]
    print(f"  {'渲染器':12s} {'风格':10s} {'质量':8s} {'速度'}")
    print("  " + "-" * 45)
    for name, style, quality, speed in renderers:
        print(f"  {name:12s} {style:10s} {quality:8s} {speed}")
    
    print("\n✓ 多种渲染器说明完成\n")


def example_hierarchical_structure():
    """示例4: 层次化结构"""
    print("=" * 60)
    print("示例4: 层次化场景结构")
    print("=" * 60)
    
    print("""
复杂场景的层次化表示:

大型场景需要层次化管理，Scene Language自动构建层次结构。

示例: 城市场景
```
Level 0: 城市 (City)
├── Level 1: 区域 (Districts)
│   ├── 商业区
│   │   ├── Level 2: 建筑物
│   │   │   ├── 摩天大楼
│   │   │   │   └── Level 3: 楼层/房间
│   │   │   └── 购物中心
│   │   └── Level 2: 街道/车辆
│   └── 住宅区
│       ├── 公寓楼
│       └── 公园
```

层次化的优势:
  1. 模块化: 每个层次可以独立编辑
  2. 细节控制: 选择展开特定层次
  3. 渲染优化: 视距裁剪
  4. 语义理解: 符合人类认知

导出层次:
  ```python
  scene.export_hierarchy(
      output_dir='city_hierarchy/',
      max_depth=3
  )
  ```

结果:
  - city_hierarchy/level_0/: 整体城市
  - city_hierarchy/level_1/: 区域分解
  - city_hierarchy/level_2/: 建筑物分解
""")
    
    print("\n层次化操作:")
    operations = [
        "展开/折叠特定层次",
        "编辑某层次的属性",
        "导出特定层次的几何",
        "层次间复制/粘贴",
    ]
    for op in operations:
        print(f"  - {op}")
    
    print("\n✓ 层次化结构说明完成\n")


def example_image_conditioned():
    """示例5: 图像条件生成"""
    print("=" * 60)
    print("示例5: 图像条件生成")
    print("=" * 60)
    
    print("""
基于参考图像的3D生成:

除了文本描述，还可以使用图像作为条件。

应用场景:
  1. 风格迁移: 将图像风格应用到3D场景
  2. 结构参考: 基于图像布局创建3D
  3. 内容扩展: 从2D图像生成3D场景

使用方法:
  ```python
  scene = generator.generate(
      description="similar 3D scene",
      condition_image="reference.png",
      temperature=0.8
  )
  ```

工作流程:
  1. 分析参考图像
     ├── 提取主要内容
     ├── 识别物体类型
     ├── 估计空间布局
     └── 分析颜色/材质
  
  2. 生成3D场景
     ├── 匹配图像内容
     ├── 扩展深度信息
     └── 添加3D细节
  
  3. 渲染验证
     ├── 对比参考图像
     └── 调整视角匹配

temperature参数:
  - 低 (0.2): 严格遵循图像
  - 中 (0.5): 平衡创新和一致
  - 高 (0.8): 更多创意发挥
""")
    
    print("\n图像条件示例:")
    examples = [
        ("建筑照片", "生成相似3D建筑模型"),
        ("室内设计", "创建3D室内场景"),
        ("概念草图", "转化为详细3D"),
        ("游戏截图", "生成可玩关卡"),
    ]
    for image_type, result in examples:
        print(f"  {image_type}: {result}")
    
    print("\n✓ 图像条件生成说明完成\n")


def example_genesis_export():
    """示例6: 导出到Genesis"""
    print("=" * 60)
    print("示例6: 导出到Genesis物理仿真")
    print("=" * 60)
    
    print("""
将生成的场景导入Genesis进行物理仿真:

流程:
  1. 使用Scene Language生成场景
     scene = generator.generate("a robot arm")
  
  2. 导出为Genesis格式
     scene.export_to_genesis('robot_arm_scene.py')
  
  3. 在Genesis中加载
     ```python
     import genesis as gs
     from robot_arm_scene import load_scene
     
     gs.init()
     scene = load_scene()
     
     # 添加物理属性
     scene.add_physics(
         gravity=[0, 0, -9.81],
         timestep=0.01
     )
     
     # 运行仿真
     for _ in range(1000):
         scene.step()
     ```

导出内容:
  - 几何网格 (Mesh)
  - 材质属性
  - 空间变换
  - 层次结构

物理化:
  - 自动计算碰撞体
  - 设置质量和惯性
  - 添加关节约束
  - 配置材质摩擦

应用场景:
  - 机器人仿真环境生成
  - 物理AI训练场景
  - 交互式3D应用
""")
    
    print("\n导出选项:")
    options = [
        ("格式", "Genesis Python脚本"),
        ("几何", "Trimesh/MJCF"),
        ("物理", "可选添加"),
        ("材质", "纹理贴图"),
    ]
    for option, value in options:
        print(f"  {option}: {value}")
    
    print("\n✓ Genesis导出说明完成\n")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Scene Language Plugin - 使用示例")
    print("=" * 60 + "\n")
    
    # 检查依赖
    try:
        import anthropic
        print("Anthropic SDK 可用")
    except ImportError:
        print("警告: Anthropic SDK 未安装 (pip install anthropic)")
    
    try:
        import openai
        print("OpenAI SDK 可用")
    except ImportError:
        print("警告: OpenAI SDK 未安装 (pip install openai)")
    
    print()
    
    # 运行示例
    examples = [
        ("文本到3D", example_text_to_3d),
        ("程序合成", example_program_synthesis),
        ("多种渲染器", example_renderers),
        ("层次化结构", example_hierarchical_structure),
        ("图像条件生成", example_image_conditioned),
        ("Genesis导出", example_genesis_export),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"{name} 示例失败: {e}\n")
    
    print("=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
