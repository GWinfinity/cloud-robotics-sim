"""
scene_language Plugin - Scene Language: Representing Scenes with Programs, Words, and Embeddings

来源: genesis-scene-language (https://github.com/zzyunzhi/scene-language)
核心实现: 使用大语言模型将文本描述转换为3D场景

论文: "The Scene Language: Representing Scenes with Programs, Words, and Embeddings"
作者: Yunzhi Zhang, Zizhang Li, Matt Zhou, Shangzhe Wu, Jiajun Wu (CVPR 2025)

核心特性:
- 文本到3D生成: 使用自然语言描述生成3D场景
- 程序合成: LLM生成Python代码创建场景
- 多种渲染器: 支持Mitsuba, Minecraft, 3D Gaussian Splatting
- 层次化表示: 场景的层次化结构和分解
- 图像条件生成: 基于参考图像生成3D场景

支持的渲染器:
- Mitsuba: 物理渲染器，高质量光照和材质
- Minecraft: 体素风格，程序化建筑
- 3D Gaussian Splatting: 神经渲染 (即将支持)

导出格式:
- Mesh (.ply)
- Genesis场景格式
- Minecraft JSON
"""

__version__ = "0.1.0"
__source__ = "genesis-scene-language"
__paper__ = "CVPR 2025"

# 核心组件
try:
    from .core.engine.scene_generator import SceneGenerator
    from .core.engine.program_executor import ProgramExecutor
    from .core.engine.renderers.mitsuba_renderer import MitsubaRenderer
    from .core.engine.renderers.minecraft_renderer import MinecraftRenderer
    from .core.engine.utils.lm_utils import LanguageModelClient
    
    __all__ = [
        'SceneGenerator',
        'ProgramExecutor',
        'MitsubaRenderer',
        'MinecraftRenderer',
        'LanguageModelClient',
    ]
except ImportError:
    # 开发模式，部分依赖可能未安装
    __all__ = []
