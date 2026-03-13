import os
from typing import Literal
from pathlib import Path

PROJ_DIR: str = str(Path(__file__).parent.parent.absolute())
IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

try:
    from .key import ANTHROPIC_API_KEY
except:
    print("Warning: No Anthropic keys found.")
    ANTHROPIC_API_KEY = ''
try:
    from .key import OPENAI_API_KEY
except:
    # print("Warning: No OpenAI keys found.")
    OPENAI_API_KEY = ''

# 火山引擎配置
try:
    from .key import VOLC_ENGINE_API_KEY
except:
    print("Warning: No Volc Engine keys found.")
    VOLC_ENGINE_API_KEY = ''

# 阿里云Qwen配置
try:
    from .key import ALIYUN_QWEN_API_KEY
except:
    print("Warning: No Aliyun Qwen keys found.")
    ALIYUN_QWEN_API_KEY = ''

try:
    import torch
    if torch.cuda.is_available():  # hack
        os.environ['MI_DEFAULT_VARIANT'] = 'cuda_ad_rgb'
    else:
        os.environ['MI_DEFAULT_VARIANT'] = 'scalar_rgb'
except ModuleNotFoundError:
    print(f'[INFO] torch not found, setting default variant to scalar_rgb')
    os.environ['MI_DEFAULT_VARIANT'] = 'scalar_rgb'

import mitsuba as mi
mi.set_variant(os.environ['MI_DEFAULT_VARIANT'])

ENGINE_MODE: Literal['neural', 'mi', 'minecraft', 'lmd', 'mi_material', 'exposed'] = os.getenv('ENGINE_MODE', 'exposed')
# print(f'{ENGINE_MODE=}')
DEBUG: bool = os.environ.get('DEBUG', '0') == '1'

PROMPT_MODE: Literal['default', 'calc', 'assert', 'sketch'] = os.environ.get('PROMPT_MODE', 'default' if ENGINE_MODE == 'minecraft' else 'calc')
if ENGINE_MODE == 'minecraft' and PROMPT_MODE != 'default':
    print(f'WARNING {PROMPT_MODE=}')
if ENGINE_MODE == 'mi' and PROMPT_MODE != 'calc':
    print(f'WARNING {PROMPT_MODE=}')

ONLY_RENDER_ROOT = True

if 'DRY_RUN' in os.environ:
    DRY_RUN = bool(os.environ['DRY_RUN'])
else:
    DRY_RUN = False
# print(f'DRY_RUN={DRY_RUN}')

# LLM configs
LLM_PROVIDER: Literal['gpt', 'claude', 'llama', 'volc', 'qwen'] = 'claude'
TEMPERATURE: float = 0.05
NUM_COMPLETIONS: int = 1
# MAX_TOKENS: int = 8192
MAX_TOKENS: int = 16384

# 火山引擎seed配置
VOLC_ENGINE_SEED = 42  # 默认seed值
VOLC_ENGINE_SEED_RANGE = (0, 2**32 - 1)  # seed取值范围

# 阿里云Qwen配置
# 支持从环境变量读取模型名称，默认为 qwen-plus
# 阿里云官方模型名参考：
#   - 通义千问大语言模型：qwen-turbo, qwen-plus, qwen-max, qwen-max-longcontext
#   - 开源版：qwen-7b-chat, qwen-14b-chat, qwen-72b-chat 等
#   - Qwen2.5系列：qwen2.5-7b-instruct, qwen2.5-14b-instruct, qwen2.5-32b-instruct, qwen2.5-72b-instruct
#   - Qwen3系列：qwen3-8b, qwen3-14b 等
#   - 视觉模型：qwen-vl-plus, qwen-vl-max
#   - 代码模型：qwen-coder-plus, qwen-coder-turbo
# 更多模型请参考阿里云官方文档：https://help.aliyun.com/zh/dashscope/developer-reference/model-introduction
QWEN_DEFAULT_MODEL = os.getenv('QWEN_MODEL_NAME', 'qwen-plus')
QWEN_TEMPERATURE = float(os.getenv('QWEN_TEMPERATURE', '0.05'))
QWEN_MAX_TOKENS = int(os.getenv('QWEN_MAX_TOKENS', '16384'))

assert 0 <= TEMPERATURE <= 1, TEMPERATURE
if NUM_COMPLETIONS > 1:
    assert TEMPERATURE > 0, TEMPERATURE
