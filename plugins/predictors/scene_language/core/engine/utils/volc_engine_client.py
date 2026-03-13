from engine.constants import (
    PROJ_DIR, VOLC_ENGINE_API_KEY, MAX_TOKENS, 
    TEMPERATURE, NUM_COMPLETIONS, VOLC_ENGINE_SEED,
    VOLC_ENGINE_SEED_RANGE
)
import json
import os
import time
import random
from pathlib import Path


class VolcEngineClient:
    """
    火山引擎客户端，支持seed生成和管理功能
    
    Args:
        model_name (str): 模型名称，默认使用火山引擎提供的默认模型
        cache (str): 缓存文件路径，默认为"volc_cache.json"
        seed (int): 随机种子，默认为constants中配置的VOLC_ENGINE_SEED
    """
    
    def __init__(self, model_name="volc-llm-32k", cache="volc_cache.json", seed=VOLC_ENGINE_SEED):
        self.cache_file = cache
        self.model_name = model_name
        self.seed = seed
        self.exponential_backoff = 1
        
        # 验证seed值是否在有效范围内
        self._validate_seed(seed)
        
        # 加载缓存文件，如果存在的话
        if os.path.exists(cache):
            while os.path.exists(self.cache_file + ".tmp") or os.path.exists(self.cache_file + ".lock"):
                time.sleep(0.1)
            with open(cache, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
        
        # 初始化火山引擎客户端
        # 注意：这里需要根据火山引擎的API文档安装对应的SDK
        # 示例：pip install volcengine
        try:
            from volcengine.ark.runtime import Ark
            self.client = Ark(api_key=VOLC_ENGINE_API_KEY)
        except ImportError:
            print("Warning: volcengine SDK not found. Please install it with 'pip install volcengine'")
            self.client = None
        except Exception as e:
            print(f"Warning: Failed to initialize Volc Engine client: {e}")
            self.client = None
    
    def _validate_seed(self, seed):
        """
        验证种子值是否在有效范围内
        
        Args:
            seed (int): 要验证的种子值
            
        Raises:
            ValueError: 如果种子值不在有效范围内
        """
        min_val, max_val = VOLC_ENGINE_SEED_RANGE
        if not (min_val <= seed <= max_val):
            raise ValueError(f"Seed must be between {min_val} and {max_val}, got {seed}")
    
    def generate_seed(self, min_value=None, max_value=None):
        """
        生成随机种子
        
        Args:
            min_value (int, optional): 种子最小值，默认为VOLC_ENGINE_SEED_RANGE[0]
            max_value (int, optional): 种子最大值，默认为VOLC_ENGINE_SEED_RANGE[1]
            
        Returns:
            int: 生成的随机种子
        """
        min_val = min_value if min_value is not None else VOLC_ENGINE_SEED_RANGE[0]
        max_val = max_value if max_value is not None else VOLC_ENGINE_SEED_RANGE[1]
        
        if min_val > max_val:
            raise ValueError("min_value must be less than or equal to max_value")
        
        return random.randint(min_val, max_val)
    
    def set_seed(self, seed):
        """
        设置种子值
        
        Args:
            seed (int): 要设置的种子值
            
        Raises:
            ValueError: 如果种子值不在有效范围内
        """
        self._validate_seed(seed)
        self.seed = seed
    
    def generate(self, user_prompt, system_prompt, max_tokens=MAX_TOKENS, 
                 temperature=TEMPERATURE, stop_sequences=None, verbose=False,
                 num_completions=NUM_COMPLETIONS, skip_cache_completions=0, skip_cache=False):
        """
        生成文本
        
        Args:
            user_prompt (str or list): 用户提示，可以是字符串或包含文本和图像的列表
            system_prompt (str): 系统提示
            max_tokens (int): 生成的最大令牌数
            temperature (float): 生成的随机性，范围0-1
            stop_sequences (list, optional): 停止序列
            verbose (bool): 是否打印详细信息
            num_completions (int): 生成的完成数量
            skip_cache_completions (int): 跳过缓存中的前n个完成
            skip_cache (bool): 是否跳过缓存
            
        Returns:
            tuple: (cache_key, results)，其中cache_key是缓存键，results是生成的结果列表
        
        Raises:
            RuntimeError: 如果客户端未初始化成功
        """
        if self.client is None:
            raise RuntimeError("Volc Engine client not initialized successfully")
        
        print(f'[INFO] Volc Engine: querying for {num_completions=}, {skip_cache_completions=}')
        if skip_cache:
            print(f'[INFO] Volc Engine: Skipping cache')
        if verbose:
            print(user_prompt)
            print("-----\n")
        
        # 准备API请求参数
        if isinstance(user_prompt, str):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        elif isinstance(user_prompt, list):
            # 处理包含图像的提示
            messages = [{"role": "system", "content": system_prompt}]
            for content_item in user_prompt:
                if content_item['type'] == 'text':
                    messages.append({"role": "user", "content": content_item['text']})
                elif content_item['type'] == 'image_url' and os.path.exists(content_item['image_url']):
                    # 火山引擎API可能支持图像输入，这里需要根据实际API调整
                    # 目前仅处理文本输入
                    raise NotImplementedError("Image input not supported yet for Volc Engine")
                else:
                    raise ValueError(f"Unsupported content type: {content_item['type']}")
        else:
            raise ValueError(f"Unsupported user_prompt type: {type(user_prompt)}")
        
        # 生成缓存键
        cache_key = None
        results = []
        if not skip_cache:
            cache_key = str((user_prompt, system_prompt, max_tokens, temperature, stop_sequences, 'volc'))
            
            # 检查缓存
            num_completions = skip_cache_completions + num_completions
            if cache_key in self.cache:
                print(f'[INFO] Volc Engine: cache hit {len(self.cache[cache_key])}')
                if len(self.cache[cache_key]) < num_completions:
                    num_completions -= len(self.cache[cache_key])
                    results = self.cache[cache_key]
                else:
                    return cache_key, self.cache[cache_key][skip_cache_completions:num_completions]
        
        # 生成新的完成
        while num_completions > 0:
            try:
                # 调用火山引擎API生成文本
                # 注意：这里的API调用方式需要根据火山引擎的实际API文档调整
                response = self.client.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop_sequences,
                    seed=self.seed
                )
                
                # 解析响应
                # 注意：这里的响应解析需要根据火山引擎的实际API响应格式调整
                # 以下是示例代码，需要根据实际情况修改
                content = response.choices[0].message.content
                indented = content.split('\n')
                results.append(indented)
                
                # 更新种子，以便下次生成不同的结果
                self.seed = self.generate_seed()
                
                num_completions -= 1
                
            except Exception as e:
                print(f"Error calling Volc Engine API: {e}")
                # 指数退避重试
                time.sleep(self.exponential_backoff)
                self.exponential_backoff *= 2
                if self.exponential_backoff > 64:
                    raise RuntimeError(f"Failed to get response after multiple retries: {e}") from e
        
        # 更新缓存
        if not skip_cache and cache_key is not None:
            self.update_cache(cache_key, results)
        
        return cache_key, results[skip_cache_completions:]
    
    def update_cache(self, cache_key, results):
        """
        更新缓存
        
        Args:
            cache_key (str): 缓存键
            results (list): 要缓存的结果列表
        """
        # 确保缓存文件安全更新
        while os.path.exists(self.cache_file + ".tmp") or os.path.exists(self.cache_file + ".lock"):
            time.sleep(0.1)
        
        # 创建锁文件
        with open(self.cache_file + ".lock", "w") as f:
            pass
        
        try:
            # 读取最新缓存
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    current_cache = json.load(f)
            else:
                current_cache = {}
            
            # 更新缓存
            current_cache[cache_key] = results
            
            # 写入临时文件
            with open(self.cache_file + ".tmp", "w") as f:
                json.dump(current_cache, f, ensure_ascii=False, indent=2)
            
            # 原子替换
            os.replace(self.cache_file + ".tmp", self.cache_file)
            
            # 更新内存缓存
            self.cache = current_cache
        finally:
            # 移除锁文件
            if os.path.exists(self.cache_file + ".lock"):
                os.remove(self.cache_file + ".lock")


def setup_volc_engine():
    """
    设置火山引擎客户端
    
    Returns:
        VolcEngineClient: 初始化的火山引擎客户端实例
    """
    try:
        username = os.getlogin()
    except OSError:
        username = os.environ.get('USER') or os.environ.get('LOGNAME')
    
    cache_file = 'volc_cache.json' if not os.path.exists('/viscam/') else f'volc_cache_{username}.json'
    model = VolcEngineClient(cache=cache_file)
    return model


if __name__ == "__main__":
    """
    测试火山引擎客户端
    """
    # 测试seed生成功能
    client = VolcEngineClient()
    
    print("Testing seed generation...")
    seed1 = client.generate_seed()
    print(f"Generated seed 1: {seed1}")
    
    seed2 = client.generate_seed(min_value=100, max_value=200)
    print(f"Generated seed 2: {seed2}")
    
    # 测试set_seed功能
    client.set_seed(42)
    print(f"Set seed to: {client.seed}")
    
    print("Volc Engine client tests completed.")
