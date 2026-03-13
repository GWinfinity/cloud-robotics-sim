from engine.constants import (
    PROJ_DIR, ALIYUN_QWEN_API_KEY, MAX_TOKENS, 
    TEMPERATURE, NUM_COMPLETIONS, QWEN_DEFAULT_MODEL
)
import json
import os
import time
from pathlib import Path


class QwenClient:
    """
    阿里云Qwen客户端
    
    Args:
        model_name (str): 模型名称，默认为constants中配置的QWEN_DEFAULT_MODEL
        cache (str): 缓存文件路径，默认为"qwen_cache.json"
    """
    
    def __init__(self, model_name=QWEN_DEFAULT_MODEL, cache="qwen_cache.json"):
        self.cache_file = cache
        self.model_name = model_name
        self.exponential_backoff = 1
        
        # 加载缓存文件，如果存在的话
        if os.path.exists(cache):
            while os.path.exists(self.cache_file + ".tmp") or os.path.exists(self.cache_file + ".lock"):
                time.sleep(0.1)
            with open(cache, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}
        
        # 初始化阿里云Qwen客户端
        # 注意：这里需要根据阿里云的API文档安装对应的SDK
        # 示例：pip install dashscope
        try:
            import dashscope
            dashscope.api_key = ALIYUN_QWEN_API_KEY
            self.dashscope = dashscope
        except ImportError:
            print("Warning: dashscope SDK not found. Please install it with 'pip install dashscope'")
            self.dashscope = None
        except Exception as e:
            print(f"Warning: Failed to initialize Qwen client: {e}")
            self.dashscope = None
    
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
        if self.dashscope is None:
            raise RuntimeError("Qwen client not initialized successfully")
        
        print(f'[INFO] Qwen: querying for {num_completions=}, {skip_cache_completions=}')
        if skip_cache:
            print(f'[INFO] Qwen: Skipping cache')
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
                    # 阿里云Qwen API支持图像输入，这里需要根据实际API调整
                    # 目前仅处理文本输入
                    raise NotImplementedError("Image input not supported yet for Qwen")
                else:
                    raise ValueError(f"Unsupported content type: {content_item['type']}")
        else:
            raise ValueError(f"Unsupported user_prompt type: {type(user_prompt)}")
        
        # 生成缓存键
        cache_key = None
        results = []
        if not skip_cache:
            cache_key = str((user_prompt, system_prompt, max_tokens, temperature, stop_sequences, 'qwen'))
            
            # 检查缓存
            num_completions = skip_cache_completions + num_completions
            if cache_key in self.cache:
                print(f'[INFO] Qwen: cache hit {len(self.cache[cache_key])}')
                if len(self.cache[cache_key]) < num_completions:
                    num_completions -= len(self.cache[cache_key])
                    results = self.cache[cache_key]
                else:
                    return cache_key, self.cache[cache_key][skip_cache_completions:num_completions]
        
        # 生成新的完成
        while num_completions > 0:
            try:
                # 调用阿里云Qwen API生成文本
                response = self.dashscope.Generation.call(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop_sequences
                )
                
                # 解析响应
                if response.status_code == 200:
                    content = response.output.choices[0].message.content
                    indented = content.split('\n')
                    results.append(indented)
                    num_completions -= 1
                else:
                    raise RuntimeError(f"Qwen API returned error: {response.code} - {response.message}")
                
            except Exception as e:
                print(f"Error calling Qwen API: {e}")
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


def setup_qwen(model_name=None, cache=None):
    """
    设置Qwen客户端
    
    Args:
        model_name (str, optional): 模型名称，默认从环境变量 QWEN_MODEL_NAME 读取，否则使用 constants 中的默认值
        cache (str, optional): 缓存文件路径，默认为 "qwen_cache.json" 或 "qwen_cache_{username}.json"
    
    Returns:
        QwenClient: 初始化的Qwen客户端实例
    """
    try:
        username = os.getlogin()
    except OSError:
        username = os.environ.get('USER') or os.environ.get('LOGNAME')
    
    if cache is None:
        cache_file = 'qwen_cache.json' if not os.path.exists('/viscam/') else f'qwen_cache_{username}.json'
    else:
        cache_file = cache
    
    # 如果传入了 model_name 则使用传入的，否则使用默认值（从环境变量或 constants 读取）
    if model_name is None:
        model_name = os.environ.get('QWEN_MODEL_NAME', QWEN_DEFAULT_MODEL)
    
    model = QwenClient(model_name=model_name, cache=cache_file)
    print(f'[INFO] QwenClient initialized with model: {model_name}')
    return model


if __name__ == "__main__":
    """
    测试Qwen客户端
    """
    # 测试基本功能
    client = QwenClient()
    
    print("Testing Qwen client initialization...")
    print(f"Client initialized with model: {client.model_name}")
    
    print("Qwen client tests completed.")
