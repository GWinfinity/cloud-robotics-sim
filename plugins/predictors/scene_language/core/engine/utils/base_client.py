import json
import logging
import os
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """LLM客户端基类，提供缓存管理和重试逻辑"""

    def __init__(self, cache_file: str, client_name: str):
        self.cache_file = cache_file
        self.client_name = client_name
        self.exponential_backoff = 1
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_file):
            while os.path.exists(self.cache_file + ".tmp") or os.path.exists(
                self.cache_file + ".lock"
            ):
                time.sleep(0.1)
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {}

    @abstractmethod
    def _init_client(self):
        """初始化底层SDK客户端，返回客户端对象或None"""
        ...

    @abstractmethod
    def _call_api(self, messages: list, model_name: str, max_tokens: int,
                  temperature: float, stop_sequences: list) -> str:
        """调用API并返回文本内容"""
        ...

    def validate_prompt(self, user_prompt, system_prompt: str) -> list:
        """验证并构建messages列表"""
        if isinstance(user_prompt, str):
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        elif isinstance(user_prompt, list):
            messages = [{"role": "system", "content": system_prompt}]
            for item in user_prompt:
                if item["type"] == "text":
                    messages.append({"role": "user", "content": item["text"]})
                elif item["type"] == "image_url" and os.path.exists(item["image_url"]):
                    raise NotImplementedError(
                        f"Image input not supported yet for {self.client_name}"
                    )
                else:
                    raise ValueError(f"Unsupported content type: {item['type']}")
            return messages
        else:
            raise ValueError(f"Unsupported user_prompt type: {type(user_prompt)}")

    def call_with_retry(self, messages: list, model_name: str, max_tokens: int,
                        temperature: float, stop_sequences: list) -> str:
        """带指数退避重试的API调用"""
        while True:
            try:
                return self._call_api(
                    messages, model_name, max_tokens, temperature, stop_sequences
                )
            except Exception as e:
                logger.warning("Error calling %s API: %s", self.client_name, e)
                time.sleep(self.exponential_backoff)
                self.exponential_backoff *= 2
                if self.exponential_backoff > 64:
                    raise RuntimeError(
                        f"Failed to get response after multiple retries: {e}"
                    ) from e

    def update_cache(self, cache_key: str, results: list):
        while os.path.exists(self.cache_file + ".tmp") or os.path.exists(
            self.cache_file + ".lock"
        ):
            time.sleep(0.1)

        with open(self.cache_file + ".lock", "w"):
            pass

        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f:
                    current_cache = json.load(f)
            else:
                current_cache = {}

            current_cache[cache_key] = results

            with open(self.cache_file + ".tmp", "w") as f:
                json.dump(current_cache, f, ensure_ascii=False, indent=2)

            os.replace(self.cache_file + ".tmp", self.cache_file)
            self.cache = current_cache
        finally:
            if os.path.exists(self.cache_file + ".lock"):
                os.remove(self.cache_file + ".lock")
