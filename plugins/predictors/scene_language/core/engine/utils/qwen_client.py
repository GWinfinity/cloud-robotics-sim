from engine.constants import (
    ALIYUN_QWEN_API_KEY, MAX_TOKENS,
    TEMPERATURE, NUM_COMPLETIONS, QWEN_DEFAULT_MODEL
)
import logging
import os

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class QwenClient(BaseLLMClient):
    """阿里云Qwen客户端"""

    def __init__(self, model_name=QWEN_DEFAULT_MODEL, cache="qwen_cache.json"):
        super().__init__(cache_file=cache, client_name="Qwen")
        self.model_name = model_name
        self.dashscope = self._init_client()

    def _init_client(self):
        try:
            import dashscope
            dashscope.api_key = ALIYUN_QWEN_API_KEY
            return dashscope
        except ImportError:
            logger.warning("dashscope SDK not found. Please install it with 'pip install dashscope'")
            return None
        except Exception as e:
            logger.warning("Failed to initialize Qwen client: %s", e)
            return None

    def _call_api(self, messages, model_name, max_tokens, temperature, stop_sequences):
        response = self.dashscope.Generation.call(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_sequences,
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content
        raise RuntimeError(f"Qwen API returned error: {response.code} - {response.message}")

    def generate(self, user_prompt, system_prompt, max_tokens=MAX_TOKENS,
                 temperature=TEMPERATURE, stop_sequences=None, verbose=False,
                 num_completions=NUM_COMPLETIONS, skip_cache_completions=0, skip_cache=False):
        if self.dashscope is None:
            raise RuntimeError("Qwen client not initialized successfully")

        logger.info("Qwen: querying for num_completions=%s, skip_cache_completions=%s",
                     num_completions, skip_cache_completions)
        if skip_cache:
            logger.info("Qwen: Skipping cache")
        if verbose:
            logger.debug("user_prompt: %s", user_prompt)

        messages = self.validate_prompt(user_prompt, system_prompt)

        cache_key = None
        results = []
        if not skip_cache:
            cache_key = str((user_prompt, system_prompt, max_tokens, temperature, stop_sequences, 'qwen'))
            num_completions = skip_cache_completions + num_completions
            if cache_key in self.cache:
                logger.info("Qwen: cache hit %d", len(self.cache[cache_key]))
                if len(self.cache[cache_key]) < num_completions:
                    num_completions -= len(self.cache[cache_key])
                    results = self.cache[cache_key]
                else:
                    return cache_key, self.cache[cache_key][skip_cache_completions:num_completions]

        while num_completions > 0:
            content = self.call_with_retry(
                messages, self.model_name, max_tokens, temperature, stop_sequences
            )
            results.append(content.split('\n'))
            num_completions -= 1

        if not skip_cache and cache_key is not None:
            self.update_cache(cache_key, results)

        return cache_key, results[skip_cache_completions:]


def setup_qwen(model_name=None, cache=None):
    try:
        username = os.getlogin()
    except OSError:
        username = os.environ.get('USER') or os.environ.get('LOGNAME')

    if cache is None:
        cache_file = 'qwen_cache.json' if not os.path.exists('/viscam/') else f'qwen_cache_{username}.json'
    else:
        cache_file = cache

    if model_name is None:
        model_name = os.environ.get('QWEN_MODEL_NAME', QWEN_DEFAULT_MODEL)

    model = QwenClient(model_name=model_name, cache=cache_file)
    logger.info("QwenClient initialized with model: %s", model_name)
    return model


if __name__ == "__main__":
    client = QwenClient()
    print("Testing Qwen client initialization...")
    print(f"Client initialized with model: {client.model_name}")
    print("Qwen client tests completed.")
