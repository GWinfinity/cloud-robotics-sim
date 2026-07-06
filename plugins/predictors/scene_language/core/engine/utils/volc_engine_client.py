from engine.constants import (
    VOLC_ENGINE_API_KEY, MAX_TOKENS,
    TEMPERATURE, NUM_COMPLETIONS, VOLC_ENGINE_SEED,
    VOLC_ENGINE_SEED_RANGE
)
import logging
import os
import random

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class VolcEngineClient(BaseLLMClient):
    """火山引擎客户端，支持seed生成和管理功能"""

    def __init__(self, model_name="volc-llm-32k", cache="volc_cache.json", seed=VOLC_ENGINE_SEED):
        super().__init__(cache_file=cache, client_name="Volc Engine")
        self.model_name = model_name
        self.seed = seed
        self._validate_seed(seed)
        self.client = self._init_client()

    def _init_client(self):
        try:
            from volcengine.ark.runtime import Ark
            return Ark(api_key=VOLC_ENGINE_API_KEY)
        except ImportError:
            logger.warning("volcengine SDK not found. Please install it with 'pip install volcengine'")
            return None
        except Exception as e:
            logger.warning("Failed to initialize Volc Engine client: %s", e)
            return None

    def _call_api(self, messages, model_name, max_tokens, temperature, stop_sequences):
        response = self.client.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_sequences,
            seed=self.seed,
        )
        content = response.choices[0].message.content
        self.seed = self.generate_seed()
        return content

    def _validate_seed(self, seed):
        min_val, max_val = VOLC_ENGINE_SEED_RANGE
        if not (min_val <= seed <= max_val):
            raise ValueError(f"Seed must be between {min_val} and {max_val}, got {seed}")

    def generate_seed(self, min_value=None, max_value=None):
        min_val = min_value if min_value is not None else VOLC_ENGINE_SEED_RANGE[0]
        max_val = max_value if max_value is not None else VOLC_ENGINE_SEED_RANGE[1]
        if min_val > max_val:
            raise ValueError("min_value must be less than or equal to max_value")
        return random.randint(min_val, max_val)

    def set_seed(self, seed):
        self._validate_seed(seed)
        self.seed = seed

    def generate(self, user_prompt, system_prompt, max_tokens=MAX_TOKENS,
                 temperature=TEMPERATURE, stop_sequences=None, verbose=False,
                 num_completions=NUM_COMPLETIONS, skip_cache_completions=0, skip_cache=False):
        if self.client is None:
            raise RuntimeError("Volc Engine client not initialized successfully")

        logger.info("Volc Engine: querying for num_completions=%s, skip_cache_completions=%s",
                     num_completions, skip_cache_completions)
        if skip_cache:
            logger.info("Volc Engine: Skipping cache")
        if verbose:
            logger.debug("user_prompt: %s", user_prompt)

        messages = self.validate_prompt(user_prompt, system_prompt)

        cache_key = None
        results = []
        if not skip_cache:
            cache_key = str((user_prompt, system_prompt, max_tokens, temperature, stop_sequences, 'volc'))
            num_completions = skip_cache_completions + num_completions
            if cache_key in self.cache:
                logger.info("Volc Engine: cache hit %d", len(self.cache[cache_key]))
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


def setup_volc_engine():
    try:
        username = os.getlogin()
    except OSError:
        username = os.environ.get('USER') or os.environ.get('LOGNAME')

    cache_file = 'volc_cache.json' if not os.path.exists('/viscam/') else f'volc_cache_{username}.json'
    model = VolcEngineClient(cache=cache_file)
    return model


if __name__ == "__main__":
    client = VolcEngineClient()
    print("Testing seed generation...")
    seed1 = client.generate_seed()
    print(f"Generated seed 1: {seed1}")
    seed2 = client.generate_seed(min_value=100, max_value=200)
    print(f"Generated seed 2: {seed2}")
    client.set_seed(42)
    print(f"Set seed to: {client.seed}")
    print("Volc Engine client tests completed.")
