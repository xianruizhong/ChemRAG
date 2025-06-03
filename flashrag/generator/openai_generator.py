import os
from typing import List
from copy import deepcopy
import warnings
from tqdm import tqdm
import numpy as np
import random
import asyncio
import openai
from openai import AsyncOpenAI, AsyncAzureOpenAI
import tiktoken
from transformers import AutoTokenizer


class OpenaiGenerator:
    """Class for api-based openai models"""

    def __init__(self, config):
        self._config = config
        self.update_config()
        
        # load openai client
        self.tokenizer = None
        if "api_type" in self.openai_setting and self.openai_setting["api_type"] == "azure":
            del self.openai_setting["api_type"]
            self.client = AsyncAzureOpenAI(**self.openai_setting)
        elif "api_type" in self.openai_setting and self.openai_setting["api_type"] == "server":
            del self.openai_setting["api_type"]
            self.client = AsyncOpenAI(**self.openai_setting)
            self.tokenizer = AutoTokenizer.from_pretrained(self._config['model2path']['LLM'])
        else:
            self.client = AsyncOpenAI(**self.openai_setting)
        
        if self.tokenizer is None:
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model_name)
            except Exception as e:
                print("Error: ", e)
                warnings.warn("This model is not supported by tiktoken. Use gpt-3.5-turbo instead.")
                self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config_data):
        self._config = config_data
        self.update_config()
    
    def update_config(self):
        self.update_base_setting()
        self.update_additional_setting()

    def update_base_setting(self):
        self.model_name = self._config["generator_model"]
        self.batch_size = self._config["generator_batch_size"]
        self.generation_params = self._config["generation_params"]

        self.openai_setting = self._config["openai_setting"]
        if self.openai_setting["api_key"] is None:
            self.openai_setting["api_key"] = os.getenv("OPENAI_API_KEY")

    def update_additional_setting(self):
        pass
    
    '''async def get_response(self, input: List, **params):
        response = await self.client.chat.completions.create(model=self.model_name, messages=input, **params)
        return response.choices[0]'''
    async def get_response(self, input: List, max_retries: int = 5, backoff_base: float = 1.0, **params):
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name, messages=input, **params
                )
                return response.choices[0]
            except (openai.APIError, openai.APITimeoutError, openai.APIConnectionError, asyncio.TimeoutError) as e:
                wait_time = backoff_base * (2 ** attempt) + random.uniform(0, 0.5)
                print(f"[Retry {attempt + 1}/{max_retries}] Error: {e}. Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
            except Exception as e:
                print(f"Unexpected error: {e}")
                break
        raise RuntimeError("Failed to get response from OpenAI API after multiple retries.")

    async def get_batch_response(self, input_list: List[List], batch_size, **params):
        total_input = [self.get_response(input, **params) for input in input_list]
        all_result = []
        for idx in tqdm(range(0, len(input_list), batch_size), desc="Generation process: "):
            batch_input = total_input[idx : idx + batch_size]
            batch_result = await asyncio.gather(*batch_input)
            all_result.extend(batch_result)

        return all_result

    def generate(self, input_list: List[List], batch_size=None, return_scores=False, **params) -> List[str]:
        # deal with single input
        if len(input_list) == 1 and isinstance(input_list[0], dict):
            input_list = [input_list]
        if batch_size is None:
            batch_size = self.batch_size
        # deal with generation params
        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        if "do_sample" in generation_params:
            generation_params.pop("do_sample")

        max_tokens = params.pop("max_tokens", None) or params.pop("max_new_tokens", None)
        if 'o1' in self.model_name.lower():
            if max_tokens is not None:
                generation_params["max_completion_tokens"] = max_tokens
            else:
                generation_params["max_completion_tokens"] = generation_params.get(
                    "max_tokens", generation_params.pop("max_new_tokens", None)
                )
            generation_params.pop("max_tokens", None)
        else:
            generation_params.pop("max_new_tokens", None)
            if max_tokens is not None:
                generation_params["max_tokens"] = max_tokens
            else:
                generation_params["max_tokens"] = generation_params.get(
                    "max_tokens", generation_params.pop("max_new_tokens", None)
                )
        generation_params.pop("max_new_tokens", None)

        if return_scores:
            if generation_params.get("logprobs") is not None:
                generation_params["logprobs"] = True
                warnings.warn("Set logprobs to True to get generation scores.")
            else:
                generation_params["logprobs"] = True

        if generation_params.get("n") is not None:
            generation_params["n"] = 1
            warnings.warn("Set n to 1. It can minimize costs.")
        else:
            generation_params["n"] = 1

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.get_batch_response(input_list, batch_size, **generation_params))

        # parse result into response text and logprob
        scores = []
        response_text = []
        for res in result:
            response_text.append(res.message.content)
            if return_scores:
                score = np.exp(list(map(lambda x: x.logprob, res.logprobs.content)))
                scores.append(score)
        if return_scores:
            return response_text, scores
        else:
            return response_text
