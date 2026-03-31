import os
from vllm import LLM, SamplingParams
from typing import List, Dict, Optional

class VLLMClient:
    def __init__(self,
                 model_name: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 gpu_memory_utilization: float = 0.7,
                 ):
        """
        Args:
            model_name: Model name or path to load
            cache_dir: Directory to download model files
        """
        self.model_name = model_name
        # Initialize vLLM engine
        device = os.environ["CUDA_VISIBLE_DEVICES"]
        self.llm = LLM(
            model=model_name,
            # tensor_parallel_size = 2,
            pipeline_parallel_size=len(device.split(",")),  # tensor_parallel_size 대신 pipeline_parallel_size 사용
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096,
            trust_remote_code=True,
            download_dir=cache_dir
        )
        self.tokenizer = self.llm.get_tokenizer()

    def chat_completion(self,
                       message: str,
                       temperature: float = 0.7,
                       max_tokens: int = 2048,
                       top_p: float = 0.9,
                       top_k: int = -1,
                       frequency_penalty: float = 0.0,
                       presence_penalty: float = 0.0,
                       stop: Optional[List[str]] = None
                       ) -> str:
        """
        Generate chat completion for a single message
        """
        results = self.batch_chat_completion(
            [message], temperature, max_tokens, top_p, top_k,
            frequency_penalty, presence_penalty, stop
        )
        return results[0]

    def batch_chat_completion(self,
                              messages: List[str],
                              temperature: float = 0.7,
                              max_tokens: int = 2048,
                              top_p: float = 0.9,
                              top_k: int = -1,
                              frequency_penalty: float = 0.0,
                              presence_penalty: float = 0.0,
                              stop: Optional[List[str]] = None,
                              ) -> List[str]:
        """
        Generate chat completion for Llama models using chat template
        """
        batch_prompts = []
        for message in messages:
            template = [{"role": "user", "content": message}]
            prompt = self.tokenizer.apply_chat_template(
                template, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False,
            )
            batch_prompts.append(prompt)   
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop
        )
        outputs = self.llm.generate(batch_prompts, sampling_params)
        gen_results = []
        for output in outputs:
            token_ids = output.outputs[0].token_ids
            text = self.tokenizer.decode(token_ids[:-1], skip_special_tokens=False) # skip EOT token
            print("-"*100)
            print(text)
            print("-"*100)
            gen_results.append(text)
        return gen_results