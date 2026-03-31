import os
import json
import argparse
from datasets import load_dataset
from transformers.utils import get_json_schema
from llm4mp.common_utils.fx_utils import load_fx_config
from glob import glob
from llm4mp.preprocessing.llm_processor.prompt.fx_chat import get_chat_prompt, get_cot_prompt, get_llm_as_a_judge_prompt

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="test", help="Split")
parser.add_argument("--client_type", type=str, default="gemini", help="Client type")
parser.add_argument("--chat_dir", type=str, default="", help="Output directory")
parser.add_argument("--output_dir", type=str, default="", help="Output directory")
parser.add_argument("--api_key", type=str, default="", help="API key")
args = parser.parse_args()

VST_INFO_DICT = {}
for key, value in load_fx_config("ndsp")['fx_tools'].items():
    VST_INFO_DICT[key] = get_json_schema(value)

if args.client_type == "vllm":
    model_name = "Qwen/Qwen3-32B-FP8"
    cache_dir = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    from llm4mp.preprocessing.llm_processor.client.vllm_client import VLLMClient
    LLM_CLIENT = VLLMClient(model_name=model_name, cache_dir=cache_dir)
else:
    from llm4mp.preprocessing.llm_processor.client.gemini_client import GeminiClient
    LLM_CLIENT = GeminiClient(api_key=args.api_key) 

def mp_function(json_file):
    chat_data = json.load(open(json_file, 'r'))
    if os.path.exists(f"{args.output_dir}/{args.split}/{chat_data['id']}.json"):
        return None
    conversation = [{
        "role": "user",
        "content": chat_data['conversation'][0]['content']
    },{
        "role": "assistant",
        "thought": chat_data['thought'],
        "content": chat_data['conversation'][1]['content']
    }]
    evaluation_prompt = get_llm_as_a_judge_prompt(f"{conversation}", f"{chat_data['fx_chain']}")
    try:
        evaluation_response = LLM_CLIENT.chat_completion(evaluation_prompt, model_name= "gemini-2.5-pro", schema_type="evaluation")
        evaluation_response = json.loads(evaluation_response)
        os.makedirs(f"{args.output_dir}/{args.split}", exist_ok=True)
        with open(f"{args.output_dir}/{args.split}/{chat_data['id']}.json", "w") as f:
            json.dump(evaluation_response, f, indent=4)
    except Exception as e:
        return None

def main():
    chat_dir = f"{args.chat_dir}/{args.split}"
    chat_dataset = glob(f"{chat_dir}/**/*.json", recursive=True)
    target_dataset = []
    for jsonfile in chat_dataset:
        _id = os.path.basename(jsonfile)
        if os.path.exists(f"{args.output_dir}/{args.split}/{_id}"):
            continue
        target_dataset.append(jsonfile)
    print("start with len(target_dataset):", len(target_dataset))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(mp_function, instance) for instance in target_dataset]
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()