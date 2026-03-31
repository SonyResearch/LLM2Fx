import os
import json
import argparse
import random
from datasets import load_dataset
from transformers.utils import get_json_schema
from llm4mp.common_utils.fx_utils import load_fx_config
import ast
from uuid import uuid4
from tqdm import tqdm
from llm4mp.preprocessing.llm_processor.prompt.fx_chat import get_chat_prompt, get_cot_prompt, get_llm_as_a_judge_prompt
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="test", help="Split")
parser.add_argument("--client_type", type=str, default="gemini", help="Client type")
parser.add_argument("--output_dir", type=str, default="", help="Output directory")
parser.add_argument("--api_key", type=str, default="AIzaSyBvjl5ANQTLyjICIUTryo4wdJmQgblhU44", help="API key")
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

VST_INFO_DICT = {}
for key, value in load_fx_config("ndsp")['fx_tools'].items():
    VST_INFO_DICT[key] = get_json_schema(value)

def mp_function(instance):
    output_dir = f"{args.output_dir}/{args.split}/{instance['id']}.json"
    if os.path.exists(output_dir):
        return None
    fx_chain = ast.literal_eval(instance['fx_chain'])
    tool_order = [i['name'] for i in fx_chain]
    tool_numer = len(fx_chain)
    vst_info = [VST_INFO_DICT[tool_name] for tool_name in tool_order]
    chat_prompt = get_chat_prompt(instance['instrument'], instance['genre'], fx_chain, tool_order, tool_numer)
    try:
        chat_response = LLM_CLIENT.chat_completion(chat_prompt, model_name= "gemini-2.5-flash-lite", schema_type="chat")
        chat_response = json.loads(chat_response)
    except Exception as e:
        print(e)
        return None
    cot_prompt = get_cot_prompt(f'{chat_response}', vst_info)
    try:
        cot_response = LLM_CLIENT.chat_completion(cot_prompt, model_name= "gemini-2.5-flash", schema_type="cot")
        cot_response = json.loads(cot_response)
    except Exception as e:
        print(e)
        return None
    try:
        conversation = [{
            "role": "user",
            "content": chat_response[0]['content']
        },{
            "role": "assistant",
            "content": chat_response[1]['content']
        }]
        print(conversation)
        results = {
            "id": instance['id'],
            "fname": instance['fname'],
            "filename": instance['filename'],
            "instrument": instance['instrument'],
            "genre": instance['genre'],
            "num_of_fx": instance['num_of_fx'],
            "tools": instance['fx_chain'],
            "chain_of_thought": cot_response['chain_of_thought'],
            "conversations": conversation,
        }
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        with open(f"{output_dir}", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
    except:
        print(e)
        return None

def main():
    source_json = glob(f"{args.output_dir}/{args.split}/*.json", recursive=True)
    target_dataset = []
    for path in source_json:
        instance = json.load(open(path, "r"))
        output_dir = f"{args.output_dir}/{args.split}/{instance['id']}.json"
        if os.path.exists(output_dir):
            continue
        target_dataset.append(instance)
    print("start with len(target_dataset):", len(target_dataset))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(mp_function, instance) for instance in target_dataset]
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()



