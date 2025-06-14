import os
import random
import json
import ast
import argparse
import torch
from uuid import uuid4
from datasets import load_dataset
import random
import re
from tqdm import tqdm
import torch
import torchaudio
from llm2fx.common_utils.dasp_utils import parametric_eq, noise_shaped_reverberation

parser = argparse.ArgumentParser(description='Generate audio effect chain parameters from text description')
parser.add_argument('--output_path', type=str, default="./outputs/llm2fx", help='Path to save the output JSON file')
parser.add_argument('--fx_type', type=str, default="reverb", help='Type of dsp', choices=["eq", "reverb"])
parser.add_argument('--use_incontext', type=bool, default=True, help='Use incontext')
parser.add_argument('--inst_type', type=str, default="guitar", help='Instrument type')
parser.add_argument('--timbre_word', type=str, default="church", help='Timbre word')
parser.add_argument('--model_name', type=str, default='mistral_7b',help='Model name', choices=["llama3_1b", "llama3_3b", "llama3_8b", "llama3_70b", "mistral_7b"])
parser.add_argument('--max_new_tokens', type=int, default=2048, help='Max new tokens')
parser.add_argument('--device', type=str, default="cuda", help='Device')
parser.add_argument('--cache_dir', type=str, default="/data3/seungheon/.cache/huggingface/hub", help='Cache directory')
args = parser.parse_args()

use_quantization = False
if args.model_name == "llama3_1b":
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
elif args.model_name == "llama3_3b":
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
elif args.model_name == "llama3_8b":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
elif args.model_name == "llama3_70b":
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    use_quantization = True
elif args.model_name == "mistral_7b":
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
else:
    raise ValueError(f"Invalid model name: {args.model_name}")

# GLOBAL VARIABLES
AUDIO_PATH = os.path.join(os.path.dirname(__file__), "assets")
INCONTEXT_PATH = os.path.join(os.path.dirname(__file__), "incontext")
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompt")
SYSTEM_PROMPT = """
You are an expert audio engineer and music producer specializing in sound design and audio processing.
Your task is to translate descriptive timbre words into specific audio effects parameters that will achieve the desired sound character.
You have deep knowledge of equalizers and understand how they shape timbre. You MUST respond with ONLY a valid JSON object.
"""

def load_llama_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
        )
    else:
        quantization_config = None
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=args.cache_dir, torch_dtype=torch.bfloat16, quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
    return model, tokenizer

@torch.no_grad()
def llm_inference(model, tokenizer, input_text):
    """Generate JSON response from LLM using chat template.
    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        system_prompt: System prompt to guide model behavior
        input_text: User input text/query

    Returns:
        str: Generated response with JSON structure
    """
    # Construct chat messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
    ]

    # Tokenize inputs using chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate response with controlled parameters
    outputs = model.generate(
        inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=1.0,  # Slightly reduced for more focused outputs
        top_p=0.9,      # Slightly reduced but maintains diversity
        top_k=250,       # Reduced to filter out less relevant tokens
        repetition_penalty=1.0,  # Slightly increased to reduce repetition
        do_sample=True,
        num_beams=1,
        early_stopping=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    # Decode only the new tokens
    response = tokenizer.decode(
        outputs[0][inputs.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )
    return response


def apply_eq(audio, sr, prediction, inst_type, eq_word, model_name, output_path, str_uuid):
    audio = audio.unsqueeze(0).to(args.device)
    try:
        eq_params = {k: torch.tensor(v).view(1, -1).to(args.device) for k, v in prediction['eq'].items()}
        eq_audio = parametric_eq(x=audio, sample_rate=sr, **eq_params)
        eq_audio = eq_audio.squeeze(0).detach().cpu()
        os.makedirs(f"{output_path}/audio", exist_ok=True)
        torchaudio.save(f"{output_path}/audio/{str_uuid}.wav", eq_audio, sr, format="wav")
    except Exception as e:
        print(f"Failed to apply eq: {e}")

def apply_reverb(audio, sr, prediction, inst_type, reverb_word, model_name, output_path, str_uuid):
    audio = audio.unsqueeze(0).to(args.device)
    try:
        reverb_params = {k: torch.tensor(v).view(1, -1).to(args.device) for k, v in prediction['reverb'].items()}
        reverb_audio = noise_shaped_reverberation(x=audio, sample_rate=sr, **reverb_params)
        reverb_audio = reverb_audio.squeeze(0).detach().cpu()
        os.makedirs(f"{output_path}/audio", exist_ok=True)
        torchaudio.save(f"{output_path}/audio/{str_uuid}.wav", reverb_audio, sr, format="wav")
    except Exception as e:
        print(f"Failed to apply reverb: {e}")


def get_dsp_function(args):
    with open(f"{INCONTEXT_PATH}/functions/dsp_{args.fx_type}.py", "r") as f:
        dsp_function_code = f.read()
    return dsp_function_code

def get_dsp_feature(args):
    from incontext.audio.dsp_feature import extract_audio_features
    dsp_feature = extract_audio_features(f"{AUDIO_PATH}/{args.inst_type}.wav")
    return dsp_feature

def get_incontext_example(args):
    incontext_db = []
    for fname in os.listdir(f"{INCONTEXT_PATH}/examples"):
        icl_fx_type, icl_inst_type, icl_timbre_word = fname.replace(".json", "").split("_")
        if icl_timbre_word == args.timbre_word:
            continue # DON'T CHEAT!
        if icl_fx_type != args.fx_type:
            continue # Focus on the target fx_type
        json_data = json.load(open(f"{INCONTEXT_PATH}/examples/{fname}", "r"))
        incontext_example = f"QUESTION: please design a {icl_fx_type} audio effects for a {icl_timbre_word} {icl_inst_type} sound.\n ANSWER: {json_data}"
        incontext_db.append(incontext_example)
    sampled_incontext_db = random.sample(incontext_db, 5)
    incontext_example = ""
    for example in sampled_incontext_db:
        incontext_example += f"{example}\n\n"
    return incontext_example

def main():
    with open(f"{PROMPT_PATH}/instruction_{args.fx_type}.txt", "r") as f:
        instruction = f.read()
    model, tokenizer = load_llama_model(model_name)
    model.eval()
    model.to(args.device)
    audio, sr = torchaudio.load(f"{AUDIO_PATH}/{args.inst_type}.wav")
    user_query = f"QUESTION: please design a {args.fx_type} audio effects for a {args.timbre_word} {args.inst_type} sound.\n ANSWER:"
    output_path = f"{args.output_path}/{args.model_name}/{args.fx_type}/{args.inst_type}/{args.timbre_word}"
    os.makedirs(f"{output_path}/json", exist_ok=True)
    input_text = f"{instruction}"
    if args.use_incontext: # fewshot case
        dsp_function_code = get_dsp_function(args)
        dsp_feature = get_dsp_feature(args)
        incontext_example = get_incontext_example(args)
        input_text += f"\n\n # Signal processing function\n\n {dsp_function_code}" # add dsp function
        input_text += f"\n\n # Input audio feature\n\n {dsp_feature}" # add dsp feature
        input_text += f"\n\n # Incontext examples\n\n {incontext_example}" # add incontext examples
    input_text += f"\n\n {user_query}"
    print(input_text)
    str_uuid = str(uuid4())
    response = llm_inference(model, tokenizer, input_text)
    response = response.replace("```json", "").replace("```", "").strip()
    print(response)
    json_data = json.loads(response)
    if args.fx_type == "eq":
        apply_eq(audio, sr, json_data, args.inst_type, args.timbre_word, model_name, output_path, str_uuid)
    elif args.fx_type == "reverb":
        apply_reverb(audio, sr, json_data, args.inst_type, args.timbre_word, model_name, output_path, str_uuid)
    with open(f"{output_path}/json/{str_uuid}.json", "w") as f:
        json.dump(json_data, f, indent=4)

if __name__ == "__main__":
    main()
