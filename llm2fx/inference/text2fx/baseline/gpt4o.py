import os
import json
import ast
import argparse
from uuid import uuid4
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import torch
import torchaudio
from llm4mp.common_utils.dasp_utils import parametric_eq, noise_shaped_reverberation

parser = argparse.ArgumentParser(description='Generate audio effect chain parameters from text description')
parser.add_argument('--output_path', type=str, default="/data3/seungheon/wasppa/outputs",
                   help='Path to save the output JSON file (default: "outputs")')
parser.add_argument('--fx_type', type=str, default="eq",
                   help='Type of dsp (default: "eq")')
parser.add_argument('--model_name', type=str, default="gpt-4o",
                   help='Model name (default: "gpt-4o")')
parser.add_argument('--device', type=str, default="cuda:1",
                   help='Device (default: "cuda")')
args = parser.parse_args()


from dotenv import load_dotenv
load_dotenv(".dotenv")
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
AUDIO_PATH = "/home/seungheon/llm4mp/llm4mp/inference/assets"
SYSTEM_PROMPT = """
You are an expert audio engineer and music producer specializing in sound design and audio processing.
Your task is to translate descriptive timbre words into specific audio effect parameters that will achieve the desired sound character.
You have deep knowledge of equalizers and understand how they shape timbre.
You MUST respond with ONLY a valid JSON object. Do not include any explanatory text, markdown formatting, or code blocks. Your entire response should parse as valid JSON.
"""


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

def llm_inference(input_text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text}
            ],
            response_format={"type": "json_object"}
        )
    json_response = response.choices[0].message.content
    return json_response

def mp_helper(instance):
    inst_type = instance['inst_type']
    timbre_word = instance['timbre_word']
    input_text = instance['input_text']
    audio, sr = torchaudio.load(f"{AUDIO_PATH}/{inst_type}.wav")
    try:
        str_uuid = str(uuid4())
        output_path = f"{args.output_path}/base/gpt4o/{args.fx_type}/{inst_type}/{timbre_word}"
        json_response = llm_inference(input_text)
        json_data = ast.literal_eval(json_response)
        os.makedirs(f"{output_path}/json", exist_ok=True)
        os.makedirs(f"{output_path}/audio", exist_ok=True)
        with open(f"{output_path}/json/{str_uuid}.json", "w") as f:
            json.dump(json_data, f, indent=4)
        if args.fx_type == "eq":
            apply_eq(audio, sr, json_data, inst_type, timbre_word, args.model_name, output_path, str_uuid)
        elif args.fx_type == "reverb":
            apply_reverb(audio, sr, json_data, inst_type, timbre_word, args.model_name, output_path, str_uuid)
    except Exception as e:
        print(f"Failed to infer: {e}")
        return None

def main():
    eval_db = load_dataset("seungheondoh/socialfx-gen-eval", split=args.fx_type)
    with open(f"prompt/{args.fx_type}/instruction_{args.fx_type}.txt", "r") as f:
        instruction = f.read()
    timbre_words = list(eval_db['input'])
    mp_samples = []
    for timbre_word in timbre_words:
        for inst_type in ["drums", "guitar", "piano"]:
            for _ in range(50):
                user_query = f"QUESTION: please design {args.fx_type} audio effect for {timbre_word} sound for {inst_type}.\n ANSWER:"
                input_text = f"{instruction}\n\n {user_query}"
                mp_samples.append({
                    'inst_type': inst_type,
                    'timbre_word': timbre_word,
                    'input_text': input_text
                })
    print(len(mp_samples)) # eq: 1350
    import multiprocessing as mp
    with mp.Pool(processes=20) as pool:
        pool.map(mp_helper, mp_samples)

if __name__ == "__main__":
    main()
