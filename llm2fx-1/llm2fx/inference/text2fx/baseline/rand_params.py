import os
import dasp_pytorch
import torchaudio
from llm4mp.common_utils.dasp_utils.augment import Random_FX_Augmentation
import argparse
from uuid import uuid4
from tqdm import tqdm
import json
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:1")
args = parser.parse_args()


INST_TYPES = ['drums', 'guitar', 'piano']
ASSETS_PATH = "/home/seungheon/llm4mp/llm4mp/inference/assets"
SAVE_PATH = "/data3/seungheon/wasppa/outputs/lowerbound/random_params"
def main():
    for fx_type in ['eq', 'reverb']:
        augment_class = Random_FX_Augmentation(sample_rate=44100, tgt_fx_names=[fx_type])
        for inst_type in INST_TYPES:
            audio, sr = torchaudio.load(f"{ASSETS_PATH}/{inst_type}.wav")
            for _ in tqdm(range(50)):
                uuid_str = str(uuid4())
                processed_audio, denorm_param_dict = augment_class.forward(audio.unsqueeze(0).to(args.device))
                processed_audio = processed_audio.squeeze(0).detach().cpu()
                os.makedirs(f"{SAVE_PATH}/{fx_type}/{inst_type}/audio", exist_ok=True)
                os.makedirs(f"{SAVE_PATH}/{fx_type}/{inst_type}/json", exist_ok=True)
                torchaudio.save(f"{SAVE_PATH}/{fx_type}/{inst_type}/audio/{uuid_str}.wav", processed_audio, sr, format="wav")
                json.dump(denorm_param_dict, open(f"{SAVE_PATH}/{fx_type}/{inst_type}/json/{uuid_str}.json", "w"))

if __name__ == "__main__":
    main()
