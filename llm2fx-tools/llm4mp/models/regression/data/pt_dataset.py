import os
import torch
import ast
import pandas as pd
import torchaudio
import librosa
import numpy as np
import random
from torch.utils.data import Dataset
from llm4mp.common_utils.fx_utils import load_fx_config
from llm4mp.common_utils.fx_utils import normalize_param

class AudioFXDataset(Dataset):
    def __init__(self, db, audio_dir, fx_config_type="ndsp", target_duration=10):
        self.metadata = db
        self.audio_dir = audio_dir
        self.target_duration = target_duration
        self.fx_config_type = fx_config_type
        self.fx_config = load_fx_config(fx_config_type)
        self.target_placeholder = {}
        self.mask_placeholder = {}
        for fx_name, fx_param_keys in self.fx_config['fx_param_keys'].items():
            self.target_placeholder[f"{fx_name}_target"] = torch.tensor([-1] * len(fx_param_keys), dtype=torch.float32)
            self.mask_placeholder[f"{fx_name}_mask"] = torch.tensor([0], dtype=torch.float32)
        
    def __len__(self):
        return len(self.metadata)

    def get_fx_regression_label(self, fx_chain):
        target = self.target_placeholder.copy()
        mask = self.mask_placeholder.copy()
        for fx in fx_chain:
            fx_name = fx['name']
            fx_params = fx['arguments']
            normalized_params = []
            for fx_param_key in self.fx_config['fx_param_keys'][fx_name]:
                original_param = fx_params[fx_param_key]
                min_val = self.fx_config['fx_param_ranges'][fx_name][fx_param_key]['min']
                max_val = self.fx_config['fx_param_ranges'][fx_name][fx_param_key]['max']
                normalized_param = normalize_param(original_param, min_val, max_val)
                normalized_params.append(normalized_param)
            target[f"{fx_name}_target"] = torch.tensor(normalized_params, dtype=torch.float32)
            mask[f"{fx_name}_mask"] = torch.tensor([1], dtype=torch.float32)
        return target, mask
    
    def get_fx_classification_label(self, fx_chain):
        fx_activations = []
        fx_labels = [0] * len(self.fx_config["fx_list"])
        current_fx_chain = [i["name"] for i in fx_chain]
        for fx_name in self.fx_config["fx_list"]:
            if fx_name in current_fx_chain:
                fx_idx = current_fx_chain.index(fx_name)
                fx_activations.append(fx_name)
                fx_labels[fx_idx] = 1
        fx_labels = torch.tensor(fx_labels, dtype=torch.float32)
        fx_str = ",".join(fx_activations)   
        return fx_str, fx_labels

    def get_dry_wet_audio_pair(self, row):
        _id = row['id']
        filename = row['filename']
        fname = row['fname']
        random_chunk_idx = random.choice(range(3))
        dry_path = f"{self.audio_dir}/{fname}/{filename}/{_id}/dry_{random_chunk_idx}.flac"
        wet_path = f"{self.audio_dir}/{fname}/{filename}/{_id}/wet_{random_chunk_idx}.flac"
        if not os.path.exists(dry_path) or not os.path.exists(wet_path):
            dry_path = f"{self.audio_dir}/{fname}/{filename}/{_id}/dry_0.flac"
            wet_path = f"{self.audio_dir}/{fname}/{filename}/{_id}/wet_0.flac"
        audio_dry, _ = librosa.load(dry_path, sr=None, mono=False)
        audio_wet, _ = librosa.load(wet_path, sr=None, mono=False)
        return audio_dry, audio_wet

    def __getitem__(self, idx):
        index = random.randint(0, len(self.metadata) - 1)
        row = self.metadata[index]
        fx_chain = ast.literal_eval(row['fx_chain'])
        audio_dry, audio_wet = self.get_dry_wet_audio_pair(row)
        fx_str, fx_labels = self.get_fx_classification_label(fx_chain)
        target, mask = self.get_fx_regression_label(fx_chain)
        batch = {
            "audio_dry": torch.from_numpy(audio_dry),
            "audio_wet": torch.from_numpy(audio_wet),
            "fx_str": fx_str,
            "fx_labels": fx_labels,
            **target,
            **mask,
        }
        return batch


# from tqdm import tqdm
# from datasets import load_dataset
# from torch.utils.data import DataLoader

# db = load_dataset("seungheondoh/medleydb_ndsp_fx", split="train")

# dataset = AudioFXDataset(db, audio_dir, fx_config_type="ndsp")
# dataloader = DataLoader(dataset, batch_size=1024, num_workers=32, shuffle=True)
# for i in tqdm(dataloader):
#     try:
#         pass
#     except:
#         print(i)