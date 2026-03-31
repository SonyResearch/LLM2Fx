import pandas as pd
import argparse
import random
from uuid import uuid4
import numpy as np
from llm4mp.common_utils.fx_utils import load_fx_config
from datasets import load_dataset, Dataset
import math
from llm4mp.common_utils.audio_utils.audio_io import get_dry_audio, get_wet_audio
import torch
import json
import os
import ast
import multiprocessing as mp
from auraloss.freq import MultiResolutionSTFTLoss
mrstft_fn = MultiResolutionSTFTLoss(sample_rate=44100)
fx_config = load_fx_config("ndsp")
DATA_DIR = ""
SAVE_DIR = ""
def uniform_sampling(param_range, num_samples=1):
    """
    Uniform sampling within parameter range with given step size.
    
    Args:
        param_range (dict): Parameter range with 'min' and 'max' keys
        num_samples (int): Number of samples to generate
    
    Returns:
        list: Sampled parameter values
    """
    param_step = param_range['step']
    scale = param_range['scale']
    min_val = param_range['min'] + param_step
    max_val = param_range['max']
    # Generate possible values with step size
    num_steps = int((max_val - min_val) / param_step) + 1
    if scale == "linear":
        possible_values = [min_val + i * param_step for i in range(num_steps)]
        possible_values = [val for val in possible_values if val <= max_val]
        sampled_values = np.random.choice(possible_values, size=num_samples, replace=True)
    elif scale == "log":
        possible_values = np.logspace(np.log10(min_val), np.log10(max_val), num_steps)
        possible_values = [val for val in possible_values if val <= max_val]
        sampled_values = np.random.choice(possible_values, size=num_samples, replace=True)
    sampled_values = sampled_values.tolist() if num_samples > 1 else sampled_values[0]
    return float(sampled_values)

def sample_fx_parameters(fx_param_range, num_samples=1, fine_grained=False):
    """
    Sample parameters for a single FX effect.
    
    Args:
        fx_param_range (dict): Parameter ranges for the FX
        sampling_method (str): 'uniform' or 'gaussian'
        num_samples (int): Number of parameter sets to generate
    
    Returns:
        list: List of parameter dictionaries
    """
    sampled_params_list = []
    for _ in range(num_samples):
        sampled_params = {}
        for param_name in fx_param_range.keys():
            if fine_grained:
                param_range = fx_param_range[param_name]['fine_grained']
            else:
                param_range = fx_param_range[param_name]
            sampled_value = uniform_sampling(param_range, num_samples=1)
            if "db" in param_name and int(sampled_value) == 0:
                sampled_value = uniform_sampling(param_range, num_samples=1)
            if "ms" in param_name:
                sampled_params[param_name] = float(int(sampled_value))
            else:
                sampled_params[param_name] = round(float(sampled_value), 3)
        sampled_params_list.append(sampled_params)
    return sampled_params_list if num_samples > 1 else sampled_params_list[0]

def fx_chain_sampling(num_of_fx, fx_list, num_samples, fx_param_ranges, fine_grained=False):
    multi_fx_samples = []
    for _ in range(num_samples):
        fx_pool = fx_list.copy()
        random.shuffle(fx_pool)
        sampled_fx_chain = random.sample(fx_pool, num_of_fx)
        fx_chain = []
        for fx_name in sampled_fx_chain:
            fx_param_range = fx_param_ranges[fx_name]
            samples = sample_fx_parameters(fx_param_range, num_samples=1, fine_grained=fine_grained)
            fx_chain.append({"name": fx_name, "arguments": samples})
        multi_fx_samples.append(fx_chain)
    return multi_fx_samples

def print_param_stats(fx_param_ranges):
    total_fx, total_parmas, total_labels = 0, 0, 0
    for fx_name, param_ranges in fx_param_ranges.items():
        print(f"\nFX: {fx_name}")
        total_fx += 1
        for param_name, param_range in param_ranges.items():
            total_parmas += 1
            min_val = param_range['min']
            max_val = param_range['max']
            step = param_range['step']
            range_span = abs(max_val - min_val)
            num_steps = int(range_span / step) + 1
            total_labels += num_steps
            print(f"  {param_name}: {num_steps} unique possible values")
    print(f"Total FX: {total_fx}, Total Params: {total_parmas}, Total Labels: {total_labels}, Total Params per FX: {total_parmas / total_fx}")


def mp_helper(data_sample):
    _id = data_sample['id']
    split = data_sample['split']
    num_of_fx = data_sample['num_of_fx']
    fx_chain = ast.literal_eval(data_sample['fx_chain'])
    audio_path = f"{DATA_DIR}/{data_sample['fname']}/{data_sample['fname']}_RAW/{data_sample['filename']}.flac"
    dry_audio = get_dry_audio(audio_path)[0]
    wet_audio = get_wet_audio(dry_audio, fx_chain, fx_config)            
    wet_audios = torch.from_numpy(wet_audio.astype(np.float32)).unsqueeze(0).contiguous()
    dry_audios = torch.from_numpy(dry_audio.astype(np.float32)).unsqueeze(0).contiguous()
    mrstft_score = mrstft_fn(wet_audios, dry_audios)
    score = mrstft_score.detach().cpu().item()
    fx_list_name = [i['name'] for i in fx_chain]
    # pass too simple manipulation 
    if float(score) < 0.3 and num_of_fx < 5:
        print(f"Fx: {fx_list_name} MRSTFT score: {score}, let's pass")
        return None
    if float(score) < 2 and num_of_fx >= 5:
        print(f"Fx: {fx_list_name} MRSTFT score: {score}, let's pass")
        return None
    os.makedirs(f"{SAVE_DIR}/{split}", exist_ok=True)
    with open(f"{SAVE_DIR}/{split}/{_id}.json", "w") as f:
        json.dump(data_sample, f, indent=2)

def sampling_trainset():
    db_pool = load_dataset("seungheondoh/medleydb_raw", split="pool")
    db_test = load_dataset("seungheondoh/medleydb_raw", split="test")
    test_pool = set(db_test['filename'])
    db_dict = {i["filename"]: i for i in db_pool if i["filename"] not in test_pool}
    df = pd.DataFrame(list(db_dict.values()))
    inst2raw = df.groupby("instrument")["filename"].apply(list).to_dict()
    inst_pool = list(inst2raw.keys())
    freq = [math.log(len(inst2raw[inst]) + 1) for inst in inst_pool]
    all_samples = []
    for num_of_fx in range(1, 10):
        print(f"Sampling {num_of_fx} FX, {len(fx_config['fx_list'])} FXs")
        target_sample = 10000 // num_of_fx
        print(f"Sampling {target_sample} samples for {num_of_fx} FX")
        corase_sample = fx_chain_sampling(num_of_fx, fx_config['fx_list'], target_sample, fx_config['fx_param_ranges'], fine_grained=False)
        # fine_sample = fx_chain_sampling(num_of_fx, fx_config['fx_list'], target_sample, fx_config['fx_param_ranges'], fine_grained=True)
        total_sample = corase_sample  #+ fine_sample
        for fx_sample in total_sample:
            inst = random.choices(inst_pool, weights=freq, k=1)[0]
            raw = random.choice(inst2raw[inst])
            raw_data = db_dict[raw]
            _id = str(uuid4())
            all_samples.append({
                "id": _id,
                "instrument": raw_data['instrument'],
                "genre": raw_data['genre'],
                "fname": raw_data['fname'],
                "filename": raw_data['filename'],
                "fx_chain": f"{fx_sample}",
                "num_of_fx": num_of_fx,
                "split": "train"
            })
    print(f"Sampling {len(all_samples)} samples")
    random.shuffle(all_samples)
    with mp.Pool(processes=mp.cpu_count()-5) as pool:
        pool.map(mp_helper, all_samples)

def sampling_testset():
    db_test = load_dataset("seungheondoh/medleydb_raw", split="test")
    db_dict = {i["filename"]: i for i in db_test}
    df = pd.DataFrame(list(db_dict.values()))
    inst2raw = df.groupby("instrument")["filename"].apply(list).to_dict()
    inst_pool = list(inst2raw.keys())
    freq = [math.log(len(inst2raw[inst]) + 1) for inst in inst_pool]
    all_samples = []
    for num_of_fx in range(1, 10):
        gen_num = 100 // num_of_fx
        print(f"Sampling {gen_num} samples for {num_of_fx} FX")
        corase_sample = fx_chain_sampling(num_of_fx, fx_config['fx_list'], gen_num, fx_config['fx_param_ranges'], fine_grained=False)
        for fx_sample in corase_sample:
            inst = random.choices(inst_pool, weights=freq, k=1)[0]
            raw = random.choice(inst2raw[inst])
            raw_data = db_dict[raw]
            _id = str(uuid4())
            data_sample = {
                "id": _id,
                "instrument": raw_data['instrument'],
                "genre": raw_data['genre'],
                "fname": raw_data['fname'],
                "filename": raw_data['filename'],
                "fx_chain": f"{fx_sample}",
                "num_of_fx": num_of_fx,
                "split": "test"
            }
            all_samples.append(data_sample)
    random.shuffle(all_samples)
    with mp.Pool(processes=20) as pool:
        pool.map(mp_helper, all_samples)
    
def check_distribution():
    from glob import glob
    for num_of_fx in range(1, 10):
        fx_chain_dir = f"{SAVE_DIR}/fx_chain/sampling/{num_of_fx}"
        all_json_files = glob(f"{fx_chain_dir}/*.json", recursive=True)
        print(len(all_json_files))
        results = []
        for json_file in all_json_files:
            fname = os.path.basename(json_file).replace(".json", "")
            data_sample = json.load(open(json_file, "r"))
            fx_chain = ast.literal_eval(data_sample['fx_chain'])
            fx_list_name = [i['name'] for i in fx_chain]
            results.append({
                "fname": fname,
                "instrument": data_sample['instrument'],
                "genre": data_sample['genre'],
                "fx_list_name": fx_list_name
            })
        df = pd.DataFrame(results)
        print(len(df))
        df['fx_list_name_tuple'] = df['fx_list_name'].apply(lambda x: tuple(x))
        group_cols = ['instrument', 'genre', 'fx_list_name_tuple']
        balanced_samples = []
        group = df.groupby(group_cols)
        for name, group_df in group:
            balanced_samples.append(group_df.sample(n=1, random_state=42))
        balanced_df = pd.concat(balanced_samples, ignore_index=True)
        balanced_df['fx_list_name_tuple'].value_counts()
        balanced_df = balanced_df.sample(100)
        os.makedirs(f"{SAVE_DIR}/fx_chain/test", exist_ok=True)
        for name in balanced_df['fname']:
            os.system(f"cp {SAVE_DIR}/fx_chain/sampling/{num_of_fx}/{name}.json {SAVE_DIR}/fx_chain/test/{name}.json")        

def main():
    print_param_stats(fx_config['fx_param_ranges'])
    sampling_trainset()
    # sampling_testset()
    # check_distribution()

if __name__ == "__main__":
    main()
