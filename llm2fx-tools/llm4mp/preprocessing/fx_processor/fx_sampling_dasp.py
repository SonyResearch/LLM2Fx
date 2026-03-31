import os
import json
import torchaudio
import pandas as pd
import argparse
import torch
import random
import multiprocessing
from collections import Counter
from uuid import uuid4
import numpy as np
from llm4mp.common_utils.dasp_utils import dasp_fx_list, dasp_fx_module_dict, dasp_fx_param_keys, dasp_fx_num_params
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--medleydb_path",default="",help="path to inputs",)
parser.add_argument("--output_dir",default="",help="path to outputs",)
parser.add_argument("--dataset_type", default="medleydb")
parser.add_argument("--target_sr",default=44100,type=int)
parser.add_argument("--sample_size",default=50000,type=int)
parser.add_argument("--batch_size",default=1,type=int)
parser.add_argument("--device",default="cuda:3",type=str)
args = parser.parse_args()

REMOVE_INST = ['main_system', 'fx_processed_sound', 'unlabeled', 'scratches']# 4
SEEN_INST = [
    'drum_set', 'bass_drum', 'kick_drum', 'snare_drum', 'toms', 'cymbal', 'auxiliary_percussion',
    'bongo', 'tabla', 'tambourine', 'vibraphone', 'glockenspiel', 'darbuka', 'doumbek',
    'violin', 'viola', 'violin_section', 'viola_section', 'string_section','double_bass', 'harp',
    'electric_bass', 'acoustic_guitar', 'clean_electric_guitar', 'distorted_electric_guitar',
    'lap_steel_guitar', 'banjo', 'mandolin',
    'piano', 'tack_piano', 'electronic_organ', 'accordion','bass_clarinet', 'piccolo', 'dizi', 'baritone_saxophone',
    'flute', 'flute_section', 'bamboo_flute', 'clarinet', 'clarinet_section',
    'oboe', 'alto_saxophone', 'tenor_saxophone', 'soprano_saxophone',
    'trumpet', 'trumpet_section', 'french_horn', 'french_horn_section', 'brass_section',
    'synthesizer', 'sampler', 'drum_machine',
    'female_singer', 'vocalists',
    'guzheng', 'yangqin', 'erhu', 'zhongruan', 'liuqin'
] # 60

UNSEEN_INST = [
    'cello', 'cello_section',
    'tuba', 'trombone', 'horn_section',
    'bassoon',
    'timpani',
    'electric_piano',
    'male_singer', 'male_singer_vocalists', 'male_singer,_male_speaker',
    'male_singer_distorted_electric_guitar', 'male_screamer', 'male_rapper', 'crowd',
    'gu', 'oud',
    'harmonica',
] # 18

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_list):
        self.metadata_list = metadata_list

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, idx):
        metadata = self.metadata_list[idx]
        dry_audio_path = f"{args.output_dir}/{args.dataset_type}/audio/{metadata['audio_path']}"
        audio, sr = torchaudio.load(dry_audio_path)
        return {
            'audio': audio,
            'sr': sr,
            'metadata': metadata
        }
    def collate_fn(self, batch):
        filtered_batch = []
        for item in batch:
            if item['audio'].size(-1) == 441000:
                filtered_batch.append(item)
        return filtered_batch

def process_single_instance(instance, save_audio_file=False):
    audios = instance['audio'].unsqueeze(0).to(args.device)
    srs = instance['sr']
    metadata = instance['metadata']
    processed_audios = audios # init processed_audios
    fx_pool = dasp_fx_list.copy()
    random.shuffle(fx_pool)
    if len(fx_pool) > 1:
        k = random.choice(range(1, len(fx_pool)+1)) # 1-8 => 5
        sampled_fx_chain = random.sample(fx_pool, k) # 5
    else:
        sampled_fx_chain = fx_pool
    fx_chain_dict = []
    for fx_name in sampled_fx_chain:
        fx_module = dasp_fx_module_dict[fx_name]
        num_params = fx_module.num_params
        rand_params = torch.rand(1, num_params, device=args.device)
        processed_audios, denorm_param_dicts = fx_module.process_normalized(processed_audios, rand_params)
        fx_keys = list(denorm_param_dicts.keys())
        denorm_param = torch.stack([denorm_param_dicts[k] for k in fx_keys], dim=1)
        denorm_param = denorm_param.flatten().detach().cpu().tolist()
        normalized_params = rand_params.flatten().detach().cpu().tolist()
        params_dict = {k:v for k,v in zip(fx_keys, denorm_param)}
        denorm_param_dicts = {k:v for k,v in zip(fx_keys, normalized_params)}
        fx_chain_dict.append({
                "fx_name": fx_name,
                "params": params_dict,
                "nomalized_params": denorm_param_dicts,
            })
    wet_id = str(uuid4())
    inst = metadata['audio_path'].split("/")[0]
    if save_audio_file:
        wet_audio_path = f"{args.output_dir}/{args.dataset_type}/dasp/wet_audio/{wet_id}.flac"
        os.makedirs(f"{args.output_dir}/{args.dataset_type}/dasp/wet_audio", exist_ok=True)
        torchaudio.save(wet_audio_path, processed_audios.squeeze(0).detach().cpu(), srs, format="flac")
    result = {
        "wet_id": wet_id,
        "dry_id": metadata['id'],
        "instrument": inst,
        "fx_chain": fx_chain_dict,
        "fx_pool": dasp_fx_list,
        "dry_audio_path": metadata['audio_path'],
        "wet_audio_path": f"dasp/wet_audio/{wet_id}.flac",
        "seen/unseen": "seen" if inst in SEEN_INST else "unseen"
    }
    return result

def wet_audio_sampling(df, inst_type, sample_size, save_audio_file=False):
    df = df.set_index("instrument").loc[inst_type].reset_index()
    batch_size = args.batch_size
    inst_groups = df.groupby('instrument')
    samples_per_inst = sample_size // len(inst_groups) + batch_size
    multiply_samples = []
    for inst, group in inst_groups:
        inst_samples = group.sample(n=samples_per_inst, replace=True).to_dict(orient="records")
        multiply_samples.extend(inst_samples)
    random.shuffle(multiply_samples)
    dataset = AudioDataset(multiply_samples)
    db = []
    for item in tqdm(dataset):
        db.append(process_single_instance(item, save_audio_file=save_audio_file))
    return db[:sample_size]

def main():
    df = pd.read_csv(f"{args.output_dir}/medleydb/metadata/metadata.csv")
    train_sample_size = args.sample_size
    seen_test_sample_size = 500
    unseen_test_sample_size = 500

    seen_testing_db = wet_audio_sampling(df = df, inst_type = SEEN_INST, sample_size = seen_test_sample_size, save_audio_file=True)
    unseen_testing_db = wet_audio_sampling(df = df, inst_type = UNSEEN_INST, sample_size = unseen_test_sample_size, save_audio_file=True)
    seen_testing_df = pd.DataFrame(seen_testing_db)
    unseen_testing_df = pd.DataFrame(unseen_testing_db)
    test_db = seen_testing_db + unseen_testing_db
    random.shuffle(test_db)
    test_df = pd.DataFrame(test_db)
    os.makedirs(f"{args.output_dir}/{args.dataset_type}/dasp/metadata", exist_ok=True)
    test_df.to_csv(f"{args.output_dir}/{args.dataset_type}/dasp/metadata/test.csv", index=False)
    seen_testing_df.to_csv(f"{args.output_dir}/{args.dataset_type}/dasp/metadata/seen_test.csv", index=False)
    unseen_testing_df.to_csv(f"{args.output_dir}/{args.dataset_type}/dasp/metadata/unseen_test.csv", index=False)

    train_db = wet_audio_sampling(df = df, inst_type = SEEN_INST, sample_size = train_sample_size, save_audio_file=True)
    training_df = pd.DataFrame(train_db)
    training_df.to_csv(f"{args.output_dir}/{args.dataset_type}/dasp/metadata/train.csv", index=False)

    from datasets import Dataset, DatasetDict
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_db),
        "test": Dataset.from_list(test_db),
    })
    dataset_dict.push_to_hub("seungheon/llm2fx2-dasp-same-audio", private=True)


if __name__ == "__main__":
    main()
