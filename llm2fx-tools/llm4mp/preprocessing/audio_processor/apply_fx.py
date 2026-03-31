
import os
import random
import ast
import numpy as np
import librosa
from datasets import load_dataset, concatenate_datasets
from llm4mp.common_utils.fx_utils import load_fx_config
from llm4mp.common_utils.audio_utils.audio_io import save_audio
from tqdm import tqdm
import uuid

fx_config = load_fx_config(fx_config_type="ndsp")
audio_dir = ""
save_dir = ""

def get_dry_audio(fname, filename, audio_dir):
    raw_path = f"{audio_dir}/{fname}/{fname}_RAW/{filename}.flac"
    dry_audio, sr = librosa.load(raw_path, sr=None, mono=False)
    duration = dry_audio.shape[1] / sr
    target_duration = 10
    num_chunks = int(duration // target_duration)
    chunk_samples = int(target_duration * sr)
    if dry_audio.shape[1] < chunk_samples:
        print(f"Padding {raw_path}")
        pad_audio = np.zeros((dry_audio.shape[0], chunk_samples))
        pad_audio[:, :dry_audio.shape[1]] = dry_audio
        dry_audio = pad_audio
        num_chunks = 1
    chunk_audio = []
    for idx, i in enumerate(range(num_chunks)):
        chunk_audio.append(dry_audio[:, i * chunk_samples:(i + 1) * chunk_samples])
        if idx == 2:
            break
    return chunk_audio, sr

def get_wet_audio(dry_audio, fx_chain, fx_config):
    wet_audio = dry_audio.copy()
    for fx in fx_chain:
        fx_name = fx['name']
        fx_params = fx['arguments']
        fx_function = fx_config['fx_function_dict'][fx_name]
        wet_audio = fx_function(wet_audio, **fx_params)
    return wet_audio

def mp_function(raw_instance):
    print(f"start processing {raw_instance['filename']}")
    data_id = raw_instance['id']
    fx_audio_path = f"{save_dir}/fx_audio/RAW_instwise_remfx_eq_imag_loud/{raw_instance['fname']}/{raw_instance['filename']}/{data_id}"
    os.makedirs(fx_audio_path, exist_ok=True)
    fx_chain = ast.literal_eval(raw_instance['tools'])
    chunk_audio, sr = get_dry_audio(raw_instance['fname'], raw_instance['filename'], audio_dir)
    for idx, dry_audio in enumerate(chunk_audio):
        wet_audio = get_wet_audio(dry_audio, fx_chain, fx_config)
        save_audio(wet_audio, sr, f"{fx_audio_path}/wet_{idx}.flac", audio_format="flac")
        save_audio(dry_audio, sr, f"{fx_audio_path}/dry_{idx}.flac", audio_format="flac")
        if idx == 3:
            break

def main():
    # db_src = load_dataset("seungheondoh/medleydb_fx_chat")
    db_src = load_dataset("seungheondoh/lp-fx") # db_src['train'],
    db = concatenate_datasets([db_src['test']])
    from multiprocessing import Pool, cpu_count
    num_workers = cpu_count() - 5
    print(f"Using {num_workers} workers")
    with Pool(num_workers) as pool:
        pool.map(mp_function, db)

if __name__ == "__main__":
    main()