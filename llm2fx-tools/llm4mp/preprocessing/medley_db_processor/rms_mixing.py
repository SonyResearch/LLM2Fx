import os
import json
import scipy
import librosa
import torch
import soundfile as sf
import torchaudio
import numpy as np
import pandas as pd
import argparse
import multiprocessing
from itertools import combinations
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import uuid
from llm4mp.preprocessing.utils.utils_audio import torch_resample, save_audio, lufs_normalize
# from llm4mp.preprocessing.utils.utils_tags import MEDELYDB_TAG_INFO

parser = argparse.ArgumentParser()
parser.add_argument("--medleydb_path",default="",help="path to inputs",)
parser.add_argument("--output_dir",default="",help="path to outputs",)
parser.add_argument("--dataset_type", default="medleydb")
parser.add_argument("--target_sr",default=44100,type=int)
parser.add_argument("--activation_threshold",default=256,type=int)
parser.add_argument("--target_duration",default=10,type=int)
parser.add_argument("--interval",default=1,type=int)
parser.add_argument("--min_inst",default=1,type=int)
args = parser.parse_args()

def full_song_process(args, metadata):
    _id = metadata['fname']
    genre = metadata['genre'].lower()
    os.makedirs(f"{args.output_dir}/{args.dataset_type}/audio_{args.target_sr}/{full_song_id}", exist_ok=True)
    os.system(f"cp {args.medleydb_path}/Audio/{_id}/{_id}_MIX.wav {args.output_dir}/{args.dataset_type}/audio_{args.target_sr}/{full_song_id}/{full_song_id}.wav")
    full_song_id = str(uuid.uuid4())
    id_to_inst = get_id_to_inst(metadata)
    all_inst = list(set(id_to_inst.values()))
    save_metadata = [
        {
            "id": full_song_id,
            "type": "full_track",
            "medleydb_id": _id,
            "genre": [genre],
            "instrument": all_inst,
            "audio_path": f"{full_song_id}/{full_song_id}.wav",
        }
    ]
    return save_metadata

def get_id_to_inst(metadata):
    stem_dir = metadata["stem_dir"]
    fname = metadata['fname']
    id_to_inst = {}
    for stem_id, stem_item in metadata['stems'].items():
        instrument = stem_item['instrument'].lower()
        id_to_inst[stem_id] = instrument
    return id_to_inst

def metadata_to_df(args, metadata):
    stem_dir = metadata["stem_dir"]
    fname = metadata['fname']
    instruments_audio = []
    for stem_id, stem_item in metadata['stems'].items():
        if not isinstance(stem_item['instrument'], str):
            continue
        stem = stem_item['instrument'].lower()
        stem = stem.replace(" ", "_").replace("/", "_")
        instrument_path = stem_item['filename']
        s_path = f"{args.medleydb_path}/Audio/{fname}/{stem_dir}/{instrument_path}"
        y, sr = sf.read(s_path)
        if sr != args.target_sr:
            raise ValueError(f"sr is not {args.target_sr}")
        loudness_normalized_audio = lufs_normalize(y, sr, peak=-1.0, lufs=-12.0)
        loudness_normalized_audio = np.clip(loudness_normalized_audio, -1.0, 1.0)
        instruments_audio.append({
            "stem_id": stem_id,
            "stem": stem,
            "audio": y.T,
            "sr": sr,
            "audio_length": len(y),
        })
    return pd.DataFrame(instruments_audio)

def get_source_energy(df_source, audio_type="audio"):
    audios = np.stack(df_source[audio_type]).mean(axis=1) # get mono
    duration = int(audios.shape[-1] // args.target_sr)
    interval_sample = int(args.interval * args.target_sr)
    source_energy = []
    for idx in range(0, duration - args.interval):
        start = int(idx * args.target_sr)
        end = start + interval_sample
        audio_frame = audios[:, start: end]
        rms_values = np.sum(np.abs(audio_frame), axis=-1)
        rms_info = {source:rms for source, rms in zip(df_source.index, rms_values)}
        rms_info.update({"onset": idx})
        source_energy.append(rms_info)
    return pd.DataFrame(source_energy).set_index("onset")

def trim_and_mix(sources):
    min_len = min(s.shape[-1] for s in sources)
    audio = np.stack([s[..., :min_len] for s in sources]).sum(0)
    return audio

def audio_mixing(df, stems, audio_type="audio", onset=None, offset=None):
    target_source = df.loc[stems][audio_type]
    min_length = min(audio.shape[-1] for audio in target_source)
    stack_audio = np.stack([audio[:,:min_length] for audio in target_source])
    if onset is not None and offset is not None:
        stack_audio = stack_audio[:,:,onset * args.target_sr : offset * args.target_sr]
    linear_mix = trim_and_mix(stack_audio) # linear sum
    return linear_mix

def crop_remove_silence(metadata):
    df_source = metadata_to_df(args, metadata)
    df_stem = df_source.groupby("stem").agg({"audio": "sum"})
    all_stem = list(df_stem.index)
    df_energy = get_source_energy(df_stem, audio_type="audio")
    df_activation = df_energy > args.activation_threshold
    processed_audio_metadata = {}
    for time_bin in range(0, len(df_activation), args.target_duration):
        onset = time_bin
        offset = (time_bin + args.target_duration)
        if int(offset - onset) < args.target_duration:
            continue
        df_activation_bin = df_activation.iloc[onset:offset]
        df_activation_bin_sum = df_activation_bin.sum(axis=0)
        dominant_stem = df_activation_bin_sum[df_activation_bin_sum >= 9].index
        if len(dominant_stem) == 0:
            continue
        for stem in dominant_stem:
            single_track_id = str(uuid.uuid4())
            single_track = audio_mixing(df_stem, [stem], onset=onset, offset=offset, audio_type="audio")
            processed_audio_metadata[single_track_id] = {
                "audio": single_track,
                "type": f"{stem}",
            }
    return processed_audio_metadata

def mp_helper(metadata):
    _id = metadata['fname']
    genre = metadata['genre'].lower()
    processed_audio_metadata = crop_remove_silence(metadata)
    if processed_audio_metadata is None:
        return None
    save_metadata = []
    for stem_id, audio_metadata in list(processed_audio_metadata.items()):
        audio = audio_metadata['audio']
        _type = audio_metadata['type']
        os.makedirs(f"{args.output_dir}/{args.dataset_type}/audio/{_type}", exist_ok=True)
        save_audio(f"{args.output_dir}/{args.dataset_type}/audio/{_type}/{stem_id}.wav", audio, args.target_sr)
        save_metadata.append({
            "id": stem_id,
            "type": _type,
            "medleydb_id": _id,
            "license": "cc-by-nc-sa",
            "genre": [genre],
            "instrument": _type,
            "audio_path": f"{_type}/{stem_id}.wav",
        })
    return save_metadata

def main():
    metadatas = []
    for idx, fname in enumerate(tqdm(os.listdir(f"{args.medleydb_path}/Audio"))):
        try:
            metadata = OmegaConf.load(f"{args.medleydb_path}/Audio/{fname}/{fname}_METADATA.yaml")
            metadata.fname = fname
            if metadata['has_bleed'] == "yes":
                continue
            metadatas.append(metadata)
        except:
            print(f"Error loading metadata for {fname}")
    worker_num = multiprocessing.cpu_count() - 2
    print(len(metadatas))
    with multiprocessing.Pool(worker_num) as pool:
        results = pool.map(mp_helper, metadatas)
    db = []
    for stem_metadatas in results:
        if stem_metadatas:
            for stem_metadata in stem_metadatas:
                db.append(stem_metadata)
    os.makedirs(f"{args.output_dir}/{args.dataset_type}/metadata", exist_ok=True)
    pd.DataFrame(db).to_csv(f"{args.output_dir}/{args.dataset_type}/metadata/metadata.csv", index=False)
    print(f"Saved {len(db)} metadata to {args.output_dir}/{args.dataset_type}/metadata/metadata.csv")

if __name__ == "__main__":
    main()
