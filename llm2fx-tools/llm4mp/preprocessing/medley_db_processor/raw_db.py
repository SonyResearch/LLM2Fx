import os
from omegaconf import OmegaConf
from glob import glob
import argparse
from tqdm import tqdm
from datasets import Dataset, load_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa

parser = argparse.ArgumentParser()
parser.add_argument("--split_path", type=str, default='', help="Path to the target directory containing audio files.")
parser.add_argument("--output_dir", type=str, default='', help="Path to the output directory where trimmed files will be saved.")
parser.add_argument("--dataset_type", type=str, default='medleydb', help="Dataset type (default: medleydb).")
args = parser.parse_args()

STOP_INST = ['main_system', 'fx_processed_sound', "fx/processed sound", 'unlabeled', 'scratches']# 4
STOP_RAWS = [
    "AClassicEducation_NightOwl_RAW_02_09.wav",
    "AClassicEducation_NightOwl_RAW_02_10.wav",
    "AClassicEducation_NightOwl_RAW_02_13.wav",
    "AClassicEducation_NightOwl_RAW_13_01.wav",
    "AClassicEducation_NightOwl_RAW_13_02.wav",
    "CelestialShore_DieForUs_RAW_01_01.wav",
    "CelestialShore_DieForUs_RAW_01_02.wav",
    "CelestialShore_DieForUs_RAW_02_01.wav",
    "CelestialShore_DieForUs_RAW_02_02.wav",
    "Creepoid_OldTree_RAW_02_01.wav",
    "MusicDelta_Country2_RAW_04_01.wav",
    "MusicDelta_Country2_RAW_03_01.wav",
    "MusicDelta_Pachelbel_RAW_04_01.wav"
]
norm_path = "RAW_instwise_remfx_eq_imag_loud"
raw_dir = f""

def process_raw_db():
    yaml_files = glob(os.path.join(args.split_path, '**', f"*.yaml"), recursive=True)
    raw_db = []
    for yaml_file in tqdm(yaml_files):
        fname = os.path.basename(yaml_file).replace("_METADATA.yaml", "")
        metadata = OmegaConf.load(yaml_file)
        if metadata.has_bleed == "yes":
            continue
        genre = metadata.genre
        for stem_id, stem_info in metadata.stems.items():
            for raw_id, raw_info in stem_info.raw.items():
                if raw_info.filename in STOP_RAWS:
                    continue
                if raw_info.instrument in STOP_INST:
                    continue
                raw_info.filename = raw_info.filename.replace(".wav", "")
                raw_path = f"{raw_dir}/{fname}/{fname}_RAW/{raw_info.filename}.flac"
                if os.path.exists(raw_path):
                    raw_info.duration = librosa.get_duration(filename=raw_path)
                    raw_info.fname = fname
                    raw_info.genre = genre
                    if raw_info.duration < 10:
                        continue
                    raw_db.append(raw_info)
    
    db = Dataset.from_list(raw_db)
    db.push_to_hub(f"seungheondoh/medleydb_raw", split="pool", private=True)
    
def balanced_sampling():
    db = load_dataset("seungheondoh/medleydb_raw", split="pool")
    rng = np.random.RandomState(42)
    df = db.to_pandas()
    inst_counts = df["instrument"].value_counts()
    # Get top 50 instruments
    top_inst = inst_counts.nlargest(50).index.tolist()
    # Filter dataframe to only include top 50 instruments 
    df_filtered = df[df['instrument'].isin(top_inst)]
    # Initialize empty list to store sampled rows
    balanced_samples = []
    used_fnames = set()
    # Sample instruments with sufficient data
    samples_per_inst = 10
    for instrument in top_inst:
        inst_df = df_filtered[df_filtered['instrument'] == instrument]
        if len(inst_df) >= samples_per_inst:
            sampled = inst_df.sample(n=samples_per_inst, random_state=rng)
            balanced_samples.append(sampled)
            used_fnames.update(sampled['fname'].tolist())
    balanced_df = pd.concat(balanced_samples)
    print(balanced_df)
    host_hop_sampling = df_filtered[df_filtered['genre'] == "Musical Theatre"].sample(5).to_dict("records")
    original_samples = balanced_df.sample(95).to_dict("records")
    testset = Dataset.from_list(original_samples + host_hop_sampling)
    testset.push_to_hub(f"seungheondoh/medleydb_raw", split="test", private=True)
    
def main():
    process_raw_db()
    balanced_sampling()

    db = load_dataset("seungheondoh/medleydb_raw")
    print(len(db['pool']), len(db['test']))
    # # Check instrument distribution
    # instrument_counts = {}
    # for item in db:
    #     inst = item["instrument"]
    #     instrument_counts[inst] = instrument_counts.get(inst, 0) + 1
    # print("\nInstrument distribution:")
    # for inst, count in sorted(instrument_counts.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{inst}: {count}")
        
    # # Check genre distribution  
    # genre_counts = {}
    # for item in db:
    #     genre = item["genre"]
    #     genre_counts[genre] = genre_counts.get(genre, 0) + 1
    # print("\nGenre distribution:")
    # for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{genre}: {count}")

if __name__ == "__main__":
    main()