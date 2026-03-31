import os
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
import librosa
import numpy as np
import soundfile as sf
import multiprocessing as mp

parser = argparse.ArgumentParser()
parser.add_argument("--raw_path", type=str, default="",
                    help="Path to raw MedleyDB audio files")
parser.add_argument("--output_path", type=str, default="")
args = parser.parse_args()

SAMPLE_RATE = 44100

def linear_mix(stem_metadata):
    audio_paths = stem_metadata['stem_wise_audio_path']
    stem_filename = stem_metadata['stem_filename']
    fname = stem_metadata['fname']
    audio_list = [sf.read(audio_path)[0] for audio_path in audio_paths]
    min_duration = min([len(audio) for audio in audio_list])
    audio_list = np.array([audio[:min_duration] for audio in audio_list])
    mixture = np.sum(audio_list, axis=0) # mono audio
    dry_stem_filename = stem_filename.replace('.wav', '.flac')
    os.makedirs(f"{args.output_path}/{fname}/{fname}_STEMS", exist_ok=True)
    sf.write(f"{args.output_path}/{fname}/{fname}_STEMS/{dry_stem_filename}", mixture, SAMPLE_RATE, format='flac')

def main():
    stem_metadata = []
    for fname in tqdm(os.listdir(args.raw_path)):
        metadata = OmegaConf.load(f"{args.raw_path}/{fname}/{fname}_METADATA.yaml")        
        for stem_name, stem_info in metadata['stems'].items():
            stem_wise_audio_path = []
            for raw_name, raw_info in stem_info['raw'].items():
                stem_wise_audio_path.append(f"{args.raw_path}/{fname}/{fname}_RAW/{raw_info['filename']}")
            stem_metadata.append({
                'fname': fname,
                'stem_name': stem_name,
                'stem_filename': stem_info['filename'],
                'stem_wise_audio_path': stem_wise_audio_path,
            })
    print(len(stem_metadata)) # Total STEM Dataset : 2091
    with mp.Pool(processes=mp.cpu_count()-2) as pool:
        pool.map(linear_mix, stem_metadata)

if __name__ == "__main__":
    main()