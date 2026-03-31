import os
from glob import glob
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm
from llm4mp.common_utils.audio_utils import trim_silence
import argparse
import multiprocessing as mp
parser = argparse.ArgumentParser()
parser.add_argument("--tgt_dir_path", type=str, default='', help="Path to the target directory containing audio files.")
parser.add_argument("--output_dir_path", type=str, default='', help="Path to the output directory where trimmed files will be saved.")
parser.add_argument("--audio_extension", type=str, default="wav", help="Audio file extension to process (default: wav).")
parser.add_argument("--top_db", type=int, default=30, help="Threshold (in dB) below reference to consider as silence (default: 30).")
args = parser.parse_args()

def process_audio(audio_file):
    trimmed_audio, sr = trim_silence(audio_file)
    output_file_path = audio_file.replace(args.tgt_dir_path, args.output_dir_path).replace(".wav", ".flac")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    sf.write(output_file_path, trimmed_audio, sr, format='flac')
    
def main():
    audio_files = glob(os.path.join(args.tgt_dir_path, '**', f"*.{args.audio_extension}"), recursive=True)
    audio_files = [audio_file for audio_file in audio_files if "RAW" in audio_file]
    
    with mp.Pool(processes=mp.cpu_count()-2) as pool:
        list(tqdm(pool.imap(process_audio, audio_files), total=len(audio_files)))

if __name__ == "__main__":
    main()