import os
from glob import glob
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm
# trim silence from the input wav file path and merge it into one wav data
def trim_silence(input_file_path, top_db=30):
    # load and convert to mono channel
    cur_aud, sample_rate = librosa.load(input_file_path, sr=None, mono=True)
    # trim silence
    cur_aud_non_silence_sections = librosa.effects.split(cur_aud, \
                                                            top_db=top_db, \
                                                            frame_length=sample_rate, \
                                                            hop_length=sample_rate//2)
    # merge non-silence parts together
    for cur_idx, cur_section in enumerate(cur_aud_non_silence_sections):
        cur_start, cur_end = cur_section
        cur_silence_trimmed_wav, cur_sr = sf.read(input_file_path, \
                                                    start=cur_start, \
                                                    frames=cur_end-cur_start)
        silence_trimmed_wav = cur_silence_trimmed_wav if cur_idx==0 else np.concatenate((silence_trimmed_wav, cur_silence_trimmed_wav), axis=0)
    return silence_trimmed_wav, cur_sr