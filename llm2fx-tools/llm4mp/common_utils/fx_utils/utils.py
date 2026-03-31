import math
import random
import librosa
import numpy as np
def normalize_param(x, low: float, high: float):
    return (x - low) / (high - low)

def denormalize_param(x, low: float, high: float):
    return x * (high - low) + low

def get_random_wet_audio(dry_audio, fx_chain, fx_config):
    random_fx_manipulation = fx_config['fx_random_module_dict']
    for fx in fx_chain:
        fx_name = fx['name']
        fx_function = random_fx_manipulation[fx_name]
        wet_audio, _, _ = fx_function(dry_audio)
    return wet_audio

def get_dry_audio(audio_dir, fname, filename, sampling=True, target_duration=10):
    raw_path = f"{audio_dir}/{fname}/{fname}_RAW/{filename}.flac"
    audio_dry, sr = librosa.load(raw_path, sr=None, mono=False)
    duration = audio_dry.shape[1] / sr
    if sampling:
        if duration > target_duration:
            random_start = random.randint(0, int(duration - target_duration) * sr)
            audio_dry = audio_dry[:, random_start:random_start + target_duration * sr]
        else:
            audio_dry = np.pad(audio_dry, (0, int(target_duration * sr - audio_dry.shape[1])), mode='constant')
    else: # split and make a pre-batch
        num_chunks = math.ceil(duration / target_duration)
        chunks = []
        for i in range(num_chunks):
            start = i * target_duration * sr
            end = start + target_duration * sr
            chunks.append(audio_dry[:, start:end])
        audio_dry = np.array(chunks[:-1])
    return audio_dry

def get_wet_audio(dry_audio, fx_chain, fx_config):
    wet_audio = dry_audio.copy()
    for fx in fx_chain:
        fx_name = fx['name']
        fx_params = fx['params']
        fx_function = fx_config['fx_function_dict'][fx_name]
        wet_audio = fx_function(wet_audio, **fx_params)
    return wet_audio