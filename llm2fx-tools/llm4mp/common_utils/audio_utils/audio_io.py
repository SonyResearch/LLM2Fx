import soundfile as sf
import numpy as np
import librosa

def load_audio(input_path, channel_first=True, random_sample=True, duration=10):
    y, sr = sf.read(input_path)
    if random_sample:
        start_idx = np.random.randint(0, len(y) - int(duration*sr))
        y = y[start_idx:start_idx+int(duration*sr)]
    if channel_first and y.ndim > 1:
        y = y.T
    return y, sr

def save_audio(data, samplerate, output_path, audio_format="flac"):
    if data.shape[0] == 2 or data.shape[0] == 1:
        data = data.T # (2, N) -> (N, 2)
    sf.write(output_path, data, samplerate, format=audio_format)

def get_dry_audio(raw_path, target_duration=10, padding=False):
    dry_audio, sr = librosa.load(raw_path, sr=None, mono=False)
    duration = dry_audio.shape[-1] / sr
    if len(dry_audio.shape) == 1:
        dry_audio = np.stack([dry_audio, dry_audio], axis=0) # simple stero_code
    num_chunks = int(duration // target_duration)
    chunk_samples = int(target_duration * sr)
    if dry_audio.shape[-1] < chunk_samples and padding:
        raise ValueError(f"Dry audio is too short: {raw_path}")
    chunk_audio = []
    for idx, i in enumerate(range(num_chunks)):
        chunk_audio.append(dry_audio[:, i * chunk_samples:(i + 1) * chunk_samples])
    return chunk_audio

def get_wet_audio(dry_audio, fx_chain, fx_config):
    wet_audio = dry_audio.copy()
    for fx in fx_chain:
        fx_name = fx['name']
        fx_params = fx['arguments']
        fx_function = fx_config['fx_function_dict'][fx_name]
        wet_audio = fx_function(wet_audio, **fx_params)
    return wet_audio