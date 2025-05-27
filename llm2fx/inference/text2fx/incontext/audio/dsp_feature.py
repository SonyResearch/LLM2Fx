import librosa
import numpy as np
import json
import os
def extract_audio_features(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    # Calculate RMS energy
    rms = librosa.feature.rms(y=y)[0].mean()
    # Calculate crest factor
    peak = np.abs(y).max()
    crest_factor = peak / (np.sqrt(np.mean(np.square(y))) + 1e-8)
    # Calculate dynamic spread
    percentile_95 = np.percentile(np.abs(y), 95)
    percentile_5 = np.percentile(np.abs(y), 5)
    dynamic_spread = percentile_95 - percentile_5
    # Calculate spectral features
    spec = np.abs(librosa.stft(y))
    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(S=spec)[0].mean()
    # Spectral flatness
    spectral_flatness = librosa.feature.spectral_flatness(S=spec)[0].mean()
    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=spec)[0].mean()
    return {
        "sample_rate": sr,
        'rms_energy': round(float(rms), 3),
        'crest_factor': round(float(crest_factor), 3),
        'dynamic_spread': round(float(dynamic_spread), 3),
        'spectral_centroid': round(float(spectral_centroid), 3),
        'spectral_flatness': round(float(spectral_flatness), 3),
        'spectral_bandwidth': round(float(spectral_bandwidth), 3)
    }

if __name__ == "__main__":
    audio_dir = "/home/seungheon/llm4mp/llm4mp/preprocessing/fx_processor/dsp_utils/socialfx/original/raw/audio"
    for inst in ["drums", "guitar", "piano"]:
        audio_path = f"{audio_dir}/{inst}.wav"
        features = extract_audio_features(audio_path)
        with open(f"./{inst}.json", "w") as f:
            json.dump(features, f, indent=4)
