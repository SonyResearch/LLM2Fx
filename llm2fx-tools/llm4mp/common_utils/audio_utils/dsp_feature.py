import librosa
import numpy as np
import json
from pyroomacoustics import measure_rt60
import os
from sklearn.metrics.pairwise import cosine_similarity
import pyloudnorm as pyln

def dsp_similarity(pred_wet_audio, gt_wet_audio, sr):
    pred_features = extract_audio_features(pred_wet_audio.numpy(), sr)
    gt_features = extract_audio_features(gt_wet_audio.numpy(), sr)
    feature_keys = list(pred_features.keys())
    pred_feature_values = np.array([pred_features[key] for key in feature_keys])
    gt_feature_values = np.array([gt_features[key] for key in feature_keys])
    pred_feature_values = pred_feature_values.reshape(1, -1)
    gt_feature_values = gt_feature_values.reshape(1, -1)
    dsp_sim = cosine_similarity(pred_feature_values, gt_feature_values)[0][0]
    return dsp_sim

def extract_audio_features(y, sr):
    rms = librosa.feature.rms(y=y)[0].mean()
    peak = np.abs(y).max()
    crest_factor = peak / (np.sqrt(np.mean(np.square(y))) + 1e-8)
    mean_amplitude = np.mean(np.abs(y))
    dynamic_spread = np.sqrt(np.mean((np.abs(y) - mean_amplitude) ** 2))
    spec = np.abs(librosa.stft(y))
    spectral_centroid = librosa.feature.spectral_centroid(S=spec)[0].mean()
    spectral_flatness = librosa.feature.spectral_flatness(S=spec)[0].mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=spec)[0].mean()
    rt60 = measure_rt60(y, sr)
    meter = pyln.Meter(sr)  # create BS.1770 meter
    loudness = meter.integrated_loudness(y)    
    return {
        'loudness': float(loudness),
        'rms_energy': float(rms),
        'crest_factor': float(crest_factor),
        'dynamic_spread': float(dynamic_spread),
        'spectral_centroid': float(spectral_centroid),
        'spectral_flatness': float(spectral_flatness),
        'spectral_bandwidth': float(spectral_bandwidth),
        "reverberation_time_60": float(rt60)
    }