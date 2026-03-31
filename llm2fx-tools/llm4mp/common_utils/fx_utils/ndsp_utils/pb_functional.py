# Adapted from https://github.com/aim-qmul/sdx23-aimless/blob/master/data/augment.py
import torch
import random
import pedalboard
import torchaudio
import numpy as np
import scipy.stats
import scipy.signal
import dasp_pytorch
import pyloudnorm as pyln
from typing import List, Tuple

import numpy as np
import math
import scipy.signal
from scipy.fft import rfft, irfft
from functools import partial

from pedalboard import (
    Pedalboard,
    Gain,
    Reverb,
    Compressor,
    Delay,
    Distortion,
    Limiter,
    NoiseGate,
)

from .constants import pedalboard_fx_param_ranges

def normalize_param(x, low: float, high: float):
    return (x - low) / (high - low)

def denormalize_param(x, low: float, high: float):
    return x * (high - low) + low

def db2linear(x):
    return 10 ** (x / 20)

def loguniform(low=0, high=1):
    return scipy.stats.loguniform.rvs(low, high)


def rand(low=0, high=1):
    return (random.random() * (high - low)) + low

def randint(low=0, high=1):
    return random.randint(low, high)

def pedalboard_distortion(x: np.ndarray, drive_db: float, sample_rate: int=44100):
    board = pedalboard.Pedalboard([pedalboard.Distortion(drive_db=drive_db)])
    output = board(x, sample_rate)
    return output

def pedalboard_reverb(x: np.ndarray, room_size: float, width:float, damping: float, mix_ratio: float, sample_rate: int=44100):
    dry_level = 1 - mix_ratio
    board = pedalboard.Pedalboard(
        [
            pedalboard.Reverb(
                room_size=room_size,
                width=width,
                damping=damping,
                wet_level=mix_ratio,
                dry_level=dry_level,
            )
        ]
    )
    output = board(x, sample_rate)
    return output

def pedalboard_compressor(x: np.ndarray, threshold_db: float, ratio: float, attack_ms: float, release_ms: float, sample_rate: int=44100):
    board = pedalboard.Pedalboard([pedalboard.Compressor(threshold_db=threshold_db, ratio=ratio, attack_ms=attack_ms, release_ms=release_ms)])
    output = board(x, sample_rate)
    return output

def pedalboard_delay(x: np.ndarray, delay_seconds: float, feedback: float, mix_ratio: float, sample_rate: int=44100):
    board = pedalboard.Pedalboard([pedalboard.Delay(delay_seconds=delay_seconds, feedback=feedback, mix=mix_ratio)])
    output = board(x, sample_rate)
    return output

def pedalboard_limiter(x: np.ndarray, threshold_db: float, release_ms:float, sample_rate: int=44100):
    board = pedalboard.Pedalboard([pedalboard.Limiter(threshold_db=threshold_db, release_ms=release_ms)])
    output = board(x, sample_rate)
    return output

def pedalboard_gain(x: np.ndarray, gain_db: float, sample_rate: int=44100):
    board = pedalboard.Pedalboard([pedalboard.Gain(gain_db=gain_db)])
    output = board(x, sample_rate)
    return output

def pedalboard_noise_gate(x: np.ndarray, threshold_db: float, ratio: float, attack_ms:float, release_ms:float, sample_rate: int=44100):
    board = pedalboard.Pedalboard([pedalboard.NoiseGate(
            threshold_db=threshold_db, 
            ratio=ratio, 
            attack_ms=attack_ms, 
            release_ms=release_ms,    
    )])
    output = board(x, sample_rate)
    return output

def apply_random_pedalboard_gain(
    x: np.ndarray,
    sample_rate: int = 44100,
    min_gain_db: float = -20.0,
    max_gain_db: float = 20.0,
):
    board = Pedalboard()
    gain_db = np.random.rand(1) * (max_gain_db - min_gain_db) + min_gain_db
    board.append(Gain(gain_db=float(gain_db)))
    y = board(x, sample_rate)
    params = {"gain_db": float(normalize_param(gain_db, min_gain_db, max_gain_db))}
    denorm_params = {"gain_db": float(gain_db)}
    return y, params, denorm_params

# def apply_random_pedalboard_noise_gate(
#     x: np.ndarray,
#     sample_rate: int = 44100,
#     min_threshold_db: float = pedalboard_fx_param_ranges["noise_gate"]["threshold_db"]["min"],
#     max_threshold_db: float = pedalboard_fx_param_ranges["noise_gate"]["threshold_db"]["max"],
#     min_ratio: float = pedalboard_fx_param_ranges["noise_gate"]["ratio"]["min"],
#     max_ratio: float = pedalboard_fx_param_ranges["noise_gate"]["ratio"]["max"],
#     min_attack_ms: float = pedalboard_fx_param_ranges["noise_gate"]["attack_ms"]["min"],
#     max_attack_ms: float = pedalboard_fx_param_ranges["noise_gate"]["attack_ms"]["max"],
#     min_release_ms: float = pedalboard_fx_param_ranges["noise_gate"]["release_ms"]["min"],
#     max_release_ms: float = pedalboard_fx_param_ranges["noise_gate"]["release_ms"]["max"],
# ):
#     board = Pedalboard()
#     threshold_db = np.random.rand(1) * (max_threshold_db - min_threshold_db) + min_threshold_db
#     ratio = np.random.rand(1) * (max_ratio - min_ratio) + min_ratio
#     attack_ms = np.random.rand(1) * (max_attack_ms - min_attack_ms) + min_attack_ms
#     release_ms = np.random.rand(1) * (max_release_ms - min_release_ms) + min_release_ms
#     board.append(NoiseGate(threshold_db=float(threshold_db), ratio=float(ratio), attack_ms=float(attack_ms), release_ms=float(release_ms)))
#     y = board(x, sample_rate)
#     params = {
#         "threshold_db": float(normalize_param(threshold_db, min_threshold_db, max_threshold_db)), 
#         "ratio": float(normalize_param(ratio, min_ratio, max_ratio)),
#         "attack_ms": float(normalize_param(attack_ms, min_attack_ms, max_attack_ms)),
#         "release_ms": float(normalize_param(release_ms, min_release_ms, max_release_ms)),
#     }
#     denorm_params = {"threshold_db": float(threshold_db), "ratio": float(ratio), "attack_ms": float(attack_ms), "release_ms": float(release_ms)}
#     return y, params, denorm_params

def apply_random_pedalboard_noise_gate():
    return None

def apply_random_pedalboard_limiter(
    x: np.ndarray,
    sample_rate: int = 44100,
    min_threshold_db: float = pedalboard_fx_param_ranges["limiter"]["threshold_db"]["min"],
    max_threshold_db: float = pedalboard_fx_param_ranges["limiter"]["threshold_db"]["max"],
    min_release_ms: float = pedalboard_fx_param_ranges["limiter"]["release_ms"]["min"],
    max_release_ms: float = pedalboard_fx_param_ranges["limiter"]["release_ms"]["max"],
):
    board = Pedalboard()
    threshold_db = np.random.rand(1) * (max_threshold_db - min_threshold_db) + min_threshold_db
    release_ms = np.random.rand(1) * (max_release_ms - min_release_ms) + min_release_ms
    board.append(Limiter(threshold_db=float(threshold_db), release_ms=float(release_ms)))
    y = board(x, sample_rate)
    params = {"threshold_db": float(normalize_param(threshold_db, min_threshold_db, max_threshold_db)), "release_ms": float(normalize_param(release_ms, min_release_ms, max_release_ms))}
    denorm_params = {"threshold_db": float(threshold_db), "release_ms": float(release_ms)}
    return y, params, denorm_params

def apply_random_stereo_widener(
    x: np.ndarray,
    sample_rate: int = 44100,
    min_width: float = pedalboard_fx_param_ranges["stereo_widener"]["width"]["min"],
    max_width: float = pedalboard_fx_param_ranges["stereo_widener"]["width"]["max"],
):
    width = rand(min_width, max_width)
    y = stereo_widener(x, width)
    params = {"width": float(normalize_param(width, min_width, max_width))}
    denorm_params = {"width": float(width)}
    return y, params, denorm_params

def apply_random_panner(
    x: np.ndarray,
    sample_rate: int = 44100,
    min_pan: float = pedalboard_fx_param_ranges["panner"]["pan"]["min"],
    max_pan: float = pedalboard_fx_param_ranges["panner"]["pan"]["max"],
):
    pan = rand(min_pan, max_pan)
    y = stereo_panner(x, pan)
    params = {"pan": float(normalize_param(pan, min_pan, max_pan))}
    denorm_params = {"pan": float(pan)}
    return y, params, denorm_params

def apply_random_three_band_eq(
    x: np.ndarray,
    sample_rate: int = 44100,
    min_gain_db: float = pedalboard_fx_param_ranges["equalizer"]["low_gain_db"]["min"],
    max_gain_db: float = pedalboard_fx_param_ranges["equalizer"]["low_gain_db"]["max"],
    min_q_factor: float = pedalboard_fx_param_ranges["equalizer"]["low_q_factor"]["min"],
    max_q_factor: float = pedalboard_fx_param_ranges["equalizer"]["low_q_factor"]["max"],
):
    # Frequency ranges for three-band EQ
    low_shelf_cutoff_freq_min, low_shelf_cutoff_freq_max = pedalboard_fx_param_ranges["equalizer"]["low_cutoff_freq"]["min"], pedalboard_fx_param_ranges["equalizer"]["low_cutoff_freq"]["max"]
    mid_shelf_cutoff_freq_min, mid_shelf_cutoff_freq_max = pedalboard_fx_param_ranges["equalizer"]["mid_cutoff_freq"]["min"], pedalboard_fx_param_ranges["equalizer"]["mid_cutoff_freq"]["max"]
    high_shelf_cutoff_freq_min, high_shelf_cutoff_freq_max = pedalboard_fx_param_ranges["equalizer"]["high_cutoff_freq"]["min"], pedalboard_fx_param_ranges["equalizer"]["high_cutoff_freq"]["max"]

    # Randomly sample parameters for each band
    low_gain_db = rand(min_gain_db, max_gain_db)
    low_cutoff_freq = rand(low_shelf_cutoff_freq_min, low_shelf_cutoff_freq_max)
    low_q_factor = rand(min_q_factor, max_q_factor)

    mid_gain_db = rand(min_gain_db, max_gain_db)
    mid_cutoff_freq = rand(mid_shelf_cutoff_freq_min, mid_shelf_cutoff_freq_max)
    mid_q_factor = rand(min_q_factor, max_q_factor)

    high_gain_db = rand(min_gain_db, max_gain_db)
    high_cutoff_freq = rand(high_shelf_cutoff_freq_min, high_shelf_cutoff_freq_max)
    high_q_factor = rand(min_q_factor, max_q_factor)
    # Apply the three-band parametric EQ
    y = parametric_eq(
        x,
        low_gain_db, low_cutoff_freq, low_q_factor,
        mid_gain_db, mid_cutoff_freq, mid_q_factor,
        high_gain_db, high_cutoff_freq, high_q_factor,
        sample_rate,
    )

    params = {
        "low_gain_db": float(normalize_param(low_gain_db, min_gain_db, max_gain_db)),
        "low_cutoff_freq": float(normalize_param(low_cutoff_freq, low_shelf_cutoff_freq_min, low_shelf_cutoff_freq_max)),
        "low_q_factor": float(normalize_param(low_q_factor, min_q_factor, max_q_factor)),
        "mid_gain_db": float(normalize_param(mid_gain_db, min_gain_db, max_gain_db)),
        "mid_cutoff_freq": float(normalize_param(mid_cutoff_freq, mid_shelf_cutoff_freq_min, mid_shelf_cutoff_freq_max)),
        "mid_q_factor": float(normalize_param(mid_q_factor, min_q_factor, max_q_factor)),
        "high_gain_db": float(normalize_param(high_gain_db, min_gain_db, max_gain_db)),
        "high_cutoff_freq": float(normalize_param(high_cutoff_freq, high_shelf_cutoff_freq_min, high_shelf_cutoff_freq_max)),
        "high_q_factor": float(normalize_param(high_q_factor, min_q_factor, max_q_factor)),
    }

    denorm_params = {
        "low_gain_db": float(low_gain_db),
        "low_cutoff_freq": float(low_cutoff_freq),
        "low_q_factor": float(low_q_factor),
        "mid_gain_db": float(mid_gain_db),
        "mid_cutoff_freq": float(mid_cutoff_freq),
        "mid_q_factor": float(mid_q_factor),
        "high_gain_db": float(high_gain_db),
        "high_cutoff_freq": float(high_cutoff_freq),
        "high_q_factor": float(high_q_factor),
    }

    return y, params, denorm_params

def apply_random_pedalboard_distortion(
    x: np.ndarray,
    sample_rate: float = 44100.0,
    min_drive_db: float = pedalboard_fx_param_ranges["distortion"]["drive_db"]["min"],
    max_drive_db: float = pedalboard_fx_param_ranges["distortion"]["drive_db"]["max"],
):
    board = Pedalboard()
    drive_db = np.random.rand(1) * (max_drive_db - min_drive_db) + min_drive_db
    board.append(Distortion(drive_db=float(drive_db)))
    params = {"drive_db": float(normalize_param(drive_db, min_drive_db, max_drive_db))}
    denorm_params =  {"drive_db": float(drive_db)}
    return board(x, sample_rate), params, denorm_params

def apply_random_pedalboard_delay(
    x: np.ndarray,
    sample_rate: float = 44100.0,
    min_delay_seconds: float = pedalboard_fx_param_ranges["delay"]["delay_seconds"]["min"],
    max_delay_seconds: float = pedalboard_fx_param_ranges["delay"]["delay_seconds"]["max"],
    min_feedback: float = pedalboard_fx_param_ranges["delay"]["feedback"]["min"],
    max_feedback: float = pedalboard_fx_param_ranges["delay"]["feedback"]["max"],
    min_mix: float = pedalboard_fx_param_ranges["delay"]["mix_ratio"]["min"],
    max_mix: float = pedalboard_fx_param_ranges["delay"]["mix_ratio"]["max"],
):
    board = Pedalboard()
    delay_seconds = rand(min_delay_seconds, max_delay_seconds)
    feedback = rand(min_feedback, max_feedback)
    mix = rand(min_mix, max_mix)
    board.append(Delay(delay_seconds=delay_seconds, feedback=feedback, mix=mix))
    y = board(x, sample_rate)
    params = {
        "delay_seconds": float(normalize_param(delay_seconds, min_delay_seconds, max_delay_seconds)),
        "feedback": float(normalize_param(feedback, min_feedback, max_feedback)),
        "mix_ratio": float(normalize_param(mix, min_mix, max_mix)),
    }
    denorm_params =  {"delay_seconds": float(delay_seconds), "feedback": float(feedback), "mix_ratio": float(mix)}
    return y, params, denorm_params

def apply_random_pedalboard_reverb(
    x: np.ndarray,
    sample_rate: float = 44100.0,
    min_room_size: float = pedalboard_fx_param_ranges["reverb"]["room_size"]["min"],
    max_room_size: float = pedalboard_fx_param_ranges["reverb"]["room_size"]["max"],
    min_damping: float = pedalboard_fx_param_ranges["reverb"]["damping"]["min"],
    max_damping: float = pedalboard_fx_param_ranges["reverb"]["damping"]["max"],
    min_wet_dry: float = pedalboard_fx_param_ranges["reverb"]["mix_ratio"]["min"],
    max_wet_dry: float = pedalboard_fx_param_ranges["reverb"]["mix_ratio"]["max"],
    min_width: float = pedalboard_fx_param_ranges["reverb"]["width"]["min"],
    max_width: float = pedalboard_fx_param_ranges["reverb"]["width"]["max"],    
):
    board = Pedalboard()
    room_size = rand(min_room_size, max_room_size)
    damping = rand(min_damping, max_damping)
    wet_dry = rand(min_wet_dry, max_wet_dry)
    width = rand(min_width, max_width)
    board.append(
        Reverb(
            room_size=room_size,
            damping=damping,
            wet_level=wet_dry,
            dry_level=(1 - wet_dry),
            width=width,
        )
    )
    y = board(x, sample_rate)
    params = {
        "room_size": float(normalize_param(room_size, min_room_size, max_room_size)),
        "damping": float(normalize_param(damping, min_damping, max_damping)),
        "width": float(normalize_param(width, min_width, max_width)),
        "mix_ratio": float(normalize_param(wet_dry, min_wet_dry, max_wet_dry)),
    }
    denorm_params =  {"room_size": float(room_size), "damping": float(damping), "width": float(width), "mix_ratio": float(wet_dry)}
    return y, params, denorm_params

def apply_random_pedalboard_compressor(
    x: np.ndarray,
    sample_rate: float = 44100.0,
    min_threshold_db: float = pedalboard_fx_param_ranges["compressor"]["threshold_db"]["min"],
    max_threshold_db: float = pedalboard_fx_param_ranges["compressor"]["threshold_db"]["max"],
    min_ratio: float = pedalboard_fx_param_ranges["compressor"]["ratio"]["min"],
    max_ratio: float = pedalboard_fx_param_ranges["compressor"]["ratio"]["max"],
    min_attack_ms: float = pedalboard_fx_param_ranges["compressor"]["attack_ms"]["min"],
    max_attack_ms: float = pedalboard_fx_param_ranges["compressor"]["attack_ms"]["max"],
    min_release_ms: float = pedalboard_fx_param_ranges["compressor"]["release_ms"]["min"],
    max_release_ms: float = pedalboard_fx_param_ranges["compressor"]["release_ms"]["max"],
):
    board = Pedalboard()
    threshold_db = rand(min_threshold_db, max_threshold_db)
    ratio = rand(min_ratio, max_ratio)
    attack_ms = rand(min_attack_ms, max_attack_ms)
    release_ms = rand(min_release_ms, max_release_ms)
    board.append(
        Compressor(
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
        )
    )

    y = board(x, sample_rate)

    params = {
        "threshold_db": float(normalize_param(threshold_db, min_threshold_db, max_threshold_db)),
        "ratio": float(normalize_param(ratio, min_ratio, max_ratio)),
        "attack_ms": float(normalize_param(attack_ms, min_attack_ms, max_attack_ms)),
        "release_ms": float(normalize_param(release_ms, min_release_ms, max_release_ms)),
    }
    denorm_params =  {"threshold_db": float(threshold_db), "ratio": float(ratio), "attack_ms": float(attack_ms), "release_ms": float(release_ms)}
    return y, params, denorm_params

def denormalize(p: np.ndarray, min_val: float, max_val: float):
    """When p is on (0,1) restore the original range of the parameter values.

    Args:
        p (np.ndarray): [bs, num_params]
        min_val (float): minimum value of the parameter
        max_val (float): maximum value of the parameter

    Returns:
        np.ndarray: [bs, num_params]
    """
    return p * (max_val - min_val) + min_val

def stereo_widener(x: np.ndarray, width: float):
    """
    Stereo width control using Mid-Side processing.
    Args:
        x (np.ndarray): Stereo input array of shape (2, N)
        width (float): Stereo width factor. 0.0 = mono, 1.0 = original, >1.0 = wider
    Returns:
        np.ndarray: Stereo output array of shape (2, N)
    """
    if x.shape[0] == 1:
        x = np.vstack((x, x))
    assert x.ndim == 2 and x.shape[0] == 2, "x must be stereo"
    # Mid-Side encoding
    left = x[0, ...]
    right = x[1, ...]
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    # Apply width control to side signal
    side_processed = side * width
    # Mid-Side decoding
    left_out = mid + side_processed
    right_out = mid - side_processed
    x_imaged = np.stack((left_out, right_out), axis=0)
    return x_imaged

def stereo_panner(x: np.ndarray, pan: float):
    """
    Constant power stereo panning.
    Args:
        x (np.ndarray): Stereo input array of shape (2, N)
        pan (float): Pan position from -1 (left) to 1 (right)
    Returns:
        np.ndarray: Stereo output array of shape (2, N)
    """
    if x.shape[0] == 1:
        x = np.vstack((x, x))
    assert x.ndim == 2 and x.shape[0] == 2, "x must be stereo"
    # Convert pan from [-1, 1] to [0, pi/2]
    pan_angle = (pan + 1.0) * np.pi / 4.0
    left_gain = np.cos(pan_angle)
    right_gain = np.sin(pan_angle)
    # Apply constant power panning law
    left = x[0] * left_gain
    right = x[1] * right_gain
    x_panned = np.stack((left, right), axis=0)
    return x_panned

def parametric_eq(
    x,  # Shape: (channels, seq_len)
    low_gain_db,
    low_cutoff_freq,
    low_q_factor,
    mid_gain_db,
    mid_cutoff_freq,
    mid_q_factor,
    high_gain_db,
    high_cutoff_freq,
    high_q_factor,
    sample_rate=44100,
):
    """Three-band Parametric Equalizer for audio signal processing.
    sample_rate: 44100
    min_gain_db: -20.0
    max_gain_db: 20.0
    min_q_factor: 0.1
    max_q_factor: 6.0
    param_ranges = {
        "low_gain_db": (min_gain_db, max_gain_db),
        "low_cutoff_freq": (20, 2000),
        "low_q_factor": (min_q_factor, max_q_factor),
        "mid_gain_db": (min_gain_db, max_gain_db),
        "mid_cutoff_freq": (80, (sample_rate // 2) - 1000),
        "mid_q_factor": (min_q_factor, max_q_factor),
        "high_gain_db": (min_gain_db, max_gain_db),
        "high_cutoff_freq": (4000, (sample_rate // 2) - 1000),
        "high_q_factor": (min_q_factor, max_q_factor),
    }
    This function implements a parametric EQ with three bands in the following configuration:
    Low-shelf -> Peaking Band -> High-shelf
    Each band is implemented as a biquad filter with adjustable gain, frequency, and Q-factor,
    allowing control over the audio's frequency response. The EQ is implemented
    as a cascade of second-order sections for optimal numerical stability.
    Args:
        x (numpy.ndarray): Time domain audio array with shape (channels, sequence_length)
        sample_rate (float): Audio sample rate in Hz.
        low_gain_db (float): Low-shelf filter gain in dB. Controls bass boost/cut.
        low_cutoff_freq (float): Low-shelf filter cutoff frequency in Hz.
        low_q_factor (float): Low-shelf filter Q-factor. Controls transition steepness.
        mid_gain_db (float): Peaking band filter gain in dB.
        mid_cutoff_freq (float): Peaking band filter center frequency in Hz.
        mid_q_factor (float): Peaking band filter Q-factor. Controls bandwidth.
        high_gain_db (float): High-shelf filter gain in dB. Controls treble boost/cut.
        high_cutoff_freq (float): High-shelf filter cutoff frequency in Hz.
        high_q_factor (float): High-shelf filter Q-factor. Controls transition steepness.
    Returns:
        y (numpy.ndarray): Filtered audio signal with the same shape as the input.
    """
    chs, seq_len = x.shape
    sos = np.zeros((3, 6))
    b, a = biquad(
        low_gain_db,
        low_cutoff_freq,
        low_q_factor,
        sample_rate,
        "low_shelf",
    )
    sos[0, :] = np.concatenate((b, a))
    b, a = biquad(
        mid_gain_db,
        mid_cutoff_freq,
        mid_q_factor,
        sample_rate,
        "peaking",
    )
    sos[1, :] = np.concatenate((b, a))
    b, a = biquad(
        high_gain_db,
        high_cutoff_freq,
        high_q_factor,
        sample_rate,
        "high_shelf",
    )
    sos[2, :] = np.concatenate((b, a))

    x_out = sosfilt_via_fsm(sos, x)

    return x_out

def biquad(
    gain_db: float,
    cutoff_freq: float,
    q_factor: float,
    sample_rate: float,
    filter_type: str = "peaking",
):
    """Implements a digital biquad filter with various filter types for audio processing.

    A biquad filter is a second-order recursive linear filter with two poles and two zeros.
    This implementation includes several common audio filter types used in digital signal processing.

    Args:
        gain_db (float): Gain in decibels.
            For peaking and shelving filters, this controls the boost/cut amount.
            For lowpass/highpass filters, this parameter is used but typically set to 0dB.

        cutoff_freq (float): Filter cutoff/center frequency in Hz.
            This is the frequency at which the filter takes effect.

        q_factor (float): Quality factor.
            Controls filter bandwidth: higher values = narrower filter

        sample_rate (float): Audio sample rate in Hz.
            Used to normalize frequency for digital filter computation.

        filter_type (str, optional): Type of biquad filter to implement. Defaults to "peaking".
            Supported types:
            - "peaking": Bell/parametric EQ filter that boosts/cuts around center frequency
            - "low_shelf": Boosts/cuts frequencies below cutoff, flat above
            - "high_shelf": Boosts/cuts frequencies above cutoff, flat below
            - "low_pass": Passes signals below cutoff, attenuates above
            - "high_pass": Passes signals above cutoff, attenuates below

    Returns:
        tuple: A tuple containing:
            - b (numpy.ndarray): Numerator coefficients [b0, b1, b2].
            - a (numpy.ndarray): Denominator coefficients [a0, a1, a2] where a0 = 1.
    """
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * math.pi * (cutoff_freq / sample_rate)
    alpha = math.sin(w0) / (2 * q_factor)
    cos_w0 = math.cos(w0)
    sqrt_A = math.sqrt(A)
    if filter_type == "high_shelf":
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "low_shelf":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + (alpha / A)
        a1 = -2 * cos_w0
        a2 = 1 - (alpha / A)
    elif filter_type == "low_pass":
        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = (1 - cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    elif filter_type == "high_pass":
        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = (1 + cos_w0) / 2
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return b, a

def fft_freqz(b, a, n_fft=512):
    """Compute frequency response using FFT method."""
    B = rfft(b, n_fft)
    A = rfft(a, n_fft)
    H = B / A
    return H

def fft_sosfreqz(sos, n_fft=512):
    """Compute the complex frequency response via FFT of cascade of biquads.
    Args:
        sos (numpy.ndarray): Second order filter sections with shape (n_sections, 6)
        n_fft (int): FFT size. Default: 512

    Returns:
        H (numpy.ndarray): Overall complex frequency response
    """
    n_sections = sos.shape[0]
    H = None

    for section_idx in range(n_sections):
        b = sos[section_idx, :3]
        a = sos[section_idx, 3:]
        if section_idx == 0:
            H = fft_freqz(b, a, n_fft=n_fft)
        else:
            H *= fft_freqz(b, a, n_fft=n_fft)
    return H

def freqdomain_fir(x, H, n_fft):
    """Filter signal in frequency domain."""
    X = rfft(x, n_fft)
    Y = X * H
    y = irfft(Y, n_fft)
    return y

def sosfilt_via_fsm(sos, x):
    """Use the frequency sampling method to approximate a cascade of second order IIR filters.

    Args:
        sos (numpy.ndarray): Tensor of coefficients with shape (n_sections, 6).
        x (numpy.ndarray): Time domain signal with shape (channels, timesteps)

    Returns:
        y (numpy.ndarray): Filtered time domain signal with shape (channels, timesteps)
    """
    n_fft = 2 ** math.ceil(math.log2(x.shape[-1] + x.shape[-1] - 1))
    n_fft = int(n_fft)
    H = fft_sosfreqz(sos, n_fft=n_fft)
    if x.ndim > 1:
        H = H[np.newaxis, :]
    y = freqdomain_fir(x, H, n_fft)
    y = y[..., :x.shape[-1]]
    return y